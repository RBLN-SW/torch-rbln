# Owner(s): ["module: PrivateUse1"]

"""Unit tests for the RBLN Generator initial implementation.

Coverage:
  1. Seed initialization — ``manual_seed`` correctly sets the generator's
     initial seed, which is observable through ``initial_seed``.

  2. State save and restore — the generator state can be saved and restored
     to reproduce the same random number sequence, both through explicit
     ``Generator`` objects and through the per-device default generators
     (``torch.rbln.get_rng_state`` / ``torch.rbln.set_rng_state``).

  3. Reproducibility — an RBLN generator initialized with the same seed as a
     CPU generator produces the same random sequence for the tested random
     operation; default-generator ops (``generator=None``) reproduce after
     ``torch.rbln.manual_seed``.

  4. Isolation — RBLN random ops do not consume the global CPU RNG stream,
     and CPU random ops do not consume the RBLN default generator stream.

  5. Validation — generators from another backend or wrong device index are
     rejected with a clear
     error; malformed RNG state blobs (wrong device, wrong dtype, too short,
     wrong size, non-contiguous) are rejected, and a failed ``set_state``
     leaves the generator unmodified.

  6. Multi-device — per-device default generators are independent, and
     ``manual_seed_all`` seeds every device.

  7. Integration — ``torch.random.fork_rng(device_type="rbln")`` saves and
     restores the default generator state.
"""

from __future__ import annotations

import threading
import unittest

import pytest
import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase

import torch_rbln  # noqa: F401  (registers the rbln backend)


RBLN_DEVICE_COUNT = torch_rbln._C.device_count()


def _sample(device="rbln", generator=None, n=16):
    """One canonical fallback-path random op used across the tests."""
    return torch.randint(0, 1024, (n,), device=device, generator=generator)


@pytest.mark.test_set_ci
class TestGeneratorBasic(TestCase):
    """Basic tests for the RBLN ``torch.Generator`` implementation."""

    def test_manual_seed(self):
        torch.manual_seed(42)

        self.assertEqual(torch_rbln._C.get_default_generator().initial_seed(), 42)

    def test_generator_set_seed(self):
        g = torch.Generator(device="rbln").manual_seed(4242)

        self.assertEqual(g.initial_seed(), 4242)

    def test_generator_get_and_set_state(self):
        g0 = torch.Generator(device="rbln").manual_seed(424242)
        initial_state = g0.get_state()
        x = _sample(generator=g0, n=10)

        g1 = torch.Generator(device="rbln")
        g1.set_state(initial_state)
        y = _sample(generator=g1, n=10)

        self.assertEqual(x.cpu(), y.cpu())

    def test_generator_reproducibility(self):
        rbln_g = torch.Generator(device="rbln").manual_seed(42424242)
        cpu_g = torch.Generator(device="cpu").manual_seed(42424242)

        rbln_x = _sample(device="rbln", generator=rbln_g, n=10)
        cpu_x = _sample(device="cpu", generator=cpu_g, n=10)

        self.assertEqual(rbln_x.cpu(), cpu_x)

    def test_set_state_restores_initial_seed(self):
        g0 = torch.Generator(device="rbln").manual_seed(987654321)
        state = g0.get_state()

        g1 = torch.Generator(device="rbln")
        g1.set_state(state)

        self.assertEqual(g1.initial_seed(), 987654321)

    def test_generator_device(self):
        g = torch.Generator(device="rbln")
        self.assertEqual(g.device.type, "rbln")

        default_gen = torch_rbln._C.get_default_generator()
        self.assertEqual(default_gen.device.type, "rbln")

    def test_seed_randomizes(self):
        # Generator.seed() must draw a fresh nondeterministic seed and use it,
        # not merely return the current one.
        g = torch.Generator(device="rbln").manual_seed(0)
        new_seed = g.seed()
        self.assertEqual(g.initial_seed(), new_seed)


@pytest.mark.test_set_ci
class TestDefaultGenerator(TestCase):
    """Default (generator=None) path: seeding, reproducibility, isolation."""

    def test_manual_seed_reproducibility_without_explicit_generator(self):
        # The reviewer's original repro: default-generator ops must be
        # reproducible after torch.rbln.manual_seed().
        torch.rbln.manual_seed(1234)
        x = _sample()

        torch.rbln.manual_seed(1234)
        y = _sample()

        self.assertEqual(x.cpu(), y.cpu())

    def test_default_and_explicit_generator_streams_match(self):
        # generator=None must consume the *default* generator's stream, so it
        # should produce the same sequence as an explicit generator with the
        # same seed.
        torch.rbln.manual_seed(5678)
        x = _sample()

        g = torch.Generator(device="rbln").manual_seed(5678)
        y = _sample(generator=g)

        self.assertEqual(x.cpu(), y.cpu())

    def test_rbln_ops_do_not_consume_cpu_rng(self):
        cpu_state_before = torch.get_rng_state()

        _ = _sample(n=1024)

        self.assertEqual(torch.get_rng_state(), cpu_state_before)

    def test_cpu_ops_do_not_consume_rbln_rng(self):
        torch.rbln.manual_seed(31337)
        rbln_state_before = torch.rbln.get_rng_state()

        _ = torch.randint(0, 1024, (1024,), device="cpu")

        self.assertEqual(torch.rbln.get_rng_state(), rbln_state_before)

    def test_cpu_rng_interleaving_does_not_break_rbln_reproducibility(self):
        torch.rbln.manual_seed(2024)
        x = _sample()

        torch.rbln.manual_seed(2024)
        _ = torch.rand(4096)  # perturb the global CPU stream in between
        y = _sample()

        self.assertEqual(x.cpu(), y.cpu())

    def test_default_rng_state_round_trip(self):
        torch.rbln.manual_seed(4321)
        state = torch.rbln.get_rng_state()
        x = _sample()

        torch.rbln.set_rng_state(state)
        y = _sample()

        self.assertEqual(x.cpu(), y.cpu())

    def test_get_rng_state_does_not_advance_stream(self):
        torch.rbln.manual_seed(999)
        _ = torch.rbln.get_rng_state()
        _ = torch.rbln.get_rng_state()
        x = _sample()

        torch.rbln.manual_seed(999)
        y = _sample()

        self.assertEqual(x.cpu(), y.cpu())


@pytest.mark.test_set_ci
class TestGeneratorValidation(TestCase):
    """Rejection of foreign generators and malformed state blobs."""

    def test_rejects_cpu_generator_for_rbln_op(self):
        cpu_g = torch.Generator(device="cpu").manual_seed(1234)

        with self.assertRaisesRegex(RuntimeError, "(?i)rbln|privateuse"):
            _sample(generator=cpu_g)

    def test_rejects_rbln_generator_for_cpu_op(self):
        rbln_g = torch.Generator(device="rbln").manual_seed(1234)

        with self.assertRaises(RuntimeError):
            _sample(device="cpu", generator=rbln_g)

    def test_set_state_rejects_empty_state(self):
        g = torch.Generator(device="rbln")

        with self.assertRaisesRegex(RuntimeError, "size"):
            g.set_state(torch.empty(0, dtype=torch.uint8))

    def test_set_state_rejects_short_state(self):
        g = torch.Generator(device="rbln")

        # Shorter than the seed header alone.
        with self.assertRaisesRegex(RuntimeError, "size"):
            g.set_state(torch.zeros(4, dtype=torch.uint8))

    def test_set_state_rejects_wrong_size_state(self):
        g = torch.Generator(device="rbln")
        good_numel = g.get_state().numel()

        for bad_numel in (good_numel - 1, good_numel + 1):
            with self.assertRaisesRegex(RuntimeError, "size"):
                g.set_state(torch.zeros(bad_numel, dtype=torch.uint8))

    def test_set_state_rejects_non_contiguous_state(self):
        g = torch.Generator(device="rbln")
        good = g.get_state()
        non_contiguous = good.repeat(2)[::2]
        self.assertFalse(non_contiguous.is_contiguous())

        with self.assertRaisesRegex(RuntimeError, "contiguous"):
            g.set_state(non_contiguous)

    def test_set_state_rejects_wrong_dtype_state(self):
        g = torch.Generator(device="rbln")
        good_numel = g.get_state().numel()

        with self.assertRaisesRegex(RuntimeError, "ByteTensor"):
            g.set_state(torch.zeros(good_numel, dtype=torch.float32))

    def test_failed_set_state_leaves_generator_unmodified(self):
        g = torch.Generator(device="rbln").manual_seed(13)
        before = g.get_state()

        for bad in (
            torch.empty(0, dtype=torch.uint8),
            torch.zeros(before.numel() - 1, dtype=torch.uint8),
            before.repeat(2)[::2],
        ):
            with self.assertRaises(RuntimeError):
                g.set_state(bad)

        self.assertEqual(g.get_state(), before)
        self.assertEqual(g.initial_seed(), 13)

        # The generator must still be usable and on the original stream.
        h = torch.Generator(device="rbln").manual_seed(13)
        self.assertEqual(_sample(generator=g).cpu(), _sample(generator=h).cpu())

    def test_set_state_rejects_non_cpu_state(self):
        # set_state consumes a CPU ByteTensor; a device-resident state blob
        # must be rejected before any header reads.
        g = torch.Generator(device="rbln")
        good_numel = g.get_state().numel()
        rbln_state = torch.zeros(good_numel, dtype=torch.uint8, device="rbln")

        with self.assertRaisesRegex(RuntimeError, "(?i)cpu|device"):
            g.set_state(rbln_state)

    def test_get_state_contract(self):
        # The blob round-tripped by get/set_state and torch.random.fork_rng
        # must be a contiguous CPU ByteTensor of a stable size.
        g = torch.Generator(device="rbln").manual_seed(7)
        state = g.get_state()

        self.assertEqual(state.device.type, "cpu")
        self.assertEqual(state.dtype, torch.uint8)
        self.assertTrue(state.is_contiguous())
        self.assertEqual(state.numel(), torch.Generator(device="rbln").get_state().numel())


@pytest.mark.test_set_ci
@unittest.skipUnless(RBLN_DEVICE_COUNT >= 2, "requires at least 2 RBLN devices")
class TestMultiDeviceGenerator(TestCase):
    """Per-device default generator independence."""

    def test_default_generators_are_distinct_objects(self):
        g0 = torch_rbln._C.get_default_generator(0)
        g1 = torch_rbln._C.get_default_generator(1)

        g0.manual_seed(111)
        g1.manual_seed(222)

        self.assertEqual(g0.initial_seed(), 111)
        self.assertEqual(g1.initial_seed(), 222)

    def test_sampling_on_one_device_does_not_advance_another(self):
        torch.rbln.manual_seed_all(1234)
        state1_before = torch.rbln.get_rng_state(1)

        _ = _sample(device="rbln:0", n=1024)

        self.assertEqual(torch.rbln.get_rng_state(1), state1_before)

    def test_same_seed_gives_same_stream_on_each_device(self):
        torch.rbln.manual_seed_all(4242)

        x0 = _sample(device="rbln:0")
        x1 = _sample(device="rbln:1")

        print(x0, x1)

        self.assertEqual(x0.cpu(), x1.cpu())

    def test_manual_seed_all_reproducibility(self):
        torch.rbln.manual_seed_all(31415)
        x0 = _sample(device="rbln:0")
        x1 = _sample(device="rbln:1")

        torch.rbln.manual_seed_all(31415)
        y0 = _sample(device="rbln:0")
        y1 = _sample(device="rbln:1")

        self.assertEqual(x0.cpu(), y0.cpu())
        self.assertEqual(x1.cpu(), y1.cpu())

    def test_per_device_rng_state_round_trip(self):
        torch.rbln.manual_seed_all(2718)
        states = [torch.rbln.get_rng_state(i) for i in range(RBLN_DEVICE_COUNT)]

        xs = [_sample(device=f"rbln:{i}") for i in range(RBLN_DEVICE_COUNT)]

        for i, state in enumerate(states):
            torch.rbln.set_rng_state(state, i)
        ys = [_sample(device=f"rbln:{i}") for i in range(RBLN_DEVICE_COUNT)]

        for x, y in zip(xs, ys):
            self.assertEqual(x.cpu(), y.cpu())

    def test_rejects_wrong_device_index_generator(self):
        # Wrong-index rejection is a distinct check from wrong-backend
        # rejection: check_generator passes (it IS an RBLNGeneratorImpl),
        # and the device-index TORCH_CHECK must fire instead.
        g1 = torch.Generator(device="rbln:1").manual_seed(1234)

        with self.assertRaisesRegex(RuntimeError, "(?i)device|rbln"):
            _sample(device="rbln:0", generator=g1)


@pytest.mark.test_set_ci
class TestForkRng(TestCase):
    """torch.random.fork_rng integration through get/set_rng_state."""

    def test_fork_rng_restores_default_state(self):
        torch.rbln.manual_seed(1234)
        before = torch.rbln.get_rng_state()

        with torch.random.fork_rng(device_type="rbln"):
            _ = torch.rand(10, device="rbln")
            # The op inside the block must have consumed the *RBLN* default
            # stream (not the global CPU stream); otherwise this save/restore
            # would be vacuous.
            self.assertNotEqual(torch.rbln.get_rng_state(), before)

        self.assertEqual(torch.rbln.get_rng_state(), before)

    def test_fork_rng_replays_same_sequence(self):
        torch.rbln.manual_seed(5678)

        with torch.random.fork_rng(device_type="rbln"):
            x = _sample()

        y = _sample()

        self.assertEqual(x.cpu(), y.cpu())

    def test_fork_rng_restores_on_exception(self):
        torch.rbln.manual_seed(4321)
        before = torch.rbln.get_rng_state()

        with self.assertRaises(ValueError):
            with torch.random.fork_rng(device_type="rbln"):
                _ = torch.rand(10, device="rbln")
                raise ValueError("boom")

        self.assertEqual(torch.rbln.get_rng_state(), before)

    def test_fork_rng_explicit_device_list(self):
        torch.rbln.manual_seed(8888)
        before = torch.rbln.get_rng_state(0)

        with torch.random.fork_rng(devices=[0], device_type="rbln"):
            _ = torch.rand(10, device="rbln:0")

        self.assertEqual(torch.rbln.get_rng_state(0), before)


@pytest.mark.test_set_ci
class TestGeneratorThreadSafety(TestCase):
    """Smoke test for concurrent seeding vs. fallback sampling.

    This cannot prove the absence of a data race (run under TSan for that),
    but it crashes or corrupts state loudly if the state APIs and the CPU
    fallback sampling path do not synchronize on the same mutex.
    """

    def test_concurrent_seed_and_sample(self):
        g = torch.Generator(device="rbln").manual_seed(1234)
        errors = []

        def seeder():
            try:
                for i in range(200):
                    g.manual_seed(i)
                    _ = g.get_state()
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        def sampler():
            try:
                for _ in range(200):
                    _ = torch.rand(256, device="rbln", generator=g)
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=seeder), threading.Thread(target=sampler)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [])

        # The generator must still be in a coherent, usable state.
        g.manual_seed(42)
        h = torch.Generator(device="rbln").manual_seed(42)
        self.assertEqual(_sample(generator=g).cpu(), _sample(generator=h).cpu())


@pytest.mark.test_set_ci
class TestFallbackBorrowSafety(TestCase):
    """Regression tests for borrow release when generator validation throws.

    The CPU fallback borrows tensor args off the stack before the generator
    argument is validated. If check_generator / the device-index TORCH_CHECK
    throws, the BorrowReleaseGuard must release those borrows; otherwise the
    next free on a still-borrowed vaddr throws from a noexcept ~TensorImpl
    deleter and terminates the process. A factory op like randint carries no
    tensor inputs and cannot exercise this path, so these tests use in-place
    ops whose `self` is borrowed at the time of the throw.
    """

    def _assert_tensor_still_healthy(self, x):
        # The failed call must not have leaked x's borrow: x must remain
        # readable and writable, and freeing it (plus fresh alloc/free
        # churn) must not crash the process.
        x.fill_(0)
        self.assertEqual(x.cpu(), torch.zeros_like(x, device="cpu"))
        del x
        for _ in range(4):
            t = torch.empty(1024, device="rbln")
            del t

    def test_failed_generator_check_releases_borrows(self):
        x = torch.empty(64, device="rbln")
        cpu_g = torch.Generator(device="cpu").manual_seed(1234)

        with self.assertRaisesRegex(RuntimeError, "(?i)rbln|privateuse"):
            x.uniform_(generator=cpu_g)

        self._assert_tensor_still_healthy(x)

    def test_failed_generator_check_releases_tensorlist_borrows(self):
        # Same throw point, but with additional borrowed tensor args in the
        # stack (weights input to multinomial) beyond the output.
        weights = torch.ones(32, device="rbln")
        cpu_g = torch.Generator(device="cpu").manual_seed(1234)

        with self.assertRaises(RuntimeError):
            torch.multinomial(weights, 8, replacement=True, generator=cpu_g)

        self._assert_tensor_still_healthy(weights)

    @unittest.skipUnless(RBLN_DEVICE_COUNT >= 2, "requires at least 2 RBLN devices")
    def test_failed_device_index_check_releases_borrows(self):
        # The second throw site in the generator block: an RBLN generator
        # whose device index does not match the op's target device.
        x = torch.empty(64, device="rbln:0")
        g1 = torch.Generator(device="rbln:1").manual_seed(1234)

        with self.assertRaises(RuntimeError):
            x.uniform_(generator=g1)

        self._assert_tensor_still_healthy(x)


@pytest.mark.test_set_ci
class TestGeneratorCoverageBreadth(TestCase):
    """Broader op coverage and mid-stream state semantics."""

    # Generator-consuming fallback ops beyond randint. Each entry must hit
    # the OptionalGenerator path in get_or_populate_schema_info for its own
    # schema. Factory-style and in-place ops are both represented.
    _OPS = {
        "rand": lambda device, g: torch.rand(64, device=device, generator=g),
        "randperm": lambda device, g: torch.randperm(64, device=device, generator=g),
        "uniform_": lambda device, g: torch.empty(64, device=device).uniform_(generator=g),
        "normal_": lambda device, g: torch.empty(64, device=device).normal_(generator=g),
        "multinomial": lambda device, g: torch.multinomial(
            torch.ones(64, device=device), 16, replacement=True, generator=g
        ),
    }

    def test_ops_reproducible_with_same_seed(self):
        for name, op in self._OPS.items():
            with self.subTest(op=name):
                g0 = torch.Generator(device="rbln").manual_seed(1234)
                g1 = torch.Generator(device="rbln").manual_seed(1234)

                self.assertEqual(op("rbln", g0).cpu(), op("rbln", g1).cpu())

    def test_ops_match_cpu_stream_with_same_seed(self):
        # The fallback runs the CPU kernel against the associated fallback
        # CPU generator, so an equally seeded pure-CPU run must match.
        for name, op in self._OPS.items():
            with self.subTest(op=name):
                rbln_g = torch.Generator(device="rbln").manual_seed(4242)
                cpu_g = torch.Generator(device="cpu").manual_seed(4242)

                self.assertEqual(op("rbln", rbln_g).cpu(), op("cpu", cpu_g))

    def test_ops_reproducible_via_default_generator(self):
        for name, op in self._OPS.items():
            with self.subTest(op=name):
                torch.rbln.manual_seed(31337)
                x = op("rbln", None)

                torch.rbln.manual_seed(31337)
                y = op("rbln", None)

                self.assertEqual(x.cpu(), y.cpu())

    def test_mid_stream_state_transplant(self):
        # Snapshot a generator mid-stream (not at its initial state) and
        # transplant it into a fresh generator: the continuation must match.
        # This is the case that actually verifies the fallback CPU state is
        # captured, not just the seed header.
        g0 = torch.Generator(device="rbln").manual_seed(2718)
        _ = _sample(generator=g0, n=100)  # advance past the initial state
        snapshot = g0.get_state()
        x = _sample(generator=g0, n=50)

        g1 = torch.Generator(device="rbln")  # different seed, different state
        g1.set_state(snapshot)
        y = _sample(generator=g1, n=50)

        self.assertEqual(x.cpu(), y.cpu())

    def test_mid_stream_default_rng_state_transplant(self):
        # Same mid-stream semantics through the module-level default path.
        torch.rbln.manual_seed(1618)
        _ = _sample(n=100)
        snapshot = torch.rbln.get_rng_state()
        x = _sample(n=50)

        torch.rbln.manual_seed(0)  # deliberately clobber
        torch.rbln.set_rng_state(snapshot)
        y = _sample(n=50)

        self.assertEqual(x.cpu(), y.cpu())

    def test_concurrent_generators_are_independent(self):
        # Two threads sampling from two distinct generators must each
        # reproduce their own serial sequence: the shared-mutex fix must
        # provide correctness without cross-generator interference.
        results = {}

        def worker(tag, seed):
            g = torch.Generator(device="rbln").manual_seed(seed)
            results[tag] = [torch.rand(256, device="rbln", generator=g).cpu() for _ in range(50)]

        threads = [
            threading.Thread(target=worker, args=("a", 111)),
            threading.Thread(target=worker, args=("b", 222)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        for tag, seed in (("a", 111), ("b", 222)):
            g = torch.Generator(device="rbln").manual_seed(seed)
            for step, expected in enumerate(results[tag]):
                self.assertEqual(
                    torch.rand(256, device="rbln", generator=g).cpu(),
                    expected,
                    msg=f"thread {tag} diverged from serial replay at step {step}",
                )


instantiate_device_type_tests(TestGeneratorBasic, globals(), only_for="privateuse1")


if __name__ == "__main__":
    run_tests()
