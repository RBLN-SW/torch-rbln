import os
from contextlib import nullcontext
from unittest.mock import patch

import torch
from torch.testing._internal.common_utils import run_tests, TestCase

from torch_rbln._internal.bucketing import (
    _get_eager_bucket_policy,
    BucketPolicy,
    pad_tensor_along_axis,
    select_bucket_size,
    slice_tensor_to_shape,
    try_create_bucketing_plan,
)


class TestEagerBucketingHelpers(TestCase):
    def tearDown(self) -> None:
        _get_eager_bucket_policy.cache_clear()

    def test_select_bucket_size_returns_smallest_fit(self):
        buckets = (1, 2, 4, 8)

        self.assertEqual(select_bucket_size(1, buckets), 1)
        self.assertEqual(select_bucket_size(3, buckets), 4)
        self.assertEqual(select_bucket_size(8, buckets), 8)
        self.assertIsNone(select_bucket_size(9, buckets))

    def test_pad_and_slice_tensor_along_axis(self):
        x = torch.arange(6).view(2, 3)

        padded = pad_tensor_along_axis(x, 0, 4, pad_value=-1)
        self.assertEqual(tuple(padded.shape), (4, 3))
        self.assertEqual(padded[2:].tolist(), [[-1, -1, -1], [-1, -1, -1]])

        sliced = slice_tensor_to_shape(padded, (2, 3))
        self.assertEqual(sliced.tolist(), x.tolist())

    def test_env_policy_is_opt_in_and_requires_ops(self):
        with patch.dict(os.environ, {}, clear=True):
            _get_eager_bucket_policy.cache_clear()
            self.assertIsNone(_get_eager_bucket_policy(None, "1,2,4", None, "aten::add:0", None))

        policy = _get_eager_bucket_policy("1", "4,1,2", "3.0", "aten::add:0", "-1")
        assert policy is not None
        self.assertEqual(policy.buckets, (1, 2, 4))
        self.assertEqual(policy.axis_for("aten::add"), 0)
        self.assertEqual(policy.max_pad_ratio, 3.0)
        self.assertEqual(policy.pad_value, -1.0)

        self.assertIsNone(_get_eager_bucket_policy("1", "1,2,4", None, None, None))

    def test_try_create_plan_pads_only_explicit_safe_pointwise_ops(self):
        policy = BucketPolicy(
            buckets=(4, 8),
            op_axes={"aten::add": 0},
            max_pad_ratio=2.0,
            pad_value=0.0,
        )
        x = torch.ones(5, 2)
        y = torch.ones(5, 2) * 2

        plan = try_create_bucketing_plan("aten::add", (x, y), {}, policy=policy)

        assert plan is not None
        self.assertEqual(plan.original_size, 5)
        self.assertEqual(plan.bucket_size, 8)
        self.assertEqual(plan.output_shape, (5, 2))
        self.assertEqual(tuple(plan.args[0].shape), (8, 2))
        self.assertEqual(tuple(plan.args[1].shape), (8, 2))

    def test_try_create_plan_rejects_ambiguous_shapes(self):
        policy = BucketPolicy(
            buckets=(4, 8),
            op_axes={"aten::add": 0, "aten::sum": 0},
            max_pad_ratio=2.0,
            pad_value=0.0,
        )

        self.assertIsNone(try_create_bucketing_plan("aten::sum", (torch.ones(5, 2),), {}, policy=policy))
        self.assertIsNone(
            try_create_bucketing_plan("aten::add", (torch.ones(5, 2), torch.ones(1, 2)), {}, policy=policy)
        )
        self.assertIsNone(try_create_bucketing_plan("aten::add", (torch.ones(5, 2).t(),), {}, policy=policy))
        self.assertIsNone(try_create_bucketing_plan("aten::add", (torch.ones(5, 2),), {}, policy=None))

    def test_compile_and_run_uses_bucket_shape_and_slices_output(self):
        from torch_rbln._internal import compile_cache
        try:
            from torch_rbln._internal.ops_utils import compile_and_run_view_aware
        except ImportError as exc:
            self.skipTest(f"RBLN native extension is unavailable: {exc}")

        compile_calls = []
        context_out_tensors = []

        def fake_compile(model, **kwargs):
            compile_calls.append(kwargs)

            def compiled(*args, **inner_kwargs):
                return args[0] + args[1]

            return compiled

        def fake_context(out_tensor=None):
            context_out_tensors.append(out_tensor)
            return nullcontext()

        env = {
            "TORCH_RBLN_EAGER_BUCKETING": "1",
            "TORCH_RBLN_EAGER_BUCKETS": "4,8",
            "TORCH_RBLN_EAGER_BUCKET_OPS": "aten::add:0",
        }
        x = torch.ones(5, 64)
        y = torch.ones(5, 64) * 2

        with patch.dict(os.environ, env, clear=False):
            _get_eager_bucket_policy.cache_clear()
            with patch.object(compile_cache, "compile_rbln_cached", side_effect=fake_compile):
                with patch("torch_rbln.device.context_holder.out_tensor_context", side_effect=fake_context):
                    with patch("torch_rbln._internal.warm_cache.install_pending") as mock_install:
                        out = compile_and_run_view_aware(torch.add, "aten::add", (x, y), {}, None)

        self.assertEqual(tuple(out.shape), (5, 64))
        self.assertTrue(torch.equal(out, torch.ones(5, 64) * 3))
        self.assertEqual(len(compile_calls), 1)
        cache_key = compile_calls[0]["device_cache_key"]
        self.assertEqual(cache_key[1][0][0], (8, 64))
        self.assertEqual(cache_key[1][1][0], (8, 64))
        self.assertEqual(context_out_tensors, [None])
        mock_install.assert_not_called()

    def test_compile_and_run_copies_sliced_bucket_result_to_out_tensor(self):
        from torch_rbln._internal import compile_cache
        try:
            from torch_rbln._internal.ops_utils import compile_and_run_view_aware
        except ImportError as exc:
            self.skipTest(f"RBLN native extension is unavailable: {exc}")

        def fake_compile(model, **kwargs):
            def compiled(*args, **inner_kwargs):
                return args[0] + args[1]

            return compiled

        env = {
            "TORCH_RBLN_EAGER_BUCKETING": "1",
            "TORCH_RBLN_EAGER_BUCKETS": "4,8",
            "TORCH_RBLN_EAGER_BUCKET_OPS": "aten::add:0",
        }
        x = torch.ones(5, 64)
        y = torch.ones(5, 64) * 2
        out_tensor = torch.empty(5, 64)

        with patch.dict(os.environ, env, clear=False):
            _get_eager_bucket_policy.cache_clear()
            with patch.object(compile_cache, "compile_rbln_cached", side_effect=fake_compile):
                with patch("torch_rbln.device.context_holder.out_tensor_context", return_value=nullcontext()):
                    result = compile_and_run_view_aware(torch.add, "aten::add", (x, y), {}, out_tensor)

        self.assertIs(result, out_tensor)
        self.assertTrue(torch.equal(out_tensor, torch.ones(5, 64) * 3))


if __name__ == "__main__":
    run_tests()
