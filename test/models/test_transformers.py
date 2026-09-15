# Owner(s): ["module: PrivateUse1"]

import inspect
import os

import pandas as pd
import pytest
import torch
from torch.profiler import profile, ProfilerActivity
from torch.profiler._pattern_matcher import Pattern
from torch.testing._internal.common_device_type import dtypes, instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, subtest, TestCase
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from test.utils import SUPPORTED_DTYPES

from torch_rbln._internal.device_arch_utils import is_atom_device


TORCH_RBLN_SAVE_PATH = os.getenv("TORCH_RBLN_SAVE_PATH", os.getcwd())


def _slice_lm_head_to_last_token(model):
    """Slice lm_head to the last token for full-sequence-logits models.

    Custom modeling files like EXAONE predate HF's ``logits_to_keep`` and run
    ``lm_head`` over the whole sequence.
    No-op for models that already slice (llama/qwen).
    """
    if "logits_to_keep" in inspect.signature(type(model).forward).parameters:
        return
    lm_head = getattr(model, "lm_head", None)
    if not isinstance(lm_head, torch.nn.Module):
        return

    class _LastTokenLMHead(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, hidden_states):
            if hidden_states.dim() >= 2 and hidden_states.shape[-2] > 1:
                hidden_states = hidden_states[..., -1:, :]
            return self.inner(hidden_states)

    model.lm_head = _LastTokenLMHead(lm_head)


class TestCausalLMBase(TestCase):
    """Base class for causal language model tests."""

    rbln_device = torch.device("rbln:0")
    cpu_device = torch.device("cpu")
    num_hidden_layers = 1
    attn_implementations = ["eager", "sdpa"]
    batch_sizes = [1, 2, 4]
    seq_lens = [16, 128, 1024]
    max_new_tokens = 2  # Run prefill & decode phase once each.

    # Pin the EXAONE HF revision (shared by every EXAONE test here, incl. TestCausalLMPerf)
    # so model-hub updates can't shift the comparison.
    _EXAONE_REVISION = "e949c91dec92095908d34e6b560af77dd0c993f8"

    def _prepare_model_and_inputs(
        self,
        model_id: str,
        config_kwargs: dict,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ):
        """Load a causal language model and prepare tokenized inputs on the given device."""
        # The pinned transformers takes ``torch_dtype``; the rename to ``dtype`` lands in a
        # later 4.x. Cases keep spelling it ``dtype``, so translate once here.
        load_kwargs = dict(config_kwargs)
        load_kwargs["torch_dtype"] = load_kwargs.pop("dtype")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            ignore_mismatched_sizes=True,
            **load_kwargs,
        )
        model.to(device)
        _slice_lm_head_to_last_token(model)
        self.assertEqual(model.config.torch_dtype, config_kwargs["dtype"])
        self.assertEqual(model.config._attn_implementation, config_kwargs["attn_implementation"])
        self.assertEqual(model.config.num_hidden_layers, config_kwargs["num_hidden_layers"])

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            padding_side="left",
            **load_kwargs,
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = 0

        prompt = "Hey, are you conscious? Can you talk to me?"
        inputs = tokenizer(
            [prompt] * batch_size,
            padding="max_length",
            truncation=True,
            max_length=seq_len,
            return_tensors="pt",
        )
        # Ensure padding tokens are 0
        inputs.input_ids[inputs.attention_mask == 0] = 0
        inputs = inputs.to(model.device)
        self.assertEqual(inputs.input_ids.size(), (batch_size, seq_len))
        self.assertEqual(inputs.attention_mask.size(), (batch_size, seq_len))

        return model, inputs

    def _generate(self, model, inputs) -> torch.Tensor:
        """Run deterministic greedy generation."""
        return model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            temperature=None,
            top_k=None,
            top_p=None,
            eos_token_id=None,
        )

    def _run(
        self,
        model_id: str,
        config_kwargs: dict,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Run a causal language model and return the generated token ids."""
        model, inputs = self._prepare_model_and_inputs(model_id, config_kwargs, batch_size, seq_len, device)
        outputs = self._generate(model, inputs)
        self.assertEqual(outputs.size(), (batch_size, seq_len + self.max_new_tokens))
        return outputs

    # RBLN runs 16-bit ops in custom float and tracks an fp32 CPU reference only to
    # within a bound; a real regression diverges by orders of magnitude. The bound
    # follows the target's compute precision policy, so re-derive it if that changes.
    # test/rbln/test_fp16_numerics.py is the faster check when this trips.
    LOGIT_ATOL = 5.0

    def _prefill_logits(self, model_id, config_kwargs, batch_size, seq_len, device):
        """Next-token logits at the prompt's last position (via generate, one step:
        no autoregressive cascade, and unlike a bare forward it works at batch size 1)."""
        model, inputs = self._prepare_model_and_inputs(model_id, config_kwargs, batch_size, seq_len, device)
        gen = model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=False,
            temperature=None,
            top_k=None,
            top_p=None,
            eos_token_id=None,
            return_dict_in_generate=True,
            output_logits=True,
        )
        return gen.logits[0].float().cpu()

    def _assert_logits_match_fp32(self, model_id, config_kwargs, batch_size, seq_len):
        """Compare RBLN's next-token logits against an fp32 CPU reference (ground truth).

        Greedy token-id equality between RBLN and a same-dtype CPU run is too strict: the
        two are different 16-bit formats (RBLN custom float vs CPU bfloat16) and disagree by
        a few logits even when both are correct, which flips near-tie argmaxes and cascades
        during generation (an intermittent failure). We instead check that RBLN tracks the
        fp32 truth, where its deviation is small and bounded by its own precision. Configs
        whose dtype overflows the model (e.g. fp16 BMM) are skipped — a dtype limitation,
        not an RBLN bug.
        """
        # ATOM promotes bfloat16 to the device's custom float, whose range HuggingFace's causal
        # mask value (torch.finfo(bfloat16).min) overflows into NaN rather than saturating, wiping
        # out every eager-attention output. sdpa lowers to a fused kernel that never adds the
        # mask this way, so only the eager cases go. test/rbln/test_bf16_range.py holds the
        # cause and fails until it is fixed; drop this when that test starts passing.
        if (
            is_atom_device()
            and config_kwargs.get("dtype") is torch.bfloat16
            and config_kwargs.get("attn_implementation") == "eager"
        ):
            self.skipTest("bfloat16 x eager attention on ATOM: see test/rbln/test_bf16_range.py")

        fp32_config_kwargs = dict(config_kwargs, dtype=torch.float32)
        cpu_logits = self._prefill_logits(model_id, fp32_config_kwargs, batch_size, seq_len, self.cpu_device)
        rbln_logits = self._prefill_logits(model_id, config_kwargs, batch_size, seq_len, self.rbln_device)

        if not torch.isfinite(rbln_logits).all():
            cpu_dtype_logits = self._prefill_logits(model_id, config_kwargs, batch_size, seq_len, self.cpu_device)
            if not torch.isfinite(cpu_dtype_logits).all():
                self.skipTest(f"dtype {config_kwargs['dtype']} overflows {model_id} on CPU too")

        torch.testing.assert_close(rbln_logits, cpu_logits, atol=self.LOGIT_ATOL, rtol=0.0)


def _small_model_cover_array(full_matrix_at=None, skip_shapes=()):
    """Strength-2 (pairwise) covering array over dtype x attn x (batch, seq) for a small model.

    The full 4-way cross product (2 dtype x 2 attn x 9 shapes = 36 cases/model) re-runs paths the
    op-level and CI tests already cover point-by-point. This replaces it with an array of one row
    per dtype at each shape, rotating the dtype<->attn pairing across shapes, which guarantees:

      * every dtype and every attn runs (1-way),
      * every attn x shape, dtype x shape, and attn x dtype pair appears over the shapes the
        model runs (2-way),
      * every (batch, seq) shape the caller asks for is compiled at least once -- each shape is a
        distinct compiled artifact, and which model's geometry compiles it is not what the shape
        axis is about.

    Dropped are the 3-/4-way interactions, and the duplication between the two small models at
    the shapes that dominate the run: a case at seq 1024 costs an order of magnitude more than
    one at seq 16, so ``skip_shapes`` lets one small model carry the expensive shapes for both,
    and lets the widest batch sit on the large models instead of on every model.

    ``full_matrix_at`` names the one shape that also gets the full dtype x attn matrix and the
    test_set_ci marker -- the CI representative point, carried by a single model per family. The
    3-way dtype x attn x shape is worth its cost at one point, not at every large shape. Pass None
    for the release-only variant used where another model already covers the family's CI slot.
    """
    shapes = [
        (bs, sl)
        for bs in TestCausalLMBase.batch_sizes
        for sl in TestCausalLMBase.seq_lens
        if (bs, sl) not in skip_shapes
    ]
    attns = TestCausalLMBase.attn_implementations
    n_dtype, n_attn = len(SUPPORTED_DTYPES), len(attns)

    array = []
    rotate = 0
    for bs, sl in shapes:
        if (bs, sl) == full_matrix_at:
            combos = [(dtype, attn) for dtype in SUPPORTED_DTYPES for attn in attns]
        else:
            combos = [
                (SUPPORTED_DTYPES[(rotate + j) % n_dtype], attns[j % n_attn]) for j in range(max(n_dtype, n_attn))
            ]
            rotate += 1
        for dtype, attn in combos:
            # Omit dtype from the subtest name: instantiate_device_type_tests appends it,
            # which keeps the ids unique (e.g. ..._eager_b2_s1024_privateuse1_float16).
            array.append(
                subtest(
                    (dtype, attn, bs, sl),
                    name=f"{attn}_b{bs}_s{sl}",
                    decorators=[pytest.mark.test_set_ci] if (bs, sl) == full_matrix_at else [],
                )
            )
    return array


@pytest.mark.single_worker
class TestCausalLM(TestCausalLMBase):
    """Test correctness of causal language model outputs across various configurations."""

    # Small models (Llama-1B, Qwen-1.5B) run a pairwise covering array over dtype x attn x
    # (batch, seq) instead of the full 4-way cross product; see _small_model_cover_array for the
    # exact coverage guarantee.
    #
    # The batch axis is split by cost rather than run twice: the small models take batch 1 and 2,
    # and batch 4 -- the widest, and the one where per-batch tiling and memory pressure actually
    # show -- is the large models' point below. Qwen-1.5B carries what is left of the shape grid
    # and, at the representative point (2, 1024), the full dtype x attn matrix and the CI marker.
    # Llama-1B is release-only and skips that point too, which Qwen-1.5B compiles at both dtypes;
    # it keeps (1, 1024), so a long sequence is still compiled on Llama geometry.
    _LARGE_BATCH_SHAPES = tuple((4, seq_len) for seq_len in TestCausalLMBase.seq_lens)
    _COSTLY_SHAPES = ((2, 1024),)
    small_model_cover_array = _small_model_cover_array(full_matrix_at=(2, 1024), skip_shapes=_LARGE_BATCH_SHAPES)
    small_model_cover_array_release_only = _small_model_cover_array(skip_shapes=_LARGE_BATCH_SHAPES + _COSTLY_SHAPES)

    # Large models (Llama-3B, EXAONE-2.4B) share the small models' code paths but are 3-5x slower
    # to compile, so they buy geometry, not breadth: one case each, at the largest shape, where
    # size-specific tiling and memory issues surface, and the suite's only batch 4. float16 and
    # sdpa, as the largest-shape smoke these two used to carry was -- dtype and attn breadth is
    # the small models' axis. This subsumes that separate smoke.
    max_shape_batch_seq = [subtest((4, 1024), decorators=[pytest.mark.test_set_ci])]

    @pytest.mark.usefixtures("enable_deploy_mode")
    @dtypes(torch.float16)
    @parametrize(
        "model_id",
        [
            "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct",
        ],
    )
    @parametrize("batch_size,seq_len", max_shape_batch_seq)
    def test_exaone(self, dtype, model_id, batch_size, seq_len):
        config_kwargs = dict(
            # Set a specific revision to avoid compatibility issues with the latest transformers version.
            # This can be removed when the transformers version is updated to 5.1.0 or higher.
            revision=self._EXAONE_REVISION,
            trust_remote_code=True,
            dtype=dtype,
            attn_implementation="sdpa",
            num_hidden_layers=self.num_hidden_layers,
        )

        self._assert_logits_match_fp32(model_id, config_kwargs, batch_size, seq_len)

    @pytest.mark.usefixtures("enable_deploy_mode")
    @parametrize("dtype,attn_implementation,batch_size,seq_len", small_model_cover_array_release_only)
    def test_llama(self, dtype, attn_implementation, batch_size, seq_len):
        config_kwargs = dict(
            dtype=dtype,
            attn_implementation=attn_implementation,
            num_hidden_layers=self.num_hidden_layers,
        )

        self._assert_logits_match_fp32("meta-llama/Llama-3.2-1B-Instruct", config_kwargs, batch_size, seq_len)

    # Llama-3B is the same architecture as Llama-1B (covering array above) but far slower to
    # compile. It is the Llama family's CI slot: the largest shape, where the larger geometry
    # surfaces failures first.
    @pytest.mark.usefixtures("enable_deploy_mode")
    @dtypes(torch.float16)
    @parametrize("batch_size,seq_len", max_shape_batch_seq)
    def test_llama_3b(self, dtype, batch_size, seq_len):
        config_kwargs = dict(
            dtype=dtype,
            attn_implementation="sdpa",
            num_hidden_layers=self.num_hidden_layers,
        )

        self._assert_logits_match_fp32("meta-llama/Llama-3.2-3B-Instruct", config_kwargs, batch_size, seq_len)

    # Deploy mode is disabled for Qwen2.5 due to float16 overflow issues in certain BMM operations.
    # This will be enabled in the future when cf16 host padding is supported.
    @parametrize("dtype,attn_implementation,batch_size,seq_len", small_model_cover_array)
    def test_qwen2(self, dtype, attn_implementation, batch_size, seq_len):
        config_kwargs = dict(
            # Pin the revision so model updates can't shift the comparison.
            revision="989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
            dtype=dtype,
            attn_implementation=attn_implementation,
            num_hidden_layers=self.num_hidden_layers,
            sliding_window=0,  # Disable sliding window attention.
        )
        self._assert_logits_match_fp32("Qwen/Qwen2.5-1.5B-Instruct", config_kwargs, batch_size, seq_len)


@pytest.mark.single_worker
class TestCausalLMGraph(TestCausalLMBase):
    """Run a causal language model through ``torch.compile(backend="rbln")``.

    TestCausalLM compiles one program per operator, as eager dispatch reaches it.
    This compiles the whole forward as a single program -- the path the RBLN PyTorch
    tutorial takes, and the only one where a graph-level break shows up.

    One model at one shape, over both attention implementations, which are the axis
    that changes the traced graph rather than its geometry. Failures here are graph
    failures; the shape and dtype grids stay in TestCausalLM.

    A compile that fails does not raise on its own -- it falls back to CPU and returns
    the right logits. conftest's autouse ``disable_compile_error_fallback`` is what
    turns that back into an error, and without it these cases would pass while graph
    mode was broken.
    """

    @pytest.mark.usefixtures("enable_deploy_mode")
    @dtypes(torch.float16)
    @parametrize("attn_implementation", TestCausalLMBase.attn_implementations)
    @parametrize("batch_size,seq_len", [subtest((1, 128), decorators=[pytest.mark.test_set_ci])])
    def test_llama(self, dtype, attn_implementation, batch_size, seq_len):
        config_kwargs = dict(
            dtype=dtype,
            attn_implementation=attn_implementation,
            num_hidden_layers=self.num_hidden_layers,
        )
        model, inputs = self._prepare_model_and_inputs(
            "meta-llama/Llama-3.2-1B", config_kwargs, batch_size, seq_len, self.rbln_device
        )
        compiled = torch.compile(model, backend="rbln", dynamic=False)

        with torch.inference_mode():
            compiled(**inputs)  # compile here, so the measured call replays the program
            with torch.rbln.explain() as steady:
                graph_logits = compiled(**inputs).logits
            eager_logits = model(**inputs).logits

        # A graph that silently fell back to CPU, or recompiled on every call, still
        # returns the right logits -- assert the program actually ran on the device.
        verdict = steady.verdict()
        self.assertEqual(verdict["cpu_fallbacks"], 0, steady.report())
        self.assertEqual(verdict["recompiles"], 0, steady.report())

        # Only the last position is comparable: the left padding carries the causal mask's
        # minimum through lm_head, and those positions come back inf in both runs.
        self.assertEqual(graph_logits[:, -1].argmax(-1), eager_logits[:, -1].argmax(-1))


@pytest.mark.test_set_perf
@pytest.mark.single_worker
class TestCausalLMPerf(TestCausalLMBase):
    """Profile causal language model performance across various configurations and generate reports."""

    reports = []
    reports_path = os.path.join(TORCH_RBLN_SAVE_PATH, "transformers_causallm.md")
    attn_implementations = ["sdpa"]

    def tearDown(self):
        df_reports = pd.json_normalize(self.reports)
        df_reports.to_markdown(self.reports_path)

        pd.set_option("display.max_columns", None)
        pd.set_option("display.max_colwidth", None)
        pd.set_option("display.max_rows", None)
        pd.set_option("display.width", None)
        print(df_reports)

    def _analyze_events(self, events: list, model: PreTrainedModel) -> dict:
        analysis_result = {
            "prefill_phase": {"duration_ms": []},
            "decode_phase": {"duration_ms": []},
        }

        target_event_name = f"nn.Module: {model.__class__.__name__}_0"
        target_events = [event for event in events if event.name == target_event_name]
        self.assertEqual(len(target_events), self.max_new_tokens)

        for idx, event in enumerate(target_events):
            phase = "prefill_phase" if idx == 0 else "decode_phase"
            duration_ms = int(event.duration_time_ns / 1e6)
            analysis_result[phase]["duration_ms"].append(duration_ms)

        return analysis_result

    def _run(
        self,
        model_id: str,
        config_kwargs: dict,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Run a causal language model under profiling, collect reports, and return outputs."""
        model, inputs = self._prepare_model_and_inputs(model_id, config_kwargs, batch_size, seq_len, device)

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
            with_stack=True,
        ) as prof:
            outputs = self._generate(model, inputs)

        def has_valid_event_name(event):
            try:
                return event.name
            except UnicodeDecodeError:
                return None

        events = [event for event in Pattern(prof).eventTreeTraversal() if has_valid_event_name(event)]
        analysis_result = self._analyze_events(events, model)

        report = {
            "model_id": model_id,
            "num_hidden_layers": config_kwargs["num_hidden_layers"],
            "dtype": config_kwargs["dtype"],
            "attn_implementation": config_kwargs["attn_implementation"],
            "num_threads": torch.get_num_threads(),
            "batch_size": batch_size,
            "seq_len": seq_len,
            "device": device,
        }
        report.update(analysis_result)
        self.reports.append(report)

        self.assertEqual(outputs.size(), (batch_size, seq_len + self.max_new_tokens))
        return outputs

    @pytest.mark.usefixtures("enable_deploy_mode")
    @dtypes(*SUPPORTED_DTYPES)
    @parametrize(
        "model_id",
        [
            "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct",
        ],
    )
    @parametrize("attn_implementation", attn_implementations)
    @parametrize("batch_size", TestCausalLMBase.batch_sizes)
    @parametrize("seq_len", TestCausalLMBase.seq_lens)
    def test_exaone(self, dtype, model_id, attn_implementation, batch_size, seq_len):
        config_kwargs = dict(
            revision=self._EXAONE_REVISION,
            trust_remote_code=True,
            dtype=dtype,
            attn_implementation=attn_implementation,
            num_hidden_layers=self.num_hidden_layers,
        )
        self._run(model_id, config_kwargs, batch_size, seq_len, self.rbln_device)

    @pytest.mark.usefixtures("enable_deploy_mode")
    @dtypes(*SUPPORTED_DTYPES)
    @parametrize(
        "model_id",
        [
            "meta-llama/Llama-3.2-1B-Instruct",
            "meta-llama/Llama-3.2-3B-Instruct",
        ],
    )
    @parametrize("attn_implementation", attn_implementations)
    @parametrize("batch_size", TestCausalLMBase.batch_sizes)
    @parametrize("seq_len", TestCausalLMBase.seq_lens)
    def test_llama(self, dtype, model_id, attn_implementation, batch_size, seq_len):
        config_kwargs = dict(
            dtype=dtype,
            attn_implementation=attn_implementation,
            num_hidden_layers=self.num_hidden_layers,
        )
        self._run(model_id, config_kwargs, batch_size, seq_len, self.rbln_device)

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize(
        "model_id",
        [
            "Qwen/Qwen2.5-1.5B-Instruct",
        ],
    )
    @parametrize("attn_implementation", attn_implementations)
    @parametrize("batch_size", TestCausalLMBase.batch_sizes)
    @parametrize("seq_len", TestCausalLMBase.seq_lens)
    def test_qwen2(self, dtype, model_id, attn_implementation, batch_size, seq_len):
        config_kwargs = dict(
            dtype=dtype,
            attn_implementation=attn_implementation,
            num_hidden_layers=self.num_hidden_layers,
            sliding_window=0,
        )
        self._run(model_id, config_kwargs, batch_size, seq_len, self.rbln_device)


instantiate_device_type_tests(TestCausalLM, globals(), only_for="privateuse1")
instantiate_device_type_tests(TestCausalLMGraph, globals(), only_for="privateuse1")
instantiate_device_type_tests(TestCausalLMPerf, globals(), only_for="privateuse1")


if __name__ == "__main__":
    run_tests()
