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

    def _prepare_model_and_inputs(
        self,
        model_id: str,
        config_kwargs: dict,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ):
        """Load a causal language model and prepare tokenized inputs on the given device."""
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            ignore_mismatched_sizes=True,
            **config_kwargs,
        )
        model.to(device)
        _slice_lm_head_to_last_token(model)
        self.assertEqual(model.config.dtype, config_kwargs["dtype"])
        self.assertEqual(model.config._attn_implementation, config_kwargs["attn_implementation"])
        self.assertEqual(model.config.num_hidden_layers, config_kwargs["num_hidden_layers"])

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            padding_side="left",
            **config_kwargs,
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

    # RBLN runs 16-bit ops in custom float and matches an fp32 CPU reference to within
    # ~1.3 logits (measured); a real regression diverges by orders of magnitude.
    LOGIT_ATOL = 3.0

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
        fp32_config_kwargs = dict(config_kwargs, dtype=torch.float32)
        cpu_logits = self._prefill_logits(model_id, fp32_config_kwargs, batch_size, seq_len, self.cpu_device)
        rbln_logits = self._prefill_logits(model_id, config_kwargs, batch_size, seq_len, self.rbln_device)

        if not torch.isfinite(rbln_logits).all():
            cpu_dtype_logits = self._prefill_logits(model_id, config_kwargs, batch_size, seq_len, self.cpu_device)
            if not torch.isfinite(cpu_dtype_logits).all():
                self.skipTest(f"dtype {config_kwargs['dtype']} overflows {model_id} on CPU too")

        torch.testing.assert_close(rbln_logits, cpu_logits, atol=self.LOGIT_ATOL, rtol=0.0)


@pytest.mark.single_worker
class TestCausalLM(TestCausalLMBase):
    """Test correctness of causal language model outputs across various configurations."""

    # CI runs only a representative subset (batch_size=2, seq_len=1024);
    # release tests cover all batch_size x seq_len combinations.
    ci_tests_batch_seq = [
        subtest((bs, sl), decorators=[pytest.mark.test_set_ci] if (bs, sl) == (2, 1024) else [])
        for bs in TestCausalLMBase.batch_sizes
        for sl in TestCausalLMBase.seq_lens
    ]

    @pytest.mark.usefixtures("enable_deploy_mode")
    @dtypes(*SUPPORTED_DTYPES)
    @parametrize(
        "model_id",
        [
            "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct",
        ],
    )
    @parametrize("attn_implementation", TestCausalLMBase.attn_implementations)
    @parametrize("batch_size,seq_len", ci_tests_batch_seq)
    def test_exaone(self, dtype, model_id, attn_implementation, batch_size, seq_len):
        config_kwargs = dict(
            # Set a specific revision to avoid compatibility issues with the latest transformers version.
            # This can be removed when the transformers version is updated to 5.1.0 or higher.
            revision="e949c91dec92095908d34e6b560af77dd0c993f8",
            trust_remote_code=True,
            dtype=dtype,
            attn_implementation=attn_implementation,
            num_hidden_layers=self.num_hidden_layers,
        )

        self._assert_logits_match_fp32(model_id, config_kwargs, batch_size, seq_len)

    @pytest.mark.usefixtures("enable_deploy_mode")
    @dtypes(*SUPPORTED_DTYPES)
    @parametrize(
        "model_id",
        [
            "meta-llama/Llama-3.2-1B-Instruct",
            "meta-llama/Llama-3.2-3B-Instruct",
        ],
    )
    @parametrize("attn_implementation", TestCausalLMBase.attn_implementations)
    @parametrize("batch_size,seq_len", ci_tests_batch_seq)
    def test_llama(self, dtype, model_id, attn_implementation, batch_size, seq_len):
        config_kwargs = dict(
            dtype=dtype,
            attn_implementation=attn_implementation,
            num_hidden_layers=self.num_hidden_layers,
        )

        self._assert_logits_match_fp32(model_id, config_kwargs, batch_size, seq_len)

    # Deploy mode is disabled for Qwen2.5 due to float16 overflow issues in certain BMM operations.
    # This will be enabled in the future when cf16 host padding is supported.
    @dtypes(*SUPPORTED_DTYPES)
    @parametrize(
        "model_id",
        [
            "Qwen/Qwen2.5-1.5B-Instruct",
        ],
    )
    @parametrize("attn_implementation", TestCausalLMBase.attn_implementations)
    @parametrize("batch_size,seq_len", ci_tests_batch_seq)
    def test_qwen2(self, dtype, model_id, attn_implementation, batch_size, seq_len):
        config_kwargs = dict(
            # Pin the revision so model updates can't shift the comparison.
            revision="989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
            dtype=dtype,
            attn_implementation=attn_implementation,
            num_hidden_layers=self.num_hidden_layers,
            sliding_window=0,  # Disable sliding window attention.
        )
        self._assert_logits_match_fp32(model_id, config_kwargs, batch_size, seq_len)


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
            revision="e949c91dec92095908d34e6b560af77dd0c993f8",
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
instantiate_device_type_tests(TestCausalLMPerf, globals(), only_for="privateuse1")


if __name__ == "__main__":
    run_tests()
