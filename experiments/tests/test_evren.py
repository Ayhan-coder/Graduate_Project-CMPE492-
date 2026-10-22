import os
import unittest
from unittest.mock import MagicMock, patch

from evren_llm import (
    EVREN_MODELS,
    generate_evren,
    get_evren_api_key,
    get_evren_base_url,
    is_evren_model,
    normalize_evren_model_id,
    strip_think_tags,
)
from gui.server import get_models, get_allowed_model_ids
from gui.pipeline import extract_terms


def _resp(content="", finish_reason="stop", reasoning=""):
    message = MagicMock()
    message.content = content
    message.model_extra = {"reasoning": reasoning}
    choice = MagicMock()
    choice.message = message
    choice.finish_reason = finish_reason
    resp = MagicMock()
    resp.choices = [choice]
    return resp


class TestEvrenIntegration(unittest.TestCase):
    def test_evren_model_detection(self):
        self.assertTrue(is_evren_model("evren/qwen3.8-flash-next"))
        self.assertTrue(is_evren_model("evren/deepseek-v4.1-flash"))
        self.assertTrue(is_evren_model("evren/glm-5.3"))
        self.assertTrue(is_evren_model("qwen3.8-flash-next"))
        self.assertTrue(is_evren_model("deepseek-v4.1-flash"))
        self.assertFalse(is_evren_model("Qwen/Qwen3.5-4B"))
        self.assertFalse(is_evren_model("openai/whisper-large-v3"))

    def test_normalize_evren_model_id(self):
        self.assertEqual(normalize_evren_model_id("evren/qwen3.8-flash-next"), "qwen3.8-flash-next")
        self.assertEqual(normalize_evren_model_id("evren:deepseek-v4.1-flash"), "deepseek-v4.1-flash")
        self.assertEqual(normalize_evren_model_id("glm-5.3"), "glm-5.3")

    def test_api_key_is_not_hardcoded(self):
        env = {k: v for k, v in os.environ.items() if k not in ("EVREN_API_KEY", "EVREN_TOKEN")}
        with patch.dict(os.environ, env, clear=True):
            self.assertRaises(RuntimeError, get_evren_api_key)
        with patch.dict(os.environ, {"EVREN_API_KEY": "evren_llm_from_env"}, clear=True):
            self.assertEqual(get_evren_api_key(), "evren_llm_from_env")
        self.assertEqual(get_evren_base_url(), "https://evren-llmapi.ssyz.org.tr/v1")

    def test_strip_think_tags(self):
        raw = "<think>Analyzing context...</think>[{\"term\": \"test\", \"explanation\": \"desc\"}]"
        cleaned = strip_think_tags(raw)
        self.assertEqual(cleaned, "[{\"term\": \"test\", \"explanation\": \"desc\"}]")

    def test_generate_disables_reasoning_by_default(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _resp('[{"term": "EVM", "explanation": "d"}]')
        with patch("evren_llm.get_evren_client", return_value=client), patch.dict(
            os.environ, {"EVREN_API_KEY": "evren_llm_test", "EVREN_REASONING_EFFORT": ""}
        ):
            out = generate_evren("evren/qwen3.8-flash-next", "Transcript: ...")
        self.assertEqual(out, '[{"term": "EVM", "explanation": "d"}]')
        self.assertEqual(client.chat.completions.create.call_args.kwargs["reasoning_effort"], "none")

    def test_empty_answer_raises_instead_of_recording_zero_terms(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _resp("", finish_reason="length", reasoning="t" * 400)
        with patch("evren_llm.get_evren_client", return_value=client), patch.dict(
            os.environ, {"EVREN_API_KEY": "evren_llm_test", "EVREN_REASONING_EFFORT": ""}
        ):
            with self.assertRaises(RuntimeError) as ctx:
                generate_evren("evren/qwen3.8-flash-next", "Transcript: ...", max_tokens=1024)
        self.assertIn("no answer text", str(ctx.exception))
        self.assertIn("finish_reason=length", str(ctx.exception))

    def test_retry_with_reasoning_off_when_effort_enabled(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            _resp("", finish_reason="length", reasoning="t" * 400),
            _resp('[{"term": "EVM", "explanation": "d"}]'),
        ]
        with patch("evren_llm.get_evren_client", return_value=client), patch.dict(
            os.environ, {"EVREN_API_KEY": "evren_llm_test", "EVREN_REASONING_EFFORT": "xhigh"}
        ):
            out = generate_evren("evren/qwen3.8-flash-next", "Transcript: ...")
        self.assertEqual(out, '[{"term": "EVM", "explanation": "d"}]')
        self.assertEqual(
            [c.kwargs["reasoning_effort"] for c in client.chat.completions.create.call_args_list],
            ["xhigh", "none"],
        )

    def test_terms_are_not_accepted_without_optin(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("403 terms_not_accepted")
        with patch("evren_llm.get_evren_client", return_value=client), patch(
            "evren_llm.accept_evren_terms"
        ) as accept, patch.dict(os.environ, {"EVREN_API_KEY": "evren_llm_test", "EVREN_ACCEPT_TERMS": ""}, clear=True):
            self.assertRaises(RuntimeError, generate_evren, "evren/qwen3.8-flash-next", "Transcript: ...")
        accept.assert_not_called()

    def test_terms_acceptance_honours_optin(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            RuntimeError("403 terms_not_accepted"),
            _resp('[{"term": "EVM", "explanation": "d"}]'),
        ]
        with patch("evren_llm.get_evren_client", return_value=client), patch(
            "evren_llm.accept_evren_terms", return_value=True
        ) as accept, patch.dict(os.environ, {"EVREN_API_KEY": "evren_llm_test", "EVREN_ACCEPT_TERMS": "1"}):
            out = generate_evren("evren/qwen3.8-flash-next", "Transcript: ...")
        self.assertEqual(out, '[{"term": "EVM", "explanation": "d"}]')
        accept.assert_called_once()

    def test_server_models_include_evren(self):
        models = get_models()
        llm_ids = [m["id"] for m in models["llm_models"]]
        self.assertIn("evren/qwen3.8-flash-next", llm_ids)
        self.assertIn("evren/deepseek-v4.1-flash", llm_ids)

        _, allowed_llm = get_allowed_model_ids()
        self.assertIn("evren/qwen3.8-flash-next", allowed_llm)
        self.assertIn("evren/deepseek-v4.1-flash", allowed_llm)

    @patch("gui.pipeline.generate_evren")
    def test_extract_terms_routes_evren(self, mock_generate_evren):
        mock_generate_evren.return_value = '[{"term": "yapay zeka", "explanation": "Makine akli."}]'
        res = extract_terms(
            transcript="Bugün yapay zeka konusunu işleyeceğiz.",
            lang="tr",
            model_id="evren/qwen3.8-flash-next",
        )
        self.assertTrue(mock_generate_evren.called)
        self.assertEqual(res["n_terms"], 1)
        self.assertEqual(res["terms"][0]["term"], "yapay zeka")
        self.assertEqual(res["model"], "evren/qwen3.8-flash-next")


if __name__ == "__main__":
    unittest.main()
