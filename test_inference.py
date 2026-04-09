import importlib
import sys
import types


def _import_inference_with_openai_stub(monkeypatch, openai_factory=None):
    class _DefaultOpenAI:
        def __init__(self, *args, **kwargs):
            pass

    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = openai_factory or _DefaultOpenAI

    class _Obs:
        def __init__(self, invoice_text):
            self.invoice_text = invoice_text

    class _InvoiceEnv:
        def reset(self, mode="easy"):
            return _Obs("Invoice from ABC Ltd. Amount: $5000. Date: 10 Oct")

        def step(self, action):
            return self.reset(), 0.0, False, {}

    class _Action:
        def __init__(self, action_type, field_name=None, value=None):
            self.action_type = action_type
            self.field_name = field_name
            self.value = value

    fake_env = types.ModuleType("env")
    fake_env.InvoiceEnv = _InvoiceEnv

    fake_models = types.ModuleType("models")
    fake_models.Action = _Action

    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setitem(sys.modules, "env", fake_env)
    monkeypatch.setitem(sys.modules, "models", fake_models)
    sys.modules.pop("inference", None)
    return importlib.import_module("inference")


def test_parse_fields_payload_handles_fenced_json_with_noise(monkeypatch):
    inference = _import_inference_with_openai_stub(monkeypatch)

    raw = """```json
{'amount': '5000', 'vendor': 'ABC Ltd', 'date': '10 Oct'}
```
extra text"""

    parsed = inference._parse_fields_payload(raw)

    assert parsed["amount"] == "5000"
    assert parsed["vendor"] == "ABC Ltd"
    assert parsed["date"] == "10 Oct"


def test_run_episode_fallback_when_client_missing(monkeypatch):
    inference = _import_inference_with_openai_stub(monkeypatch)

    monkeypatch.setattr(inference, "client", None)

    score = inference.run_episode("easy")

    assert score == 0.01


def test_run_episode_fallback_on_api_failure(monkeypatch):
    inference = _import_inference_with_openai_stub(monkeypatch)

    class _FailingChatCompletions:
        def create(self, *args, **kwargs):
            raise RuntimeError("simulated API failure")

    class _FailingChat:
        completions = _FailingChatCompletions()

    class _FailingClient:
        chat = _FailingChat()

    monkeypatch.setattr(inference, "client", _FailingClient())

    score = inference.run_episode("easy")

    assert score == 0.01


def test_proxy_env_initializes_client(monkeypatch):
    captured = {}

    def _fake_openai(base_url, api_key):
        captured["base_url"] = base_url
        captured["api_key"] = api_key
        return object()

    monkeypatch.setenv("API_BASE_URL", "https://proxy.example/v1")
    monkeypatch.setenv("API_KEY", "proxy-test-key")

    reloaded = _import_inference_with_openai_stub(monkeypatch, _fake_openai)

    assert captured["base_url"] == "https://proxy.example/v1"
    assert captured["api_key"] == "proxy-test-key"
    assert reloaded.client is not None

    monkeypatch.delenv("API_BASE_URL", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)
    importlib.reload(reloaded)
