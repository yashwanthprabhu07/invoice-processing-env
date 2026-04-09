import importlib
import os


def test_parse_fields_payload_handles_fenced_json_with_noise():
    import inference

    raw = """```json
{'amount': '5000', 'vendor': 'ABC Ltd', 'date': '10 Oct'}
```
extra text"""

    parsed = inference._parse_fields_payload(raw)

    assert parsed["amount"] == "5000"
    assert parsed["vendor"] == "ABC Ltd"
    assert parsed["date"] == "10 Oct"


def test_run_episode_fallback_when_client_missing(monkeypatch):
    import inference

    monkeypatch.setattr(inference, "client", None)

    score = inference.run_episode("easy")

    assert score == 0.01


def test_run_episode_fallback_on_api_failure(monkeypatch):
    import inference

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
    import inference

    captured = {}

    class _DummyClient:
        pass

    def _fake_openai(base_url, api_key):
        captured["base_url"] = base_url
        captured["api_key"] = api_key
        return _DummyClient()

    monkeypatch.setenv("API_BASE_URL", "https://proxy.example/v1")
    monkeypatch.setenv("API_KEY", "proxy-test-key")

    monkeypatch.setattr("openai.OpenAI", _fake_openai)

    reloaded = importlib.reload(inference)

    assert captured["base_url"] == "https://proxy.example/v1"
    assert captured["api_key"] == "proxy-test-key"
    assert reloaded.client is not None

    monkeypatch.delenv("API_BASE_URL", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)
    importlib.reload(inference)
