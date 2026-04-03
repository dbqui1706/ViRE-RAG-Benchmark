from rag_bench.config import RagConfig


def test_config_defaults():
    cfg = RagConfig(csv_path="data/CSConDa.csv", embed_model="default")
    assert cfg.top_k == 5
    assert cfg.max_samples == 200
    assert cfg.sample_seed == 42
    assert cfg.llm_provider == "fpt"
    assert cfg.llm_model == "Qwen3-32B"


def test_config_from_env(monkeypatch):
    monkeypatch.setenv("FPT_API_KEY", "test-key")
    monkeypatch.setenv("FPT_BASE_URL", "https://test.api")
    cfg = RagConfig.from_env(csv_path="data/CSConDa.csv", embed_model="default")
    assert cfg.llm_api_key == "test-key"
    assert cfg.llm_base_url == "https://test.api"
