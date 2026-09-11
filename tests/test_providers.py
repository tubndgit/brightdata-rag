import pytest

from brightdata_rag.providers import BrightDataProvider


def test_provider_requires_key(monkeypatch):
    monkeypatch.delenv("BRIGHTDATA_API_KEY", raising=False)
    with pytest.raises(ValueError, match="BRIGHTDATA_API_KEY"):
        BrightDataProvider()


def test_optional_adapters_import_lazily():
    import brightdata_rag

    assert brightdata_rag.__version__ == "0.1.1"
