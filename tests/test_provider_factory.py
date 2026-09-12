"""Tests for market-data provider construction."""

from pathlib import Path

import pytest

from src import provider_factory
from src.market_data import (
    ProviderError,
    YFinanceProvider,
)


def _master_csv(path: Path) -> None:
    """Create a minimal Dhan instrument-master fixture."""

    path.write_text(
        "SECURITY_ID,EXCH_ID,SEGMENT,"
        "INSTRUMENT,SYMBOL_NAME,TRADING_SYMBOL\n"
        "1333,NSE,E,EQUITY,"
        "RELIANCE,RELIANCE\n",
        encoding="utf-8",
    )


def test_build_yfinance_provider():
    provider = (
        provider_factory.build_market_data_provider(
            "yfinance"
        )
    )

    assert isinstance(
        provider,
        YFinanceProvider,
    )

    assert provider.name == "yfinance"


def test_provider_name_is_case_insensitive():
    provider = (
        provider_factory.build_market_data_provider(
            "YFINANCE"
        )
    )

    assert provider.name == "yfinance"


def test_unknown_provider_is_rejected():
    with pytest.raises(
        ProviderError,
        match="Unsupported market-data provider",
    ):
        provider_factory.build_market_data_provider(
            "unknown"
        )


def test_dhan_requires_access_token(
    monkeypatch,
    tmp_path,
):
    monkeypatch.delenv(
        "DHAN_ACCESS_TOKEN",
        raising=False,
    )

    with pytest.raises(
        ProviderError,
        match="DHAN_ACCESS_TOKEN",
    ):
        provider_factory.build_market_data_provider(
            "dhan",
            env_file=tmp_path / "missing.env",
        )


def test_dhan_uses_cached_instrument_master(
    monkeypatch,
    tmp_path,
):
    cache_path = (
        tmp_path / "api-scrip-master.csv"
    )

    _master_csv(
        cache_path
    )

    monkeypatch.setenv(
        "DHAN_ACCESS_TOKEN",
        "test-token",
    )

    def fail_download():
        raise AssertionError(
            "Network download should not occur"
        )

    monkeypatch.setattr(
        provider_factory,
        "_download_instrument_master",
        fail_download,
    )

    provider = (
        provider_factory.build_market_data_provider(
            "dhan",
            dhan_master_cache=cache_path,
        )
    )

    assert provider.name == "dhan"

    assert (
        provider.instrument_master
        .resolve("RELIANCE")
        .security_id
        == "1333"
    )


def test_dhan_downloads_and_caches_master(
    monkeypatch,
    tmp_path,
):
    cache_path = (
        tmp_path
        / "nested"
        / "api-scrip-master.csv"
    )

    csv_data = (
        b"SECURITY_ID,EXCH_ID,SEGMENT,"
        b"INSTRUMENT,SYMBOL_NAME,TRADING_SYMBOL\n"
        b"1333,NSE,E,EQUITY,"
        b"RELIANCE,RELIANCE\n"
    )

    monkeypatch.setenv(
        "DHAN_ACCESS_TOKEN",
        "test-token",
    )

    monkeypatch.setattr(
        provider_factory,
        "_download_instrument_master",
        lambda: csv_data,
    )

    provider = (
        provider_factory.build_market_data_provider(
            "dhan",
            dhan_master_cache=cache_path,
        )
    )

    assert cache_path.exists()

    assert provider.name == "dhan"

    assert (
        provider.instrument_master
        .resolve("RELIANCE")
        .symbol
        == "RELIANCE"
    )


def test_dhan_can_use_explicit_master_path(
    monkeypatch,
    tmp_path,
):
    cache_path = (
        tmp_path / "cache.csv"
    )

    explicit_path = (
        tmp_path / "explicit.csv"
    )

    _master_csv(
        explicit_path
    )

    monkeypatch.setenv(
        "DHAN_ACCESS_TOKEN",
        "test-token",
    )

    monkeypatch.setattr(
        provider_factory,
        "_download_instrument_master",
        lambda: (
            _ for _ in ()
        ).throw(
            AssertionError(
                "Unexpected master download"
            )
        ),
    )

    provider = (
        provider_factory.build_market_data_provider(
            "dhan",
            instrument_master_path=explicit_path,
            dhan_master_cache=cache_path,
        )
    )

    assert not cache_path.exists()

    assert (
        provider.instrument_master
        .resolve("RELIANCE")
        .security_id
        == "1333"
    )


def test_dhan_env_file_is_supported(
    monkeypatch,
    tmp_path,
):
    env_file = (
        tmp_path / ".env"
    )

    env_file.write_text(
        "DHAN_ACCESS_TOKEN="
        "test-token-from-file\n",
        encoding="utf-8",
    )

    master_path = (
        tmp_path / "master.csv"
    )

    _master_csv(
        master_path
    )

    monkeypatch.delenv(
        "DHAN_ACCESS_TOKEN",
        raising=False,
    )

    provider = (
        provider_factory.build_market_data_provider(
            "dhan",
            env_file=env_file,
            instrument_master_path=master_path,
        )
    )

    assert provider.name == "dhan"


def test_dhan_does_not_fallback_to_yfinance(
    monkeypatch,
    tmp_path,
):
    monkeypatch.delenv(
        "DHAN_ACCESS_TOKEN",
        raising=False,
    )

    with pytest.raises(
        ProviderError
    ):
        provider_factory.build_market_data_provider(
            "dhan",
            env_file=tmp_path / "missing.env",
        )