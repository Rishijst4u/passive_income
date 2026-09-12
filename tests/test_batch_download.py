from pathlib import Path

from src.batch_download import (
    DownloadResult,
    build_parser,
    download_symbol,
    download_universe,
)


class FakeProvider:
    name = "fake"

    def download(
        self,
        symbol,
        start,
        end,
        interval,
    ):
        raise AssertionError(
            "FakeProvider.download should not be called "
            "directly in this test."
        )


def test_download_result_success():
    result = DownloadResult(
        symbol="RELIANCE",
        success=True,
        message="downloaded successfully",
    )

    assert result.symbol == "RELIANCE"
    assert result.success is True
    assert result.message == "downloaded successfully"


def test_download_result_failure():
    result = DownloadResult(
        symbol="RELIANCE",
        success=False,
        message="failed",
    )

    assert result.symbol == "RELIANCE"
    assert result.success is False
    assert result.message == "failed"


def test_parser_defaults_to_dhan_and_5m():
    parser = build_parser()

    args = parser.parse_args(
        [
            "--start",
            "2026-08-01",
            "--end",
            "2026-09-01",
        ]
    )

    assert args.provider == "dhan"
    assert args.interval == "5m"


def test_parser_accepts_instrument_master():
    parser = build_parser()

    args = parser.parse_args(
        [
            "--start",
            "2026-08-01",
            "--end",
            "2026-09-01",
            "--instrument-master",
            "master.csv",
        ]
    )

    assert args.instrument_master == "master.csv"


def test_download_symbol_success(monkeypatch):
    class FakeIngestProvider:
        name = "fake"

    def fake_build_provider(
        name,
        *,
        instrument_master_path=None,
    ):
        assert name == "dhan"
        return FakeIngestProvider()

    def fake_ingest_market_data(
        *,
        provider,
        symbol,
        start,
        end,
        interval,
    ):
        assert provider.name == "fake"
        assert symbol == "RELIANCE"
        assert start == "2026-08-01"
        assert end == "2026-09-01"
        assert interval == "5m"

    monkeypatch.setattr(
        "src.batch_download.build_market_data_provider",
        fake_build_provider,
    )

    monkeypatch.setattr(
        "src.batch_download.ingest_market_data",
        fake_ingest_market_data,
    )

    result = download_symbol(
        symbol="RELIANCE",
        provider_name="dhan",
        start="2026-08-01",
        end="2026-09-01",
        interval="5m",
    )

    assert result.success is True
    assert result.symbol == "RELIANCE"


def test_download_symbol_failure(monkeypatch):
    def fake_build_provider(
        name,
        *,
        instrument_master_path=None,
    ):
        raise RuntimeError("test failure")

    monkeypatch.setattr(
        "src.batch_download.build_market_data_provider",
        fake_build_provider,
    )

    result = download_symbol(
        symbol="RELIANCE",
        provider_name="dhan",
        start="2026-08-01",
        end="2026-09-01",
        interval="5m",
    )

    assert result.success is False
    assert result.symbol == "RELIANCE"
    assert "test failure" in result.message


def test_download_universe_processes_all_symbols(monkeypatch):
    downloaded = []

    def fake_download_symbol(
        *,
        symbol,
        provider_name,
        start,
        end,
        interval,
        instrument_master=None,
    ):
        downloaded.append(symbol)

        return DownloadResult(
            symbol=symbol,
            success=True,
            message="ok",
        )

    monkeypatch.setattr(
        "src.batch_download.download_symbol",
        fake_download_symbol,
    )

    symbols = (
        "RELIANCE",
        "HDFCBANK",
        "ICICIBANK",
    )

    results = download_universe(
        provider_name="dhan",
        start="2026-08-01",
        end="2026-09-01",
        interval="5m",
        symbols=symbols,
    )

    assert downloaded == list(symbols)
    assert len(results) == 3
    assert all(result.success for result in results)


def test_download_universe_preserves_failed_symbols(monkeypatch):
    def fake_download_symbol(
        *,
        symbol,
        provider_name,
        start,
        end,
        interval,
        instrument_master=None,
    ):
        if symbol == "HDFCBANK":
            return DownloadResult(
                symbol=symbol,
                success=False,
                message="bad data",
            )

        return DownloadResult(
            symbol=symbol,
            success=True,
            message="ok",
        )

    monkeypatch.setattr(
        "src.batch_download.download_symbol",
        fake_download_symbol,
    )

    results = download_universe(
        provider_name="dhan",
        start="2026-08-01",
        end="2026-09-01",
        interval="5m",
        symbols=(
            "RELIANCE",
            "HDFCBANK",
            "ICICIBANK",
        ),
    )

    assert len(results) == 3
    assert results[0].success is True
    assert results[1].success is False
    assert results[2].success is True