import json
from datetime import date

import pandas as pd
import pytest

from src.dhan_instruments import DhanInstrumentMaster
from src.dhan_market_data import (
    DhanProvider,
    DhanProviderError,
    MAX_INTRADAY_REQUEST_DAYS,
    _response_to_dataframe,
)


def make_master():
    return DhanInstrumentMaster(
        pd.DataFrame(
            {
                "SEM_SMST_SECURITY_ID": ["1333"],
                "SEM_EXM_EXCH_ID": ["NSE"],
                "SEM_SEGMENT": ["E"],
                "SEM_INSTRUMENT_NAME": ["EQUITY"],
                "SM_SYMBOL_NAME": ["RELIANCE"],
                "SEM_TRADING_SYMBOL": ["RELIANCE"],
            }
        )
    )


def make_response():
    return {
        "open": [100.0, 101.0],
        "high": [102.0, 103.0],
        "low": [99.0, 100.0],
        "close": [101.0, 102.0],
        "volume": [1000, 1200],
        "timestamp": [1770000300, 1770000600],
    }


def test_response_is_normalized_to_canonical_schema():
    result = _response_to_dataframe(make_response(), "RELIANCE")

    assert list(result.columns) == [
        "timestamp",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
    assert str(result["timestamp"].dt.tz) == "Asia/Kolkata"
    assert result["symbol"].unique().tolist() == ["RELIANCE"]
    assert result["volume"].tolist() == [1000, 1200]


def test_response_rejects_missing_field():
    response = make_response()
    response.pop("volume")

    with pytest.raises(DhanProviderError, match="missing fields"):
        _response_to_dataframe(response, "RELIANCE")


def test_response_rejects_mismatched_array_lengths():
    response = make_response()
    response["volume"] = [1000]

    with pytest.raises(DhanProviderError, match="inconsistent lengths"):
        _response_to_dataframe(response, "RELIANCE")


def test_response_rejects_empty_response():
    response = {key: [] for key in make_response()}

    with pytest.raises(DhanProviderError, match="no historical candles"):
        _response_to_dataframe(response, "RELIANCE")


class FakeResponse:
    def __init__(self, payload):
        self._payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return self._payload


class RecordingOpener:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def __call__(self, request, timeout):
        self.requests.append(request)
        return FakeResponse(self.responses.pop(0))


def test_download_builds_dhan_request_and_normalizes_response():
    opener = RecordingOpener([make_response()])
    provider = DhanProvider(
        "test-token",
        make_master(),
        opener=opener,
    )

    result = provider.download(
        "RELIANCE",
        "2026-08-01",
        "2026-08-02",
        "5m",
    )

    assert len(opener.requests) == 1
    request = opener.requests[0]
    body = json.loads(request.data.decode("utf-8"))

    assert body["securityId"] == "1333"
    assert body["exchangeSegment"] == "NSE_EQ"
    assert body["instrument"] == "EQUITY"
    assert body["interval"] == "5"
    assert request.get_header("Access-token") == "test-token"
    assert result["symbol"].eq("RELIANCE").all()


def test_download_chunks_ranges_at_90_days():
    response = make_response()
    opener = RecordingOpener([response, response, response])
    provider = DhanProvider(
        "test-token",
        make_master(),
        opener=opener,
    )

    provider.download(
        "RELIANCE",
        date(2026, 1, 1),
        date(2026, 7, 1),
        "5",
    )

    assert len(opener.requests) == 3
    bodies = [
        json.loads(request.data.decode("utf-8"))
        for request in opener.requests
    ]
    assert bodies[0]["fromDate"].startswith("2026-01-01")
    assert bodies[1]["fromDate"].startswith("2026-04-01")
    assert bodies[2]["fromDate"].startswith("2026-06-30")


def test_download_rejects_unsupported_interval():
    provider = DhanProvider("test-token", make_master())

    with pytest.raises(ValueError, match="interval"):
        provider.download("RELIANCE", "2026-01-01", "2026-01-02", "10m")


def test_download_requires_token():
    with pytest.raises(ValueError, match="access token"):
        DhanProvider("", make_master())


def test_download_rejects_api_error():
    opener = RecordingOpener(
        [
            {
                "errorCode": "DH-902",
                "errorMessage": "Invalid access token",
            }
        ]
    )
    provider = DhanProvider(
        "test-token",
        make_master(),
        opener=opener,
    )

    with pytest.raises(DhanProviderError, match="Invalid access token"):
        provider.download("RELIANCE", "2026-08-01", "2026-08-02", "5")
