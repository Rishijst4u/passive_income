# Milestone 6B.3 — DhanHQ Configuration and CLI Integration

This milestone connects the existing Dhan historical market-data
provider to the command-line interface.

## Added

- `--provider yfinance|dhan`
- Dhan access token loaded from `.env` or environment
- No credentials accepted as CLI arguments
- Public Dhan instrument-master download
- Local instrument-master caching
- Optional `--instrument-master` override
- Existing normalization pipeline remains unchanged
- Existing market-data validation remains unchanged
- Provider-selection tests
- Credential handling tests
- Instrument-master cache tests

## Existing Yahoo workflow

The existing Yahoo workflow remains available:

```powershell
.\.venv\Scripts\python.exe -m src.cli download --provider yfinance --symbol RELIANCE --start 2026-08-01 --end 2026-09-01 --interval 5m