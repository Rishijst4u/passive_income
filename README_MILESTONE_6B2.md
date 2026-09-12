# Milestone 6B.2 — DhanHQ historical provider

Copy the `src/` and `tests/` files into the existing project.

This milestone adds:
- Dhan instrument-master normalization/resolution
- Dhan 5-minute historical-data provider
- 90-day request chunking
- Dhan response validation/parsing
- offline fixture tests

It does NOT add live order placement or authentication automation.

After copying, run:

.\.venv\Scripts\python.exe -m pytest -q

Do not commit until the test result and diff have been reviewed.

When Dhan KYC/API access is ready, we will add credentials through `.env` and perform the first real RELIANCE qualification download.
