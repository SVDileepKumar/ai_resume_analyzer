# Contributing

Thanks for your interest in improving this project.

## Development setup

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8899
```

Run tests before submitting a change:

```bash
pytest -q
```

## Pull requests

1. Fork the repository and create a branch from `main`.
2. Keep changes focused on one topic when possible.
3. Add or update tests for behavior changes.
4. Run `pytest` and fix failures.
5. Describe what changed and why in the PR description.

## Code style

- Match existing patterns in the file you edit (imports, typing, logging).
- Avoid drive-by refactors unrelated to the fix or feature.

## Security

Do not commit secrets, API keys, or personal data. If you find a security issue, see `SECURITY.md`.
