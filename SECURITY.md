# Security and data-handling policy

FU-TCM accepts security reports for accidental credential or sensitive-data
exposure, unsafe dependency use, and vulnerabilities in its workflow code.

## Credentials

- Copy `.env.example` to a local `.env` file and keep the real file untracked.
- Load keys only from environment variables. Never print a key, its prefix,
  suffix, hash, or exact length.
- If a credential is committed, revoke and rotate it before removing it from
  the repository. Deleting the current file does not remove Git history.

## Data

- Store source and generated datasets only in paths ignored by `.gitignore`.
- Use a separate private data repository or controlled object storage when
  data must be shared.
- Run `python scripts/check_repository_hygiene.py` before every push.

## Reporting

Report accidental disclosure privately to the repository owner. Do not place
real credentials or patient data in a public GitHub issue.
