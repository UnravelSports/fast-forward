# Before Release Checklist

## Branch References

- [ ] Update `.github/workflows/fetch-wheels.yml` to use `main` branch instead of `feat/separate-rust`:
  - Line 20: `ref: feat/separate-rust` → `ref: main` (fetch-wheels job)
  - Line 60: `ref: feat/separate-rust` → `ref: main` (run-tests job)
  - Line 63: `git pull origin feat/separate-rust` → `git pull origin main`
  - Line 89: `ref: feat/separate-rust` → `ref: main` (create-release job)
  - Line 92: `git pull origin feat/separate-rust` → `git pull origin main`

## Workflow Triggers

- [ ] Update `rust/.github/workflows/build-wheels.yml` to only trigger on tags (remove `main` branch trigger):
  - Remove `branches: - main` from the `on.push` section
  - Keep only `tags: - 'v*'` to trigger releases on version tags
