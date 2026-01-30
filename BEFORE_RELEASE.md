# Before Release Checklist

## Build Matrix (Currently Minimal for Testing)

- [ ] **Expand `rust/.github/workflows/build-wheels.yml`** to build all platforms:
  - Uncomment all commented-out platform jobs (Linux x86_64, Linux aarch64, macOS x86_64, macOS arm64, Windows)
  - Update `create-release.needs` to include all platform jobs
  - Each job should build for Python 3.11, 3.12, 3.13

- [ ] **Expand `.github/workflows/fetch-wheels.yml`** to test all platforms:
  - Change `run-tests` job to use matrix strategy:
    ```yaml
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python-version: ['3.11', '3.12', '3.13']
    ```
  - Update `runs-on: ${{ matrix.os }}`
  - Update Python version to `${{ matrix.python-version }}`

## Branch References

- [ ] Update `.github/workflows/fetch-wheels.yml` to use `main` branch:
  - Line 54: `ref: feat/separate-rust` → `ref: main`

- [ ] Update `rust/.github/workflows/build-wheels.yml` to use `main` branch:
  - All `git clone` commands: change `--branch feat/separate-rust` → `--branch main`

## Workflow Triggers

- [ ] Update `rust/.github/workflows/build-wheels.yml` to only trigger on tags (remove `main` branch trigger):
  - Remove `branches: - main` from the `on.push` section
  - Keep only `tags: - 'v*'` to trigger releases on version tags
