# Before Release Checklist

## ### TODO:

- [ ] Host Docs on readthedocs.io
- [ ] Reset github repo to completely clean main branch, no commit history

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

## Dependencies

- [ ] Switch kloppy dependency from git URL to PyPI version:

  - In `rust/.github/workflows/build-wheels.yml`: change `kloppy @ git+https://github.com/PySport/kloppy.git` → `kloppy` (or `kloppy>=X.Y.Z`)
  - In `.github/workflows/fetch-wheels.yml`: change `pip install "kloppy @ git+https://..."` → `pip install kloppy`
  - This requires a compatible kloppy version to be published on PyPI
- [ ] Remove debug/inspect steps from both workflows (Verify build inputs, Inspect wheel contents)

## Disabled Features (see DISABLED_FEATURES.md)

- [ ] Re-enable lazy loading (`lazy=True`) — remove `NotImplementedError` guards
- [ ] Re-enable cache loading (`from_cache=True`) — remove guards, re-add cache exports to `__init__.py`

## Workflow Triggers

- [X] ~~Update `rust/.github/workflows/build-wheels.yml` to only trigger on tags~~ (done — removed `branches: - main`)
