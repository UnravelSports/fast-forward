- [ ] We want to add more tracking data providers, this should be a standard procedure that relies on:

  - [ ] To do this, first make an action plan to add a provider, look at the existing provider implementations, and read REMINDER.md. Create a file ADD_PROVIDER.md that will outline a step by step procedure of everything we need to implement, take note of etc when adding a new provider
  - [ ] Checking original kloppy supported tracking data providers: https://github.com/PySport/kloppy/tree/master/kloppy/infra/serializers/tracking
    - [ ] Compare to see which providers we have not implemented yet, we can ignore `metrica`
    - [ ] Use Kloppy unittests for that specific provider https://github.com/PySport/kloppy/tree/master/kloppy/tests (there are test files in the files folder)
  - [ ] Check active (open) kloppy pull requests to see any not yet implemented Tracking data provider PRs https://github.com/PySport/kloppy/pulls
    - [ ] Verify that the PR is correct and is not missing anything important. If anything is missing, like `only_alive`, or `Orientation.NOT_SET`
    - [ ] Verify that the PR has unit tests associated
    - [ ] Verify that the PR has relevant test files associated
  - [ ] Update scripts/benchmark
  - [ ] Update REMINDER.md



- [ ] Update ADD_PROVIDER
- [ ] Parquet cache > SHA hash > lazy load
- [ ] collect(engine="gpu")
