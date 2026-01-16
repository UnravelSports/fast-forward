# Bugs

## HawkEye

- [ ] [MEDIUM] Coordinates possibly incorrect - needs verification

## Provider

- [X] [HIGH] SecondSpectrum is missing frame_id=0

## Collect

- [ ] [HIGH]  `.collect` on dataset has not the same features as `.collect()` on df.




python scripts/benchmark_memory.py gradientsports kloppy --no-pushdown
======================================================================

kloppy-light Benchmark
======================

Memory measurement: memory_profiler (accurate)
Providers: gradientsports
Layouts: long, long_ball, wide
Engines: polars
Features: kloppy

Data directory: /Users/jbekkers/PycharmProjects/kloppy-light/data

Operation                     Memory         Time          Rows
---------------------------------------------------------------

Gradientsports tracking file: 727.6 MiB

## Gradientsports  - long layout

Eager loading                2339.62 MiB     1.420s     3,617,344
Lazy (no collect)               1.28 MiB     0.240s           N/A
Lazy (with collect)          1650.28 MiB     1.395s     3,617,344

## Gradientsports  - long_ball layout

Eager loading                1671.47 MiB     1.426s     3,460,495
Lazy (no collect)               0.00 MiB     0.243s           N/A
Lazy (with collect)          1635.55 MiB     1.419s     3,460,495

## Gradientsports  - wide layout

Eager loading                1655.08 MiB     1.493s       156,849
Lazy (no collect)               3.75 MiB     0.242s           N/A

Log saved to: /Users/jbekkers/PycharmProjects/kloppy-light/log/feat_new-provs/20260115_212629.txt
Traceback (most recent call last):
  File "/Users/jbekkers/PycharmProjects/kloppy-light/scripts/benchmark_memory.py", line 846, in `<module>`
    main()
  File "/Users/jbekkers/PycharmProjects/kloppy-light/scripts/benchmark_memory.py", line 835, in main
    run_provider_benchmarks(logger, provider, config)
  File "/Users/jbekkers/PycharmProjects/kloppy-light/scripts/benchmark_memory.py", line 645, in run_provider_benchmarks
    run_layout_benchmarks(logger, provider_name, layout, engine, config)
  File "/Users/jbekkers/PycharmProjects/kloppy-light/scripts/benchmark_memory.py", line 432, in run_layout_benchmarks
    mem, t, result = measure_memory_and_time(lazy_collect)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jbekkers/PycharmProjects/kloppy-light/scripts/benchmark_memory.py", line 208, in measure_memory_and_time
    mem_usage = memory_usage((func,), interval=0.01, max_iterations=1, retval=True)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jbekkers/PycharmProjects/kloppy-light/venv/lib/python3.12/site-packages/memory_profiler.py", line 379, in memory_usage
    returned = f(*args, **kw)
               ^^^^^^^^^^^^^^
  File "/Users/jbekkers/PycharmProjects/kloppy-light/scripts/benchmark_memory.py", line 430, in lazy_collect
    return dataset.tracking.collect(), dataset.metadata, dataset.teams, dataset.players
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jbekkers/PycharmProjects/kloppy-light/venv/lib/python3.12/site-packages/polars/_utils/deprecation.py", line 97, in wrapper
    return function(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jbekkers/PycharmProjects/kloppy-light/venv/lib/python3.12/site-packages/polars/lazyframe/opt_flags.py", line 328, in wrapper
    return function(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jbekkers/PycharmProjects/kloppy-light/venv/lib/python3.12/site-packages/polars/lazyframe/frame.py", line 2429, in collect
    return wrap_df(ldf.collect(engine, callback))
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
polars.exceptions.ComputeError: The output schema of 'LazyFrame.map' is incorrect. Expected: Schema:
name: game_id, field: String
name: frame_id, field: UInt32
name: period_id, field: Int32
name: timestamp, field: Duration('ms')
name: ball_state, field: String
name: ball_owning_team_id, field: String
name: ball_x, field: Float32
name: ball_y, field: Float32
name: ball_z, field: Float32
name: 11488_x, field: Float32
name: 11488_y, field: Float32
name: 11488_z, field: Float32
name: 11853_x, field: Float32
name: 11853_y, field: Float32
name: 11853_z, field: Float32
name: 127_x, field: Float32
name: 127_y, field: Float32
name: 127_z, field: Float32
name: 12960_x, field: Float32
name: 12960_y, field: Float32
name: 12960_z, field: Float32
name: 13118_x, field: Float32
name: 13118_y, field: Float32
name: 13118_z, field: Float32
name: 133_x, field: Float32
name: 133_y, field: Float32
name: 133_z, field: Float32
name: 13880_x, field: Float32
name: 13880_y, field: Float32
name: 13880_z, field: Float32
name: 13881_x, field: Float32
name: 13881_y, field: Float32
name: 13881_z, field: Float32
name: 1393_x, field: Float32
name: 1393_y, field: Float32
name: 1393_z, field: Float32
name: 157_x, field: Float32
name: 157_y, field: Float32
name: 157_z, field: Float32
name: 17_x, field: Float32
name: 17_y, field: Float32
name: 17_z, field: Float32
name: 1876_x, field: Float32
name: 1876_y, field: Float32
name: 1876_z, field: Float32
name: 1907_x, field: Float32
name: 1907_y, field: Float32
name: 1907_z, field: Float32
name: 2066_x, field: Float32
name: 2066_y, field: Float32
name: 2066_z, field: Float32
name: 227_x, field: Float32
name: 227_y, field: Float32
name: 227_z, field: Float32
name: 240_x, field: Float32
name: 240_y, field: Float32
name: 240_z, field: Float32
name: 27_x, field: Float32
name: 27_y, field: Float32
name: 27_z, field: Float32
name: 320_x, field: Float32
name: 320_y, field: Float32
name: 320_z, field: Float32
name: 4056_x, field: Float32
name: 4056_y, field: Float32
name: 4056_z, field: Float32
name: 4566_x, field: Float32
name: 4566_y, field: Float32
name: 4566_z, field: Float32
name: 4656_x, field: Float32
name: 4656_y, field: Float32
name: 4656_z, field: Float32
name: 4661_x, field: Float32
name: 4661_y, field: Float32
name: 4661_z, field: Float32
name: 4666_x, field: Float32
name: 4666_y, field: Float32
name: 4666_z, field: Float32
name: 4691_x, field: Float32
name: 4691_y, field: Float32
name: 4691_z, field: Float32
name: 4695_x, field: Float32
name: 4695_y, field: Float32
name: 4695_z, field: Float32
name: 4703_x, field: Float32
name: 4703_y, field: Float32
name: 4703_z, field: Float32
name: 4717_x, field: Float32
name: 4717_y, field: Float32
name: 4717_z, field: Float32
name: 4732_x, field: Float32
name: 4732_y, field: Float32
name: 4732_z, field: Float32
name: 4912_x, field: Float32
name: 4912_y, field: Float32
name: 4912_z, field: Float32
name: 5006_x, field: Float32
name: 5006_y, field: Float32
name: 5006_z, field: Float32
name: 5035_x, field: Float32
name: 5035_y, field: Float32
name: 5035_z, field: Float32
name: 5131_x, field: Float32
name: 5131_y, field: Float32
name: 5131_z, field: Float32
name: 57_x, field: Float32
name: 57_y, field: Float32
name: 57_z, field: Float32
name: 6020_x, field: Float32
name: 6020_y, field: Float32
name: 6020_z, field: Float32
name: 8000_x, field: Float32
name: 8000_y, field: Float32
name: 8000_z, field: Float32
name: 8008_x, field: Float32
name: 8008_y, field: Float32
name: 8008_z, field: Float32
name: 8011_x, field: Float32
name: 8011_y, field: Float32
name: 8011_z, field: Float32
name: 8077_x, field: Float32
name: 8077_y, field: Float32
name: 8077_z, field: Float32
name: 8079_x, field: Float32
name: 8079_y, field: Float32
name: 8079_z, field: Float32
name: 8089_x, field: Float32
name: 8089_y, field: Float32
name: 8089_z, field: Float32
name: 8113_x, field: Float32
name: 8113_y, field: Float32
name: 8113_z, field: Float32
name: 8328_x, field: Float32
name: 8328_y, field: Float32
name: 8328_z, field: Float32
name: 8397_x, field: Float32
name: 8397_y, field: Float32
name: 8397_z, field: Float32
name: 8400_x, field: Float32
name: 8400_y, field: Float32
name: 8400_z, field: Float32
name: 8416_x, field: Float32
name: 8416_y, field: Float32
name: 8416_z, field: Float32
name: 8538_x, field: Float32
name: 8538_y, field: Float32
name: 8538_z, field: Float32
name: 8555_x, field: Float32
name: 8555_y, field: Float32
name: 8555_z, field: Float32
name: 8564_x, field: Float32
name: 8564_y, field: Float32
name: 8564_z, field: Float32
name: 8593_x, field: Float32
name: 8593_y, field: Float32
name: 8593_z, field: Float32

Got: Schema:
name: game_id, field: String
name: frame_id, field: UInt32
name: period_id, field: Int32
name: timestamp, field: Duration('ms')
name: ball_state, field: String
name: ball_owning_team_id, field: String
name: ball_x, field: Float32
name: ball_y, field: Float32
name: ball_z, field: Float32
name: 11488_x, field: Float32
name: 11488_y, field: Float32
name: 11488_z, field: Float32
name: 127_x, field: Float32
name: 127_y, field: Float32
name: 127_z, field: Float32
name: 12960_x, field: Float32
name: 12960_y, field: Float32
name: 12960_z, field: Float32
name: 13118_x, field: Float32
name: 13118_y, field: Float32
name: 13118_z, field: Float32
name: 133_x, field: Float32
name: 133_y, field: Float32
name: 133_z, field: Float32
name: 13881_x, field: Float32
name: 13881_y, field: Float32
name: 13881_z, field: Float32
name: 1393_x, field: Float32
name: 1393_y, field: Float32
name: 1393_z, field: Float32
name: 157_x, field: Float32
name: 157_y, field: Float32
name: 157_z, field: Float32
name: 1876_x, field: Float32
name: 1876_y, field: Float32
name: 1876_z, field: Float32
name: 1907_x, field: Float32
name: 1907_y, field: Float32
name: 1907_z, field: Float32
name: 2066_x, field: Float32
name: 2066_y, field: Float32
name: 2066_z, field: Float32
name: 240_x, field: Float32
name: 240_y, field: Float32
name: 240_z, field: Float32
name: 27_x, field: Float32
name: 27_y, field: Float32
name: 27_z, field: Float32
name: 320_x, field: Float32
name: 320_y, field: Float32
name: 320_z, field: Float32
name: 4566_x, field: Float32
name: 4566_y, field: Float32
name: 4566_z, field: Float32
name: 4661_x, field: Float32
name: 4661_y, field: Float32
name: 4661_z, field: Float32
name: 4666_x, field: Float32
name: 4666_y, field: Float32
name: 4666_z, field: Float32
name: 4691_x, field: Float32
name: 4691_y, field: Float32
name: 4691_z, field: Float32
name: 4703_x, field: Float32
name: 4703_y, field: Float32
name: 4703_z, field: Float32
name: 4717_x, field: Float32
name: 4717_y, field: Float32
name: 4717_z, field: Float32
name: 4732_x, field: Float32
name: 4732_y, field: Float32
name: 4732_z, field: Float32
name: 5035_x, field: Float32
name: 5035_y, field: Float32
name: 5035_z, field: Float32
name: 57_x, field: Float32
name: 57_y, field: Float32
name: 57_z, field: Float32
name: 6020_x, field: Float32
name: 6020_y, field: Float32
name: 6020_z, field: Float32
name: 8008_x, field: Float32
name: 8008_y, field: Float32
name: 8008_z, field: Float32
name: 8011_x, field: Float32
name: 8011_y, field: Float32
name: 8011_z, field: Float32
name: 8113_x, field: Float32
name: 8113_y, field: Float32
name: 8113_z, field: Float32
name: 8328_x, field: Float32
name: 8328_y, field: Float32
name: 8328_z, field: Float32
name: 8538_x, field: Float32
name: 8538_y, field: Float32
name: 8538_z, field: Float32
name: 8555_x, field: Float32
name: 8555_y, field: Float32
name: 8555_z, field: Float32
name: 8564_x, field: Float32
name: 8564_y, field: Float32
name: 8564_z, field: Float32
name: 8593_x, field: Float32
name: 8593_y, field: Float32
name: 8593_z, field: Float32
