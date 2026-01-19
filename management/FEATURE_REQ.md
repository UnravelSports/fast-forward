# Feature Requests

## PySpark

## New Providers

- [X] [HIGH] Stats Perform
- [X] [MEDIUM] Gradient
- [X] [MEDIUM] Signality
- [ ] [MEDIUM] SciSports
- [ ] [LOW] RespoVision
- [X] [LOW] CDF - needs support for untracked/dangling JSON objects
- [ ] HawkEye skeletal
  - [ ] COCO-WholeBody
    - [ ] image?
    - [ ] Do we have a T-pose for this, how does it relate to HawkEye T-pose?
    - [ ] Infer midHip and neck from middle of hips and shoulders
  - [ ] transform.inverse_kinematics() only for SkeletalTrackingDataset
  - [ ] transform.add_rotation_2d(hip, head, shoulder)
  - [ ] `skeletal_model="hawkeye", "coco_wholebody_23"`
    - [ ] "coco_17" easy, leave things out only
    - [ ] "coco_wholebody_23"
      - [ ] assume neck is middle of left shoulder and right shoulder in 3D
      - [ ] assume midHip is middle of left hip and right hip in 3D
      - [ ] Assume left_thumb, right_thumb, left_pinky, right_pinky are equal to hand_left_20 etc.
  - [ ] Body hierarchy
    - [ ] Use categories for "extra_long" format

## HawkEye Enhancements

- [X] [HIGH] Parallelize Hawkeye file loading (might need sort at the end)
- [X] [HIGH] Futhter parallelize parsing jsons within the files similar to other parsers
- [ ] [MEDIUM] Skeletal transformations - reduce granularity, bone-based long format
- [ ] [MEDIUM] Additional long format with joint column

## Core Features

- [X] [MEDIUM] engine parameter in load_tracking - support Pandas, Polars, PySpark
- [X] [LOW] Pitch dimensions as function on dataset.df [feat/pitch-dimensions]
- [X] [LOW] dataset.df.transform_orientation() method [feat/pitch-dimensions]
- [X] [LOW] dataset.transform_coordinates() method [feat/pitch-dimensions]

## Documentation

- [ ] [LOW] Interactive real-time filtering tool for docs/website
- [ ] Analysis of number or rows, fps, filesize, speed, memory and different approaches, kloppy and kloppy-light, polars, pyspark
- [ ] Add warning that it's a beta release, if you have an issue report it at GitHub, include minimal working example, stacktrace and some anonymized data. Or make a feature request.
- [ ] Add a warning when lazy loading about incomplete meta data
