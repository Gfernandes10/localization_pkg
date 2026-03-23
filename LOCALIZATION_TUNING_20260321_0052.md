# Localization Tuning Analysis - 2026-03-21 00:52

## Context
This note records the offline evaluation of the localization pipeline after running:

```bash
ros2 launch localization_pkg tune_localization_offline.launch.py \
  kalman_measurement_var_xy:=1.5 \
  kalman_process_sigma_a_xy:=5.0
```

The result was compared against the previous run `20260321_0045`.

Logs used:
- `csvLogs/20260321_0052/localization_pkg/filtered_pose.csv`
- `csvLogs/20260321_0052/localization_pkg/ground_truth.csv`
- `csvLogs/20260321_0045/localization_pkg/filtered_pose.csv`
- `csvLogs/20260321_0045/localization_pkg/ground_truth.csv`

The comparison focused mainly on the final stable segment of the replay, especially the last 60 s.

## Main Result
The `0052` tuning is better than `0045`, mainly on the lateral axis:
- `y` noise decreased
- `dy` noise decreased significantly
- `x/dx` improved slightly
- `yaw/dyaw` improved slightly
- `z` stayed essentially the same
- `dz` got slightly worse, but not enough to invalidate the tuning

## Comparison: 0045 vs 0052
Using the last 60 s of each run:

- `x`: `0.00486 -> 0.00452 m`
- `dx`: `0.00177 -> 0.00153 m/s`
- `y`: `0.02335 -> 0.01921 m`
- `dy`: `0.01312 -> 0.00872 m/s`
- `z`: `0.02032 -> 0.02042 m`
- `dz`: `0.00599 -> 0.00686 m/s`
- `yaw`: `0.01454 -> 0.01322 rad`
- `dyaw`: `0.00552 -> 0.00529 rad/s`

Interpretation:
- `x/dx`: acceptable and slightly improved
- `y/dy`: clear improvement; this was the main objective
- `z/dz`: mostly unchanged, with small degradation in `dz`
- `yaw/dyaw`: acceptable and slightly improved

## Axis-by-Axis Assessment
### X / dX
Current behavior is good.
- `x` noise is low
- `dx` noise is low
- no immediate need for additional tuning in `XY` because of `x`

### Y / dY
This remains the weakest axis, but the new tuning is clearly better.
- `y` dropped from about `2.34 cm RMS` to `1.92 cm RMS`
- `dy` dropped from about `1.31 cm/s RMS` to `0.87 cm/s RMS`

This is a meaningful improvement and makes the estimate more usable for control.

### Z / dZ
This axis is still acceptable.
- `z` stayed at about the same noise level
- `dz` became slightly noisier

This is the only relevant trade-off introduced by the new `XY` tuning.

### Yaw / dYaw
This axis is in good condition.
- moderate angular noise
- slightly better than the previous run
- no immediate need to retune yaw based on this comparison

## Conclusion
At this point, the `0052` tuning is a better compromise than `0045`.

Recommended interpretation:
- `x/dx`: OK
- `y/dy`: acceptable now, and clearly better than before
- `z/dz`: OK
- `yaw/dyaw`: OK

## Recommendation
Do not continue pushing `XY` smoothing aggressively right now.

Reason:
- the main target, `y/dy`, already improved significantly
- further smoothing may start adding too much delay
- `dz` already showed a small degradation

Practical recommendation:
- keep this tuning as the current baseline for `XY`
- only do one more `XY` test if there is a clear control-driven reason to reduce `y/dy` noise further

If another `XY` test is needed, the next conservative step should be:

```bash
ros2 launch localization_pkg tune_localization_offline.launch.py \
  kalman_measurement_var_xy:=1.5 \
  kalman_process_sigma_a_xy:=4.5
```

This should be treated as a fine adjustment, not as a necessary correction.

## Current Best XY Offline Tuning
```bash
ros2 launch localization_pkg tune_localization_offline.launch.py \
  kalman_measurement_var_xy:=1.5 \
  kalman_process_sigma_a_xy:=5.0
```
