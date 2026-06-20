# Experiment Record: longer_cosine_schedule

Time: 2026-06-19 15:27:35 UTC

## Goal

Train for more data epochs while avoiding the previous cosine schedule decaying too early.

## Changes From Previous Config

File: `configs/rf_final.yaml`

- Changed `max_steps` from `50000` to `70000`.
  - Reason: target a longer training run, roughly 30 data epochs under the expected 8-GPU setup.
- Changed `warmup_steps` from `200` to `500`.
  - Reason: make warmup less abrupt for the large audio adapter branch.
- Changed `lr_decay_steps` from `12000` to `70000`.
  - Reason: previous schedule reached the minimum LR around step 12000, leaving most of the long run at very low LR.
- Changed `min_lr_ratio` from `0.1` to `0.2`.
  - Reason: keep the tail LR at `2e-6` instead of `1e-6`.
- Kept old values commented next to the new values for traceability.

## Expected Impact

- Peak LR remains `1e-5`.
- LR decays across the full 70000-step run instead of saturating at the minimum after 12000 steps.
- Approximate LR values:
  - step 500: `1.000e-05`
  - step 6000: `9.877e-06`
  - step 12000: `9.472e-06`
  - step 30000: `6.940e-06`
  - step 50000: `3.526e-06`
  - step 70000: `2.000e-06`

## Validation Results

- Config parse passed:
  - `max_steps == 70000`
  - `warmup_steps == 500`
  - `lr_decay_steps == 70000`
  - `min_lr_ratio == 0.2`
- LR schedule sanity check passed using the trainer formula.

## Not Validated

- No training smoke was run for this scheduler-only config change.
- Long-run convergence and lip-sync quality remain unvalidated.
