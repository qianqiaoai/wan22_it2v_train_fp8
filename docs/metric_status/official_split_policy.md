# Official Split Policy

Status: implemented for smoke/dev split generation  
Paper table status: unchanged  
Current date: 2026-06-14

This document defines versioned official split policy for profile-aware offline digital-human evaluation.

## Implemented Splits

Implemented in this stage:

- `official_smoke_10.v1`
- `official_dev_50.v1`

Planned, not implemented:

- `official_paper_100.v1`
- `official_paper_full.v1`

## Split Identity

Split membership is defined by `benchmark_key`.

Forbidden as formal identity:

- numeric row index alone
- basename guessing
- row-order fallback
- debug-only numeric binding

The row index may be stored as metadata, but it is not the identity.

## Benchmark Key Dependency

Official splits depend on a benchmark index generated from `benchmark/benchmark.jsonl` through a benchmark profile.

Current key policy:

- If benchmark rows have explicit key fields, use them.
- If rows lack explicit keys, derive:
  - `zero_padded_row_index_6 + "_" + row_hash_prefix_8`
- Row hash input fields are canonical fields from the active benchmark profile.

Current profile:

```text
configs/benchmark_profiles/audio_ref_text2video.current.yaml
```

maps current benchmark row names to canonical identity fields.

The split generator must fail if `collision_count != 0`.

## Hash Policy

Each split records:

- `split_sha256`
- `benchmark_index_sha256`
- `benchmark_profile_sha256` for new splits

`split_sha256` is computed from split JSON content excluding the `split_sha256` field itself.

`benchmark_index_sha256` is the SHA256 of the canonical benchmark index JSON file referenced by `benchmark_index_path`.

Validation must fail if split or benchmark index hash does not match. For legacy split files that predate benchmark profile hashing, validation emits a warning instead of silently assuming compatibility.

## Subset Policy

Official splits should not be subsetted by default.

If `--max_samples` is used with `--split`, the run must fail unless:

```text
--allow_split_subset true
```

When a subset is explicitly allowed, resolved config must record:

- `split_subset = true`
- `not_paper_comparable = true`

Subset runs are debug/development runs and are not paper comparable.

## Paper Comparability

Paper comparability requires:

- `eval_profile = paper_model_native_v1` or another implemented paper profile
- official split
- `split_sha256`
- `benchmark_index_sha256`
- strict target binding where required by the metric
- frozen metric provenance
- no fallback binding
- no split subset

Ad-hoc sample selection is not paper comparable by default.

## Validation Outputs

Validation writes:

```text
eval_outputs/official_split_validation_v1/
  official_smoke_10.validation.json
  official_dev_50.validation.json
  summary.json
```

Each validation report records:

- split name/version
- sample count
- unique key count
- recorded and computed split hash
- recorded and computed benchmark index hash
- collision count
- duplicate key count
- missing key count
- status
- errors/warnings
