# Benchmark Key Policy

## Goal

Every benchmark row used by the offline eval agent must have a stable `benchmark_key` so generated outputs, materialized benchmark assets, and paired metrics can be linked without row-order guessing.

This key is required before formal paired diagnostics such as AKD/PCK can be reported.

## Key Priority

1. If the original benchmark row already has an explicit stable field, use it:
   - `key`
   - `sample_id`
   - `benchmark_key`
   - `id`
2. If no explicit key exists, derive a key from row index plus row content hash.

## Derived Key Format

```text
benchmark_key = zero_padded_row_index_6 + "_" + short_row_hash_8
```

Example:

```text
000001_a1b2c3d4
```

The row index is retained for readability and traceability, but it is not the identity by itself. The hash makes the key sensitive to the benchmark row contents.

## Profile-Aware Row Hash Inputs

Row hash inputs are now defined by the benchmark profile, not by generic evaluator code.

Current profile:

```text
configs/benchmark_profiles/audio_ref_text2video.current.yaml
```

maps canonical fields to the current benchmark row names:

- `imgpath`
- `videopath`
- `wav_path`
- `posepath`
- `prompt`, if present
- `text`, if present

The key policy hashes canonical fields:

- `reference_image`
- `target_video`
- `audio`
- `pose`
- `prompt`
- `text`

`benchmark_index.json` records:

- `benchmark_profile.name/version/hash`
- `row_hash_input_fields`
- canonical media fields
- raw row for traceability

If a future benchmark uses different row field names, update the YAML profile instead of evaluator code.

## Key Source

Each indexed row must record:

- `key_source=explicit_<field>` when an original benchmark field is used
- `key_source=derived_row_hash_key` when the agent derives the key

## Collision Checks

The benchmark index builder must check collisions over the final `benchmark_key`.

Rules:

- `collision_count=0` is required for formal binding.
- If a collision exists, strict binding must fail.
- Missing media paths should be reported as warnings but must not prevent key generation.

## Required Artifacts

Every key materialization run must write:

- `benchmark_index.json`
- `benchmark_index.csv`
- `selected_samples.jsonl`
- `benchmark_materialized/manifest.jsonl`

The materialized manifest must propagate:

- `benchmark_key`
- `benchmark_row_index`
- `benchmark_row_hash`
- original benchmark paths
- materialized ref, target clip, audio, and pose paths when available
- generation/eval profile used for materialization
- benchmark profile name/version/hash

## Forbidden Identity Inference

Formal binding must not use:

- row number alone
- basename guessing
- silent fallback to the N-th row
- old generated-output keys that only identify output order

Old runs that lack `benchmark_key` remain `debug_only` or `debug_not_for_paper`.
