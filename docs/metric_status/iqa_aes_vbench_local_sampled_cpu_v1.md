# IQA/AES VBench Local Sampled CPU V1

## Metric Variant

- Name: `iqa_aes_sampled`
- Full variant: `iqa_aes_vbench_local_sampled_cpu_v1`
- Partial variants:
  - `iqa_vbench_local_sampled_cpu_v1`
  - `aes_vbench_local_sampled_cpu_v1`
- Cost level: `high`
- Scope: sampled no-reference video technical quality and aesthetic quality.
- Status: `dev_full_admitted`
- Paper ready: `false`
- Eval profile: `dev_native`

## Backend

- Implementation: local VBench reference adapter
- VBench version: `0.1.5`
- VBench commit: `61dfa7d9136e35023edd1266a13de679b52fdd31`
- IQA reference: `/mnt/data/cv/yutan/Baselines/VBench/vbench/imaging_quality.py`
- AES reference: `/mnt/data/cv/yutan/Baselines/VBench/vbench/aesthetic_quality.py`
- Adapter: `eval_agent/third_party_adapters/vbench_quality.py`
- Runtime device used in validation: CPU / CPUExecution
- Offline mode: `true`
- Allow download: `false`
- `pyiqa`: `0.1.15.post2`
- Current Torch CUDA availability: `false`

The adapter does not call VBench download helpers. It preflights local files and loads model weights explicitly.

`pyiqa` installation happened before this hardening run and accessed the package index. Metric runtime itself does not access the network or download model files.

## Local Models

Model root:

```text
/mnt/data/nlp/user/qiaoqian/newproject/eval_agent_pretrained
```

Required files:

| Component | Path | SHA256 |
| --- | --- | --- |
| IQA MUSIQ | `pyiqa_model/musiq_spaq_ckpt-358bb6af.pth` | `358bb6af275e28ea56821d44fb55c6cb83645db11f394d3ad65b2d149965ab50` |
| AES linear head | `aesthetic_model/emb_reader/sa_0_4_vit_l_14_linear.pth` | `2cd4e60f4f24ae3bcd57b847b13c1f3ba27edc28cc1a7f9ce74ee9f421243cba` |
| CLIP ViT-L/14 | `clip_model/ViT-L-14.pt` | `b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836` |

## Score Scale

- IQA: VBench MUSIQ raw score divided by `100`.
- AES: LAION aesthetic linear head score divided by `10`.
- Direction: higher is better for both IQA and AES.

These values are backend-specific and must not be compared against other IQA/AES backends without matching `metric_variant` and `provenance_fingerprint`.

The current `dev_native` profile is not directly comparable to the 5s/720p HOIVG-Bench paper protocol because it evaluates generated outputs at their native resolution and duration with sampled frames.

## Sampling And Artifacts

- Frame extraction: ffmpeg sampled frame extraction.
- Default sampling: uniform `8` frames.
- OpenCV video IO: not used.
- Frame directory: `artifacts/<sample_id>/iqa_aes/frames/`

Each sample writes:

```text
artifacts/<sample_id>/iqa_aes/
  frames/
  sampled_frames_contact_sheet.jpg
  iqa_aes_per_frame.jsonl
  worst_iqa_frames.jpg
  worst_aes_frames.jpg
  frame_extract_command.txt
  frame_extract_stdout.txt
  frame_extract_stderr.txt
  sampled_frame_paths.json
  input_video_probe.json
  iqa_aes_error.json
```

`iqa_aes_error.json` is written when the metric fails or is skipped because the backend is unavailable.

## Status Rules

- Generated video unreadable: `failed`, null scores.
- ffmpeg extraction failure: `failed`, null scores.
- Full backend requested but only IQA or AES is available: `degraded`, available component scores retained.
- IQA and AES both unavailable: `failed` or `skipped`, controlled by `--iqa_aes_backend_unavailable_status`.
- High sampled black/frozen frame rate: `degraded`, scores retained, `failure_taxonomy=visual_quality`.
- Failed/skipped numeric scores are always `null`.
- Failed/skipped samples are not aggregated as zero.

## Validation Summary

Commands were run after installing `pyiqa==0.1.15.post2`.

Environment audit:

- `pip freeze`: saved to `eval_outputs/env_audit_iqa_aes/pip_freeze.txt`
- `pip check`: saved to `eval_outputs/env_audit_iqa_aes/pip_check.txt`
- Dependency summary: saved to `eval_outputs/env_audit_iqa_aes/dependency_conflicts.json`
- Torch CUDA status: saved to `eval_outputs/env_audit_iqa_aes/torch_cuda_status.json`
- Import smoke: saved to `eval_outputs/env_audit_iqa_aes/import_smoke.txt`
- `pip install pyiqa` changed `platformdirs`, `fsspec`, and `dill`.
- `pip check` currently crashes with `ValueError: not enough values to unpack`, so dependency consistency cannot be fully trusted.
- Recommendation: use a separate eval environment/container or restore the training environment from lockfile before long training jobs.

| Case | Result |
| --- | --- |
| `py_compile` | Passed |
| `--list_metrics` | Shows `iqa_aes_sampled` |
| `--list_presets` | `full` includes `iqa_aes_sampled`; `quick` unchanged |
| Missing model paths | `failed`, null scores, concrete missing-path errors |
| CUDA preflight while Torch CUDA unavailable | `failed`, null scores, `torch.cuda.is_available() is false` |
| Real manifest 1 sample, 2 frames | `ok`, full IQA+AES |
| Real manifest 5 samples, 8 frames | `ok`: 5/5 |
| Black video | `degraded`, scores retained, black/frozen warnings |
| Fingerprint changes with `num_frames` | Passed |
| Cache key readiness | Includes input video hash, sampled frame hashes, sampled indices/timestamps, model hashes, preprocess, score scale, and provenance fingerprint |
| VBench parity | Passed on one real sample, same sampled PNG frames, absolute diff `0.0` for both IQA and AES |
| Synthetic degradation sanity | Generated 7 variants; severe blur/noise/black generally lowered scores, frozen video raised AES and is flagged |
| `dev_full` baseline | Passed on 11-sample manifest |

5-sample smoke:

```text
manifest: inference_outputs/audio_i2v_batch_20260608_1610/manifest.jsonl
output_dir: /tmp/eval_iqa_5sample
metrics: sanity, iqa_aes_sampled
num_frames: 8
device: cpu
status: 5 ok / 0 degraded / 0 failed / 0 skipped
mean iqa_mean: 0.7366528129577636
mean aes_mean: 0.6223196014761925
aggregate provenance status: ok_single_provenance
```

VBench parity:

```text
output: eval_outputs/iqa_aes_parity/vbench_parity.csv
sample_id: 0001
adapter_iqa: 0.7068966770172119
vbench_iqa: 0.7068966770172119
iqa_abs_diff: 0.0
adapter_aes: 0.6011482626199722
vbench_aes: 0.6011482626199722
aes_abs_diff: 0.0
parity_status: pass
```

Synthetic degradation sanity:

```text
output: eval_outputs/iqa_aes_degradation/degradation_sanity.csv
original: iqa_mean=0.7133741188, aes_mean=0.5762067288
gaussian_blur: iqa_mean=0.2904206133, aes_mean=0.4694230407
downsample_upscale: iqa_mean=0.6337967587, aes_mean=0.5163284615
jpeg_low_quality: iqa_mean=0.6738369942, aes_mean=0.4950476736
noise_added: iqa_mean=0.5209259129, aes_mean=0.5756254345
frozen_video: iqa_mean=0.7132896042, aes_mean=0.6544932872, score_discriminativeness_warning=true
black_video: iqa_mean=0.1643444252, aes_mean=0.3357174024
```

`frozen_video` is a known diagnostic warning case: AES increased versus the original even though the video is frozen. The metric marks the sample `degraded` because sampled frozen-frame rate is `1.0`.

`dev_full` baseline:

```text
output_dir: eval_outputs/dev_full_baseline_v1
manifest: inference_outputs/audio_i2v_batch_20260608_1610/manifest.jsonl
sample_count: 11
metrics: sanity, facesim_sampled, sync_global, iqa_aes_sampled
sanity: 11 degraded
facesim_sampled: 11 ok
sync_global: 11 ok, score_interpretation=uncalibrated
iqa_aes_sampled: 11 ok
mean iqa_mean: 0.7239207021
mean aes_mean: 0.6091635444
```

Admission decision:

- `iqa_aes_sampled` is admitted to `full` / `dev_full`.
- `iqa_aes_sampled` is not admitted to `quick`.
- `paper_ready=false` until a paper protocol is fixed, including official split, resolution/duration normalization, dense or protocol-specific sampling, and parity/admission thresholds.

## Known Limitations

- CPU inference is slow; 5 samples with 8 frames took several minutes.
- Torch currently reports `cuda_available=false` in this environment despite visible GPUs from `nvidia-smi`.
- IQA/AES are sampled, not dense.
- This metric does not evaluate text alignment, motion quality, temporal flicker, FVD/FID, or realtime behavior.
