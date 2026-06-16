# Eval Dashboard

`scripts/build_eval_dashboard.py` builds a report-only static dashboard from one or more existing eval output directories. It does not rerun inference, rerun metrics, or modify source eval outputs.

## Single Run

```bash
python scripts/build_eval_dashboard.py \
  --eval_dir eval_outputs/paper_model_native_full_pose_paper50_v1 \
  --output_dir eval_outputs/eval_dashboard_paper50_v1 \
  --title "Paper Model Native Paper50 Dashboard"
```

Open:

```text
eval_outputs/eval_dashboard_paper50_v1/index.html
```

## Multiple Runs

Repeat `--eval_dir` to compare experiments, checkpoints, or eval profiles:

```bash
python scripts/build_eval_dashboard.py \
  --eval_dir eval_outputs/experiment_a_checkpoint_010000 \
  --eval_dir eval_outputs/experiment_a_checkpoint_020000 \
  --eval_dir eval_outputs/experiment_b_checkpoint_020000 \
  --output_dir eval_outputs/eval_dashboard_multi_run_v1 \
  --title "Experiment Comparison"
```

The generated page includes:

- run-level metric summaries
- metric distribution charts
- scatter plots for cross-metric inspection
- representative sample cards
- all-sample browse mode with filtering, sorting, and Load More
- inline generated/target video playback
- in-page image modal for artifact review

## Representative Samples

Representative samples are selected from:

- worst FaceSim / IQA / AES / AKD / PCK cases
- high generated or target OOB rate
- low matched keypoint rate
- failed, skipped, or degraded metrics
- median reference samples
- disagreement cases such as high FaceSim with low PCK

All samples remain available in the `All Samples` tab.

## Path Mode

By default, media links are written as relative paths from the dashboard output directory. This works when opening the HTML file directly or serving the repository root with a local HTTP server.

Use file URIs if needed:

```bash
python scripts/build_eval_dashboard.py \
  --eval_dir eval_outputs/paper_model_native_full_pose_paper50_v1 \
  --output_dir eval_outputs/eval_dashboard_paper50_fileuri_v1 \
  --link_mode file_uri
```

