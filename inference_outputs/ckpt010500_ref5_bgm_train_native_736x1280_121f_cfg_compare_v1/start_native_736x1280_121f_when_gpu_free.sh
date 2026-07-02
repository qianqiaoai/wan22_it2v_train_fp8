#!/usr/bin/env bash
set -euo pipefail

ROOT="inference_outputs/ckpt010500_ref5_bgm_train_native_736x1280_121f_cfg_compare_v1"
SOURCE_INPUT_ROOT="inference_outputs/ckpt010500_ref5_bgm_train_720p_cfg_compare_v1/inputs"
CKPT="logs/wan2.2_5B_fp8_audio_20260621_210556/checkpoint_model_010500/model.pt"
WIDTH=736
HEIGHT=1280
FRAME_NUM=121
FPS=25
MAX_JOBS=5
GPU_MEM_FREE_THRESHOLD_MB=2000
GPU_UTIL_FREE_THRESHOLD=20

mkdir -p "$ROOT/logs" "$ROOT/inputs"
if [ ! -e "$ROOT/inputs/ref_5" ]; then
  ln -s "$(pwd)/$SOURCE_INPUT_ROOT/ref_5" "$ROOT/inputs/ref_5"
fi
if [ ! -e "$ROOT/inputs/audio_5" ]; then
  ln -s "$(pwd)/$SOURCE_INPUT_ROOT/audio_5" "$ROOT/inputs/audio_5"
fi

cat > "$ROOT/run_config.json" <<EOF
{
  "checkpoint": "$CKPT",
  "width": $WIDTH,
  "height": $HEIGHT,
  "frame_num": $FRAME_NUM,
  "fps": $FPS,
  "max_jobs": $MAX_JOBS,
  "cfgs": [
    {"name": "no_cfg", "cfg_mode": "off", "guide_scale": 1.0, "gpu": 0},
    {"name": "textcfg3", "cfg_mode": "text", "guide_scale": 3.0, "gpu": 1}
  ],
  "note": "Native inference at 736x1280, no post-resize."
}
EOF

echo "[$(date -Is)] Waiting for GPUs 0 and 1 to become available for native ${WIDTH}x${HEIGHT}, ${FRAME_NUM} frames, ${FPS}fps..." | tee -a "$ROOT/logs/launcher.log"
while true; do
  mapfile -t GPU_LINES < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)
  gpu0_ok=0
  gpu1_ok=0
  for line in "${GPU_LINES[@]}"; do
    IFS=',' read -r idx mem util <<< "$line"
    idx="${idx// /}"
    mem="${mem// /}"
    util="${util// /}"
    if [ "$idx" = "0" ] && [ "$mem" -le "$GPU_MEM_FREE_THRESHOLD_MB" ] && [ "$util" -le "$GPU_UTIL_FREE_THRESHOLD" ]; then
      gpu0_ok=1
    fi
    if [ "$idx" = "1" ] && [ "$mem" -le "$GPU_MEM_FREE_THRESHOLD_MB" ] && [ "$util" -le "$GPU_UTIL_FREE_THRESHOLD" ]; then
      gpu1_ok=1
    fi
  done
  if [ "$gpu0_ok" = "1" ] && [ "$gpu1_ok" = "1" ]; then
    break
  fi
  echo "[$(date -Is)] GPUs busy: ${GPU_LINES[*]}" >> "$ROOT/logs/launcher.log"
  sleep 60
done

echo "[$(date -Is)] GPUs are available. Launching native ${WIDTH}x${HEIGHT}, ${FRAME_NUM}f inference." | tee -a "$ROOT/logs/launcher.log"

CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python scripts/infer_audio_i2v.py \
  --checkpoint_path "$CKPT" \
  --ref_dir "$ROOT/inputs/ref_5" \
  --audio_dir "$ROOT/inputs/audio_5" \
  --output_dir "$ROOT/no_cfg_native_736x1280_121f" \
  --width "$WIDTH" \
  --height "$HEIGHT" \
  --frame_num "$FRAME_NUM" \
  --fps "$FPS" \
  --cfg_mode off \
  --guide_scale 1.0 \
  --pairing match \
  --max_jobs "$MAX_JOBS" \
  > "$ROOT/logs/no_cfg_native_736x1280_121f_stdout.log" \
  2> "$ROOT/logs/no_cfg_native_736x1280_121f_stderr.log" &
PID0=$!

CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 python scripts/infer_audio_i2v.py \
  --checkpoint_path "$CKPT" \
  --ref_dir "$ROOT/inputs/ref_5" \
  --audio_dir "$ROOT/inputs/audio_5" \
  --output_dir "$ROOT/textcfg3_native_736x1280_121f" \
  --width "$WIDTH" \
  --height "$HEIGHT" \
  --frame_num "$FRAME_NUM" \
  --fps "$FPS" \
  --cfg_mode text \
  --guide_scale 3.0 \
  --pairing match \
  --max_jobs "$MAX_JOBS" \
  > "$ROOT/logs/textcfg3_native_736x1280_121f_stdout.log" \
  2> "$ROOT/logs/textcfg3_native_736x1280_121f_stderr.log" &
PID1=$!

echo "$PID0" > "$ROOT/logs/no_cfg_native_736x1280_121f.pid"
echo "$PID1" > "$ROOT/logs/textcfg3_native_736x1280_121f.pid"
echo "[$(date -Is)] Launched no_cfg PID=$PID0, textcfg3 PID=$PID1" | tee -a "$ROOT/logs/launcher.log"

set +e
wait "$PID0"
STATUS0=$?
wait "$PID1"
STATUS1=$?
set -e
echo "[$(date -Is)] Finished no_cfg status=$STATUS0, textcfg3 status=$STATUS1" | tee -a "$ROOT/logs/launcher.log"
exit $(( STATUS0 != 0 || STATUS1 != 0 ))
