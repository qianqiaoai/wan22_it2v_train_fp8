#!/usr/bin/env bash
set -euo pipefail

ROOT="inference_outputs/ckpt010500_ref5_bgm_train_native_736x1280_10s_cfg_compare_v1"
SOURCE_INPUT_ROOT="inference_outputs/ckpt010500_ref5_bgm_train_720p_cfg_compare_v1/inputs"
CKPT="logs/wan2.2_5B_fp8_audio_20260621_210556/checkpoint_model_010500/model.pt"
WIDTH=736
HEIGHT=1280
FRAME_NUM=249
FPS=25
MAX_JOBS=5

mkdir -p "$ROOT/logs" "$ROOT/inputs"
if [ ! -e "$ROOT/inputs/ref_5" ]; then
  ln -s "$(pwd)/$SOURCE_INPUT_ROOT/ref_5" "$ROOT/inputs/ref_5"
fi
if [ ! -e "$ROOT/inputs/audio_5" ]; then
  ln -s "$(pwd)/$SOURCE_INPUT_ROOT/audio_5" "$ROOT/inputs/audio_5"
fi

echo "[$(date -Is)] Launching native ${WIDTH}x${HEIGHT}, ${FRAME_NUM} frames, ${FPS}fps inference." | tee -a "$ROOT/logs/launcher.log"

CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python scripts/infer_audio_i2v.py \
  --checkpoint_path "$CKPT" \
  --ref_dir "$ROOT/inputs/ref_5" \
  --audio_dir "$ROOT/inputs/audio_5" \
  --output_dir "$ROOT/no_cfg_native_736x1280_249f" \
  --width "$WIDTH" \
  --height "$HEIGHT" \
  --frame_num "$FRAME_NUM" \
  --fps "$FPS" \
  --cfg_mode off \
  --guide_scale 1.0 \
  --pairing match \
  --max_jobs "$MAX_JOBS" \
  > "$ROOT/logs/no_cfg_native_736x1280_249f_stdout.log" \
  2> "$ROOT/logs/no_cfg_native_736x1280_249f_stderr.log" &
PID0=$!

CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 python scripts/infer_audio_i2v.py \
  --checkpoint_path "$CKPT" \
  --ref_dir "$ROOT/inputs/ref_5" \
  --audio_dir "$ROOT/inputs/audio_5" \
  --output_dir "$ROOT/textcfg3_native_736x1280_249f" \
  --width "$WIDTH" \
  --height "$HEIGHT" \
  --frame_num "$FRAME_NUM" \
  --fps "$FPS" \
  --cfg_mode text \
  --guide_scale 3.0 \
  --pairing match \
  --max_jobs "$MAX_JOBS" \
  > "$ROOT/logs/textcfg3_native_736x1280_249f_stdout.log" \
  2> "$ROOT/logs/textcfg3_native_736x1280_249f_stderr.log" &
PID1=$!

echo "$PID0" > "$ROOT/logs/no_cfg_native_736x1280_249f.pid"
echo "$PID1" > "$ROOT/logs/textcfg3_native_736x1280_249f.pid"
echo "[$(date -Is)] Launched no_cfg PID=$PID0, textcfg3 PID=$PID1" | tee -a "$ROOT/logs/launcher.log"

set +e
wait "$PID0"
STATUS0=$?
wait "$PID1"
STATUS1=$?
set -e
echo "[$(date -Is)] Finished no_cfg status=$STATUS0, textcfg3 status=$STATUS1" | tee -a "$ROOT/logs/launcher.log"
exit $(( STATUS0 != 0 || STATUS1 != 0 ))
