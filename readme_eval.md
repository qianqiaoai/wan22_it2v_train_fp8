# 离线数字人评测说明

本文档是当前项目中 `offline-digital-human-eval` Skill 的中文使用说明。评测入口统一为：

```bash
python auto_eval/eval_digital_human_agent.py <subcommand>
```

不要直接依赖零散的一次性脚本。后续迁移或打包 Skill 时，也应只暴露这个公共 CLI。

## 1. 这个评测工具做什么

用于离线评测数字人 / 视频生成结果，当前支持：

- `sanity`：视频/音频可读性、帧率、分辨率、黑屏/冻结等基础检查。
- `facesim_sampled`：基于 InsightFace 的身份相似度采样评测。
- `sync_global`：基于 SyncNet v2 的全局音画同步诊断，目前仍是 diagnostic / uncalibrated。
- `iqa_aes_sampled`：基于本地 VBench 资源的 IQA/AES 质量和美学评分。
- `dwpose_sampled`：DWPose 采样姿态提取诊断。
- `akd_pck_body18`：基于 body18 的 AKD/PCK，要求 key-preserving manifest、严格 target binding 和可比较 target video。

不包含实时评测，不包含 TA/VQ/MQ/temporal flicker 等未实现指标。

## 2. 这个 Skill 不包含什么

Skill 包不应该包含：

- 真实模型权重
- `resources.local.yaml`
- `eval_outputs`
- checkpoints
- benchmark 数据
- 私有机器路径

真实权重由使用者自己准备，许可证、来源、下载方式也由使用者自行确认。

## 3. 外部权重目录

推荐准备如下目录结构。目录实际位置可以自定义：

```text
eval_agent_pretrained/
  dwpose/
    yolox_l.onnx
    dw-ll_ucoco_384.onnx

  syncnet/
    syncnet_v2.model

  insightface/
    models/
      antelopev2/
        scrfd_10g_bnkps.onnx
        glintr100.onnx
        1k3d68.onnx
        2d106det.onnx
        genderage.onnx

  aesthetic_model/
  clip_model/
  pyiqa_model/
```

文件名和相对路径可以改，但必须在 `resources.local.yaml` 中写清楚。

## 4. resources.local.yaml

不要把 `resources.local.yaml` 放进 Skill 包。可以在项目外或本地 eval 输出目录中创建，例如：

```yaml
pretrained_root: "/ABS/PATH/TO/eval_agent_pretrained"

resources:
  dwpose:
    adapter: dwpose_onnx
    detector_model_relpath: dwpose/yolox_l.onnx
    pose_model_relpath: dwpose/dw-ll_ucoco_384.onnx
    provider: cpu

  syncnet:
    adapter: syncnet_v2
    model_relpath: syncnet/syncnet_v2.model

  facesim:
    adapter: insightface_antelopev2
    model_root_relpath: insightface
    model_pack: antelopev2
    provider: cpu

  vbench_quality:
    adapter: vbench_local
    model_root_relpath: .
    device: cpu
    offline: true
```

示例文件在：

```text
.codex/skills/offline-digital-human-eval/templates/resources.example.yaml
configs/resources.example.yaml
```

这些示例只保留占位符，不能直接当真实配置使用。

## 5. 资源检查

跑任何 metric 前，先执行：

```bash
python auto_eval/eval_digital_human_agent.py doctor_resources \
  --resource_config /path/to/resources.local.yaml
```

如果失败，先修 `resources.local.yaml` 或权重目录。评测代码不允许自动猜测 `/mnt/data` 或其他私有路径。

## 6. Benchmark profile

benchmark 字段通过 YAML profile 映射到统一 schema。换 benchmark schema 时，只改 profile，不改评测代码。

当前示例：

```text
auto_eval/configs/benchmark_profiles/audio_ref_text2video.current.yaml
.codex/skills/offline-digital-human-eval/templates/benchmark_profile.example.yaml
```

验证命令：

```bash
python auto_eval/eval_digital_human_agent.py benchmark profile_validate \
  --benchmark benchmark/benchmark.jsonl \
  --benchmark_profile auto_eval/configs/benchmark_profiles/audio_ref_text2video.current.yaml \
  --output_dir eval_outputs/<run_id>/benchmark_profile_validate
```

禁止正式 binding 使用 numeric row fallback、basename fallback 或第 N 行 fallback。

## 7. 常用命令

列出指标：

```bash
python auto_eval/eval_digital_human_agent.py --list_metrics
```

列出 preset：

```bash
python auto_eval/eval_digital_human_agent.py --list_presets
```

运行 quick：

```bash
python auto_eval/eval_digital_human_agent.py manifest_eval \
  --manifest /path/to/manifest.jsonl \
  --output_dir eval_outputs/<run_id> \
  --resource_config /path/to/resources.local.yaml \
  --benchmark_profile /path/to/benchmark_profile.yaml \
  --preset quick
```

运行 full_pose：

```bash
python auto_eval/eval_digital_human_agent.py manifest_eval \
  --manifest /path/to/manifest.jsonl \
  --output_dir eval_outputs/<run_id> \
  --resource_config /path/to/resources.local.yaml \
  --benchmark_profile /path/to/benchmark_profile.yaml \
  --preset full_pose
```

生成 report-only paper candidate：

```bash
python auto_eval/eval_digital_human_agent.py report_paper_candidate \
  --source_eval_dir eval_outputs/<source_eval_dir> \
  --output_dir eval_outputs/<candidate_output_dir>
```

## 8. 用自然语言发起评测

在当前项目里，也可以直接用自然语言描述评测需求，让 Codex 帮你选择命令并执行。例如：

```text
帮我用 quick 评测这个 manifest：
eval_outputs/key_preserving_infer_actual_50samples_v1/inference/manifest.jsonl
资源配置用：
eval_outputs/portability_refactor_v1/resources.local.yaml
只跑前 3 个样本
```

或者：

```text
帮我看这个实验的 10 个视频质量和身份保持怎么样。
```

通常需要提供：

- manifest 路径，或视频目录
- `resources.local.yaml` 路径
- 想看的指标范围，例如身份、质量、口型、姿态，或者“都看一下”
- 样本数，例如 1、3、10、50

默认建议先跑小样本 smoke，确认资源、manifest、benchmark profile 都正常，再跑更大的评测。

Codex 执行时应该遵守：

1. 先跑 `doctor_resources`。
2. 确认或验证 benchmark profile。
3. 根据需求选择 preset 或 explicit metrics。
4. 执行 `manifest_eval`。
5. 汇总 status、指标均值、失败原因和报告路径。

如果涉及 AKD/PCK，必须确认 manifest 有 `benchmark_key`、严格 target binding 和 materialized target clip，不允许使用 numeric fallback、basename fallback 或第 N 行 fallback。

## 9. 评测结果如何查看

每次 `manifest_eval` 跑完后，输出目录都会包含标准报告文件：

```text
<output_dir>/
  metrics.json
  resolved_config.json
  metric_plan.json
  current_run_table.csv
  paper_table.csv
  html_report.html
```

最常用的是：

```text
<output_dir>/html_report.html
```

例如：

```text
eval_outputs/internal_skill_smoke_v1/full_pose_1/html_report.html
```

如果在远程服务器上查看，可以在项目目录启动 HTTP 服务，再通过端口转发在本地浏览器打开：

```bash
python -m http.server 8898 --bind 127.0.0.1
```

然后在本地浏览器访问：

```text
http://127.0.0.1:8898/
```

进入对应的 `eval_outputs/.../html_report.html` 即可。

对于大量实验和大量视频，单个表格会不够直观。推荐额外生成 dashboard / review HTML，总览多个实验和权重：

- 代表性样本优先展示，避免 50/100/300 个视频一次性铺满页面。
- 其余样本通过“更多”展开。
- 视频直接内嵌播放，不只显示文件路径。
- 指标既显示数字，也用图表展示分布和对比。
- 卡片式展示每个样本，适合人工快速查看生成效果。
- 对 worst cases、low FaceSim、low IQA/AES、high AKD、low PCK、high OOB 等样本单独标记。

后续可以直接说：

```text
帮我把这次评测结果做成一个中文 HTML 总览页，视频直接显示，代表性样本优先，其余点击更多。
```

Codex 会基于已有 `metrics.json` / `current_run_table.csv` / artifacts 生成或更新适合浏览的 HTML，不需要重跑 metrics。

### 多实验 / 多权重对比 Dashboard

如果你有很多训练实验，每个实验又有很多 checkpoint，可以生成专门的 compare dashboard。它用于两类对比：

- **纵向对比**：同一个实验内，不同 checkpoint 在同一个 case 上的生成结果。
- **横向对比**：不同实验之间，各自最优 checkpoint 在同一个 case 上的生成结果。

页面会用 `benchmark_key` 对齐同一个 case，并展示：

- 生成视频 / target video / reference image
- FaceSim / IQA / AES / LSE-C / LSE-D / AKD / PCK / 关键点匹配率
- DWPose / AKD-PCK 姿态 overlay
- checkpoint 趋势图
- 代表 case / worst case

其中 SyncNet 相关指标为：

- `LSE-C↑`：音画同步置信分，越高越好。
- `LSE-D↓`：音画同步距离，越低越好。

它们当前仍是 diagnostic / uncalibrated 指标，用于可视化和调试，不进入正式 paper table。

推荐主入口：

```bash
python auto_eval/eval_digital_human_agent.py compare dashboard \
  --run_spec "exp_a|step_10000|eval_outputs/exp_a_step_10000_eval" \
  --run_spec "exp_a|step_20000|eval_outputs/exp_a_step_20000_eval" \
  --run_spec "exp_b|step_15000|eval_outputs/exp_b_step_15000_eval" \
  --output_dir eval_outputs/compare_dashboard_v1 \
  --title "多实验多权重对比"
```

也可以让系统自动从 eval 目录推断实验名和 checkpoint：

```bash
python auto_eval/eval_digital_human_agent.py compare dashboard \
  --eval_dir eval_outputs/run_a \
  --eval_dir eval_outputs/run_b \
  --output_dir eval_outputs/compare_dashboard_v1
```

如果要在浏览器看：

```text
http://127.0.0.1:<端口>/eval_outputs/compare_dashboard_v1/index.html
```

注意：compare dashboard 只读取已有 `metrics.json` 和 artifact，不会重新跑评测，也不会修改 paper table。

## 10. Preset 当前含义

- `smoke = sanity`
- `quick = sanity + facesim_sampled`
- `quick_lipsync = sanity + sync_global`
- `full = sanity + facesim_sampled + sync_global + iqa_aes_sampled`
- `full_pose = sanity + facesim_sampled + sync_global + iqa_aes_sampled + akd_pck_body18`

`full` 不包含 AKD/PCK；需要姿态指标时使用 `full_pose` 或显式 metrics。

## 11. Paper table 注意事项

正式 paper table 只应由已确认的 paper profile / official split / candidate promotion 流程生成。不要把 diagnostic-only 指标直接加入正式表。

当前重要限制：

- `sync_global` 仍是 uncalibrated diagnostic，不进入正式主表。
- `iqa_aes_sampled` 依赖本地 VBench 资源和采样策略。
- `akd_pck_body18` 只适用于 paired / key-preserving / target-video-comparable 场景。
- `paper_hoivg_720p25fps5s_v1` 仍是 planned-only，不代表当前已实现 profile。

## 12. Skill 发布状态

当前可作为 internal Skill 使用。公开发布前仍需要：

- 第三方 runtime glue / vendored code license review
- 预训练权重来源和许可证说明
- 确认 Skill 包不包含真实权重和私有路径

真实权重不得进入 git，也不得进入 Skill 包。
