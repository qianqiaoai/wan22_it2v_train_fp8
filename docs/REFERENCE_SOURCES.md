# Reference Sources for Offline Digital Human Eval

## Source A: Hallo-Live

Repo:
https://github.com/fudan-generative-vision/Hallo-Live

Use for:
- understanding audio-video avatar generation evaluation conventions
- locating possible SyncNet / reward / scoring utilities
- learning inference-eval output organization

Do not use for:
- realtime metrics in the MVP
- changing our generation pipeline
- copying large training/inference code into eval_agent

## Source B: Local Baselines

Path:
/mnt/data/cv/yutan/Baselines

Use for:
- existing video generation metric implementations
- possible FID/FVD/IQA/AES/FaceSim wrappers
- dependency and preprocessing conventions

Do not modify this path.

## Source C: TalkVid dataprocess

Path:
/mnt/data/nlp/user/qiaoqian/TalkVid/dataprocess

Use for:
- DWPose extraction
- face bbox extraction
- LSE-C / LSE-D / SyncNet-style metrics
- audio/video preprocessing utilities

Do not modify this path.

## Integration rule

External code is reference-only by default.
Prefer thin adapters and wrappers.
If copying small code snippets is necessary:
1. record source path
2. record license if available
3. record why wrapper/import is insufficient
4. preserve provenance in metric result