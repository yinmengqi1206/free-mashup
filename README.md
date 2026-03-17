# 音乐电影混剪

```
这是一个自动化剪辑视频，放置好原始音视频素材，原素材不动，根据音乐自动选择使用素材，音乐必须选，自动解析音乐节奏视频适配视频
```

---

### 初始化素材库（只做一次）

```bash
uv run python main.py scan --videos input/videos --db index/clip_db.sqlite --verbose
```

大体积视频会非常慢，可选加速（降低精度）：

```bash
# 每 N 帧取一帧做场景检测，数值越大越快
SCAN_STRIDE=3 uv run python main.py scan --videos input/videos --db index/clip_db.sqlite --verbose

# 每个场景采样帧数（越小越快）
SCAN_SAMPLES=6 uv run python main.py scan --videos input/videos --db index/clip_db.sqlite --verbose

# 每 N 个片段输出一次进度日志
SCAN_LOG_EVERY=10 uv run python main.py scan --videos input/videos --db index/clip_db.sqlite --verbose
```

### 高质量输出与字幕

```bash
# 高质量输出
uv run python main.py generate --audio input/audios/music.mp3 --db index/clip_db.sqlite --output output/meme_001.mp4 --quality high

# 加歌词字幕（LRC）
uv run python main.py generate --audio input/audios/music.mp3 --db index/clip_db.sqlite --output output/meme_001.mp4 --lyrics input/audios/lyrics.lrc --quality high --verbose
```

### 依赖说明

- 基础依赖：`opencv-python`、`moviepy`、`numpy<2`
- 向量模型：`open_clip_torch`（会自动下载模型权重）

安装可选 AI 依赖：

```bash
uv sync --extra ai
```

注意：Intel Mac 上官方 PyTorch 没有可用的 wheel。`--extra ai` 会自动跳过 torch/open_clip，
此时会降级为可复现的随机向量。若要用真实模型，请在 Apple Silicon 或 Linux/Windows 环境运行。

Intel Mac 可用的固定版本（已验证可安装）：

```bash
uv sync --extra ai_intel
```

---
