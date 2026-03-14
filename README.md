# 鬼畜视频混剪

```
这是一个自动化鬼畜生成视频，放置好原始音视频素材，原素材不动，根据音乐自动选择使用素材，音乐必须选，自动解析音乐节奏视频适配视频，视频的原声音可以做适当取舍，需要有一套适用性很强的鬼畜解析算法
```

---

# V2 方案（向量库版）

核心思路：

```
素材视频
↓
一次性扫描 + 向量化
↓
素材数据库

生成时
↓
音频分析
↓
全量匹配
↓
拼接渲染
```

### 初始化素材库（只做一次）

```bash
uv run python main.py scan --videos input/videos --db index/clip_db.sqlite --verbose
```

### 一键执行（推荐顺序）

```bash
# 1. 扫描素材（首次或新增素材后执行）
uv run python main.py scan --videos input/videos --db index/clip_db.sqlite --verbose

# 2. 生成视频
uv run python main.py generate --audio input/audios/music.mp3 --db index/clip_db.sqlite --output output/meme_001.mp4 --verbose
```

### 生成视频（反复使用）

```bash
uv run python main.py generate --audio input/audios/music.mp3 --db index/clip_db.sqlite --output output/meme_001.mp4 --verbose
```

### 新增素材

把新素材放进新目录后，再执行一次 scan：

```bash
uv run python main.py scan --videos /path/to/new/videos --db index/clip_db.sqlite --verbose
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
