# DiCoW 本地推理适配修改记录

本文档记录了对 DiCoW 仓库的所有修改，用于适配昇腾 NPU（torch_npu）环境、新版 transformers 以及本地模型路径加载。

## 修改总览

| 文件 | 修改类型 | 说明 |
|---|---|---|
| `inference.py` | 修改 | 内嵌 weights_only 兼容补丁 + 新增 `--embedding-model` 参数 |
| `pipeline.py` | 修改 | 修正 `model_type`，避免 transformers 误判 DiCoW 为 CTC 模型 |
| `run.sh` | 新增 | 启动脚本（本地模型路径） |

---

## 1. `inference.py` — weights_only 兼容补丁（内嵌）

### 问题

```
_pickle.UnpicklingError: Unsupported global: GLOBAL torch.torch_version.TorchVersion
```

### 根因

1. **PyTorch 2.6+** 的 `torch.load` 默认 `weights_only=True`，只允许加载白名单内的 pickle 类型，旧 checkpoint 中的 `TorchVersion`、`pyannote.audio.core.task.Specifications` 等类型被拒绝。
2. **torch_npu 的 `transfer_to_npu`** 劫持了 `torch.load`，其 `serialization.load` 实现**硬编码使用 weights-only unpickler，完全忽略 `weights_only` 参数**（见 `torch_npu/utils/serialization.py` 第 252 行 `_load(opened_zipfile, map_location, _weights_only_unpickler, ...)`）。
3. **PyTorch Lightning 的 `pl_load`** 会显式传 `weights_only=True`。

### 修复

在 `inference.py` 顶部（`from torch_npu.contrib import transfer_to_npu` 之后）内嵌补丁，改用**原生 `torch.serialization.load`** 绕过 torch_npu 劫持，并强制 `weights_only=False`：

```python
import torch.serialization as torch_serialization

_original_torch_load = torch.load

def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return torch_serialization.load(*args, **kwargs)

torch.load = _patched_torch_load
```

内嵌在 `inference.py` 中，不依赖外部文件；`torch.load` 被替换后，Lightning 的 `pl_load` → `cloud_io._load` 也会走此补丁（因为最终调用的是 `torch.load`）。

---

## 2. `inference.py` — 新增 `--embedding-model` 参数

### 背景

DiariZen 的 `DiariZenPipeline.from_pretrained` 已支持 `embedding_model_path` 参数指定本地 embedding 模型，但 DiCoW 的 CLI 未透传。

### 修改

新增参数（默认 HuggingFace 官方模型）：

```python
parser.add_argument(
    "--embedding-model",
    type=str,
    default="pyannote/wespeaker-voxceleb-resnet34-LM",
    help="Diarization model name or path"
)
```

加载时透传：

```python
diar_pipeline = DiariZenPipeline.from_pretrained(
    args.diarization_model,
    embedding_model_path=args.embedding_model
).to(device)
```

使用示例：

```bash
--embedding-model /home/wjh/wespeaker-voxceleb-resnet34-LM/pytorch_model.bin
```

---

## 3. `pipeline.py` — 修正 `model_type`，避免误判为 CTC

### 问题

```
ValueError: CTC can either predict character level timestamps, or word level timestamps.
Set return_timestamps='char' or return_timestamps='word' as required.
```

### 根因

新版 transformers（4.5x+）的 `AutomaticSpeechRecognitionPipeline.__init__` 在 `super().__init__()` **之前**根据模型类型判定 `self.type`：

```python
if model.config.model_type == "whisper":
    self.type = "seq2seq_whisper"
elif model.__class__.__name__ in MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES.values():
    self.type = "seq2seq"
elif model.config.model_type in ("parakeet_tdt", ...):
    self.type = "tdt"
elif decoder is not None:
    self.type = "ctc_with_lm"
else:
    self.type = "ctc"   # ← DiCoW 掉进这里
```

DiCoW 是**自定义 Whisper 变体**（`trust_remote_code` 加载）：
- `config.model_type` 不是 `"whisper"`
- 模型类名（如 `DiCoWForConditionalGeneration`）不在 transformers 标准的 seq2seq 类名映射中
- → 被误判为 **CTC 模型**

而 `_sanitize_parameters` 中：

```python
if hasattr(self.generation_config, "return_timestamps"):
    return_timestamps = return_timestamps or self.generation_config.return_timestamps
    # DiCoW 的 generation_config.return_timestamps = True（Whisper 风格）
if self.type == "ctc" and return_timestamps not in ["char", "word"]:
    raise ValueError(...)   # ← 报错
```

CTC 校验要求 `return_timestamps` 必须是 `'char'` 或 `'word'`，而 DiCoW 的是 `True` → 初始化直接失败。

### 修复

在 `DiCoWPipeline.__init__` 中，调用 `super().__init__()` 之前把 `model.config.model_type` 临时修正为 `"whisper"`：

```python
class DiCoWPipeline(AutomaticSpeechRecognitionPipeline):
    def __init__(self, *args, diarization_pipeline, **kwargs):
        model = args[0] if args else kwargs.get("model")
        if model is not None and getattr(model.config, "model_type", None) != "whisper":
            model.config.model_type = "whisper"
        super().__init__(*args, **kwargs)
        self.diarization_pipeline = diarization_pipeline
        self.type = "seq2seq_whisper"
```

这样 transformers 正确判定为 `seq2seq_whisper`：
- `return_timestamps=True` 合法（Whisper 段级时间戳）
- `self.type == "seq2seq_whisper"` 判定正确，`_forward` 中时间戳相关逻辑正常工作

---

## 4. `run.sh` — 启动脚本（新增）

```bash
python inference.py \
    --dicow-model /home/wjh/DiCoW_v3_3_large \
    --diarization-model /home/wjh/diarizen-wavlm-large-s80-md \
    --input-folder /home/wjh/DiariZen/example \
    --output-folder ./output
```

---

## 相关依赖问题（DiariZen 仓库）

DiCoW 依赖的 DiariZen 子模块（独立仓库 `Hu1Lcode/DiariZen`，见其 `CHANGES.md`）包含：

1. `diarizen/pipelines/inference.py`：`from_pretrained` 支持本地路径 + `embedding_model_path` 参数 + 嵌入模型改用 `Model.from_pretrained` 加载（绕过 ONNX 误判）+ 音频解码替换为 soundfile
2. `pyannote-audio/.../core/model.py`：`hf_hub_download` 的 `use_auth_token` 参数改为 `token`（兼容新版 huggingface_hub）
3. `patch_diarizen.py`：独立的 weights_only 三层拦截补丁（DiCoW 已内嵌，无需再引入）
