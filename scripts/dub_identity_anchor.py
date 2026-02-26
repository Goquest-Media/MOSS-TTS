#!/usr/bin/env python3
"""
MOSS-TTS dubbing pipeline that separates:
1) stable voice identity (fixed anchor clip), and
2) per-line emotion/prosody (line Turkish audio).

Manifest format: JSONL, one object per line:
{
  "line_id": "ep01_sc03_l012",
  "turkish_audio": "/abs/path/to/line_012_tr.wav",
  "turkish_text": "Turkish transcript of the same line",
  "english_text": "Syllable-matched English line",
  "target_seconds": 3.84
}
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import inspect
import json
import platform
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import torch
import torchaudio
from transformers import (
    AutoModel,
    AutoProcessor,
    GenerationConfig,
    __version__ as transformers_version,
)


DEFAULT_MODEL_PATH = "OpenMOSS-Team/MOSS-TTS"
DEFAULT_CODEC_PATH = "OpenMOSS-Team/MOSS-Audio-Tokenizer"
TOKENS_PER_SECOND = 12.5
LOCAL_TEXT_LAYER_TOP_K = 50
LOCAL_TEXT_LAYER_TOP_P = 1.0
LOCAL_TEXT_LAYER_TEMPERATURE = 1.5


@dataclass
class LineItem:
    line_id: str
    turkish_audio: str
    turkish_text: str
    english_text: str
    target_seconds: Optional[float]


class DelayGenerationConfig(GenerationConfig):
    """GenerationConfig with MOSS local-transformer per-layer controls."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layers = kwargs.get("layers", [{} for _ in range(32)])
        self.do_samples = kwargs.get("do_samples", None)
        self.n_vq_for_inference = kwargs.get("n_vq_for_inference", 32)


def resolve_attn_implementation(device: torch.device, dtype: torch.dtype) -> str:
    if (
        device.type == "cuda"
        and importlib.util.find_spec("flash_attn") is not None
        and dtype in {torch.float16, torch.bfloat16}
    ):
        major, _ = torch.cuda.get_device_capability(device)
        if major >= 8:
            return "flash_attention_2"
    if device.type == "cuda":
        return "sdpa"
    return "eager"


def load_manifest(path: Path) -> list[LineItem]:
    items: list[LineItem] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            obj = json.loads(line)
            line_id = str(obj.get("line_id", f"line_{idx:04d}")).strip()
            turkish_audio = str(obj["turkish_audio"]).strip()
            turkish_text = str(obj["turkish_text"]).strip()
            english_text = str(obj["english_text"]).strip()
            target_seconds_raw = obj.get("target_seconds", None)
            target_seconds = (
                float(target_seconds_raw) if target_seconds_raw is not None else None
            )
            if not turkish_audio:
                raise ValueError(f"Missing turkish_audio for line {line_id}")
            if not turkish_text:
                raise ValueError(f"Missing turkish_text for line {line_id}")
            if not english_text:
                raise ValueError(f"Missing english_text for line {line_id}")
            items.append(
                LineItem(
                    line_id=line_id,
                    turkish_audio=turkish_audio,
                    turkish_text=turkish_text,
                    english_text=english_text,
                    target_seconds=target_seconds,
                )
            )
    return items


def compute_max_new_tokens(
    item: LineItem,
    cap_ratio: float,
    fallback_chars_ratio: float,
    min_tokens_with_target: int,
    min_tokens_without_target: int,
) -> int:
    if item.target_seconds is not None and item.target_seconds > 0:
        target_tokens = max(1, int(round(item.target_seconds * TOKENS_PER_SECOND)))
        return max(int(min_tokens_with_target), int(target_tokens * cap_ratio))
    est_tokens = max(1, int(len(item.english_text) * fallback_chars_ratio))
    return max(int(min_tokens_without_target), int(est_tokens * 2.0))


def as_mono_2d(wav: torch.Tensor) -> torch.Tensor:
    if wav.ndim == 1:
        return wav.unsqueeze(0)
    if wav.ndim == 2:
        if wav.shape[0] == 1:
            return wav
        return wav.mean(dim=0, keepdim=True)
    return wav.reshape(1, -1)


def format_text(item: LineItem, text_join_style: str) -> str:
    if text_join_style == "space":
        return f"{item.turkish_text} {item.english_text}".strip()
    if text_join_style == "newline":
        return f"{item.turkish_text}\n{item.english_text}".strip()
    if text_join_style == "lang_tags":
        return f"[TR]{item.turkish_text}[/TR] [EN]{item.english_text}[/EN]".strip()
    raise ValueError(f"Unsupported text_join_style: {text_join_style}")


def maybe_snapshot_download(path_or_repo: str, revision: Optional[str]) -> str:
    if Path(path_or_repo).exists():
        return str(Path(path_or_repo).expanduser().resolve())
    if revision is None:
        return path_or_repo
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise RuntimeError(
            "revision pinning requires huggingface_hub to be importable"
        ) from exc
    return snapshot_download(repo_id=path_or_repo, revision=revision)


def get_git_commit(repo_root: Path) -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return None


def safe_commit_hash_from_obj(obj: Any) -> Optional[str]:
    if obj is None:
        return None
    return getattr(obj, "_commit_hash", None)


def detect_generation_api(model: Any) -> str:
    sig = inspect.signature(model.generate)
    params = sig.parameters
    if {"audio_temperature", "audio_top_p", "audio_top_k", "audio_repetition_penalty"}.issubset(params.keys()):
        return "delay_kwargs"
    if "generation_config" in params:
        return "local_generation_config"
    raise RuntimeError(
        "Unable to detect model.generate API. Expected delay kwargs or generation_config."
    )


def map_entry_to_device(entry: Any) -> Optional[torch.device]:
    if isinstance(entry, torch.device):
        return entry
    if isinstance(entry, int):
        if torch.cuda.is_available():
            return torch.device(f"cuda:{entry}")
        return torch.device("cpu")
    if isinstance(entry, str):
        if entry in {"cpu", "mps"}:
            return torch.device(entry)
        if entry.startswith("cuda"):
            if torch.cuda.is_available():
                return torch.device(entry)
            return torch.device("cpu")
    return None


def resolve_model_input_device(model: Any, fallback_device: torch.device) -> torch.device:
    hf_device_map = getattr(model, "hf_device_map", None)
    if isinstance(hf_device_map, dict):
        preferred_keys = (
            "model.embed_tokens",
            "transformer.wte",
            "wte",
            "embeddings.word_embeddings",
        )
        for key in preferred_keys:
            if key in hf_device_map:
                dev = map_entry_to_device(hf_device_map[key])
                if dev is not None:
                    return dev
        for entry in hf_device_map.values():
            dev = map_entry_to_device(entry)
            if dev is not None:
                return dev
    try:
        first_param = next(model.parameters())
        return first_param.device
    except Exception:
        return fallback_device


def build_local_base_generation_config(
    model_source: str,
    processor: Any,
    model: Any,
) -> DelayGenerationConfig:
    if not hasattr(model, "channels"):
        raise RuntimeError("Local generation path requires model.channels.")
    channels = int(model.channels)
    cfg = DelayGenerationConfig.from_pretrained(model_source)
    cfg.pad_token_id = processor.tokenizer.pad_token_id
    cfg.eos_token_id = 151653
    cfg.use_cache = True
    cfg.n_vq_for_inference = channels - 1
    cfg.do_samples = [True] * channels
    if not isinstance(cfg.layers, list):
        cfg.layers = [{} for _ in range(channels)]
    if len(cfg.layers) < channels:
        cfg.layers = list(cfg.layers) + ([{}] * (channels - len(cfg.layers)))
    return cfg


def build_local_generation_config(
    base_cfg: DelayGenerationConfig,
    max_new_tokens: int,
    audio_temperature: float,
    audio_top_p: float,
    audio_top_k: int,
    audio_repetition_penalty: float,
) -> DelayGenerationConfig:
    cfg = copy.deepcopy(base_cfg)
    cfg.max_new_tokens = int(max_new_tokens)
    channels = int(cfg.n_vq_for_inference) + 1
    cfg.layers = [
        {
            "repetition_penalty": 1.0,
            "temperature": float(LOCAL_TEXT_LAYER_TEMPERATURE),
            "top_p": float(LOCAL_TEXT_LAYER_TOP_P),
            "top_k": int(LOCAL_TEXT_LAYER_TOP_K),
        }
    ] + [
        {
            "repetition_penalty": float(audio_repetition_penalty),
            "temperature": float(max(audio_temperature, 1e-5)),
            "top_p": float(audio_top_p),
            "top_k": int(audio_top_k),
        }
    ] * max(0, channels - 1)
    if audio_temperature <= 0:
        cfg.do_samples = [True] + ([False] * max(0, channels - 1))
    else:
        cfg.do_samples = [True] * channels
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dub English with stable identity anchor + per-line Turkish expressive prefix."
    )
    parser.add_argument("--manifest", required=True, help="JSONL manifest path.")
    parser.add_argument("--anchor-audio", required=True, help="Fixed identity anchor wav/mp3/m4a.")
    parser.add_argument("--out-dir", required=True, help="Output folder for wav files.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--model-revision", default=None, help="Optional pinned revision for model repo.")
    parser.add_argument("--codec-path", default=DEFAULT_CODEC_PATH, help="Codec repo/path for processor codec_path.")
    parser.add_argument("--codec-revision", default=None, help="Optional pinned revision for codec repo.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--cpu-offload",
        action="store_true",
        help="Load with device_map=auto for lower VRAM usage (requires accelerate).",
    )
    parser.add_argument(
        "--text-join-style",
        choices=("space", "newline", "lang_tags"),
        default="space",
        help="How Turkish transcript and English line are joined for C2 prompt text.",
    )
    parser.add_argument("--audio-temperature", type=float, default=1.2)
    parser.add_argument("--audio-top-p", type=float, default=0.7)
    parser.add_argument("--audio-top-k", type=int, default=20)
    parser.add_argument("--audio-repetition-penalty", type=float, default=1.0)
    parser.add_argument(
        "--cap-ratio",
        type=float,
        default=1.35,
        help="max_new_tokens multiplier vs target tokens (if target_seconds exists).",
    )
    parser.add_argument(
        "--fallback-chars-ratio",
        type=float,
        default=0.9,
        help="English chars->token estimate when target_seconds is missing.",
    )
    parser.add_argument("--min-max-new-tokens-with-target", type=int, default=128)
    parser.add_argument("--min-max-new-tokens-without-target", type=int, default=256)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "report.jsonl"
    metadata_path = out_dir / "run_metadata.json"
    repo_root = Path(__file__).resolve().parents[1]

    model_source = maybe_snapshot_download(args.model_path, args.model_revision)
    codec_source = maybe_snapshot_download(args.codec_path, args.codec_revision)

    if args.seed is not None:
        torch.manual_seed(int(args.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(args.seed))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    attn_implementation = resolve_attn_implementation(device=device, dtype=dtype)

    print(f"[INFO] Loading processor/model from: {model_source}")
    print(f"[INFO] codec_source={codec_source}")
    print(
        f"[INFO] device={device} dtype={dtype} attn={attn_implementation} "
        f"text_join_style={args.text_join_style} cpu_offload={bool(args.cpu_offload)}"
    )
    processor = AutoProcessor.from_pretrained(
        model_source,
        trust_remote_code=True,
        codec_path=codec_source,
    )

    model_load_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
        "attn_implementation": attn_implementation,
    }
    if args.cpu_offload:
        model_load_kwargs["device_map"] = "auto"
    model = AutoModel.from_pretrained(model_source, **model_load_kwargs)
    if not args.cpu_offload:
        model = model.to(device)
    model.eval()
    model_input_device = resolve_model_input_device(model=model, fallback_device=device)
    if hasattr(processor, "audio_tokenizer"):
        processor.audio_tokenizer = processor.audio_tokenizer.to(model_input_device)
    generation_api = detect_generation_api(model)
    local_base_generation_config: Optional[DelayGenerationConfig] = None
    if generation_api == "local_generation_config":
        local_base_generation_config = build_local_base_generation_config(
            model_source=model_source,
            processor=processor,
            model=model,
        )

    sample_rate = int(getattr(processor.model_config, "sampling_rate", 24000))
    print(f"[INFO] sample_rate={sample_rate}")
    print(f"[INFO] generation_api={generation_api} model_input_device={model_input_device}")

    print("[INFO] Encoding fixed identity anchor once...")
    anchor_codes = processor.encode_audios_from_path([str(args.anchor_audio)])[0]

    lines = load_manifest(manifest_path)
    print(f"[INFO] Loaded {len(lines)} lines from {manifest_path}")

    with report_path.open("w", encoding="utf-8") as report_fp:
        for i, item in enumerate(lines, start=1):
            max_new_tokens = compute_max_new_tokens(
                item=item,
                cap_ratio=float(args.cap_ratio),
                fallback_chars_ratio=float(args.fallback_chars_ratio),
                min_tokens_with_target=int(args.min_max_new_tokens_with_target),
                min_tokens_without_target=int(args.min_max_new_tokens_without_target),
            )

            full_text = format_text(item=item, text_join_style=str(args.text_join_style))
            conversation = [
                [
                    processor.build_user_message(
                        text=full_text,
                        reference=[anchor_codes],
                    ),
                    processor.build_assistant_message(audio_codes_list=[item.turkish_audio]),
                ]
            ]

            batch = processor(conversation, mode="continuation")
            input_ids = batch["input_ids"].to(model_input_device)
            attention_mask = batch["attention_mask"].to(model_input_device)

            with torch.no_grad():
                if generation_api == "delay_kwargs":
                    outputs = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        audio_temperature=float(args.audio_temperature),
                        audio_top_p=float(args.audio_top_p),
                        audio_top_k=int(args.audio_top_k),
                        audio_repetition_penalty=float(args.audio_repetition_penalty),
                    )
                elif generation_api == "local_generation_config":
                    if local_base_generation_config is None:
                        raise RuntimeError(
                            "Missing local_base_generation_config for local generation API."
                        )
                    generation_config = build_local_generation_config(
                        base_cfg=local_base_generation_config,
                        max_new_tokens=max_new_tokens,
                        audio_temperature=float(args.audio_temperature),
                        audio_top_p=float(args.audio_top_p),
                        audio_top_k=int(args.audio_top_k),
                        audio_repetition_penalty=float(args.audio_repetition_penalty),
                    )
                    outputs = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        generation_config=generation_config,
                    )
                else:
                    raise RuntimeError(f"Unsupported generation_api: {generation_api}")

            messages = processor.decode(outputs)
            if not messages or messages[0] is None:
                raise RuntimeError(f"No decodable audio for line {item.line_id}")

            generated = messages[0].audio_codes_list[0]
            if not isinstance(generated, torch.Tensor):
                generated = torch.tensor(generated, dtype=torch.float32)
            wav = as_mono_2d(generated.detach().float().cpu())

            out_path = out_dir / f"{item.line_id}.wav"
            torchaudio.save(str(out_path), wav, sample_rate)

            out_seconds = float(wav.shape[-1] / sample_rate)
            duration_error = None
            duration_rel_error = None
            if item.target_seconds is not None and item.target_seconds > 0:
                duration_error = out_seconds - float(item.target_seconds)
                duration_rel_error = duration_error / float(item.target_seconds)

            report_row: dict[str, Any] = {
                "line_id": item.line_id,
                "out_wav": str(out_path),
                "target_seconds": item.target_seconds,
                "out_seconds": out_seconds,
                "duration_error_seconds": duration_error,
                "duration_rel_error": duration_rel_error,
                "max_new_tokens": max_new_tokens,
                "text_join_style": args.text_join_style,
                "generation_api": generation_api,
            }
            report_fp.write(json.dumps(report_row, ensure_ascii=True) + "\n")
            report_fp.flush()

            print(
                f"[{i}/{len(lines)}] {item.line_id} "
                f"out={out_seconds:.3f}s "
                f"target={item.target_seconds if item.target_seconds is not None else 'n/a'} "
                f"max_new_tokens={max_new_tokens}"
            )

    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/dub_identity_anchor.py",
        "repo_commit": get_git_commit(repo_root),
        "system": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchaudio": torchaudio.__version__,
            "transformers": transformers_version,
        },
        "resolved_sources": {
            "model_source": model_source,
            "codec_source": codec_source,
            "model_path_arg": args.model_path,
            "model_revision_arg": args.model_revision,
            "codec_path_arg": args.codec_path,
            "codec_revision_arg": args.codec_revision,
            "processor_model_commit_hash": safe_commit_hash_from_obj(getattr(processor, "model_config", None)),
            "audio_tokenizer_commit_hash": safe_commit_hash_from_obj(
                getattr(getattr(processor, "audio_tokenizer", None), "config", None)
            ),
            "model_commit_hash": safe_commit_hash_from_obj(getattr(model, "config", None)),
        },
        "runtime": {
            "device": str(device),
            "model_input_device": str(model_input_device),
            "cpu_offload": bool(args.cpu_offload),
            "dtype": str(dtype),
            "attn_implementation": attn_implementation,
            "generation_api": generation_api,
            "sample_rate": sample_rate,
        },
        "generation_setup": {
            "text_join_style": args.text_join_style,
            "audio_temperature": float(args.audio_temperature),
            "audio_top_p": float(args.audio_top_p),
            "audio_top_k": int(args.audio_top_k),
            "audio_repetition_penalty": float(args.audio_repetition_penalty),
            "cap_ratio": float(args.cap_ratio),
            "fallback_chars_ratio": float(args.fallback_chars_ratio),
            "min_max_new_tokens_with_target": int(args.min_max_new_tokens_with_target),
            "min_max_new_tokens_without_target": int(args.min_max_new_tokens_without_target),
            "cpu_offload": bool(args.cpu_offload),
            "seed": args.seed,
            "local_text_layer_defaults": {
                "temperature": float(LOCAL_TEXT_LAYER_TEMPERATURE),
                "top_p": float(LOCAL_TEXT_LAYER_TOP_P),
                "top_k": int(LOCAL_TEXT_LAYER_TOP_K),
            },
        },
    }
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=True, indent=2)

    print(f"[DONE] Wrote audio to {out_dir}")
    print(f"[DONE] Wrote report: {report_path}")
    print(f"[DONE] Wrote metadata: {metadata_path}")


if __name__ == "__main__":
    main()
