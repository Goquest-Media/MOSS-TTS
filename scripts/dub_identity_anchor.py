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
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torchaudio
from transformers import AutoModel, AutoProcessor


DEFAULT_MODEL_PATH = "OpenMOSS-Team/MOSS-TTS"
TOKENS_PER_SECOND = 12.5


@dataclass
class LineItem:
    line_id: str
    turkish_audio: str
    turkish_text: str
    english_text: str
    target_seconds: Optional[float]


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


def compute_max_new_tokens(item: LineItem, cap_ratio: float, fallback_chars_ratio: float) -> int:
    if item.target_seconds is not None and item.target_seconds > 0:
        target_tokens = max(1, int(round(item.target_seconds * TOKENS_PER_SECOND)))
        return max(128, int(target_tokens * cap_ratio))
    # Fallback when no duration is provided.
    est_tokens = max(1, int(len(item.english_text) * fallback_chars_ratio))
    return max(256, int(est_tokens * 2.0))


def as_mono_2d(wav: torch.Tensor) -> torch.Tensor:
    if wav.ndim == 1:
        return wav.unsqueeze(0)
    if wav.ndim == 2:
        if wav.shape[0] == 1:
            return wav
        return wav.mean(dim=0, keepdim=True)
    return wav.reshape(1, -1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dub English with stable identity anchor + per-line Turkish expressive prefix."
    )
    parser.add_argument("--manifest", required=True, help="JSONL manifest path.")
    parser.add_argument("--anchor-audio", required=True, help="Fixed identity anchor wav/mp3/m4a.")
    parser.add_argument("--out-dir", required=True, help="Output folder for wav files.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
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
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "report.jsonl"

    if args.seed is not None:
        torch.manual_seed(int(args.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(args.seed))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    attn_implementation = resolve_attn_implementation(device=device, dtype=dtype)

    print(f"[INFO] Loading processor/model from: {args.model_path}")
    print(f"[INFO] device={device} dtype={dtype} attn={attn_implementation}")
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    if hasattr(processor, "audio_tokenizer"):
        processor.audio_tokenizer = processor.audio_tokenizer.to(device)

    model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        attn_implementation=attn_implementation,
    ).to(device)
    model.eval()

    sample_rate = int(getattr(processor.model_config, "sampling_rate", 24000))
    print(f"[INFO] sample_rate={sample_rate}")

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
            )

            # Key idea:
            # - reference=[anchor_codes] locks base speaker identity
            # - assistant prefix uses this line's Turkish audio for expressive carryover
            # - text starts with Turkish transcript then English line for continuation alignment
            full_text = f"{item.turkish_text} {item.english_text}".strip()
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
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    audio_temperature=float(args.audio_temperature),
                    audio_top_p=float(args.audio_top_p),
                    audio_top_k=int(args.audio_top_k),
                    audio_repetition_penalty=float(args.audio_repetition_penalty),
                )

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
            if item.target_seconds is not None and item.target_seconds > 0:
                duration_error = out_seconds - float(item.target_seconds)

            report_row: dict[str, Any] = {
                "line_id": item.line_id,
                "out_wav": str(out_path),
                "target_seconds": item.target_seconds,
                "out_seconds": out_seconds,
                "duration_error_seconds": duration_error,
                "max_new_tokens": max_new_tokens,
            }
            report_fp.write(json.dumps(report_row, ensure_ascii=True) + "\n")
            report_fp.flush()

            print(
                f"[{i}/{len(lines)}] {item.line_id} "
                f"out={out_seconds:.3f}s "
                f"target={item.target_seconds if item.target_seconds is not None else 'n/a'} "
                f"max_new_tokens={max_new_tokens}"
            )

    print(f"[DONE] Wrote audio to {out_dir}")
    print(f"[DONE] Wrote report: {report_path}")


if __name__ == "__main__":
    main()
