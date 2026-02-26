#!/usr/bin/env python3
"""
Run dubbing ablations for identity-vs-expressiveness tradeoff in one command.

Conditions:
  C0: baseline_per_line_clone_generation
      - generation mode
      - reference = per-line Turkish audio
      - text = English line
      - optional tokens control from target_seconds

  C1: anchor_only_generation
      - generation mode
      - reference = fixed anchor
      - text = English line
      - optional tokens control from target_seconds

  C2: anchor_plus_line_continuation
      - continuation mode
      - user reference = fixed anchor
      - assistant prefix audio = per-line Turkish audio
      - text = "<turkish_text> <english_text>"

  C3: C2 + candidate rerank
      - generate N candidates using C2
      - choose best by identity proxy and duration penalty
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F
import torchaudio
from transformers import AutoModel, AutoProcessor


DEFAULT_MODEL_PATH = "OpenMOSS-Team/MOSS-TTS"
TOKENS_PER_SECOND = 12.5
SUPPORTED_CONDITIONS = ("C0", "C1", "C2", "C3")


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


def parse_csv_ints(value: str) -> list[int]:
    parsed = []
    for token in (value or "").split(","):
        tok = token.strip()
        if not tok:
            continue
        parsed.append(int(tok))
    return parsed


def parse_csv_strings(value: str) -> list[str]:
    parsed = []
    for token in (value or "").split(","):
        tok = token.strip()
        if not tok:
            continue
        parsed.append(tok)
    return parsed


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


def as_mono_2d(wav: torch.Tensor) -> torch.Tensor:
    if wav.ndim == 1:
        return wav.unsqueeze(0)
    if wav.ndim == 2:
        if wav.shape[0] == 1:
            return wav
        return wav.mean(dim=0, keepdim=True)
    return wav.reshape(1, -1)


def safe_rel_error(out_seconds: float, target_seconds: Optional[float]) -> Optional[float]:
    if target_seconds is None or target_seconds <= 0:
        return None
    return (out_seconds - target_seconds) / target_seconds


def compute_expected_tokens(item: LineItem) -> Optional[int]:
    if item.target_seconds is None or item.target_seconds <= 0:
        return None
    return max(1, int(round(item.target_seconds * TOKENS_PER_SECOND)))


def compute_max_new_tokens(
    item: LineItem,
    cap_ratio: float,
    fallback_chars_ratio: float,
) -> int:
    expected = compute_expected_tokens(item)
    if expected is not None:
        return max(128, int(expected * cap_ratio))
    est_tokens = max(1, int(len(item.english_text) * fallback_chars_ratio))
    return max(256, int(est_tokens * 2.0))


def build_conversation(
    condition: str,
    item: LineItem,
    processor: Any,
    anchor_codes: torch.Tensor,
    include_tokens_in_generation: bool,
) -> tuple[list[list[dict[str, Any]]], str]:
    expected_tokens = compute_expected_tokens(item)
    if condition == "C0":
        user_kwargs: dict[str, Any] = {
            "text": item.english_text,
            "reference": [item.turkish_audio],
        }
        if include_tokens_in_generation and expected_tokens is not None:
            user_kwargs["tokens"] = int(expected_tokens)
        return [[processor.build_user_message(**user_kwargs)]], "generation"

    if condition == "C1":
        user_kwargs = {
            "text": item.english_text,
            "reference": [anchor_codes],
        }
        if include_tokens_in_generation and expected_tokens is not None:
            user_kwargs["tokens"] = int(expected_tokens)
        return [[processor.build_user_message(**user_kwargs)]], "generation"

    if condition in {"C2", "C3"}:
        full_text = f"{item.turkish_text} {item.english_text}".strip()
        conversations = [
            [
                processor.build_user_message(
                    text=full_text,
                    reference=[anchor_codes],
                ),
                processor.build_assistant_message(
                    audio_codes_list=[item.turkish_audio],
                ),
            ]
        ]
        return conversations, "continuation"

    raise ValueError(f"Unsupported condition: {condition}")


def decode_first_audio(messages: list[Any]) -> torch.Tensor:
    if not messages or messages[0] is None:
        raise RuntimeError("Model returned no decodable message.")
    generated = messages[0].audio_codes_list[0]
    if not isinstance(generated, torch.Tensor):
        generated = torch.tensor(generated, dtype=torch.float32)
    return generated.detach().float().cpu()


def mfcc_embedding(wav: torch.Tensor, sample_rate: int, target_sr: int = 16000) -> torch.Tensor:
    mono = as_mono_2d(wav)
    if sample_rate != target_sr:
        mono = torchaudio.functional.resample(mono, sample_rate, target_sr)
    mfcc = torchaudio.transforms.MFCC(
        sample_rate=target_sr,
        n_mfcc=20,
        melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 40},
    )(mono)
    # [1, n_mfcc, T] -> [n_mfcc]
    emb = mfcc.mean(dim=-1).squeeze(0)
    return emb


def identity_proxy_score(
    wav: torch.Tensor,
    sample_rate: int,
    anchor_emb: torch.Tensor,
) -> float:
    emb = mfcc_embedding(wav, sample_rate=sample_rate)
    score = F.cosine_similarity(emb.unsqueeze(0), anchor_emb.unsqueeze(0), dim=-1).item()
    return float(score)


def generate_once(
    model: Any,
    processor: Any,
    device: torch.device,
    conversations: list[list[dict[str, Any]]],
    mode: str,
    max_new_tokens: int,
    audio_temperature: float,
    audio_top_p: float,
    audio_top_k: int,
    audio_repetition_penalty: float,
) -> torch.Tensor:
    batch = processor(conversations, mode=mode)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=int(max_new_tokens),
            audio_temperature=float(audio_temperature),
            audio_top_p=float(audio_top_p),
            audio_top_k=int(audio_top_k),
            audio_repetition_penalty=float(audio_repetition_penalty),
        )
    messages = processor.decode(outputs)
    return decode_first_audio(messages)


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {
            "n_lines": 0,
            "mae_seconds": None,
            "mape_percent": None,
            "within_5pct": None,
            "within_10pct": None,
            "mean_identity_proxy": None,
        }

    abs_errors = []
    abs_rel = []
    identity = []
    within_5 = 0
    within_10 = 0
    with_target = 0

    for row in rows:
        if row["identity_proxy"] is not None:
            identity.append(float(row["identity_proxy"]))

        target = row["target_seconds"]
        out_sec = row["out_seconds"]
        if target is None or target <= 0:
            continue
        with_target += 1
        err = float(out_sec - target)
        abs_err = abs(err)
        abs_errors.append(abs_err)
        rel = abs_err / float(target)
        abs_rel.append(rel)
        if rel <= 0.05:
            within_5 += 1
        if rel <= 0.10:
            within_10 += 1

    mae = sum(abs_errors) / len(abs_errors) if abs_errors else None
    mape = (100.0 * sum(abs_rel) / len(abs_rel)) if abs_rel else None

    if with_target > 0:
        within_5pct = 100.0 * (within_5 / with_target)
        within_10pct = 100.0 * (within_10 / with_target)
    else:
        within_5pct = None
        within_10pct = None

    mean_identity = sum(identity) / len(identity) if identity else None

    return {
        "n_lines": n,
        "mae_seconds": mae,
        "mape_percent": mape,
        "within_5pct": within_5pct,
        "within_10pct": within_10pct,
        "mean_identity_proxy": mean_identity,
    }


def fmt_num(v: Optional[float], digits: int = 4) -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "n/a"
    return f"{v:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run C0/C1/C2/C3 dubbing ablation and export comparison tables."
    )
    parser.add_argument("--manifest", required=True, help="Input JSONL manifest path.")
    parser.add_argument("--anchor-audio", required=True, help="Fixed identity anchor wav/mp3/m4a.")
    parser.add_argument("--out-dir", required=True, help="Output directory root.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--conditions", default="C0,C1,C2,C3", help="Comma-separated subset.")
    parser.add_argument("--seeds", default="42", help="Comma-separated integer seeds.")
    parser.add_argument("--limit-lines", type=int, default=0, help="Optional debug limit.")
    parser.add_argument("--candidate-count", type=int, default=3, help="Candidates for C3.")
    parser.add_argument("--save-all-candidates", action="store_true")
    parser.add_argument("--audio-temperature", type=float, default=1.2)
    parser.add_argument("--audio-top-p", type=float, default=0.7)
    parser.add_argument("--audio-top-k", type=int, default=20)
    parser.add_argument("--audio-repetition-penalty", type=float, default=1.0)
    parser.add_argument("--cap-ratio", type=float, default=1.35)
    parser.add_argument("--fallback-chars-ratio", type=float, default=0.9)
    parser.add_argument("--include-tokens-in-generation", action="store_true")
    parser.add_argument("--identity-weight", type=float, default=1.0)
    parser.add_argument("--duration-weight", type=float, default=0.35)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).expanduser().resolve()
    out_root = Path(args.out_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    conditions = parse_csv_strings(args.conditions)
    if not conditions:
        raise ValueError("No conditions provided.")
    invalid = [c for c in conditions if c not in SUPPORTED_CONDITIONS]
    if invalid:
        raise ValueError(f"Unsupported conditions: {invalid}")

    seeds = parse_csv_ints(args.seeds)
    if not seeds:
        raise ValueError("At least one seed is required.")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    attn_implementation = resolve_attn_implementation(device=device, dtype=dtype)

    print(f"[INFO] Loading model={args.model_path} device={device} attn={attn_implementation}")
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

    print("[INFO] Encoding anchor...")
    anchor_codes = processor.encode_audios_from_path([str(args.anchor_audio)])[0]
    anchor_wav, anchor_sr = torchaudio.load(str(args.anchor_audio))
    anchor_wav = as_mono_2d(anchor_wav).squeeze(0)
    anchor_emb = mfcc_embedding(anchor_wav, sample_rate=int(anchor_sr))

    lines = load_manifest(manifest_path)
    if args.limit_lines and int(args.limit_lines) > 0:
        lines = lines[: int(args.limit_lines)]
    print(f"[INFO] lines={len(lines)} conditions={conditions} seeds={seeds}")

    all_rows: list[dict[str, Any]] = []

    for condition in conditions:
        for seed in seeds:
            run_name = f"{condition}_seed{seed}"
            run_dir = out_root / run_name
            run_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n[RUN] {run_name}")
            for line_idx, item in enumerate(lines, start=1):
                max_new_tokens = compute_max_new_tokens(
                    item=item,
                    cap_ratio=float(args.cap_ratio),
                    fallback_chars_ratio=float(args.fallback_chars_ratio),
                )

                if condition != "C3":
                    torch.manual_seed(int(seed) + line_idx)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(int(seed) + line_idx)

                    conversations, mode = build_conversation(
                        condition=condition,
                        item=item,
                        processor=processor,
                        anchor_codes=anchor_codes,
                        include_tokens_in_generation=bool(args.include_tokens_in_generation),
                    )
                    wav = generate_once(
                        model=model,
                        processor=processor,
                        device=device,
                        conversations=conversations,
                        mode=mode,
                        max_new_tokens=max_new_tokens,
                        audio_temperature=float(args.audio_temperature),
                        audio_top_p=float(args.audio_top_p),
                        audio_top_k=int(args.audio_top_k),
                        audio_repetition_penalty=float(args.audio_repetition_penalty),
                    )
                    out_path = run_dir / f"{item.line_id}.wav"
                    torchaudio.save(str(out_path), as_mono_2d(wav), sample_rate)

                    out_seconds = float(wav.shape[-1] / sample_rate)
                    rel_err = safe_rel_error(out_seconds=out_seconds, target_seconds=item.target_seconds)
                    identity = identity_proxy_score(wav=wav, sample_rate=sample_rate, anchor_emb=anchor_emb)

                    row = {
                        "condition": condition,
                        "seed": int(seed),
                        "line_id": item.line_id,
                        "out_wav": str(out_path),
                        "target_seconds": item.target_seconds,
                        "out_seconds": out_seconds,
                        "duration_error_seconds": (
                            None
                            if item.target_seconds is None
                            else float(out_seconds - item.target_seconds)
                        ),
                        "duration_rel_error": rel_err,
                        "identity_proxy": identity,
                        "selected_candidate": 0,
                        "candidate_count": 1,
                    }
                    all_rows.append(row)
                else:
                    # C3: proposed condition + candidate reranking.
                    conversations, mode = build_conversation(
                        condition="C3",
                        item=item,
                        processor=processor,
                        anchor_codes=anchor_codes,
                        include_tokens_in_generation=False,
                    )
                    candidate_rows = []
                    best_score = None
                    best_wav = None
                    best_idx = None

                    for cand_idx in range(int(args.candidate_count)):
                        cand_seed = int(seed) + (line_idx * 100) + (cand_idx * 100000)
                        torch.manual_seed(cand_seed)
                        if torch.cuda.is_available():
                            torch.cuda.manual_seed_all(cand_seed)

                        wav = generate_once(
                            model=model,
                            processor=processor,
                            device=device,
                            conversations=conversations,
                            mode=mode,
                            max_new_tokens=max_new_tokens,
                            audio_temperature=float(args.audio_temperature),
                            audio_top_p=float(args.audio_top_p),
                            audio_top_k=int(args.audio_top_k),
                            audio_repetition_penalty=float(args.audio_repetition_penalty),
                        )

                        out_seconds = float(wav.shape[-1] / sample_rate)
                        rel_err = safe_rel_error(
                            out_seconds=out_seconds,
                            target_seconds=item.target_seconds,
                        )
                        identity = identity_proxy_score(
                            wav=wav,
                            sample_rate=sample_rate,
                            anchor_emb=anchor_emb,
                        )

                        duration_penalty = abs(rel_err) if rel_err is not None else 0.0
                        score = (
                            float(args.identity_weight) * identity
                            - float(args.duration_weight) * duration_penalty
                        )

                        cand_row = {
                            "candidate_index": cand_idx,
                            "cand_seed": cand_seed,
                            "identity_proxy": identity,
                            "duration_rel_error": rel_err,
                            "score": score,
                            "out_seconds": out_seconds,
                        }
                        candidate_rows.append(cand_row)

                        if bool(args.save_all_candidates):
                            cand_path = run_dir / f"{item.line_id}.cand{cand_idx}.wav"
                            torchaudio.save(str(cand_path), as_mono_2d(wav), sample_rate)

                        if best_score is None or score > best_score:
                            best_score = score
                            best_wav = wav
                            best_idx = cand_idx

                    assert best_wav is not None and best_idx is not None
                    out_path = run_dir / f"{item.line_id}.wav"
                    torchaudio.save(str(out_path), as_mono_2d(best_wav), sample_rate)

                    out_seconds = float(best_wav.shape[-1] / sample_rate)
                    rel_err = safe_rel_error(out_seconds=out_seconds, target_seconds=item.target_seconds)
                    identity = identity_proxy_score(
                        wav=best_wav,
                        sample_rate=sample_rate,
                        anchor_emb=anchor_emb,
                    )
                    row = {
                        "condition": condition,
                        "seed": int(seed),
                        "line_id": item.line_id,
                        "out_wav": str(out_path),
                        "target_seconds": item.target_seconds,
                        "out_seconds": out_seconds,
                        "duration_error_seconds": (
                            None
                            if item.target_seconds is None
                            else float(out_seconds - item.target_seconds)
                        ),
                        "duration_rel_error": rel_err,
                        "identity_proxy": identity,
                        "selected_candidate": int(best_idx),
                        "candidate_count": int(args.candidate_count),
                        "candidate_details": candidate_rows,
                    }
                    all_rows.append(row)

                print(
                    f"  [{line_idx}/{len(lines)}] {item.line_id} "
                    f"cond={condition} seed={seed} done"
                )

    # Write detailed rows.
    results_jsonl = out_root / "results.jsonl"
    with results_jsonl.open("w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    # Summary by (condition, seed)
    summary_run_csv = out_root / "summary_by_run.csv"
    summary_condition_csv = out_root / "summary_by_condition.csv"
    summary_md = out_root / "summary.md"

    run_groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    condition_groups: dict[str, list[dict[str, Any]]] = {}
    for row in all_rows:
        run_groups.setdefault((row["condition"], int(row["seed"])), []).append(row)
        condition_groups.setdefault(row["condition"], []).append(row)

    run_headers = [
        "condition",
        "seed",
        "n_lines",
        "mae_seconds",
        "mape_percent",
        "within_5pct",
        "within_10pct",
        "mean_identity_proxy",
    ]
    with summary_run_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=run_headers)
        writer.writeheader()
        for (condition, seed), rows in sorted(run_groups.items()):
            s = summarize_rows(rows)
            writer.writerow(
                {
                    "condition": condition,
                    "seed": seed,
                    **s,
                }
            )

    cond_headers = [
        "condition",
        "n_lines",
        "mae_seconds",
        "mape_percent",
        "within_5pct",
        "within_10pct",
        "mean_identity_proxy",
    ]
    with summary_condition_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cond_headers)
        writer.writeheader()
        for condition, rows in sorted(condition_groups.items()):
            s = summarize_rows(rows)
            writer.writerow(
                {
                    "condition": condition,
                    **s,
                }
            )

    with summary_md.open("w", encoding="utf-8") as f:
        f.write("# Dubbing Ablation Summary\n\n")
        f.write("## By Condition\n\n")
        f.write("| condition | n_lines | mae_seconds | mape_percent | within_5pct | within_10pct | mean_identity_proxy |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for condition, rows in sorted(condition_groups.items()):
            s = summarize_rows(rows)
            f.write(
                f"| {condition} | {s['n_lines']} | {fmt_num(s['mae_seconds'])} | "
                f"{fmt_num(s['mape_percent'])} | {fmt_num(s['within_5pct'])} | "
                f"{fmt_num(s['within_10pct'])} | {fmt_num(s['mean_identity_proxy'])} |\n"
            )

        f.write("\n## By Condition + Seed\n\n")
        f.write("| condition | seed | n_lines | mae_seconds | mape_percent | within_5pct | within_10pct | mean_identity_proxy |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for (condition, seed), rows in sorted(run_groups.items()):
            s = summarize_rows(rows)
            f.write(
                f"| {condition} | {seed} | {s['n_lines']} | {fmt_num(s['mae_seconds'])} | "
                f"{fmt_num(s['mape_percent'])} | {fmt_num(s['within_5pct'])} | "
                f"{fmt_num(s['within_10pct'])} | {fmt_num(s['mean_identity_proxy'])} |\n"
            )

    print("\n[DONE] Ablation complete.")
    print(f"[DONE] results_jsonl: {results_jsonl}")
    print(f"[DONE] summary_by_run_csv: {summary_run_csv}")
    print(f"[DONE] summary_by_condition_csv: {summary_condition_csv}")
    print(f"[DONE] summary_md: {summary_md}")


if __name__ == "__main__":
    main()

