#!/usr/bin/env python3
"""
Run dubbing ablations for identity-vs-expressiveness tradeoff in one command.

Conditions:
  C0: baseline_per_line_clone_generation
      - generation mode
      - reference = per-line Turkish audio
      - text = English line

  C1: anchor_only_generation
      - generation mode
      - reference = fixed anchor
      - text = English line

  C2: anchor_plus_line_continuation
      - continuation mode
      - user reference = fixed anchor
      - assistant prefix audio = per-line Turkish audio
      - text join style is configurable (see --text-join-style)

  C3: C2 + candidate rerank
      - generate N candidates using C2
      - choose best by weighted objective
        score = identity_weight * identity
                - duration_weight * abs(duration_rel_error)
                + expressiveness_weight * expressiveness_score

Determinism/Auditability:
  - Optional model/codec revision pinning via snapshot_download
  - Run metadata output with resolved model sources and runtime versions
  - Explicit timing-control policy so cross-condition comparisons are auditable
"""

from __future__ import annotations

import argparse
import copy
import csv
import inspect
import importlib.util
import json
import math
import platform
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F
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
SUPPORTED_CONDITIONS = ("C0", "C1", "C2", "C3")
SUPPORTED_TEXT_JOIN_STYLES = ("space", "newline", "lang_tags")
SUPPORTED_TIMING_POLICIES = ("off", "generation_only")
SUPPORTED_IDENTITY_BACKENDS = ("mfcc", "xvector")
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


@dataclass
class ExpressivenessFeatures:
    f0_hz: torch.Tensor
    energy_db: torch.Tensor
    pause_mask: torch.Tensor
    speech_rate_hz: float


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


def as_mono_1d(wav: torch.Tensor) -> torch.Tensor:
    return as_mono_2d(wav).squeeze(0)


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
    min_tokens_with_target: int,
    min_tokens_without_target: int,
) -> int:
    expected = compute_expected_tokens(item)
    if expected is not None:
        return max(int(min_tokens_with_target), int(expected * cap_ratio))
    est_tokens = max(1, int(len(item.english_text) * fallback_chars_ratio))
    return max(int(min_tokens_without_target), int(est_tokens * 2.0))


def format_c2_text(item: LineItem, text_join_style: str) -> str:
    if text_join_style == "space":
        return f"{item.turkish_text} {item.english_text}".strip()
    if text_join_style == "newline":
        return f"{item.turkish_text}\n{item.english_text}".strip()
    if text_join_style == "lang_tags":
        return f"[TR]{item.turkish_text}[/TR] [EN]{item.english_text}[/EN]".strip()
    raise ValueError(f"Unsupported text_join_style: {text_join_style}")


def build_conversation(
    condition: str,
    item: LineItem,
    processor: Any,
    anchor_codes: torch.Tensor,
    timing_control_policy: str,
    text_join_style: str,
) -> tuple[list[list[dict[str, Any]]], str]:
    expected_tokens = compute_expected_tokens(item)
    include_tokens = (
        timing_control_policy == "generation_only" and expected_tokens is not None
    )

    if condition == "C0":
        user_kwargs: dict[str, Any] = {
            "text": item.english_text,
            "reference": [item.turkish_audio],
        }
        if include_tokens:
            user_kwargs["tokens"] = int(expected_tokens)
        return [[processor.build_user_message(**user_kwargs)]], "generation"

    if condition == "C1":
        user_kwargs = {
            "text": item.english_text,
            "reference": [anchor_codes],
        }
        if include_tokens:
            user_kwargs["tokens"] = int(expected_tokens)
        return [[processor.build_user_message(**user_kwargs)]], "generation"

    if condition in {"C2", "C3"}:
        full_text = format_c2_text(item=item, text_join_style=text_join_style)
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


def load_audio_features(path: str) -> tuple[torch.Tensor, int]:
    wav, sr = torchaudio.load(path)
    return as_mono_1d(wav), int(sr)


def resample_mono(wav: torch.Tensor, sample_rate: int, target_sr: int) -> torch.Tensor:
    mono = as_mono_2d(wav)
    if sample_rate != target_sr:
        mono = torchaudio.functional.resample(mono, sample_rate, target_sr)
    return mono.squeeze(0)


def interpolate_1d(signal: torch.Tensor, target_len: int, mode: str = "linear") -> torch.Tensor:
    if target_len <= 0:
        raise ValueError("target_len must be > 0")
    src = signal.reshape(1, 1, -1).to(torch.float32)
    if src.shape[-1] == target_len:
        return src.reshape(-1)
    if mode == "nearest":
        out = F.interpolate(src, size=target_len, mode=mode)
    else:
        out = F.interpolate(src, size=target_len, mode=mode, align_corners=False)
    return out.reshape(-1)


def pearson_corr(x: torch.Tensor, y: torch.Tensor) -> Optional[float]:
    if x.numel() == 0 or y.numel() == 0:
        return None
    x = x.to(torch.float32)
    y = y.to(torch.float32)
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.norm(x) * torch.norm(y)
    if float(denom.item()) <= 1e-12:
        return None
    return float(torch.dot(x, y).item() / denom.item())


def frame_rms(wav: torch.Tensor, frame_len: int, hop_len: int) -> torch.Tensor:
    if wav.numel() == 0:
        return torch.zeros(1, dtype=torch.float32)
    if wav.numel() < frame_len:
        wav = F.pad(wav, (0, frame_len - wav.numel()))
    frames = wav.unfold(0, frame_len, hop_len)
    rms = torch.sqrt(torch.mean(frames**2, dim=-1) + 1e-9)
    return rms.to(torch.float32)


def pause_mask_from_energy(energy_db: torch.Tensor, margin_db: float = 15.0) -> torch.Tensor:
    if energy_db.numel() == 0:
        return torch.zeros(1, dtype=torch.bool)
    thr = float(torch.median(energy_db).item()) - float(margin_db)
    return (energy_db < thr).to(torch.bool)


def speech_rate_proxy_hz(
    pause_mask: torch.Tensor,
    n_samples: int,
    sample_rate: int,
    hop_len: int,
) -> float:
    duration_sec = max(float(n_samples) / float(sample_rate), 1e-6)
    if pause_mask.numel() < 2:
        return 0.0
    speech_mask = ~pause_mask
    onsets = torch.logical_and(speech_mask[1:], pause_mask[:-1]).sum().item()
    return float(onsets) / duration_sec


def extract_expressiveness_features(
    wav: torch.Tensor,
    sample_rate: int,
    target_sr: int = 16000,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
) -> ExpressivenessFeatures:
    mono = resample_mono(wav=wav, sample_rate=sample_rate, target_sr=target_sr)
    frame_len = max(1, int(round(target_sr * (frame_ms / 1000.0))))
    hop_len = max(1, int(round(target_sr * (hop_ms / 1000.0))))

    energy = frame_rms(mono, frame_len=frame_len, hop_len=hop_len)
    energy_db = 20.0 * torch.log10(energy + 1e-9)
    pause_mask = pause_mask_from_energy(energy_db=energy_db)

    try:
        f0 = torchaudio.functional.detect_pitch_frequency(
            waveform=mono.unsqueeze(0),
            sample_rate=target_sr,
            frame_time=hop_ms / 1000.0,
            win_length=30,
        ).squeeze(0)
    except Exception:
        f0 = torch.zeros(max(1, energy_db.numel()), dtype=torch.float32)

    speech_rate_hz = speech_rate_proxy_hz(
        pause_mask=pause_mask,
        n_samples=int(mono.numel()),
        sample_rate=target_sr,
        hop_len=hop_len,
    )

    return ExpressivenessFeatures(
        f0_hz=f0.to(torch.float32).cpu(),
        energy_db=energy_db.to(torch.float32).cpu(),
        pause_mask=pause_mask.cpu(),
        speech_rate_hz=float(speech_rate_hz),
    )


def compare_expressiveness(
    src: ExpressivenessFeatures,
    out: ExpressivenessFeatures,
) -> dict[str, Optional[float]]:
    out_len = max(out.energy_db.numel(), 1)
    src_energy = interpolate_1d(src.energy_db, out_len, mode="linear")
    out_energy = interpolate_1d(out.energy_db, out_len, mode="linear")
    energy_similarity = pearson_corr(src_energy, out_energy)

    src_f0 = interpolate_1d(src.f0_hz, out_len, mode="linear")
    out_f0 = interpolate_1d(out.f0_hz, out_len, mode="linear")
    voiced = (src_f0 > 60.0) & (out_f0 > 60.0)
    if int(voiced.sum().item()) >= 5:
        src_f0_log = torch.log(torch.clamp(src_f0[voiced], min=1e-3))
        out_f0_log = torch.log(torch.clamp(out_f0[voiced], min=1e-3))
        f0_similarity = pearson_corr(src_f0_log, out_f0_log)
    else:
        f0_similarity = None

    src_pause = interpolate_1d(src.pause_mask.to(torch.float32), out_len, mode="nearest") > 0.5
    out_pause = interpolate_1d(out.pause_mask.to(torch.float32), out_len, mode="nearest") > 0.5
    tp = torch.logical_and(src_pause, out_pause).sum().item()
    fp = torch.logical_and(~src_pause, out_pause).sum().item()
    fn = torch.logical_and(src_pause, ~out_pause).sum().item()
    denom = (2 * tp) + fp + fn
    pause_f1 = 1.0 if denom == 0 else float((2 * tp) / denom)

    if src.speech_rate_hz > 0.0 and out.speech_rate_hz > 0.0:
        speech_rate_ratio = float(out.speech_rate_hz / src.speech_rate_hz)
        speech_rate_similarity = float(math.exp(-abs(math.log(speech_rate_ratio))))
    else:
        speech_rate_ratio = None
        speech_rate_similarity = None

    values = [
        v
        for v in (
            f0_similarity,
            energy_similarity,
            pause_f1,
            speech_rate_similarity,
        )
        if v is not None
    ]
    expressiveness_score = float(sum(values) / len(values)) if values else None
    return {
        "f0_similarity": f0_similarity,
        "energy_similarity": energy_similarity,
        "pause_f1": pause_f1,
        "speech_rate_similarity": speech_rate_similarity,
        "speech_rate_ratio": speech_rate_ratio,
        "expressiveness_score": expressiveness_score,
    }


def mfcc_embedding(wav: torch.Tensor, sample_rate: int, target_sr: int = 16000) -> torch.Tensor:
    mono = resample_mono(wav=wav, sample_rate=sample_rate, target_sr=target_sr).unsqueeze(0)
    mfcc = torchaudio.transforms.MFCC(
        sample_rate=target_sr,
        n_mfcc=20,
        melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 40},
    )(mono)
    emb = mfcc.mean(dim=-1).squeeze(0)
    return F.normalize(emb.to(torch.float32), dim=0)


class IdentityScorer:
    def __init__(
        self,
        backend: str,
        anchor_wav: torch.Tensor,
        anchor_sr: int,
        device: torch.device,
    ):
        self.backend = backend
        self.device = device

        if backend == "mfcc":
            self.target_sr = 16000
            self._xvector_model = None
            self.anchor_emb = mfcc_embedding(anchor_wav, sample_rate=anchor_sr, target_sr=self.target_sr)
            return

        if backend == "xvector":
            if not hasattr(torchaudio.pipelines, "SUPERB_XVECTOR"):
                raise RuntimeError(
                    "identity-backend=xvector requires torchaudio.pipelines.SUPERB_XVECTOR"
                )
            bundle = torchaudio.pipelines.SUPERB_XVECTOR
            self.target_sr = int(bundle.sample_rate)
            self._xvector_model = bundle.get_model().to(device)
            self._xvector_model.eval()
            self.anchor_emb = self._embedding_xvector(anchor_wav, sample_rate=anchor_sr)
            return

        raise ValueError(f"Unsupported identity backend: {backend}")

    def _embedding_xvector(self, wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if self._xvector_model is None:
            raise RuntimeError("xvector model not initialized")
        mono = resample_mono(wav=wav, sample_rate=sample_rate, target_sr=self.target_sr).to(self.device)
        with torch.no_grad():
            emb = self._xvector_model(mono.unsqueeze(0))
        if isinstance(emb, (tuple, list)):
            emb = emb[0]
        emb = emb.squeeze(0).detach().float().cpu()
        return F.normalize(emb, dim=0)

    def embedding(self, wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if self.backend == "mfcc":
            return mfcc_embedding(wav=wav, sample_rate=sample_rate, target_sr=self.target_sr)
        return self._embedding_xvector(wav=wav, sample_rate=sample_rate)

    def score(self, wav: torch.Tensor, sample_rate: int) -> float:
        emb = self.embedding(wav=wav, sample_rate=sample_rate)
        score = F.cosine_similarity(emb.unsqueeze(0), self.anchor_emb.unsqueeze(0), dim=-1).item()
        return float(score)


def detect_generation_api(model: Any) -> str:
    """
    Detect which generation API this loaded MOSS model exposes.

    - delay_kwargs: MossTTSDelayModel.generate(..., audio_temperature, ...)
    - local_generation_config: HF GenerationMixin.generate(..., generation_config=...)
    """
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


def generate_once(
    model: Any,
    processor: Any,
    input_device: torch.device,
    conversations: list[list[dict[str, Any]]],
    mode: str,
    max_new_tokens: int,
    audio_temperature: float,
    audio_top_p: float,
    audio_top_k: int,
    audio_repetition_penalty: float,
    generation_api: str,
    local_base_generation_config: Optional[DelayGenerationConfig],
) -> torch.Tensor:
    batch = processor(conversations, mode=mode)
    input_ids = batch["input_ids"].to(input_device)
    attention_mask = batch["attention_mask"].to(input_device)
    with torch.no_grad():
        if generation_api == "delay_kwargs":
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=int(max_new_tokens),
                audio_temperature=float(audio_temperature),
                audio_top_p=float(audio_top_p),
                audio_top_k=int(audio_top_k),
                audio_repetition_penalty=float(audio_repetition_penalty),
            )
        elif generation_api == "local_generation_config":
            if local_base_generation_config is None:
                raise RuntimeError("Missing local_base_generation_config for local generation API.")
            generation_config = build_local_generation_config(
                base_cfg=local_base_generation_config,
                max_new_tokens=int(max_new_tokens),
                audio_temperature=float(audio_temperature),
                audio_top_p=float(audio_top_p),
                audio_top_k=int(audio_top_k),
                audio_repetition_penalty=float(audio_repetition_penalty),
            )
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
            )
        else:
            raise RuntimeError(f"Unsupported generation_api: {generation_api}")
    messages = processor.decode(outputs)
    return decode_first_audio(messages)


def safe_std(values: list[float]) -> Optional[float]:
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    t = torch.tensor(values, dtype=torch.float32)
    return float(torch.std(t, unbiased=False).item())


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {
            "n_lines": 0,
            "mae_seconds": None,
            "mape_percent": None,
            "within_5pct": None,
            "within_10pct": None,
            "mean_identity": None,
            "std_identity": None,
            "mean_f0_similarity": None,
            "mean_energy_similarity": None,
            "mean_pause_f1": None,
            "mean_speech_rate_similarity": None,
            "mean_expressiveness_score": None,
        }

    abs_errors = []
    abs_rel = []
    identity = []
    f0_sim = []
    energy_sim = []
    pause_f1 = []
    speech_rate_sim = []
    expressiveness = []
    within_5 = 0
    within_10 = 0
    with_target = 0

    for row in rows:
        if row.get("identity_score") is not None:
            identity.append(float(row["identity_score"]))
        if row.get("f0_similarity") is not None:
            f0_sim.append(float(row["f0_similarity"]))
        if row.get("energy_similarity") is not None:
            energy_sim.append(float(row["energy_similarity"]))
        if row.get("pause_f1") is not None:
            pause_f1.append(float(row["pause_f1"]))
        if row.get("speech_rate_similarity") is not None:
            speech_rate_sim.append(float(row["speech_rate_similarity"]))
        if row.get("expressiveness_score") is not None:
            expressiveness.append(float(row["expressiveness_score"]))

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
    within_5pct = (100.0 * within_5 / with_target) if with_target > 0 else None
    within_10pct = (100.0 * within_10 / with_target) if with_target > 0 else None

    mean_identity = sum(identity) / len(identity) if identity else None
    std_identity = safe_std(identity)
    mean_f0 = sum(f0_sim) / len(f0_sim) if f0_sim else None
    mean_energy = sum(energy_sim) / len(energy_sim) if energy_sim else None
    mean_pause = sum(pause_f1) / len(pause_f1) if pause_f1 else None
    mean_speech_rate = sum(speech_rate_sim) / len(speech_rate_sim) if speech_rate_sim else None
    mean_expressiveness = (
        sum(expressiveness) / len(expressiveness) if expressiveness else None
    )

    return {
        "n_lines": n,
        "mae_seconds": mae,
        "mape_percent": mape,
        "within_5pct": within_5pct,
        "within_10pct": within_10pct,
        "mean_identity": mean_identity,
        "std_identity": std_identity,
        "mean_f0_similarity": mean_f0,
        "mean_energy_similarity": mean_energy,
        "mean_pause_f1": mean_pause,
        "mean_speech_rate_similarity": mean_speech_rate,
        "mean_expressiveness_score": mean_expressiveness,
    }


def fmt_num(v: Optional[float], digits: int = 4) -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "n/a"
    return f"{v:.{digits}f}"


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


def normalize_timing_policy(
    timing_control_policy: str,
    include_tokens_in_generation: bool,
) -> str:
    policy = timing_control_policy
    if include_tokens_in_generation and policy == "off":
        policy = "generation_only"
    return policy


def condition_mode(condition: str) -> str:
    if condition in {"C0", "C1"}:
        return "generation"
    return "continuation"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run C0/C1/C2/C3 dubbing ablation and export deterministic comparison tables."
    )
    parser.add_argument("--manifest", required=True, help="Input JSONL manifest path.")
    parser.add_argument("--anchor-audio", required=True, help="Fixed identity anchor wav/mp3/m4a.")
    parser.add_argument("--out-dir", required=True, help="Output directory root.")

    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--model-revision", default=None, help="Optional pinned revision for model repo.")
    parser.add_argument("--codec-path", default=DEFAULT_CODEC_PATH, help="Codec repo/path passed as processor codec_path.")
    parser.add_argument("--codec-revision", default=None, help="Optional pinned revision for codec repo.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--cpu-offload",
        action="store_true",
        help="Load with device_map=auto for lower VRAM usage (requires accelerate).",
    )

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
    parser.add_argument("--min-max-new-tokens-with-target", type=int, default=128)
    parser.add_argument("--min-max-new-tokens-without-target", type=int, default=256)

    parser.add_argument(
        "--timing-control-policy",
        choices=SUPPORTED_TIMING_POLICIES,
        default="off",
        help="off = never set UserMessage.tokens; generation_only = set tokens only for C0/C1",
    )
    parser.add_argument(
        "--include-tokens-in-generation",
        action="store_true",
        help="Deprecated compatibility flag. If set and timing policy is off, policy becomes generation_only.",
    )

    parser.add_argument(
        "--text-join-style",
        choices=SUPPORTED_TEXT_JOIN_STYLES,
        default="space",
        help="How C2/C3 text is built from Turkish transcript + English text.",
    )

    parser.add_argument(
        "--identity-backend",
        choices=SUPPORTED_IDENTITY_BACKENDS,
        default="mfcc",
        help="Identity scorer backend. xvector is stronger but requires loading torchaudio SUPERB_XVECTOR weights.",
    )
    parser.add_argument("--identity-weight", type=float, default=1.0)
    parser.add_argument("--duration-weight", type=float, default=0.35)
    parser.add_argument("--expressiveness-weight", type=float, default=0.0)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).expanduser().resolve()
    out_root = Path(args.out_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[1]

    timing_control_policy = normalize_timing_policy(
        timing_control_policy=str(args.timing_control_policy),
        include_tokens_in_generation=bool(args.include_tokens_in_generation),
    )
    if timing_control_policy not in SUPPORTED_TIMING_POLICIES:
        raise ValueError(f"Unsupported timing_control_policy: {timing_control_policy}")

    conditions = parse_csv_strings(args.conditions)
    if not conditions:
        raise ValueError("No conditions provided.")
    invalid = [c for c in conditions if c not in SUPPORTED_CONDITIONS]
    if invalid:
        raise ValueError(f"Unsupported conditions: {invalid}")

    seeds = parse_csv_ints(args.seeds)
    if not seeds:
        raise ValueError("At least one seed is required.")

    model_source = maybe_snapshot_download(args.model_path, args.model_revision)
    codec_source = maybe_snapshot_download(args.codec_path, args.codec_revision)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    attn_implementation = resolve_attn_implementation(device=device, dtype=dtype)

    print(
        f"[INFO] Loading model={model_source} codec={codec_source} "
        f"device={device} attn={attn_implementation} cpu_offload={bool(args.cpu_offload)}"
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
    print(
        "[INFO] timing_control_policy="
        f"{timing_control_policy} text_join_style={args.text_join_style} "
        f"identity_backend={args.identity_backend} generation_api={generation_api} "
        f"model_input_device={model_input_device}"
    )

    print("[INFO] Encoding anchor...")
    anchor_codes = processor.encode_audios_from_path([str(args.anchor_audio)])[0]
    anchor_wav_raw, anchor_sr = torchaudio.load(str(args.anchor_audio))
    anchor_wav = as_mono_1d(anchor_wav_raw)
    identity_scorer = IdentityScorer(
        backend=str(args.identity_backend),
        anchor_wav=anchor_wav,
        anchor_sr=int(anchor_sr),
        device=device,
    )

    lines = load_manifest(manifest_path)
    if args.limit_lines and int(args.limit_lines) > 0:
        lines = lines[: int(args.limit_lines)]
    print(f"[INFO] lines={len(lines)} conditions={conditions} seeds={seeds}")

    print("[INFO] Precomputing source-line expressiveness features...")
    source_features: dict[str, ExpressivenessFeatures] = {}
    for item in lines:
        src_wav, src_sr = load_audio_features(item.turkish_audio)
        source_features[item.line_id] = extract_expressiveness_features(
            wav=src_wav,
            sample_rate=src_sr,
        )

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
                    min_tokens_with_target=int(args.min_max_new_tokens_with_target),
                    min_tokens_without_target=int(args.min_max_new_tokens_without_target),
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
                        timing_control_policy=timing_control_policy,
                        text_join_style=str(args.text_join_style),
                    )
                    wav = generate_once(
                        model=model,
                        processor=processor,
                        input_device=model_input_device,
                        conversations=conversations,
                        mode=mode,
                        max_new_tokens=max_new_tokens,
                        audio_temperature=float(args.audio_temperature),
                        audio_top_p=float(args.audio_top_p),
                        audio_top_k=int(args.audio_top_k),
                        audio_repetition_penalty=float(args.audio_repetition_penalty),
                        generation_api=generation_api,
                        local_base_generation_config=local_base_generation_config,
                    )
                    out_path = run_dir / f"{item.line_id}.wav"
                    torchaudio.save(str(out_path), as_mono_2d(wav), sample_rate)

                    out_seconds = float(wav.shape[-1] / sample_rate)
                    rel_err = safe_rel_error(out_seconds=out_seconds, target_seconds=item.target_seconds)
                    identity = identity_scorer.score(wav=wav, sample_rate=sample_rate)
                    out_features = extract_expressiveness_features(
                        wav=wav,
                        sample_rate=sample_rate,
                    )
                    expr = compare_expressiveness(source_features[item.line_id], out_features)

                    row = {
                        "condition": condition,
                        "mode": condition_mode(condition),
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
                        "identity_score": identity,
                        "identity_backend": str(args.identity_backend),
                        "selected_candidate": 0,
                        "candidate_count": 1,
                        **expr,
                    }
                    all_rows.append(row)
                else:
                    conversations, mode = build_conversation(
                        condition="C3",
                        item=item,
                        processor=processor,
                        anchor_codes=anchor_codes,
                        timing_control_policy=timing_control_policy,
                        text_join_style=str(args.text_join_style),
                    )
                    candidate_rows = []
                    best_score = None
                    best_wav = None
                    best_idx = None
                    best_expr = None
                    best_identity = None
                    best_rel_err = None

                    for cand_idx in range(int(args.candidate_count)):
                        cand_seed = int(seed) + (line_idx * 100) + (cand_idx * 100000)
                        torch.manual_seed(cand_seed)
                        if torch.cuda.is_available():
                            torch.cuda.manual_seed_all(cand_seed)

                        wav = generate_once(
                            model=model,
                            processor=processor,
                            input_device=model_input_device,
                            conversations=conversations,
                            mode=mode,
                            max_new_tokens=max_new_tokens,
                            audio_temperature=float(args.audio_temperature),
                            audio_top_p=float(args.audio_top_p),
                            audio_top_k=int(args.audio_top_k),
                            audio_repetition_penalty=float(args.audio_repetition_penalty),
                            generation_api=generation_api,
                            local_base_generation_config=local_base_generation_config,
                        )

                        out_seconds = float(wav.shape[-1] / sample_rate)
                        rel_err = safe_rel_error(
                            out_seconds=out_seconds,
                            target_seconds=item.target_seconds,
                        )
                        identity = identity_scorer.score(
                            wav=wav,
                            sample_rate=sample_rate,
                        )
                        out_features = extract_expressiveness_features(
                            wav=wav,
                            sample_rate=sample_rate,
                        )
                        expr = compare_expressiveness(source_features[item.line_id], out_features)

                        duration_penalty = abs(rel_err) if rel_err is not None else 0.0
                        expr_score = expr["expressiveness_score"]
                        expr_term = float(expr_score) if expr_score is not None else 0.0
                        score = (
                            float(args.identity_weight) * identity
                            - float(args.duration_weight) * duration_penalty
                            + float(args.expressiveness_weight) * expr_term
                        )

                        cand_row = {
                            "candidate_index": cand_idx,
                            "cand_seed": cand_seed,
                            "identity_score": identity,
                            "duration_rel_error": rel_err,
                            "score": score,
                            "out_seconds": out_seconds,
                            **expr,
                        }
                        candidate_rows.append(cand_row)

                        if bool(args.save_all_candidates):
                            cand_path = run_dir / f"{item.line_id}.cand{cand_idx}.wav"
                            torchaudio.save(str(cand_path), as_mono_2d(wav), sample_rate)

                        if best_score is None or score > best_score:
                            best_score = score
                            best_wav = wav
                            best_idx = cand_idx
                            best_expr = expr
                            best_identity = identity
                            best_rel_err = rel_err

                    assert best_wav is not None and best_idx is not None
                    assert best_expr is not None and best_identity is not None

                    out_path = run_dir / f"{item.line_id}.wav"
                    torchaudio.save(str(out_path), as_mono_2d(best_wav), sample_rate)

                    out_seconds = float(best_wav.shape[-1] / sample_rate)
                    row = {
                        "condition": condition,
                        "mode": condition_mode(condition),
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
                        "duration_rel_error": best_rel_err,
                        "identity_score": best_identity,
                        "identity_backend": str(args.identity_backend),
                        "selected_candidate": int(best_idx),
                        "candidate_count": int(args.candidate_count),
                        "candidate_details": candidate_rows,
                        **best_expr,
                    }
                    all_rows.append(row)

                print(
                    f"  [{line_idx}/{len(lines)}] {item.line_id} "
                    f"cond={condition} seed={seed} done"
                )

    results_jsonl = out_root / "results.jsonl"
    with results_jsonl.open("w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    run_groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    condition_groups: dict[str, list[dict[str, Any]]] = {}
    mode_groups: dict[str, list[dict[str, Any]]] = {}
    for row in all_rows:
        run_groups.setdefault((row["condition"], int(row["seed"])), []).append(row)
        condition_groups.setdefault(row["condition"], []).append(row)
        mode_groups.setdefault(str(row["mode"]), []).append(row)

    summary_run_csv = out_root / "summary_by_run.csv"
    summary_condition_csv = out_root / "summary_by_condition.csv"
    summary_mode_csv = out_root / "summary_by_mode.csv"
    summary_md = out_root / "summary.md"
    run_metadata_json = out_root / "run_metadata.json"

    headers = [
        "n_lines",
        "mae_seconds",
        "mape_percent",
        "within_5pct",
        "within_10pct",
        "mean_identity",
        "std_identity",
        "mean_f0_similarity",
        "mean_energy_similarity",
        "mean_pause_f1",
        "mean_speech_rate_similarity",
        "mean_expressiveness_score",
    ]

    with summary_run_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["condition", "seed", "mode", *headers])
        writer.writeheader()
        for (condition, seed), rows in sorted(run_groups.items()):
            s = summarize_rows(rows)
            writer.writerow(
                {
                    "condition": condition,
                    "seed": seed,
                    "mode": condition_mode(condition),
                    **s,
                }
            )

    with summary_condition_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["condition", "mode", *headers])
        writer.writeheader()
        for condition, rows in sorted(condition_groups.items()):
            s = summarize_rows(rows)
            writer.writerow(
                {
                    "condition": condition,
                    "mode": condition_mode(condition),
                    **s,
                }
            )

    with summary_mode_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["mode", *headers])
        writer.writeheader()
        for mode, rows in sorted(mode_groups.items()):
            s = summarize_rows(rows)
            writer.writerow({"mode": mode, **s})

    with summary_md.open("w", encoding="utf-8") as f:
        f.write("# Dubbing Ablation Summary\n\n")
        f.write("## Run Setup\n\n")
        f.write(f"- timing_control_policy: `{timing_control_policy}`\n")
        f.write(f"- text_join_style: `{args.text_join_style}`\n")
        f.write(f"- identity_backend: `{args.identity_backend}`\n")
        f.write(f"- identity_weight: `{float(args.identity_weight):.4f}`\n")
        f.write(f"- duration_weight: `{float(args.duration_weight):.4f}`\n")
        f.write(f"- expressiveness_weight: `{float(args.expressiveness_weight):.4f}`\n\n")

        f.write("## By Condition\n\n")
        f.write(
            "| condition | mode | n_lines | mae_seconds | mape_percent | within_5pct | within_10pct | "
            "mean_identity | std_identity | mean_f0_similarity | mean_energy_similarity | "
            "mean_pause_f1 | mean_speech_rate_similarity | mean_expressiveness_score |\n"
        )
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for condition, rows in sorted(condition_groups.items()):
            s = summarize_rows(rows)
            f.write(
                f"| {condition} | {condition_mode(condition)} | {s['n_lines']} | {fmt_num(s['mae_seconds'])} | "
                f"{fmt_num(s['mape_percent'])} | {fmt_num(s['within_5pct'])} | {fmt_num(s['within_10pct'])} | "
                f"{fmt_num(s['mean_identity'])} | {fmt_num(s['std_identity'])} | {fmt_num(s['mean_f0_similarity'])} | "
                f"{fmt_num(s['mean_energy_similarity'])} | {fmt_num(s['mean_pause_f1'])} | "
                f"{fmt_num(s['mean_speech_rate_similarity'])} | {fmt_num(s['mean_expressiveness_score'])} |\n"
            )

        f.write("\n## By Mode\n\n")
        f.write(
            "| mode | n_lines | mae_seconds | mape_percent | within_5pct | within_10pct | "
            "mean_identity | std_identity | mean_f0_similarity | mean_energy_similarity | "
            "mean_pause_f1 | mean_speech_rate_similarity | mean_expressiveness_score |\n"
        )
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for mode, rows in sorted(mode_groups.items()):
            s = summarize_rows(rows)
            f.write(
                f"| {mode} | {s['n_lines']} | {fmt_num(s['mae_seconds'])} | {fmt_num(s['mape_percent'])} | "
                f"{fmt_num(s['within_5pct'])} | {fmt_num(s['within_10pct'])} | {fmt_num(s['mean_identity'])} | "
                f"{fmt_num(s['std_identity'])} | {fmt_num(s['mean_f0_similarity'])} | "
                f"{fmt_num(s['mean_energy_similarity'])} | {fmt_num(s['mean_pause_f1'])} | "
                f"{fmt_num(s['mean_speech_rate_similarity'])} | {fmt_num(s['mean_expressiveness_score'])} |\n"
            )

    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/run_dubbing_ablation.py",
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
        "ablation_setup": {
            "conditions": conditions,
            "seeds": seeds,
            "timing_control_policy": timing_control_policy,
            "text_join_style": args.text_join_style,
            "identity_backend": args.identity_backend,
            "identity_weight": float(args.identity_weight),
            "duration_weight": float(args.duration_weight),
            "expressiveness_weight": float(args.expressiveness_weight),
            "audio_temperature": float(args.audio_temperature),
            "audio_top_p": float(args.audio_top_p),
            "audio_top_k": int(args.audio_top_k),
            "audio_repetition_penalty": float(args.audio_repetition_penalty),
            "cap_ratio": float(args.cap_ratio),
            "fallback_chars_ratio": float(args.fallback_chars_ratio),
            "min_max_new_tokens_with_target": int(args.min_max_new_tokens_with_target),
            "min_max_new_tokens_without_target": int(args.min_max_new_tokens_without_target),
            "cpu_offload": bool(args.cpu_offload),
            "candidate_count": int(args.candidate_count),
            "limit_lines": int(args.limit_lines),
            "local_text_layer_defaults": {
                "temperature": float(LOCAL_TEXT_LAYER_TEMPERATURE),
                "top_p": float(LOCAL_TEXT_LAYER_TOP_P),
                "top_k": int(LOCAL_TEXT_LAYER_TOP_K),
            },
        },
    }
    with run_metadata_json.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=True, indent=2)

    print("\n[DONE] Ablation complete.")
    print(f"[DONE] results_jsonl: {results_jsonl}")
    print(f"[DONE] summary_by_run_csv: {summary_run_csv}")
    print(f"[DONE] summary_by_condition_csv: {summary_condition_csv}")
    print(f"[DONE] summary_by_mode_csv: {summary_mode_csv}")
    print(f"[DONE] summary_md: {summary_md}")
    print(f"[DONE] run_metadata_json: {run_metadata_json}")


if __name__ == "__main__":
    main()
