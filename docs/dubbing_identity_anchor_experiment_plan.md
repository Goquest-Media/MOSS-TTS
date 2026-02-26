# Dubbing Experiment Plan: Stable Character Identity + Per-Line Expressiveness

## 1. Objective

For Turkish-to-English dubbing with MOSS-TTS, preserve both:

1. Stable character identity (same timbre across all lines).
2. Per-line expressiveness/prosody/timing from each original Turkish line.

Current baseline (using each line's Turkish audio as clone reference) preserves (2) well but causes line-to-line timbre drift.

## 2. Core Hypothesis

Use two conditioning channels in one continuation call:

1. Identity anchor (fixed character clip) via `reference=[anchor_codes]`.
2. Per-line expressive context via `AssistantMessage(audio_codes_list=[line_turkish_audio])`.

Text is formatted as (join style is a controlled variable):

```text
space: <turkish_transcript_of_prefix_audio> <english_dub_line>
newline: <turkish_transcript_of_prefix_audio>\n<english_dub_line>
lang_tags: [TR]...[/TR] [EN]...[/EN]
```

This should keep a stable base voice while retaining local emotional cadence.

## 3. Implementation in This Repo

Script:

- `scripts/dub_identity_anchor.py`

What it does:

1. Loads a JSONL manifest of lines.
2. Encodes one fixed anchor once per run.
3. For each line, builds continuation conversation:
   1. `UserMessage(text=<join_style>(turkish_text, english_text), reference=[anchor_codes])`
   2. `AssistantMessage(audio_codes_list=[line.turkish_audio])`
4. Generates audio with conservative sampling defaults.
5. Saves one WAV per line plus `report.jsonl` with duration diagnostics.
6. Saves `run_metadata.json` for auditability (sources, revisions, runtime versions, decoding setup).
7. Auto-detects MOSS generation API:
   1. Delay model (`audio_temperature` kwargs path), or
   2. Local-transformer model (`generation_config.layers` path),
   so decoding controls are applied deterministically for either family.

## 4. Manifest Format (JSONL)

One JSON object per line:

```json
{
  "line_id": "ep01_sc03_l012",
  "turkish_audio": "/abs/path/to/line_012_tr.wav",
  "turkish_text": "Turkish transcript for this line audio",
  "english_text": "Syllable-matched English dub line",
  "target_seconds": 3.84
}
```

Fields:

1. `line_id` string.
2. `turkish_audio` absolute path to source Turkish line audio.
3. `turkish_text` transcript matching `turkish_audio`.
4. `english_text` target line.
5. `target_seconds` optional duration target for lip sync diagnostics.

## 5. Experimental Conditions

Run identical line sets under these conditions:

1. `C0` Baseline: per-line Turkish as clone reference (current pipeline).
2. `C1` Anchor-only: fixed anchor only, no per-line expressive prefix.
3. `C2` Proposed: fixed anchor + per-line Turkish continuation prefix.
4. `C3` Proposed + rerank: generate N candidates/line, select by speaker similarity and duration error.

Minimum required comparison: `C0` vs `C2`.

## 6. Recommended Starting Decoding Settings

For `C2`:

1. `audio_temperature=1.2`
2. `audio_top_p=0.7`
3. `audio_top_k=20`
4. `audio_repetition_penalty=1.0`
5. `cap_ratio=1.35`

Then small sweeps:

1. temperature: `1.1, 1.2, 1.3`
2. top_p: `0.65, 0.70, 0.75`
3. top_k: `15, 20, 25`
4. cap_ratio: `1.25, 1.35, 1.50`

## 7. Run Command

```bash
python3 scripts/dub_identity_anchor.py \
  --manifest /abs/path/character_lines.jsonl \
  --anchor-audio /abs/path/character_anchor.wav \
  --out-dir /abs/path/outputs/mut_c2_run1 \
  --model-revision <pinned_model_revision_or_tag> \
  --codec-revision <pinned_codec_revision_or_tag> \
  --cpu-offload \
  --text-join-style lang_tags \
  --audio-temperature 1.2 \
  --audio-top-p 0.7 \
  --audio-top-k 20 \
  --audio-repetition-penalty 1.0 \
  --cap-ratio 1.35 \
  --seed 42
```

Expected outputs:

1. `out_dir/<line_id>.wav` for each line.
2. `out_dir/report.jsonl` containing `target_seconds`, `out_seconds`, and `duration_error_seconds`.
3. `out_dir/run_metadata.json` with deterministic run metadata.

## 8. Evaluation Metrics

### 8.1 Identity Consistency

1. Anchor-vs-output speaker similarity per line (cosine, WeSpeaker/ECAPA).
2. Intra-character similarity variance across lines.
3. Pairwise output-output speaker distance statistics.

### 8.2 Expressiveness Retention

Compare dubbed output against source Turkish line:

1. F0 contour similarity.
2. Energy contour similarity.
3. Pause boundary alignment.
4. Relative speech-rate similarity.

### 8.3 Lip-Sync Timing

1. Absolute duration error: `|out_seconds - target_seconds|`.
2. Relative duration error percent.
3. Proportion of lines within thresholds (for example `<=5%`, `<=10%`).

### 8.4 Listening Tests

Blind A/B (`C0` vs `C2`) with ratings for:

1. Character identity consistency.
2. Emotional match.
3. Overall dubbing quality.

## 9. Acceptance Criteria

Promote `C2` if:

1. Identity metrics improve materially vs `C0` (higher mean similarity, lower variance).
2. Expressiveness does not degrade materially.
3. Timing remains within operational retime tolerance.

## 10. Data Hygiene Requirements

1. Anchor clip: clean, neutral, 5-8 seconds.
2. Turkish per-line audio: tight utterance boundaries.
3. Turkish transcript must match per-line prefix audio.
4. One fixed anchor per character across the entire episode.

## 11. Risks and Mitigations

1. Turkish carryover may affect English articulation.
   1. Test text delimiters between Turkish transcript and English line.
2. Continuation may miss exact duration.
   1. Tune `cap_ratio`, then apply light final retime.
3. Sampling variance remains.
   1. Multi-seed runs and optional candidate reranking.

## 12. Deliverables for Researcher

1. Metrics table for `C0-C3`.
2. Curated listening bundle of representative lines.
3. Failure taxonomy with examples.
4. Final production preset and fallback policy.

## 13. One-Command Ablation Runner

This repo also includes:

- `scripts/run_dubbing_ablation.py`

It runs `C0/C1/C2/C3`, supports multiple seeds, and writes:

1. `results.jsonl` (all line-level outputs and metadata)
2. `summary_by_run.csv` (condition+seed aggregates)
3. `summary_by_condition.csv` (condition aggregates)
4. `summary_by_mode.csv` (generation vs continuation aggregates)
5. `summary.md` (human-readable table)
6. `run_metadata.json` (sources, revisions, backend, weights, runtime versions)

Example:

```bash
python3 scripts/run_dubbing_ablation.py \
  --manifest /abs/path/character_lines.jsonl \
  --anchor-audio /abs/path/character_anchor.wav \
  --out-dir /abs/path/outputs/ablation_run_01 \
  --model-revision <pinned_model_revision_or_tag> \
  --codec-revision <pinned_codec_revision_or_tag> \
  --cpu-offload \
  --conditions C0,C1,C2,C3 \
  --seeds 42,1337,2026 \
  --timing-control-policy off \
  --text-join-style lang_tags \
  --identity-backend xvector \
  --candidate-count 3 \
  --identity-weight 1.0 \
  --duration-weight 0.35 \
  --expressiveness-weight 0.0 \
  --audio-temperature 1.2 \
  --audio-top-p 0.7 \
  --audio-top-k 20 \
  --audio-repetition-penalty 1.0 \
  --cap-ratio 1.35
```

## 14. Deterministic Evaluation Protocol

To make conclusions auditable and reproducible:

1. Pin model/codec revisions (`--model-revision`, `--codec-revision`) and keep `run_metadata.json`.
2. Keep timing policy explicit. For cross-mode fairness, prefer `--timing-control-policy off`.
3. Keep TR/EN boundary explicit with a fixed join style (`--text-join-style`).
4. Report identity mean and identity variance, not mean only.
5. Report expressiveness metrics (`f0_similarity`, `energy_similarity`, `pause_f1`, `speech_rate_similarity`) alongside timing metrics.
6. For constrained VRAM, use `--cpu-offload` and record it in `run_metadata.json`.
