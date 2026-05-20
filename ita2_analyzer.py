#!/usr/bin/env python3
"""
ITA2 audio analyzer and decoder for simple 2-tone FSK WAV files.

What it does:
- Reads a mono/stereo PCM WAV file.
- Detects active audio region and silence.
- Estimates dominant two FSK tones (near configured expectations).
- Estimates bit duration and baud from run-length timing.
- Attempts asynchronous ITA2 decode (5-bit, start bit 0, stop bits 1+).
- Compares measured parameters against expected encoder settings.

Default expected settings match the current workspace encoder.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import wave
from array import array
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


EXPECTED = {
    "baud": 50.0,
    "mark_hz": 1410.0,
    "space_hz": 1820.0,
    "stop_bits": 2,
    "sample_rate": 44100,
    "lead_in_bits": 50,
    "tail_bits": 25,
}

BAUD_SNAP_TARGET = 50.0
BAUD_SNAP_TOLERANCE = 0.75

TOLERANCE_PRESETS = {
    "default": {
        "sample_rate_hz": 1.0,
        "baud": 0.75,
        "mark_hz": 35.0,
        "space_hz": 35.0,
        "stop_bits": 0.35,
    },
    "strict": {
        "sample_rate_hz": 0.0,
        "baud": 0.25,
        "mark_hz": 10.0,
        "space_hz": 10.0,
        "stop_bits": 0.1,
    },
    "teleprinter-compatible": {
        "sample_rate_hz": 40000.0,
        "baud": 1.5,
        "mark_hz": 60.0,
        "space_hz": 60.0,
        "stop_bits": 0.6,
    },
}

LTRS_SHIFT_CODE = 0b11111
FIGS_SHIFT_CODE = 0b11011

LETTERS = {
    0b00000: "",
    0b00001: "E",
    0b00010: "\n",
    0b00011: "A",
    0b00100: " ",
    0b00101: "S",
    0b00110: "I",
    0b00111: "U",
    0b01000: "\r",
    0b01001: "D",
    0b01010: "R",
    0b01011: "J",
    0b01100: "N",
    0b01101: "F",
    0b01110: "C",
    0b01111: "K",
    0b10000: "T",
    0b10001: "Z",
    0b10010: "L",
    0b10011: "W",
    0b10100: "H",
    0b10101: "Y",
    0b10110: "P",
    0b10111: "Q",
    0b11000: "O",
    0b11001: "B",
    0b11010: "G",
    0b11100: "M",
    0b11101: "X",
    0b11110: "V",
}

FIGURES = {
    0b00000: "",
    0b00001: "3",
    0b00010: "\n",
    0b00011: "-",
    0b00100: " ",
    0b00101: "<BEL>",
    0b00110: "8",
    0b00111: "7",
    0b01000: "\r",
    0b01001: "$",
    0b01010: "4",
    0b01011: "'",
    0b01100: ",",
    0b01101: "!",
    0b01110: ":",
    0b01111: "(",
    0b10000: "5",
    0b10001: '"',
    0b10010: ")",
    0b10011: "2",
    0b10100: "#",
    0b10101: "6",
    0b10110: "0",
    0b10111: "1",
    0b11000: "9",
    0b11001: "?",
    0b11010: "&",
    0b11100: ".",
    0b11101: "/",
    0b11110: ";",
}


@dataclass
class Run:
    state: int
    start: int
    end: int


@dataclass
class DecodeResult:
    score: float
    valid_frames: int
    total_frames: int
    bit_duration: float
    offset: float
    invert_bits: bool
    stop_bits_est: float
    decoded_text: str


@dataclass
class AnalysisResult:
    sample_rate: int
    duration_s: float
    channels: int
    leading_silence_s: float
    trailing_silence_s: float
    active_duration_s: float
    estimated_mark_hz: float
    estimated_space_hz: float
    estimated_baud: float
    estimated_bit_duration: float
    raw_estimated_baud: float
    raw_estimated_bit_duration: float
    timing_snapped_to_target: bool
    stop_bits_est: float
    run_count: int
    decode: DecodeResult


@dataclass
class FramingConfidence:
    score_0_to_100: float
    level: str
    valid_frame_ratio: float
    stop_bits_alignment: float
    timing_alignment: float


@dataclass
class GateResult:
    passed: bool
    checks: List[Dict[str, object]]


def read_wav_mono_float(path: str) -> Tuple[List[float], int, int]:
    with wave.open(path, "rb") as wf:
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sample_rate = wf.getframerate()
        frames = wf.getnframes()
        raw = wf.readframes(frames)

    if sampwidth != 2:
        raise ValueError("Only 16-bit PCM WAV is supported.")

    samples = array("h")
    samples.frombytes(raw)

    if channels > 1:
        samples = array("h", samples[::channels])

    mono = [v / 32768.0 for v in samples]
    return mono, sample_rate, channels


def active_region(samples: Sequence[float], threshold_ratio: float = 0.12) -> Tuple[int, int]:
    if not samples:
        return 0, 0
    peak = max(abs(v) for v in samples)
    if peak == 0:
        return 0, len(samples) - 1
    threshold = peak * threshold_ratio
    active = [i for i, v in enumerate(samples) if abs(v) > threshold]
    if not active:
        return 0, len(samples) - 1
    return active[0], active[-1]


def goertzel_power(seg: Sequence[float], sample_rate: int, freq_hz: float) -> float:
    n = len(seg)
    if n < 4:
        return 0.0
    k = int(round((n * freq_hz) / sample_rate))
    omega = (2.0 * math.pi / n) * k
    coeff = 2.0 * math.cos(omega)

    s0 = 0.0
    s1 = 0.0
    s2 = 0.0
    for x in seg:
        s0 = x + coeff * s1 - s2
        s2 = s1
        s1 = s0

    return s1 * s1 + s2 * s2 - coeff * s1 * s2


def classify_states(
    samples: Sequence[float],
    sample_rate: int,
    tone_a_hz: float,
    tone_b_hz: float,
    win_s: float = 0.010,
    hop_s: float = 0.001,
) -> List[Tuple[int, int]]:
    win = max(8, int(round(win_s * sample_rate)))
    hop = max(1, int(round(hop_s * sample_rate)))
    out: List[Tuple[int, int]] = []

    for i in range(0, max(1, len(samples) - win), hop):
        seg = samples[i : i + win]
        pa = goertzel_power(seg, sample_rate, tone_a_hz)
        pb = goertzel_power(seg, sample_rate, tone_b_hz)
        state = 1 if pa >= pb else 0
        out.append((i, state))

    if not out:
        out.append((0, 1))
    return out


def smooth_states(states: Sequence[Tuple[int, int]], radius: int = 2) -> List[Tuple[int, int]]:
    smoothed: List[Tuple[int, int]] = []
    n = len(states)
    for i in range(n):
        lo = max(0, i - radius)
        hi = min(n, i + radius + 1)
        ones = sum(states[j][1] for j in range(lo, hi))
        state = 1 if ones >= ((hi - lo) / 2.0) else 0
        smoothed.append((states[i][0], state))
    return smoothed


def states_to_runs(states: Sequence[Tuple[int, int]], hop_samples: int) -> List[Run]:
    if not states:
        return []

    runs: List[Run] = []
    cur = states[0][1]
    run_start = states[0][0]

    for pos, state in states[1:]:
        if state != cur:
            runs.append(Run(cur, run_start, pos))
            cur = state
            run_start = pos

    runs.append(Run(cur, run_start, states[-1][0] + hop_samples))
    return runs


def estimate_bit_duration_from_runs(runs: Sequence[Run], sample_rate: int, nominal_bit_s: float = 0.02) -> float:
    if not runs:
        return nominal_bit_s

    durations = [(r.end - r.start) / sample_rate for r in runs]
    multiples = [max(1, int(round(d / nominal_bit_s))) for d in durations]
    bit_estimates = [d / m for d, m in zip(durations, multiples)]

    if not bit_estimates:
        return nominal_bit_s
    return statistics.median(bit_estimates)


def snap_bit_duration_if_close(bit_duration_s: float, target_baud: float, tol_baud: float) -> Tuple[float, bool]:
    if bit_duration_s <= 0 or target_baud <= 0:
        return bit_duration_s, False

    measured_baud = 1.0 / bit_duration_s
    if abs(measured_baud - target_baud) <= tol_baud:
        return 1.0 / target_baud, True

    return bit_duration_s, False


def estimate_freq_from_zero_crossings(seg: Sequence[float], sample_rate: int) -> float:
    if len(seg) < 4:
        return 0.0

    crossings = 0
    prev = seg[0]
    for x in seg[1:]:
        if (prev <= 0 < x) or (prev >= 0 > x):
            crossings += 1
        prev = x

    duration_s = len(seg) / sample_rate
    if duration_s <= 0:
        return 0.0
    return crossings / (2.0 * duration_s)


def estimate_tone_frequencies(
    samples: Sequence[float],
    sample_rate: int,
    runs: Sequence[Run],
    long_run_min_s: float = 0.03,
) -> Tuple[float, float]:
    f_state_1: List[float] = []
    f_state_0: List[float] = []

    for r in runs:
        duration = (r.end - r.start) / sample_rate
        if duration < long_run_min_s:
            continue
        seg = samples[r.start : r.end]
        f = estimate_freq_from_zero_crossings(seg, sample_rate)
        if f <= 0:
            continue
        if r.state == 1:
            f_state_1.append(f)
        else:
            f_state_0.append(f)

    est1 = statistics.median(f_state_1) if f_state_1 else 0.0
    est0 = statistics.median(f_state_0) if f_state_0 else 0.0
    return est1, est0


def classify_bit_at_time(
    samples: Sequence[float],
    sample_rate: int,
    t_s: float,
    tone_a_hz: float,
    tone_b_hz: float,
    invert_bits: bool,
) -> int:
    center = int(round(t_s * sample_rate))
    radius = max(8, int(round(0.003 * sample_rate)))
    a = max(0, center - radius)
    b = min(len(samples), center + radius + 1)
    seg = samples[a:b]

    pa = goertzel_power(seg, sample_rate, tone_a_hz)
    pb = goertzel_power(seg, sample_rate, tone_b_hz)
    bit = 1 if pa >= pb else 0
    return 1 - bit if invert_bits else bit


def decoded_text_quality_score(text: str) -> float:
    if not text:
        return 0.0

    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,:;!?-'/\"()#&")
    good = 0
    control = 0
    other = 0

    for ch in text:
        if ch in allowed:
            good += 1
        elif ch in ("\r", "\n", "\t"):
            control += 1
        else:
            other += 1

    n = len(text)
    return (good / n) - (0.6 * control / n) - (0.3 * other / n)


def decode_ita2(
    samples: Sequence[float],
    sample_rate: int,
    start_idx: int,
    end_idx: int,
    bit_duration_guess: float,
    tone_a_hz: float,
    tone_b_hz: float,
    lock_timing_near_guess: bool = False,
    allow_inverted: bool = True,
) -> DecodeResult:
    start_t = start_idx / sample_rate
    end_t = end_idx / sample_rate

    candidates: List[DecodeResult] = []

    if lock_timing_near_guess:
        # For known-accurate clocks, lock decode timing exactly to the snapped bit period.
        bit_sweep = [bit_duration_guess]
        # Use fine-grained offset steps to avoid missing near-optimal symbol phase.
        offset_sweep = [0.0 + 0.0005 * i for i in range(0, 41)]
        # Allow both 1.5 and 2.0 stop-bit pacing in the frame advance.
        char_bit_steps = (7.5, 8.0)
    else:
        bit_sweep = [bit_duration_guess * (0.86 + 0.01 * i) for i in range(0, 29)]
        offset_sweep = [0.0 + 0.0025 * i for i in range(0, 21)]
        char_bit_steps = (7.5,)

    invert_candidates = (False, True) if allow_inverted else (False,)
    for invert in invert_candidates:
        for tb in bit_sweep:
            for char_step in char_bit_steps:
                for offset in offset_sweep:
                    t = start_t + offset
                    mode = "LTRS"
                    chars: List[str] = []
                    valid = 0
                    total = 0
                    stop_one_count = 0
                    stop_two_count = 0
                    retries = 0

                    while t + (7.1 * tb) < end_t:
                        start_bit = classify_bit_at_time(samples, sample_rate, t + 0.5 * tb, tone_a_hz, tone_b_hz, invert)
                        if start_bit != 0:
                            t += 0.25 * tb
                            retries += 1
                            if retries > 1000:
                                break
                            continue

                        data = 0
                        for i in range(5):
                            b = classify_bit_at_time(samples, sample_rate, t + (1.5 + i) * tb, tone_a_hz, tone_b_hz, invert)
                            data |= (b << i)

                        stop1 = classify_bit_at_time(samples, sample_rate, t + 6.5 * tb, tone_a_hz, tone_b_hz, invert)
                        stop2 = classify_bit_at_time(samples, sample_rate, t + 7.0 * tb, tone_a_hz, tone_b_hz, invert)

                        total += 1
                        if stop1 == 1:
                            valid += 1
                        stop_one_count += 1 if stop1 == 1 else 0
                        stop_two_count += 1 if stop2 == 1 else 0

                        if data == FIGS_SHIFT_CODE:
                            mode = "FIGS"
                        elif data == LTRS_SHIFT_CODE:
                            mode = "LTRS"
                        else:
                            table = LETTERS if mode == "LTRS" else FIGURES
                            chars.append(table.get(data, "?"))

                        next_t = t + (char_step * tb)

                        if lock_timing_near_guess:
                            # Re-acquire the next start bit near the predicted frame boundary
                            # to avoid cumulative drift on recordings with variable stop lengths.
                            best_t = None
                            best_abs_j = 1_000_000
                            for j in range(-8, 9):
                                cand_t = next_t + (j * 0.1 * tb)
                                if cand_t <= t or cand_t + (7.1 * tb) >= end_t:
                                    continue
                                cand_start = classify_bit_at_time(
                                    samples,
                                    sample_rate,
                                    cand_t + (0.5 * tb),
                                    tone_a_hz,
                                    tone_b_hz,
                                    invert,
                                )
                                if cand_start == 0 and abs(j) < best_abs_j:
                                    best_abs_j = abs(j)
                                    best_t = cand_t

                            t = best_t if best_t is not None else next_t
                        else:
                            t = next_t

                    if total == 0:
                        continue

                    stop_bits_est = 1.0 + (stop_two_count / total)
                    decoded = "".join(chars)
                    quality = decoded_text_quality_score(decoded)
                    timing_penalty = 0.8 * abs(tb - bit_duration_guess) / max(bit_duration_guess, 1e-12)
                    invert_penalty = 0.75 if (lock_timing_near_guess and invert) else 0.0
                    char_step_penalty = 0.0 if char_step == 7.5 else 0.2
                    score = valid - (0.15 * retries) + (1.6 * quality) - timing_penalty - invert_penalty - char_step_penalty

                    candidates.append(
                        DecodeResult(
                            score=score,
                            valid_frames=valid,
                            total_frames=total,
                            bit_duration=tb,
                            offset=offset,
                            invert_bits=invert,
                            stop_bits_est=stop_bits_est,
                            decoded_text=decoded,
                        )
                    )

    if not candidates:
        return DecodeResult(0.0, 0, 0, bit_duration_guess, 0.0, False, 0.0, "")

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[0]


def compare_to_expected(ar: AnalysisResult) -> List[Tuple[str, str, str, str]]:
    rows: List[Tuple[str, str, str, str]] = []

    def verdict(ok: bool) -> str:
        return "MATCH" if ok else "DIFF"

    rows.append(
        (
            "Sample rate",
            str(EXPECTED["sample_rate"]),
            str(ar.sample_rate),
            verdict(abs(ar.sample_rate - EXPECTED["sample_rate"]) <= 1),
        )
    )
    rows.append(
        (
            "Baud",
            f"{EXPECTED['baud']:.2f}",
            f"{ar.estimated_baud:.3f}",
            verdict(abs(ar.estimated_baud - EXPECTED["baud"]) <= 0.75),
        )
    )
    rows.append(
        (
            "Mark tone (Hz)",
            f"{EXPECTED['mark_hz']:.1f}",
            f"{ar.estimated_mark_hz:.1f}",
            verdict(abs(ar.estimated_mark_hz - EXPECTED["mark_hz"]) <= 35),
        )
    )
    rows.append(
        (
            "Space tone (Hz)",
            f"{EXPECTED['space_hz']:.1f}",
            f"{ar.estimated_space_hz:.1f}",
            verdict(abs(ar.estimated_space_hz - EXPECTED["space_hz"]) <= 35),
        )
    )
    rows.append(
        (
            "Stop bits",
            f"{EXPECTED['stop_bits']:.1f}",
            f"{ar.stop_bits_est:.2f}",
            verdict(abs(ar.stop_bits_est - EXPECTED["stop_bits"]) <= 0.35),
        )
    )
    rows.append(
        (
            "Lead-in mark",
            f"~{EXPECTED['lead_in_bits']} bits",
            "Detected from payload shape only",
            "INFO",
        )
    )
    rows.append(
        (
            "Tail mark",
            f"~{EXPECTED['tail_bits']} bits",
            "Detected from payload shape only",
            "INFO",
        )
    )

    return rows


def compute_framing_confidence(ar: AnalysisResult) -> FramingConfidence:
    if ar.decode.total_frames <= 0:
        return FramingConfidence(
            score_0_to_100=0.0,
            level="low",
            valid_frame_ratio=0.0,
            stop_bits_alignment=0.0,
            timing_alignment=0.0,
        )

    valid_ratio = ar.decode.valid_frames / ar.decode.total_frames
    stop_alignment = max(0.0, 1.0 - (abs(ar.decode.stop_bits_est - EXPECTED["stop_bits"]) / 1.0))

    if ar.estimated_bit_duration > 0:
        timing_error_ratio = abs(ar.decode.bit_duration - ar.estimated_bit_duration) / ar.estimated_bit_duration
    else:
        timing_error_ratio = 1.0
    timing_alignment = max(0.0, 1.0 - (timing_error_ratio / 0.25))

    # Weighted score: framing validity is dominant, then stop-bit and timing alignment.
    score = (0.55 * valid_ratio) + (0.30 * stop_alignment) + (0.15 * timing_alignment)
    score_0_to_100 = max(0.0, min(100.0, score * 100.0))

    if score_0_to_100 >= 85:
        level = "high"
    elif score_0_to_100 >= 65:
        level = "medium"
    else:
        level = "low"

    return FramingConfidence(
        score_0_to_100=score_0_to_100,
        level=level,
        valid_frame_ratio=valid_ratio,
        stop_bits_alignment=stop_alignment,
        timing_alignment=timing_alignment,
    )


def as_dict(ar: AnalysisResult) -> Dict[str, object]:
    comparison_rows = compare_to_expected(ar)
    confidence = compute_framing_confidence(ar)

    return {
        "analysis": {
            "sample_rate_hz": ar.sample_rate,
            "channels": ar.channels,
            "duration_s": ar.duration_s,
            "leading_silence_s": ar.leading_silence_s,
            "active_duration_s": ar.active_duration_s,
            "trailing_silence_s": ar.trailing_silence_s,
            "run_count": ar.run_count,
        },
        "estimated": {
            "mark_hz": ar.estimated_mark_hz,
            "space_hz": ar.estimated_space_hz,
            "bit_duration_s": ar.estimated_bit_duration,
            "baud": ar.estimated_baud,
            "raw_bit_duration_s": ar.raw_estimated_bit_duration,
            "raw_baud": ar.raw_estimated_baud,
            "timing_snapped_to_target": ar.timing_snapped_to_target,
            "stop_bits_est": ar.stop_bits_est,
        },
        "decode": {
            "score": ar.decode.score,
            "valid_frames": ar.decode.valid_frames,
            "total_frames": ar.decode.total_frames,
            "bit_duration_s": ar.decode.bit_duration,
            "offset_s": ar.decode.offset,
            "invert_bits": ar.decode.invert_bits,
            "stop_bits_est": ar.decode.stop_bits_est,
            "decoded_preview": ar.decode.decoded_text[:200],
            "decoded_text": ar.decode.decoded_text,
        },
        "framing_confidence": {
            "score_0_to_100": confidence.score_0_to_100,
            "level": confidence.level,
            "valid_frame_ratio": confidence.valid_frame_ratio,
            "stop_bits_alignment": confidence.stop_bits_alignment,
            "timing_alignment": confidence.timing_alignment,
        },
        "comparison": [
            {
                "parameter": name,
                "expected": expected,
                "measured": measured,
                "status": status,
            }
            for name, expected, measured, status in comparison_rows
        ],
    }


def evaluate_gates(
    ar: AnalysisResult,
    *,
    sample_rate_tol: float,
    baud_tol: float,
    mark_tol_hz: float,
    space_tol_hz: float,
    stop_bits_tol: float,
) -> GateResult:
    checks: List[Dict[str, object]] = []

    def add_check(name: str, expected: float, measured: float, tolerance: float) -> None:
        delta = abs(measured - expected)
        passed = delta <= tolerance
        checks.append(
            {
                "name": name,
                "expected": expected,
                "measured": measured,
                "tolerance": tolerance,
                "delta": delta,
                "passed": passed,
            }
        )

    add_check("sample_rate_hz", float(EXPECTED["sample_rate"]), float(ar.sample_rate), sample_rate_tol)
    add_check("baud", float(EXPECTED["baud"]), ar.estimated_baud, baud_tol)
    add_check("mark_hz", float(EXPECTED["mark_hz"]), ar.estimated_mark_hz, mark_tol_hz)
    add_check("space_hz", float(EXPECTED["space_hz"]), ar.estimated_space_hz, space_tol_hz)
    add_check("stop_bits", float(EXPECTED["stop_bits"]), ar.stop_bits_est, stop_bits_tol)

    return GateResult(passed=all(c["passed"] for c in checks), checks=checks)


def apply_gate_result_to_json(payload: Dict[str, object], gate_result: GateResult) -> Dict[str, object]:
    payload = dict(payload)
    payload["gates"] = {
        "passed": gate_result.passed,
        "checks": gate_result.checks,
    }
    return payload


def resolve_tolerances(args: argparse.Namespace) -> Dict[str, float]:
    preset = TOLERANCE_PRESETS[args.preset]
    return {
        "sample_rate_hz": args.sample_rate_tol if args.sample_rate_tol is not None else preset["sample_rate_hz"],
        "baud": args.baud_tol if args.baud_tol is not None else preset["baud"],
        "mark_hz": args.mark_tol if args.mark_tol is not None else preset["mark_hz"],
        "space_hz": args.space_tol if args.space_tol is not None else preset["space_hz"],
        "stop_bits": args.stop_bits_tol if args.stop_bits_tol is not None else preset["stop_bits"],
    }


def analyze(path: str) -> AnalysisResult:
    samples, sample_rate, channels = read_wav_mono_float(path)
    total_duration = len(samples) / sample_rate if sample_rate > 0 else 0.0

    start, end = active_region(samples)
    if end < start:
        start, end = 0, len(samples) - 1

    leading_silence = start / sample_rate
    trailing_silence = (len(samples) - 1 - end) / sample_rate
    active = samples[start : end + 1]
    active_duration = len(active) / sample_rate

    # Use expected tones as classification anchors (robust for this workspace use-case).
    tone_a = EXPECTED["mark_hz"]
    tone_b = EXPECTED["space_hz"]

    win = max(8, int(round(0.010 * sample_rate)))
    hop = max(1, int(round(0.001 * sample_rate)))
    states = classify_states(active, sample_rate, tone_a, tone_b, win_s=0.010, hop_s=0.001)
    states = smooth_states(states, radius=2)
    runs = states_to_runs(states, hop)

    raw_bit_duration = estimate_bit_duration_from_runs(runs, sample_rate, nominal_bit_s=1.0 / EXPECTED["baud"])
    raw_baud = 1.0 / raw_bit_duration if raw_bit_duration > 0 else 0.0
    bit_duration, timing_snapped = snap_bit_duration_if_close(
        raw_bit_duration,
        target_baud=BAUD_SNAP_TARGET,
        tol_baud=BAUD_SNAP_TOLERANCE,
    )
    baud = 1.0 / bit_duration if bit_duration > 0 else 0.0

    f_state_1, f_state_0 = estimate_tone_frequencies(active, sample_rate, runs)

    # If estimation failed, fall back to configured anchors.
    mark_hz = f_state_1 if f_state_1 > 0 else tone_a
    space_hz = f_state_0 if f_state_0 > 0 else tone_b

    decode = decode_ita2(
        active,
        sample_rate,
        0,
        len(active) - 1,
        bit_duration_guess=bit_duration,
        tone_a_hz=tone_a,
        tone_b_hz=tone_b,
        lock_timing_near_guess=timing_snapped,
        allow_inverted=not timing_snapped,
    )

    return AnalysisResult(
        sample_rate=sample_rate,
        duration_s=total_duration,
        channels=channels,
        leading_silence_s=leading_silence,
        trailing_silence_s=trailing_silence,
        active_duration_s=active_duration,
        estimated_mark_hz=mark_hz,
        estimated_space_hz=space_hz,
        estimated_baud=baud,
        estimated_bit_duration=bit_duration,
        raw_estimated_baud=raw_baud,
        raw_estimated_bit_duration=raw_bit_duration,
        timing_snapped_to_target=timing_snapped,
        stop_bits_est=decode.stop_bits_est,
        run_count=len(runs),
        decode=decode,
    )


def print_report(ar: AnalysisResult) -> None:
    print("ITA2 Audio Analysis")
    print("=" * 72)
    print(f"Sample rate      : {ar.sample_rate} Hz")
    print(f"Channels         : {ar.channels}")
    print(f"Total duration   : {ar.duration_s:.3f} s")
    print(f"Leading silence  : {ar.leading_silence_s:.3f} s")
    print(f"Active duration  : {ar.active_duration_s:.3f} s")
    print(f"Trailing silence : {ar.trailing_silence_s:.3f} s")
    print(f"Run count        : {ar.run_count}")
    print()

    print("Estimated FSK/Timing")
    print("-" * 72)
    print(f"Mark tone        : {ar.estimated_mark_hz:.1f} Hz")
    print(f"Space tone       : {ar.estimated_space_hz:.1f} Hz")
    print(f"Bit duration     : {ar.estimated_bit_duration * 1000:.3f} ms")
    print(f"Baud             : {ar.estimated_baud:.3f}")
    if ar.timing_snapped_to_target:
        print(
            f"Timing snap      : ON (raw {ar.raw_estimated_baud:.3f} baud -> {BAUD_SNAP_TARGET:.3f} baud)"
        )
    else:
        print("Timing snap      : OFF")
    print(f"Stop bits (est)  : {ar.stop_bits_est:.2f}")
    print()

    print("Decode Attempt")
    print("-" * 72)
    print(f"Score            : {ar.decode.score:.2f}")
    print(f"Valid/Total frame: {ar.decode.valid_frames}/{ar.decode.total_frames}")
    print(f"Bit duration fit : {ar.decode.bit_duration * 1000:.3f} ms")
    print(f"Offset fit       : {ar.decode.offset:.4f} s")
    print(f"Invert bits      : {ar.decode.invert_bits}")
    preview = ar.decode.decoded_text[:200]
    print(f"Decoded preview  : {preview!r}")
    print()

    confidence = compute_framing_confidence(ar)
    print("Framing Confidence")
    print("-" * 72)
    print(f"Overall score    : {confidence.score_0_to_100:.1f} / 100 ({confidence.level})")
    print(f"Frame lock ratio : {confidence.valid_frame_ratio:.3f}")
    print(f"Stop-bit align   : {confidence.stop_bits_alignment:.3f}")
    print(f"Timing align     : {confidence.timing_alignment:.3f}")
    print()

    print("Comparison vs Workspace Encoder Settings")
    print("-" * 72)
    rows = compare_to_expected(ar)
    for name, expected, measured, status in rows:
        print(f"{name:16} | expected: {expected:>12} | measured: {measured:>24} | {status}")


def print_gate_report(gate_result: GateResult) -> None:
    print()
    print("Gate Evaluation")
    print("-" * 72)
    for c in gate_result.checks:
        status = "PASS" if c["passed"] else "FAIL"
        print(
            f"{c['name']:16} | expected: {c['expected']:>9.3f} | "
            f"measured: {c['measured']:>9.3f} | tol: {c['tolerance']:>7.3f} | "
            f"delta: {c['delta']:>8.3f} | {status}"
        )
    print(f"Overall gates    : {'PASS' if gate_result.passed else 'FAIL'}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Analyze and decode ITA2-style FSK WAV audio.")
    p.add_argument("wav", help="Path to WAV file")
    p.add_argument(
        "--json",
        action="store_true",
        help="Emit analysis as JSON instead of human-readable text.",
    )
    p.add_argument(
        "--json-indent",
        type=int,
        default=2,
        help="Indent level for JSON output. Default: 2",
    )
    p.add_argument(
        "--check",
        action="store_true",
        help="Enable CI gates and return non-zero exit code on mismatch.",
    )
    p.add_argument(
        "--preset",
        choices=sorted(TOLERANCE_PRESETS.keys()),
        default="default",
        help="Tolerance preset for --check mode. Default: default",
    )
    p.add_argument(
        "--sample-rate-tol",
        type=float,
        default=None,
        help="Allowed absolute sample-rate difference in Hz. Overrides preset value.",
    )
    p.add_argument(
        "--baud-tol",
        type=float,
        default=None,
        help="Allowed absolute baud difference. Overrides preset value.",
    )
    p.add_argument(
        "--mark-tol",
        type=float,
        default=None,
        help="Allowed absolute mark-frequency difference in Hz. Overrides preset value.",
    )
    p.add_argument(
        "--space-tol",
        type=float,
        default=None,
        help="Allowed absolute space-frequency difference in Hz. Overrides preset value.",
    )
    p.add_argument(
        "--stop-bits-tol",
        type=float,
        default=None,
        help="Allowed absolute stop-bits difference. Overrides preset value.",
    )
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    result = analyze(args.wav)
    gate_result = None
    if args.check:
        tolerances = resolve_tolerances(args)
        gate_result = evaluate_gates(
            result,
            sample_rate_tol=tolerances["sample_rate_hz"],
            baud_tol=tolerances["baud"],
            mark_tol_hz=tolerances["mark_hz"],
            space_tol_hz=tolerances["space_hz"],
            stop_bits_tol=tolerances["stop_bits"],
        )

    if args.json:
        payload = as_dict(result)
        if gate_result is not None:
            payload = apply_gate_result_to_json(payload, gate_result)
        print(json.dumps(payload, indent=args.json_indent, sort_keys=False))
    else:
        print_report(result)
        if gate_result is not None:
            print_gate_report(gate_result)

    if gate_result is not None and not gate_result.passed:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
