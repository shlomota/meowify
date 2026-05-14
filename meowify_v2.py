"""
Meowify v2: Replace each vocal note with a real cat meow,
pitch-shifted to match the original melody.
"""
import os
import sys
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import resample


def load_meow_sample(meow_path: str, sr: int = 44100) -> np.ndarray:
    """Load and isolate the best single meow from the reference."""
    y, _ = librosa.load(meow_path, sr=sr, mono=True)

    # Detect onsets to find individual meows
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units='samples')

    # Find the loudest/longest meow segment
    best_meow = None
    best_energy = 0

    for i in range(len(onsets)):
        start = onsets[i]
        # End is next onset or 0.5s later, whichever is shorter
        end = onsets[i + 1] if i + 1 < len(onsets) else min(start + int(0.5 * sr), len(y))
        end = min(end, start + int(0.6 * sr))  # cap at 0.6s

        segment = y[start:end]
        if len(segment) < int(0.05 * sr):  # skip tiny blips
            continue

        energy = np.sum(segment ** 2)
        if energy > best_energy:
            best_energy = energy
            best_meow = segment

    if best_meow is None:
        # fallback: just take first 0.4s
        best_meow = y[:int(0.4 * sr)]

    # Fade in/out to avoid clicks
    fade = int(0.005 * sr)
    best_meow[:fade] *= np.linspace(0, 1, fade)
    best_meow[-fade:] *= np.linspace(1, 0, fade)

    return best_meow


def get_meow_base_f0(meow: np.ndarray, sr: int = 44100) -> float:
    """Estimate the fundamental frequency of the meow sample."""
    f0, voiced, _ = librosa.pyin(meow, fmin=100, fmax=2000, sr=sr)
    voiced_f0 = f0[voiced]
    if len(voiced_f0) > 0:
        return float(np.median(voiced_f0))
    return 400.0  # reasonable cat meow default


def detect_notes(vocals_path: str, sr: int = 44100, min_note_duration: float = 0.03) -> list[dict]:
    """
    Detect note segments from vocals: each note has start_time, end_time, f0.
    Groups consecutive voiced frames into notes.
    """
    y, _ = librosa.load(vocals_path, sr=sr, mono=True)

    # Smaller frame_length = finer time resolution (1024 frames ~ 23ms at 44100)
    hop = 512
    f0, voiced, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'),
                                  fmax=librosa.note_to_hz('C6'), sr=sr,
                                  frame_length=1024, hop_length=hop)
    times = librosa.times_like(f0, sr=sr, hop_length=hop)

    # Group consecutive voiced frames into note segments
    notes = []
    in_note = False
    note_start = 0
    note_f0s = []

    for i, (t, freq, v) in enumerate(zip(times, f0, voiced)):
        if v and not in_note:
            in_note = True
            note_start = i
            note_f0s = [freq]
        elif v and in_note:
            note_f0s.append(freq)
        elif not v and in_note:
            in_note = False
            # Only keep notes longer than ~50ms
            duration = times[i] - times[note_start]
            if duration > min_note_duration and len(note_f0s) > 0:
                notes.append({
                    'start': times[note_start],
                    'end': times[i],
                    'duration': duration,
                    'f0': float(np.median(note_f0s)),
                })
            note_f0s = []

    # Handle note at end
    if in_note and len(note_f0s) > 0:
        duration = times[-1] - times[note_start]
        if duration > min_note_duration:
            notes.append({
                'start': times[note_start],
                'end': times[-1],
                'duration': duration,
                'f0': float(np.median(note_f0s)),
            })

    return notes


def place_meows(notes: list[dict], meow: np.ndarray, meow_f0: float,
                total_duration: float, sr: int = 44100) -> np.ndarray:
    """
    For each note, pitch-shift the meow to match the note's F0,
    and time-stretch to match the note's duration.
    """
    output = np.zeros(int(total_duration * sr), dtype=np.float32)
    meow_duration = len(meow) / sr

    for note in notes:
        # Pitch shift: how many semitones from meow base to target note
        semitones = 12 * np.log2(note['f0'] / meow_f0)

        # Time stretch: adjust meow length to match note duration
        stretch_ratio = meow_duration / note['duration']

        # Stretch first, then pitch shift
        if stretch_ratio > 0.1 and stretch_ratio < 10:
            stretched = librosa.effects.time_stretch(meow, rate=stretch_ratio)
        else:
            stretched = meow.copy()

        # Pitch shift
        shifted = librosa.effects.pitch_shift(stretched, sr=sr, n_steps=semitones)

        # Trim or pad to exact note duration
        target_samples = int(note['duration'] * sr)
        if len(shifted) > target_samples:
            shifted = shifted[:target_samples]
            # Fade out at end
            fade = min(int(0.01 * sr), len(shifted) // 4)
            shifted[-fade:] *= np.linspace(1, 0, fade)

        # Place in output
        start_sample = int(note['start'] * sr)
        end_sample = min(start_sample + len(shifted), len(output))
        chunk_len = end_sample - start_sample
        output[start_sample:end_sample] += shifted[:chunk_len]

    return output


def meowify_v2(
    vocals_path: str,
    instrumental_path: str,
    meow_ref_path: str,
    output_path: str = "output/meowified_v2.wav",
    sr: int = 44100,
    inst_gain: float = 1.0,
    vocal_gain: float = 0.5,
    meow_gain: float = 0.7,
    inst_pitch_semitones: float = 0.0,
    vocal_pitch_semitones: float = 12.0,
    speed_factor: float = 1.2,
) -> str:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    print("[1/6] Loading meow sample...")
    meow = load_meow_sample(meow_ref_path, sr=sr)
    meow_f0 = get_meow_base_f0(meow, sr=sr)
    print(f"       Meow: {len(meow)/sr:.2f}s, base F0={meow_f0:.0f}Hz")

    print("[2/6] Detecting notes in vocals...")
    notes = detect_notes(vocals_path, sr=sr)
    print(f"       Found {len(notes)} notes")

    print("[3/6] Placing meows on melody...")
    y_voc, _ = librosa.load(vocals_path, sr=sr, mono=True)
    total_duration = len(y_voc) / sr
    meow_track = place_meows(notes, meow, meow_f0, total_duration, sr=sr)

    # Normalize meow track
    peak = np.max(np.abs(meow_track))
    if peak > 0:
        meow_track = meow_track / peak * 0.9

    print(f"[4/6] Pitching: instrumental +{inst_pitch_semitones:.0f}st, vocals/meow +{vocal_pitch_semitones:.0f}st...")
    meow_track = librosa.effects.pitch_shift(meow_track, sr=sr, n_steps=vocal_pitch_semitones)
    inst, _ = librosa.load(instrumental_path, sr=sr, mono=True)
    inst = librosa.effects.pitch_shift(inst, sr=sr, n_steps=inst_pitch_semitones)

    print(f"[5/6] Speeding up everything by {speed_factor}x...")
    meow_track = librosa.effects.time_stretch(meow_track, rate=speed_factor)
    inst = librosa.effects.time_stretch(inst, rate=speed_factor)

    print("[6/6] Mixing and saving...")
    # Pitch and speed the original vocals too
    voc, _ = librosa.load(vocals_path, sr=sr, mono=True)
    voc = librosa.effects.pitch_shift(voc, sr=sr, n_steps=vocal_pitch_semitones)
    voc = librosa.effects.time_stretch(voc, rate=speed_factor)

    # Match lengths
    n = min(len(meow_track), len(inst), len(voc))
    mixed = inst_gain * inst[:n] + vocal_gain * voc[:n] + meow_gain * meow_track[:n]

    # Normalize
    peak = np.max(np.abs(mixed))
    if peak > 1.0:
        mixed /= peak

    sf.write(output_path, mixed, sr)
    print(f"       Done! -> {os.path.abspath(output_path)}")
    return os.path.abspath(output_path)


if __name__ == "__main__":
    vocals = "/Users/stannor/PycharmProjects/meowify/work/htdemucs/ROSE__Bruno_Mars_-_APT_Official_Music_Video/vocals_30s.wav"
    instrumental = "/Users/stannor/PycharmProjects/meowify/work/htdemucs/ROSE__Bruno_Mars_-_APT_Official_Music_Video/no_vocals_30s.wav"
    meow_ref = "/Users/stannor/PycharmProjects/meowify/cat_samples/separated/htdemucs/cat_meow_ref_trim/vocals_10s.wav"

    meowify_v2(vocals, instrumental, meow_ref,
               output_path="output/apt_meowified_v2.wav",
               inst_pitch_semitones=2.0,
               vocal_pitch_semitones=14.0)
