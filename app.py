import contextlib
import io
import json as _json
import os
import sys
import subprocess
import threading
import time
import uuid

import boto3
import librosa
import numpy as np
import requests
import soundfile as sf
import streamlit as st
from pathlib import Path
from scipy import signal as sig

from downloader import download_youtube_as_mp3, get_video_info
from meowify_v2 import meowify_v2

# ── Constants ─────────────────────────────────────────────────────────────────
WORK_DIR = "work"
OUTPUT_DIR = "output"
CAT_REF = "cat_samples/separated/htdemucs/cat_meow_ref_trim/vocals_10s.wav"
S3_BUCKET_DEFAULT = "sagemaker-us-east-1-478706476061"
S3_REGION = "us-east-1"
KIE_BASE = "https://api.kie.ai"
SITE_URL = "https://meowify.click"
SES_SENDER = "noreply@meowify.click"
NOTIFY_EMAIL = "stannor@gmail.com"

# ── Env ───────────────────────────────────────────────────────────────────────
def load_env() -> dict:
    env = {}
    p = Path(".env")
    if p.exists():
        for line in p.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env


# ── Chorus detection ─────────────────────────────────────────────────────────
def detect_chorus(audio_path: str, default_start: float = 45.0, duration: float = 30.0) -> tuple[float, str]:
    y, sr = librosa.load(audio_path, sr=22050, mono=True, duration=180.0)
    total = len(y) / sr

    if total < default_start + duration:
        return 0.0, "track too short — using 0s"

    rms = librosa.feature.rms(y=y, frame_length=sr * 2, hop_length=sr)[0]
    w = int(duration)
    def_idx = int(default_start)
    def_e = float(np.mean(rms[def_idx : def_idx + w])) if def_idx + w < len(rms) else 0.0

    best_idx, best_e = def_idx, def_e
    for s in range(30, min(71, int(total - duration))):
        if s + w >= len(rms):
            break
        e = float(np.mean(rms[s : s + w]))
        if e > best_e:
            best_e, best_idx = e, s

    if best_idx != def_idx and best_e > 1.15 * def_e:
        adjusted = max(0.0, float(best_idx) - 1.0)
        return adjusted, f"auto-detected at {best_idx}s, starting 1s early at {adjusted:.0f}s (energy ↑{100*(best_e/def_e-1):.0f}% vs default)"
    return default_start, f"default {default_start:.0f}s (no better candidate found)"


# ── Audio helpers ─────────────────────────────────────────────────────────────
def extract_clip(src: str, start: float, duration: float, dst: str) -> str:
    y, sr = librosa.load(src, sr=44100, mono=True)
    s, e = int(start * sr), int((start + duration) * sr)
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    sf.write(dst, y[s:e], sr)
    return dst


def _biquad(y: np.ndarray, b, a) -> np.ndarray:
    sos = sig.tf2sos(b, a)
    return sig.sosfilt(sos, y.T).T if y.ndim > 1 else sig.sosfilt(sos, y)

def _peaking(sr, fc, gain_db, Q=1.0):
    A = 10 ** (gain_db / 40); w0 = 2 * np.pi * fc / sr; alpha = np.sin(w0) / (2 * Q)
    return [1+alpha*A, -2*np.cos(w0), 1-alpha*A], [1+alpha/A, -2*np.cos(w0), 1-alpha/A]

def _high_shelf(sr, fc, gain_db, Q=0.707):
    A = 10**(gain_db/40); w0 = 2*np.pi*fc/sr; cw = np.cos(w0)
    al = np.sin(w0)/2 * np.sqrt((A+1/A)*(1/Q-1)+2)
    b = [A*((A+1)+(A-1)*cw+2*np.sqrt(A)*al), -2*A*((A-1)+(A+1)*cw), A*((A+1)+(A-1)*cw-2*np.sqrt(A)*al)]
    a = [(A+1)-(A-1)*cw+2*np.sqrt(A)*al, 2*((A-1)-(A+1)*cw), (A+1)-(A-1)*cw-2*np.sqrt(A)*al]
    return b, a

def _low_shelf(sr, fc, gain_db, Q=0.707):
    A = 10**(gain_db/40); w0 = 2*np.pi*fc/sr; cw = np.cos(w0)
    al = np.sin(w0)/2 * np.sqrt((A+1/A)*(1/Q-1)+2)
    b = [A*((A+1)-(A-1)*cw+2*np.sqrt(A)*al), 2*A*((A-1)-(A+1)*cw), A*((A+1)-(A-1)*cw-2*np.sqrt(A)*al)]
    a = [(A+1)+(A-1)*cw+2*np.sqrt(A)*al, -2*((A-1)+(A+1)*cw), (A+1)+(A-1)*cw-2*np.sqrt(A)*al]
    return b, a

def apply_masking(src: str, dst: str, heavy: bool = False) -> str:
    y, sr = sf.read(src)
    y = _biquad(y, *_peaking(sr, 350, -2.5, Q=1.2))
    y = _biquad(y, *_peaking(sr, 3000, 2.0))
    y = _biquad(y, *_high_shelf(sr, 10000, 2.0))
    y = _biquad(y, *_low_shelf(sr, 80, 1.5))
    if heavy:
        y = _biquad(y, *_peaking(sr, 700, -4.0, Q=0.8))
        y = _biquad(y, *_peaking(sr, 5000, 3.5, Q=1.2))
    drive = 1.8 if heavy else 1.4
    y = 0.80 * y + 0.20 * (np.tanh(y * drive) / np.tanh(drive))
    noise_db = -20 if heavy else -35
    rms = np.sqrt(np.mean(y**2))
    y = y + np.random.normal(0, rms * 10**(noise_db/20), y.shape).astype(np.float32)
    peak = np.max(np.abs(y))
    if peak > 0.98:
        y = y / peak * 0.97
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    sf.write(dst, y.astype(np.float32), sr)
    return dst


# ── S3 ────────────────────────────────────────────────────────────────────────
def s3_upload_presign(local: str, bucket: str, key: str, aws_id: str, aws_sec: str,
                      region: str = "us-east-1", expiry: int = 7200) -> str:
    client = boto3.client("s3", aws_access_key_id=aws_id, aws_secret_access_key=aws_sec, region_name=region)
    client.upload_file(local, bucket, key)
    return client.generate_presigned_url("get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=expiry)


# ── kie.ai / Suno ─────────────────────────────────────────────────────────────
def suno_submit(upload_url: str, kie_key: str, lyrics: str,
                style: str, title: str, model: str,
                style_weight: float | None = None,
                audio_weight: float | None = None,
                weirdness: float | None = None,
                vocal_gender: str | None = None) -> dict:
    payload = {
        "uploadUrl": upload_url,
        "prompt": lyrics,
        "style": style,
        "title": title,
        "customMode": True,
        "instrumental": False,
        "model": model,
        "callBackUrl": "https://api.example.com/callback",
    }
    if style_weight is not None:
        payload["styleWeight"] = round(style_weight, 2)
    if audio_weight is not None:
        payload["audioWeight"] = round(audio_weight, 2)
    if weirdness is not None:
        payload["weirdnessConstraint"] = round(weirdness, 2)
    if vocal_gender:
        payload["vocalGender"] = vocal_gender

    r = requests.post(
        f"{KIE_BASE}/api/v1/generate/upload-cover",
        headers={"Authorization": f"Bearer {kie_key}"},
        json=payload,
        timeout=30,
    )
    data = r.json()
    if data["code"] != 200:
        raise RuntimeError(f"kie.ai error {data['code']}: {data['msg']}")
    return {"taskId": data["data"]["taskId"], "payload": payload}


def dl_file(url: str, dst: str) -> str:
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    r = requests.get(url, timeout=120, stream=True)
    r.raise_for_status()
    with open(dst, "wb") as f:
        for chunk in r.iter_content(8192):
            f.write(chunk)
    return dst


# ── Email ─────────────────────────────────────────────────────────────────────
def send_result_email(title: str, tracks: list, job_url: str, aws_id: str, aws_sec: str):
    client = boto3.client("ses", region_name="us-east-1",
                          aws_access_key_id=aws_id, aws_secret_access_key=aws_sec)
    track_lines = "\n".join(f"  - {t['title']}" for t in tracks) if tracks else "  (none)"
    body = (
        f'Your meow cover of "{title}" is ready!\n\n'
        f"{len(tracks)} track(s) generated:\n{track_lines}\n\n"
        f"Listen and download here:\n{job_url}\n\n"
        "-- Meowify"
    )
    client.send_email(
        Source=SES_SENDER,
        Destination={"ToAddresses": [NOTIFY_EMAIL]},
        Message={
            "Subject": {"Data": f"Meowify: {title[:60]} is ready!"},
            "Body": {"Text": {"Data": body}},
        },
    )


# ── Background pipeline ───────────────────────────────────────────────────────
def run_pipeline(job_id: str, url: str, params: dict):
    job = JOBS[job_id]

    def log(msg: str):
        job["logs"].append(f"[{time.strftime('%H:%M:%S')}] {msg}")

    def step(label: str):
        job["step"] = label
        log(f"--- {label} ---")

    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(WORK_DIR, exist_ok=True)

        # 1. Video info
        step("Fetching video info...")
        info = get_video_info(url)
        job["video_info"] = info
        log(f"Title: {info['title']}")
        log(f"Duration: {info['duration']}s")

        # 2. Download
        step("Downloading audio...")
        mp3_path = download_youtube_as_mp3(url)
        size_mb = os.path.getsize(mp3_path) / 1e6
        log(f"Downloaded: {mp3_path} ({size_mb:.1f} MB)")
        job["files"]["mp3"] = mp3_path

        # 3. Chorus / clip selection
        base = os.path.splitext(os.path.basename(mp3_path))[0]
        full_song  = params["full_song"]
        chorus_dur = params["chorus_dur"]

        step("Selecting section...")
        if full_song:
            chorus_s   = 0.0
            clip_input = mp3_path
            clip_base  = base
            log("Full song mode — using complete track")
        else:
            manual_start = params["manual_start"]
            if manual_start > 0:
                chorus_s = float(manual_start)
                reason   = f"manual override at {manual_start:.0f}s"
            else:
                chorus_s, reason = detect_chorus(mp3_path,
                                                 default_start=params["chorus_start"],
                                                 duration=chorus_dur)
            log(f"Chorus: {reason}")
            clip_base  = f"{base}_clip_{int(chorus_s)}s"
            clip_input = os.path.join(OUTPUT_DIR, f"{clip_base}.wav")
            extract_clip(mp3_path, chorus_s, chorus_dur, clip_input)
            log(f"Clip saved: {clip_input}")
        job["files"]["chorus_start"] = chorus_s

        # 4. Demucs on clip only
        chorus_voc  = os.path.join(WORK_DIR, "htdemucs", clip_base, "vocals.wav")
        chorus_inst = os.path.join(WORK_DIR, "htdemucs", clip_base, "no_vocals.wav")

        step("Separating vocals (Demucs)...")
        if os.path.exists(chorus_voc):
            log("Using cached Demucs output")
        else:
            log("Running Demucs htdemucs --two-stems vocals...")
            cmd = [sys.executable, "-m", "demucs", "--two-stems", "vocals",
                   "-n", "htdemucs", "--out", WORK_DIR, clip_input]
            result = subprocess.run(cmd, capture_output=True, text=True)
            for line in (result.stdout + result.stderr).splitlines():
                if line.strip():
                    log(line)
            if result.returncode != 0:
                raise RuntimeError(result.stderr[-800:])
        job["files"]["chorus_voc"]  = chorus_voc
        job["files"]["chorus_inst"] = chorus_inst

        # 5. Local meowify
        step("Local meowify (note replacement)...")
        meow_local = os.path.join(OUTPUT_DIR, f"{base}_meowified.wav")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            meowify_v2(
                vocals_path=chorus_voc,
                instrumental_path=chorus_inst,
                meow_ref_path=CAT_REF,
                output_path=meow_local,
                inst_pitch_semitones=float(params["inst_pitch"]),
                vocal_pitch_semitones=float(params["vocal_pitch"]),
                speed_factor=params["speed"],
                inst_gain=params["inst_gain"],
                vocal_gain=params["vocal_gain"],
                meow_gain=params["meow_gain"],
            )
        for line in buf.getvalue().splitlines():
            log(line)
        job["files"]["meow_local"] = meow_local

        # 6-8. Masking → S3 → Suno
        kie_key = params["kie_key"]
        aws_id  = params["aws_id"]
        aws_sec = params["aws_sec"]
        s3_bucket = params["s3_bucket"]

        if kie_key and aws_id and aws_sec:
            step("Masking + uploading to S3...")
            masked = os.path.join(OUTPUT_DIR, f"{base}_meowified_masked.wav")
            apply_masking(meow_local, masked)
            job["files"]["masked"] = masked
            log(f"EQ + noise masking → {masked}")
            s3_key = f"meowify/{base}_meowified_masked.wav"
            presigned_url = s3_upload_presign(masked, s3_bucket, s3_key, aws_id, aws_sec)
            log(f"Uploaded s3://{s3_bucket}/{s3_key}")

            _suffix = " (Meow Cover)"
            cover_title = info["title"][: 95 - len(_suffix)] + _suffix
            terminal = {"SUCCESS", "FAILED", "ERROR", "FIRST_SUCCESS", "GENERATE_AUDIO_FAILED"}

            def _meow(n): return "\n".join(["meow"] * n)
            if full_song:
                suno_lyrics = (
                    f"[Intro]\n{_meow(10)}\n\n"
                    f"[Verse]\n{_meow(30)}\n\n"
                    f"[Chorus]\n{_meow(30)}\n\n"
                    f"[Verse]\n{_meow(30)}\n\n"
                    f"[Chorus]\n{_meow(30)}"
                )
            else:
                suno_lyrics = "[Chorus]\n" + _meow(params["meow_count"])

            def submit_and_poll_bg(upload_url, attempt_label):
                sub = suno_submit(
                    upload_url, kie_key, suno_lyrics,
                    params["suno_style"], cover_title, params["suno_model"],
                    style_weight=params["style_weight"] if params["style_weight"] > 0 else None,
                    audio_weight=params["audio_weight"] if params["audio_weight"] > 0 else None,
                    weirdness=params["weirdness"]       if params["weirdness"]    > 0 else None,
                    vocal_gender=params["vocal_gender"] if params["vocal_gender"] != "(none)" else None,
                )
                task_id = sub["taskId"]
                log(f"Suno task submitted ({attempt_label}): {task_id}")
                t0 = time.time()
                while True:
                    r = requests.get(
                        f"{KIE_BASE}/api/v1/generate/record-info",
                        params={"taskId": task_id},
                        headers={"Authorization": f"Bearer {kie_key}"},
                        timeout=30,
                    )
                    data = r.json()["data"]
                    s = data["status"]
                    elapsed = time.time() - t0
                    job["step"] = f"Suno cover — {s} ({elapsed:.0f}s)..."
                    log(f"Suno poll ({attempt_label}): {s} ({elapsed:.0f}s)")
                    if s in terminal:
                        return sub["payload"], data, s
                    if elapsed > 600:
                        raise TimeoutError("Suno timed out after 10 min")
                    time.sleep(10)

            step("Generating Suno cover (5-10 min)...")
            suno_payload, suno_data, suno_status = submit_and_poll_bg(presigned_url, "attempt 1")
            job["files"]["suno_payload"] = suno_payload

            # Auto-retry with heavy masking on 413 fingerprint detection
            if suno_status not in {"SUCCESS", "FIRST_SUCCESS"} and suno_data.get("errorCode") == 413:
                log("413 fingerprint detected — retrying with heavy masking + pitch shift")
                meow_retry = os.path.join(OUTPUT_DIR, f"{base}_meowified_retry.wav")
                meowify_v2(
                    vocals_path=chorus_voc,
                    instrumental_path=chorus_inst,
                    meow_ref_path=CAT_REF,
                    output_path=meow_retry,
                    inst_pitch_semitones=float(params["inst_pitch"]) + 2.0,
                    vocal_pitch_semitones=float(params["vocal_pitch"]) + 2.0,
                    speed_factor=params["speed"],
                    inst_gain=params["inst_gain"],
                    vocal_gain=0.35,
                    meow_gain=params["meow_gain"],
                )
                job["files"]["meow_retry"] = meow_retry
                heavy_masked = os.path.join(OUTPUT_DIR, f"{base}_meowified_heavy.wav")
                apply_masking(meow_retry, heavy_masked, heavy=True)
                job["files"]["heavy_masked"] = heavy_masked
                heavy_s3_key = f"meowify/{base}_meowified_heavy.wav"
                heavy_url = s3_upload_presign(heavy_masked, s3_bucket, heavy_s3_key, aws_id, aws_sec)
                log(f"Uploaded heavy-masked retry: {heavy_s3_key}")
                suno_payload, suno_data, suno_status = submit_and_poll_bg(heavy_url, "attempt 2 (heavy mask)")
                job["files"]["suno_payload"] = suno_payload

            if suno_status not in {"SUCCESS", "FIRST_SUCCESS"}:
                error_code = suno_data.get("errorCode")
                error_msg  = suno_data.get("errorMessage")
                tracks_raw = (suno_data.get("response") or {}).get("sunoData") or []
                lines = [f"Status: {suno_status}"]
                if error_code:
                    lines.append(f"Error code: {error_code}")
                if error_msg:
                    lines.append(f"Error message: {error_msg}")
                for t in tracks_raw:
                    if t and t.get("errorMessage"):
                        lines.append(f"Track error: {t['errorMessage']}")
                lines.append(f"Full response:\n```json\n{_json.dumps(suno_data, indent=2)}\n```")
                job["suno_error"] = "\n\n".join(lines)
                log(f"Suno failed: {suno_status}")
            else:
                tracks = (suno_data.get("response") or {}).get("sunoData") or []
                log(f"Suno complete: {len(tracks)} tracks")
                for i, track in enumerate(tracks, 1):
                    audio_url = track.get("audioUrl") or track.get("streamAudioUrl", "")
                    if not audio_url:
                        continue
                    out = os.path.join(OUTPUT_DIR, f"{base}_suno_{i}.mp3")
                    dl_file(audio_url, out)
                    job["suno_tracks"].append({"path": out, "title": track.get("title", f"Track {i}")})
                    log(f"Track {i}: {out}")

        job["status"] = "done"
        job["step"]   = "Complete"
        log("Pipeline complete")

        # Send notification email
        if aws_id and aws_sec:
            try:
                job_url = f"{SITE_URL}/?job={job_id}"
                send_result_email(info["title"], job["suno_tracks"], job_url, aws_id, aws_sec)
                log(f"Notification email sent to {NOTIFY_EMAIL}")
            except Exception as e:
                log(f"Email failed (non-fatal): {e}")

    except Exception as e:
        job["status"] = "error"
        job["error"]  = str(e)
        job["step"]   = f"Error: {e}"
        job["logs"].append(f"[{time.strftime('%H:%M:%S')}] FATAL: {e}")


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Meowify", page_icon="🐱", layout="wide")

# @st.cache_resource gives the same dict on every script rerun and every session.
# A plain module-level `JOBS = {}` would reset to empty on each Streamlit rerun.
@st.cache_resource
def _jobs_store() -> dict:
    return {}

JOBS = _jobs_store()

env = load_env()

# ── Auth ──────────────────────────────────────────────────────────────────────
_NO_LOGIN = "--no-login" in sys.argv
_BANNER   = "https://cdn-images-1.medium.com/v2/resize:fit:800/1*0yOLabnolhlZDzShFH2vdg.png"

if not _NO_LOGIN and not st.session_state.get("logged_in"):
    st.image(_BANNER, width=600)
    st.title("🐱 Meowify")
    with st.form("login"):
        _user = st.text_input("Username")
        _pw   = st.text_input("Password", type="password")
        if st.form_submit_button("Login"):
            if _user == "stannor" and _pw == "Password1!":
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Invalid username or password")
    st.stop()

st.image(_BANNER, width=600)
st.title("🐱 Meowify")
st.caption("Create a meow cover of your favorite song")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")

    st.subheader("Audio processing")
    inst_pitch  = st.slider("Instrumental pitch (semitones)", 0, 24, 0)
    vocal_pitch = st.slider("Vocal / meow pitch (semitones)", 0, 36, 12)
    speed       = st.slider("Speed factor", 1.0, 2.0, 1.2, 0.1)
    inst_gain   = st.slider("Instrumental volume", 0.0, 1.5, 1.0, 0.05)
    vocal_gain  = st.slider("Original vocals volume", 0.0, 1.0, 0.5, 0.05)
    meow_gain   = st.slider("Meow volume", 0.0, 1.5, 0.7, 0.05)

    st.subheader("Chorus")
    full_song    = st.checkbox("Full song (skip chorus extraction)", value=False)
    chorus_start = st.number_input("Auto-detect fallback start (s)", value=45.0, step=5.0, disabled=full_song)
    chorus_dur   = st.number_input("Duration (s)", value=45.0, step=5.0, min_value=15.0, max_value=60.0, disabled=full_song)

    st.subheader("Suno (kie.ai)")
    kie_key      = st.text_input("kie.ai API Key", value=env.get("KIE_API_KEY", ""), type="password")
    suno_model   = st.selectbox("Model", ["V5_5", "V5", "V4_5PLUS", "V4_5", "V4"])
    meow_count   = st.slider("Meows in lyrics", 50, 300, 120)
    suno_style   = st.text_input("Style", value="pop")
    vocal_gender = st.selectbox("Vocal gender", ["(none)", "f", "m"])
    style_weight = st.slider("Style weight (0=off)", 0.0, 1.0, 0.0, 0.05)
    audio_weight = st.slider("Audio weight (0=off)", 0.0, 1.0, 0.0, 0.05)
    weirdness    = st.slider("Weirdness (0=off)", 0.0, 1.0, 0.0, 0.05)

    st.subheader("AWS S3")
    aws_id    = st.text_input("Access Key ID",     value=env.get("AWS_ACCESS_KEY_ID", ""),     type="password")
    aws_sec   = st.text_input("Secret Access Key", value=env.get("AWS_SECRET_ACCESS_KEY", ""), type="password")
    s3_bucket = st.text_input("Bucket", value=S3_BUCKET_DEFAULT)


# ── Results renderer (used by both live-poll view and cached view) ─────────────
def show_results(job: dict):
    info  = job.get("video_info") or {}
    files = job.get("files", {})

    if job.get("suno_error"):
        st.error(job["suno_error"])

    if job.get("suno_tracks"):
        for i, track in enumerate(job["suno_tracks"], 1):
            path = track["path"]
            if os.path.exists(path):
                st.markdown(f"**Track {i}: {track['title']}**")
                st.audio(path, format="audio/mpeg")
                with open(path, "rb") as f:
                    st.download_button(
                        f"Download track {i}", f,
                        file_name=os.path.basename(path),
                        mime="audio/mpeg",
                        key=f"suno_{i}",
                    )

    intermediates = [
        ("Downloaded MP3",              "mp3",         "song.mp3",                  "audio/mpeg"),
        ("Clip — vocals",               "chorus_voc",  "clip_vocals.wav",           "audio/wav"),
        ("Clip — instrumental",         "chorus_inst", "clip_instrumental.wav",     "audio/wav"),
        ("Local meowified",             "meow_local",  "meowified_local.wav",       "audio/wav"),
        ("Attempt 1 — masked",          "masked",      "meowified_masked.wav",      "audio/wav"),
        ("Attempt 2 — remix (+2st)",    "meow_retry",  "meowified_retry.wav",       "audio/wav"),
        ("Attempt 2 — heavy masked",    "heavy_masked","meowified_heavy_masked.wav","audio/wav"),
    ]
    available = [(lbl, k, fn, mime) for lbl, k, fn, mime in intermediates
                 if files.get(k) and os.path.exists(str(files[k]))]
    if available:
        with st.expander("Intermediate files", expanded=False):
            cols = st.columns(min(3, len(available)))
            for i, (lbl, k, fn, mime) in enumerate(available):
                with cols[i % 3]:
                    st.caption(lbl)
                    st.audio(str(files[k]), format=mime)
                    with open(str(files[k]), "rb") as f:
                        st.download_button("Download", f, file_name=fn, mime=mime, key=f"int_{k}")

    chorus_s = files.get("chorus_start", 45.0)
    src_url  = job.get("source_url", "")
    if src_url:
        ts  = int(chorus_s)
        sep = "&" if "?" in src_url else "?"
        st.markdown(f"[Watch original from {ts}s on YouTube]({src_url}{sep}t={ts}s)")

    payload = files.get("suno_payload")
    if payload:
        with st.expander("Suno API request", expanded=False):
            display = dict(payload)
            display["prompt"] = display["prompt"][:80] + f"... ({display['prompt'].count('meow')} meows)"
            st.code(_json.dumps(display, indent=2), language="json")

    if job.get("logs"):
        with st.expander("Verbose logs", expanded=False):
            st.code("\n".join(job["logs"]), language=None)


# ── Job status view ───────────────────────────────────────────────────────────
_job_id = st.query_params.get("job", "")

if _job_id and _job_id in JOBS:
    job = JOBS[_job_id]

    if st.button("Make another cover"):
        del st.query_params["job"]
        st.rerun()

    st.divider()

    if job["status"] == "running":
        st.info(f"Running in background: **{job.get('step', '...')}**")
        st.caption("You can close this tab and return later — the pipeline keeps running.")
        if job.get("logs"):
            with st.expander("Logs so far", expanded=False):
                st.code("\n".join(job["logs"][-30:]), language=None)
        time.sleep(3)
        st.rerun()

    elif job["status"] == "error":
        st.error(f"Pipeline failed: {job.get('error', 'Unknown error')}")
        if job.get("logs"):
            with st.expander("Logs", expanded=True):
                st.code("\n".join(job["logs"]), language=None)

    else:  # done
        info = job.get("video_info") or {}
        st.subheader(f"Results — {info.get('title', '')}")
        st.balloons()
        show_results(job)

    st.stop()


# ── Input form ────────────────────────────────────────────────────────────────
_col_url, _col_start = st.columns([4, 1])
url          = _col_url.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
manual_start = _col_start.number_input("Start time (s)", value=0, step=5, min_value=0,
                                        help="Leave 0 to auto-detect chorus")
run = st.button("Meowify! 🐾", type="primary")

if run and url:
    job_id = str(uuid.uuid4())[:8]
    JOBS[job_id] = {
        "status":      "running",
        "step":        "Starting...",
        "logs":        [],
        "files":       {},
        "suno_tracks": [],
        "video_info":  None,
        "source_url":  url,
        "error":       None,
        "suno_error":  None,
    }
    params = {
        "inst_pitch":   inst_pitch,
        "vocal_pitch":  vocal_pitch,
        "speed":        speed,
        "inst_gain":    inst_gain,
        "vocal_gain":   vocal_gain,
        "meow_gain":    meow_gain,
        "full_song":    full_song,
        "chorus_start": chorus_start,
        "chorus_dur":   chorus_dur,
        "manual_start": manual_start,
        "kie_key":      kie_key,
        "suno_model":   suno_model,
        "meow_count":   meow_count,
        "suno_style":   suno_style,
        "vocal_gender": vocal_gender,
        "style_weight": style_weight,
        "audio_weight": audio_weight,
        "weirdness":    weirdness,
        "aws_id":       aws_id,
        "aws_sec":      aws_sec,
        "s3_bucket":    s3_bucket,
    }
    t = threading.Thread(target=run_pipeline, args=(job_id, url, params), daemon=False)
    t.start()
    st.query_params["job"] = job_id
    st.rerun()
