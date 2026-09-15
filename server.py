"""
Meowify — FastAPI backend
Run with: uvicorn server:app --host 127.0.0.1 --port 8503 --workers 1
(single worker required — JOBS dict lives in process memory)
"""

import contextlib
import io
import json as _json
import logging
import os
import secrets
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('server.log'),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)

import boto3
import librosa
import numpy as np
import requests as http_requests
import soundfile as sf
from scipy import signal as sig

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from downloader import download_youtube_as_mp3, get_video_info, sanitize_filename
from meowify_v2 import meowify_v2

# ── Constants ──────────────────────────────────────────────────────────────────
WORK_DIR    = "work"
OUTPUT_DIR  = "output"
CAT_REF     = "cat_samples/separated/htdemucs/cat_meow_ref_trim/vocals_10s.wav"
S3_BUCKET   = "sagemaker-us-east-1-478706476061"
KIE_BASE    = "https://api.kie.ai"
SITE_URL    = "https://meowify.click"
SES_SENDER  = "noreply@meowify.click"
ADMIN_EMAIL = "stannor@gmail.com"
DB_PATH     = "users.db"
BANNER      = "https://cdn-images-1.medium.com/v2/resize:fit:800/1*0yOLabnolhlZDzShFH2vdg.png"

GOOGLE_AUTH_URL     = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL    = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v3/userinfo"


# ── Env ────────────────────────────────────────────────────────────────────────
def _load_env() -> dict:
    env = {}
    p = Path(".env")
    if p.exists():
        for line in p.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env

_ENV = _load_env()
GOOGLE_CLIENT_ID     = _ENV.get("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = _ENV.get("GOOGLE_CLIENT_SECRET", "")
OAUTH_REDIRECT_URI   = _ENV.get("OAUTH_REDIRECT_URI", f"{SITE_URL}/oauth/callback")

# Persistent session secret — stored in file if not provided via environment
def _get_session_secret():
    if "SESSION_SECRET" in _ENV:
        return _ENV["SESSION_SECRET"]
    secret_file = Path(".session_secret")
    if secret_file.exists():
        return secret_file.read_text().strip()
    secret = secrets.token_hex(32)
    secret_file.write_text(secret)
    return secret

SESSION_SECRET = _get_session_secret()
KIE_KEY        = _ENV.get("KIE_API_KEY", "")
RAPIDAPI_KEY   = _ENV.get("RAPIDAPI_KEY", "")
AWS_KEY_ID     = _ENV.get("AWS_ACCESS_KEY_ID", "")
AWS_SECRET     = _ENV.get("AWS_SECRET_ACCESS_KEY", "")


# ── Database ───────────────────────────────────────────────────────────────────
def _db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con

SHARES_DIR = "uploads/shares"

def init_db():
    con = _db()
    con.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            email   TEXT PRIMARY KEY,
            name    TEXT,
            picture TEXT,
            credits INTEGER DEFAULT 3,
            created_at TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS shares (
            id           TEXT PRIMARY KEY,
            email        TEXT,
            screenshot   TEXT,
            submitted_at TEXT DEFAULT (datetime('now')),
            approved     INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS jobs (
            id          TEXT PRIMARY KEY,
            owner_email TEXT,
            source_url  TEXT,
            status      TEXT,
            step        TEXT,
            error       TEXT,
            suno_error  TEXT,
            title       TEXT,
            files       TEXT,
            suno_tracks TEXT,
            video_info  TEXT,
            logs        TEXT,
            created_at  TEXT DEFAULT (datetime('now'))
        );
    """)
    con.commit()
    con.close()

def save_job(job_id: str, job: dict):
    import json as _j
    con = _db()
    con.execute("""
        INSERT OR REPLACE INTO jobs
            (id, owner_email, source_url, status, step, error, suno_error, title, files, suno_tracks, video_info, logs)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        job_id,
        job.get("owner_email"),
        job.get("source_url"),
        job.get("status"),
        job.get("step"),
        job.get("error"),
        job.get("suno_error"),
        (job.get("video_info") or {}).get("title"),
        _j.dumps(job.get("files", {})),
        _j.dumps(job.get("suno_tracks", [])),
        _j.dumps(job.get("video_info")),
        _j.dumps(job.get("logs", [])),
    ))
    con.commit()
    con.close()

def load_job(job_id: str) -> Optional[dict]:
    import json as _j
    con = _db()
    row = con.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    con.close()
    if not row:
        return None
    return {
        "status":      row["status"],
        "step":        row["step"],
        "error":       row["error"],
        "suno_error":  row["suno_error"],
        "source_url":  row["source_url"],
        "owner_email": row["owner_email"],
        "files":       _j.loads(row["files"] or "{}"),
        "suno_tracks": _j.loads(row["suno_tracks"] or "[]"),
        "video_info":  _j.loads(row["video_info"] or "null"),
        "logs":        _j.loads(row["logs"] or "[]"),
    }

def get_or_create_user(email: str, name: str, picture: str) -> dict:
    con = _db()
    con.execute(
        "INSERT OR IGNORE INTO users (email, name, picture, credits) VALUES (?,?,?,3)",
        (email, name, picture),
    )
    con.execute(
        "UPDATE users SET name=?, picture=? WHERE email=?",
        (name, picture, email),
    )
    con.commit()
    row = con.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
    con.close()
    return dict(row)

def get_user(email: str) -> Optional[dict]:
    con = _db()
    row = con.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
    con.close()
    return dict(row) if row else None

def use_credit(email: str) -> bool:
    if email == ADMIN_EMAIL:
        return True
    con = _db()
    row = con.execute("SELECT credits FROM users WHERE email=?", (email,)).fetchone()
    if not row or row["credits"] <= 0:
        con.close()
        return False
    con.execute("UPDATE users SET credits=credits-1 WHERE email=?", (email,))
    con.commit()
    con.close()
    return True

def all_users() -> list[dict]:
    con = _db()
    rows = con.execute("SELECT * FROM users ORDER BY created_at DESC").fetchall()
    con.close()
    return [dict(r) for r in rows]

def pending_shares() -> list[dict]:
    con = _db()
    rows = con.execute(
        "SELECT * FROM shares WHERE approved=0 ORDER BY submitted_at DESC"
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]


# ── Audio helpers ──────────────────────────────────────────────────────────────
def detect_chorus(audio_path: str, default_start: float = 45.0, duration: float = 45.0) -> tuple[float, str]:
    y, sr = librosa.load(audio_path, sr=22050, mono=True, duration=180.0)
    total = len(y) / sr
    if total < default_start + duration:
        return 0.0, "track too short — using 0s"
    rms = librosa.feature.rms(y=y, frame_length=sr * 2, hop_length=sr)[0]
    w = int(duration)
    def_idx = int(default_start)
    def_e = float(np.mean(rms[def_idx: def_idx + w])) if def_idx + w < len(rms) else 0.0
    best_idx, best_e = def_idx, def_e
    for s in range(30, min(71, int(total - duration))):
        if s + w >= len(rms):
            break
        e = float(np.mean(rms[s: s + w]))
        if e > best_e:
            best_e, best_idx = e, s
    if best_idx != def_idx and best_e > 1.15 * def_e:
        adjusted = max(0.0, float(best_idx) - 1.0)
        return adjusted, f"auto-detected at {best_idx}s ({adjusted:.0f}s)"
    return default_start, f"default {default_start:.0f}s"

def extract_clip(src: str, start: float, duration: float, dst: str) -> str:
    y, sr = librosa.load(src, sr=44100, mono=True)
    s, e = int(start * sr), int((start + duration) * sr)
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    sf.write(dst, y[s:e], sr)
    return dst

def _biquad(y, b, a):
    sos = sig.tf2sos(b, a)
    return sig.sosfilt(sos, y.T).T if y.ndim > 1 else sig.sosfilt(sos, y)

def _peaking(sr, fc, gain_db, Q=1.0):
    A = 10**(gain_db/40); w0 = 2*np.pi*fc/sr; alpha = np.sin(w0)/(2*Q)
    return [1+alpha*A, -2*np.cos(w0), 1-alpha*A], [1+alpha/A, -2*np.cos(w0), 1-alpha/A]

def _high_shelf(sr, fc, gain_db, Q=0.707):
    A = 10**(gain_db/40); w0 = 2*np.pi*fc/sr; cw = np.cos(w0)
    al = np.sin(w0)/2*np.sqrt((A+1/A)*(1/Q-1)+2)
    b = [A*((A+1)+(A-1)*cw+2*np.sqrt(A)*al), -2*A*((A-1)+(A+1)*cw), A*((A+1)+(A-1)*cw-2*np.sqrt(A)*al)]
    a = [(A+1)-(A-1)*cw+2*np.sqrt(A)*al, 2*((A-1)-(A+1)*cw), (A+1)-(A-1)*cw-2*np.sqrt(A)*al]
    return b, a

def _low_shelf(sr, fc, gain_db, Q=0.707):
    A = 10**(gain_db/40); w0 = 2*np.pi*fc/sr; cw = np.cos(w0)
    al = np.sin(w0)/2*np.sqrt((A+1/A)*(1/Q-1)+2)
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
    y = 0.80*y + 0.20*(np.tanh(y*drive)/np.tanh(drive))
    noise_db = -20 if heavy else -35
    rms = np.sqrt(np.mean(y**2))
    y = y + np.random.normal(0, rms*10**(noise_db/20), y.shape).astype(np.float32)
    peak = np.max(np.abs(y))
    if peak > 0.98:
        y = y/peak*0.97
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    sf.write(dst, y.astype(np.float32), sr)
    return dst

def s3_upload_presign(local: str, bucket: str, key: str, expiry: int = 7200) -> str:
    client = boto3.client("s3", aws_access_key_id=AWS_KEY_ID,
                          aws_secret_access_key=AWS_SECRET, region_name="us-east-1")
    client.upload_file(local, bucket, key)
    return client.generate_presigned_url("get_object",
                                         Params={"Bucket": bucket, "Key": key},
                                         ExpiresIn=expiry)

def suno_submit(upload_url: str, lyrics: str, style: str, title: str, model: str,
                style_weight=None, audio_weight=None, weirdness=None, vocal_gender=None) -> dict:
    payload = {"uploadUrl": upload_url, "prompt": lyrics, "style": style,
               "title": title, "customMode": True, "instrumental": False,
               "model": model, "callBackUrl": "https://api.example.com/callback"}
    if style_weight: payload["styleWeight"] = round(style_weight, 2)
    if audio_weight: payload["audioWeight"] = round(audio_weight, 2)
    if weirdness:    payload["weirdnessConstraint"] = round(weirdness, 2)
    if vocal_gender: payload["vocalGender"] = vocal_gender
    r = http_requests.post(f"{KIE_BASE}/api/v1/generate/upload-cover",
                           headers={"Authorization": f"Bearer {KIE_KEY}"},
                           json=payload, timeout=30)
    data = r.json()
    if data["code"] != 200:
        raise RuntimeError(f"kie.ai error {data['code']}: {data['msg']}")
    return {"taskId": data["data"]["taskId"], "payload": payload}

def dl_file(url: str, dst: str) -> str:
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    r = http_requests.get(url, timeout=120, stream=True)
    r.raise_for_status()
    with open(dst, "wb") as f:
        for chunk in r.iter_content(8192):
            f.write(chunk)
    return dst

def send_result_email(to_email: str, title: str, tracks: list, job_url: str,
                      suno_error: str = None, pipeline_error: str = None, log_fn=None):
    import traceback

    def _log(msg):
        print(msg)
        if log_fn:
            log_fn(msg)

    if not AWS_KEY_ID or not AWS_SECRET:
        _log("Email skipped: AWS credentials not configured")
        return

    recipients = list({to_email, ADMIN_EMAIL})

    try:
        if tracks:
            track_lines = "\n".join(f"  - {t['title']}" for t in tracks)
            subject = f"Meowify: \"{title[:55]}\" is ready!"
            user_body = (f'Your meow cover of "{title}" is ready!\n\n'
                         f'{len(tracks)} track(s):\n{track_lines}\n\n'
                         f'Listen here:\n{job_url}\n\n-- Meowify')
            admin_body = (f'{to_email} generated a cover of "{title}".\n\n'
                          f'{len(tracks)} track(s):\n{track_lines}\n\n'
                          f'Listen here:\n{job_url}\n\n-- Meowify')
        elif suno_error and "413" in suno_error:
            subject = f"Meowify: \"{title[:50]}\" — fingerprint detected"
            user_body = (f'Meowify couldn\'t generate a cover for "{title}".\n\n'
                         f'Suno flagged this track for copyright fingerprinting.\n\n'
                         f'Tip: this usually happens with very popular or well-known songs. '
                         f'Try a less mainstream track — folk songs, indie artists, or lesser-known covers tend to work much better.\n\n'
                         f'Job details:\n{job_url}\n\n-- Meowify')
            admin_body = f'{to_email}: fingerprint on "{title}"\n\n{job_url}'
        elif suno_error:
            subject = f"Meowify: \"{title[:55]}\" — Suno failed"
            user_body = (f'Meowify ran into a problem generating a cover for "{title}".\n\n'
                         f'Error: {suno_error}\n\n'
                         f'Job details:\n{job_url}\n\n-- Meowify')
            admin_body = f'{to_email}: Suno failed for "{title}"\n\nError: {suno_error}\n\n{job_url}'
        elif pipeline_error:
            subject = f"Meowify: \"{title[:55]}\" — pipeline error"
            user_body = (f'The Meowify pipeline failed for "{title}".\n\n'
                         f'Error: {pipeline_error}\n\n'
                         f'Job details:\n{job_url}\n\n-- Meowify')
            admin_body = f'{to_email}: pipeline error for "{title}"\n\nError: {pipeline_error}\n\n{job_url}'
        else:
            return

        client = boto3.client("ses", region_name="us-east-1",
                              aws_access_key_id=AWS_KEY_ID, aws_secret_access_key=AWS_SECRET)

        def _send(to, body):
            client.send_email(
                Source=SES_SENDER,
                Destination={"ToAddresses": [to]},
                Message={"Subject": {"Data": subject},
                         "Body": {"Text": {"Data": body}}},
            )

        sent = []
        try:
            _send(to_email, user_body)
            sent.append(to_email)
        except Exception as e:
            _log(f"Email FAILED to {to_email}: {e}")
        if to_email != ADMIN_EMAIL:
            try:
                _send(ADMIN_EMAIL, admin_body)
                sent.append(ADMIN_EMAIL)
            except Exception as e:
                _log(f"Email FAILED to {ADMIN_EMAIL}: {e}\n{traceback.format_exc()}")
        if sent:
            _log(f"Email sent to {', '.join(sent)}")
    except Exception as e:
        _log(f"Email FAILED: {e}\n{traceback.format_exc()}")


# ── Jobs ───────────────────────────────────────────────────────────────────────
JOBS: dict = {}


# ── Pipeline ───────────────────────────────────────────────────────────────────
def run_pipeline(job_id: str, url: str, params: dict, local_mp3_path: Optional[str] = None):
    job = JOBS[job_id]

    def log(msg: str):
        job["logs"].append(f"[{time.strftime('%H:%M:%S')}] {msg}")

    def step(label: str):
        job["step"] = label
        log(f"--- {label} ---")

    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(WORK_DIR, exist_ok=True)

        if local_mp3_path:
            step("Using uploaded audio...")
            mp3_path = local_mp3_path
            title = os.path.splitext(os.path.basename(mp3_path))[0]
            info = {"title": title, "duration": 0, "thumbnail_url": "", "uploader": "(uploaded)", "id": ""}
            job["video_info"] = info
            log(f"File: {mp3_path} ({os.path.getsize(mp3_path)/1e6:.1f} MB)")
            job["files"]["mp3"] = mp3_path
        else:
            step("Fetching video info...")
            info = get_video_info(url)
            job["video_info"] = info
            log(f"Title: {info['title']} ({info['duration']}s)")

            step("Downloading audio...")
            mp3_path = download_youtube_as_mp3(url, api_key=RAPIDAPI_KEY)
            log(f"Downloaded: {mp3_path} ({os.path.getsize(mp3_path)/1e6:.1f} MB)")
            job["files"]["mp3"] = mp3_path

        base      = os.path.splitext(os.path.basename(mp3_path))[0]
        base_short = base[:50] if len(base) > 50 else base
        full_song = params["full_song"]
        chorus_dur = params["chorus_dur"]

        step("Selecting section...")
        if full_song:
            chorus_s, clip_input, clip_base = 0.0, mp3_path, base
            log("Full song mode")
        else:
            manual = params["manual_start"]
            if manual > 0:
                chorus_s, reason = float(manual), f"manual at {manual:.0f}s"
            else:
                chorus_s, reason = detect_chorus(mp3_path,
                                                 default_start=params["chorus_start"],
                                                 duration=chorus_dur)
            log(f"Chorus: {reason}")
            clip_base  = f"{base}_clip_{int(chorus_s)}s"
            clip_input = os.path.join(OUTPUT_DIR, f"{clip_base}.wav")
            extract_clip(mp3_path, chorus_s, chorus_dur, clip_input)
        job["files"]["chorus_start"] = chorus_s

        chorus_voc  = os.path.join(WORK_DIR, "htdemucs", clip_base, "vocals.wav")
        chorus_inst = os.path.join(WORK_DIR, "htdemucs", clip_base, "no_vocals.wav")

        step("Separating vocals (Demucs)...")
        if not os.path.exists(chorus_voc):
            log("Running Demucs...")
            cmd = [sys.executable, "-m", "demucs", "--two-stems", "vocals",
                   "-n", "htdemucs", "--out", WORK_DIR, clip_input]
            result = subprocess.run(cmd, capture_output=True, text=True)
            for line in (result.stdout + result.stderr).splitlines():
                if line.strip(): log(line)
            if result.returncode != 0:
                raise RuntimeError(result.stderr[-800:])
        else:
            log("Using cached Demucs output")
        job["files"]["chorus_voc"]  = chorus_voc
        job["files"]["chorus_inst"] = chorus_inst

        step("Local meowify (note replacement)...")
        meow_local = os.path.join(OUTPUT_DIR, f"{base}_meowified.wav")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            meowify_v2(
                vocals_path=chorus_voc, instrumental_path=chorus_inst,
                meow_ref_path=CAT_REF, output_path=meow_local,
                inst_pitch_semitones=float(params["inst_pitch"]),
                vocal_pitch_semitones=float(params["vocal_pitch"]),
                speed_factor=params["speed"],
                inst_gain=params["inst_gain"], vocal_gain=params["vocal_gain"],
                meow_gain=params["meow_gain"],
            )
        for line in buf.getvalue().splitlines(): log(line)
        job["files"]["meow_local"] = meow_local

        if KIE_KEY and AWS_KEY_ID and AWS_SECRET:
            step("Masking + uploading to S3...")
            masked = os.path.join(OUTPUT_DIR, f"{base}_meowified_masked.wav")
            apply_masking(meow_local, masked)
            job["files"]["masked"] = masked
            s3_key = f"meowify/{base}_meowified_masked.wav"
            presigned_url = s3_upload_presign(masked, S3_BUCKET, s3_key)
            log(f"Uploaded s3://{S3_BUCKET}/{s3_key}")

            _suffix = " (Meow Cover)"
            cover_title = info["title"][:95 - len(_suffix)] + _suffix
            terminal = {"SUCCESS", "FAILED", "ERROR", "FIRST_SUCCESS", "GENERATE_AUDIO_FAILED"}

            def _meow(n): return "\n".join(["meow"] * n)
            suno_lyrics = (
                f"[Intro]\n{_meow(10)}\n\n[Verse]\n{_meow(30)}\n\n"
                f"[Chorus]\n{_meow(30)}\n\n[Verse]\n{_meow(30)}\n\n[Chorus]\n{_meow(30)}"
            ) if full_song else f"[Chorus]\n{_meow(params['meow_count'])}"

            def poll_suno(upload_url, label):
                sub = suno_submit(
                    upload_url, suno_lyrics, params["suno_style"], cover_title,
                    params["suno_model"],
                    style_weight=params["style_weight"] or None,
                    audio_weight=params["audio_weight"] or None,
                    weirdness=params["weirdness"]    or None,
                    vocal_gender=params["vocal_gender"] if params["vocal_gender"] != "(none)" else None,
                )
                task_id = sub["taskId"]
                log(f"Suno submitted ({label}): {task_id}")
                t0 = time.time()
                while True:
                    r = http_requests.get(f"{KIE_BASE}/api/v1/generate/record-info",
                                          params={"taskId": task_id},
                                          headers={"Authorization": f"Bearer {KIE_KEY}"},
                                          timeout=30)
                    data = r.json()["data"]
                    s = data["status"]
                    elapsed = time.time() - t0
                    job["step"] = f"Suno cover — {s} ({elapsed:.0f}s)..."
                    log(f"Suno poll ({label}): {s} ({elapsed:.0f}s)")
                    if s in terminal:
                        return sub["payload"], data, s
                    if elapsed > 600:
                        raise TimeoutError("Suno timed out after 10 min")
                    time.sleep(10)

            step("Generating Suno cover (5–10 min)...")
            payload, suno_data, status = poll_suno(presigned_url, "attempt 1")
            job["files"]["suno_payload"] = payload

            if status not in {"SUCCESS", "FIRST_SUCCESS"} and suno_data.get("errorCode") == 413:
                log("413 fingerprint detected — retrying with heavy masking")
                meow_retry = os.path.join(OUTPUT_DIR, f"{base}_meowified_retry.wav")
                meowify_v2(vocals_path=chorus_voc, instrumental_path=chorus_inst,
                           meow_ref_path=CAT_REF, output_path=meow_retry,
                           inst_pitch_semitones=float(params["inst_pitch"]) + 2.0,
                           vocal_pitch_semitones=float(params["vocal_pitch"]) + 2.0,
                           speed_factor=params["speed"], inst_gain=params["inst_gain"],
                           vocal_gain=0.35, meow_gain=params["meow_gain"])
                job["files"]["meow_retry"] = meow_retry
                heavy = os.path.join(OUTPUT_DIR, f"{base}_meowified_heavy.wav")
                apply_masking(meow_retry, heavy, heavy=True)
                job["files"]["heavy_masked"] = heavy
                heavy_key = f"meowify/{base}_meowified_heavy.wav"
                heavy_url = s3_upload_presign(heavy, S3_BUCKET, heavy_key)
                payload, suno_data, status = poll_suno(heavy_url, "attempt 2 (heavy)")
                job["files"]["suno_payload"] = payload

            if status not in {"SUCCESS", "FIRST_SUCCESS"}:
                ec = suno_data.get("errorCode"); em = suno_data.get("errorMessage")
                job["suno_error"] = f"Suno {status}" + (f" ({ec})" if ec else "") + (f": {em}" if em else "")
                log(f"Suno failed: {job['suno_error']}")
            else:
                tracks = (suno_data.get("response") or {}).get("sunoData") or []
                for i, track in enumerate(tracks, 1):
                    audio_url = track.get("audioUrl") or track.get("streamAudioUrl", "")
                    if not audio_url: continue
                    out = os.path.join(OUTPUT_DIR, f"{base_short}_suno_{i}.mp3")
                    dl_file(audio_url, out)
                    job["suno_tracks"].append({"path": out, "title": track.get("title", f"Track {i}")})
                    log(f"Track {i}: {out}")
                log(f"Suno complete: {len(job['suno_tracks'])} tracks")

        job["status"] = "done"
        job["step"]   = "Complete"
        log("Pipeline complete")
        save_job(job_id, job)

        job_url = f"{SITE_URL}/job/{job_id}"
        send_result_email(job["owner_email"], info["title"], job["suno_tracks"], job_url,
                          suno_error=job.get("suno_error"), log_fn=log)

    except Exception as e:
        job["status"] = "error"
        job["error"]  = str(e)
        job["step"]   = f"Error: {e}"
        job["logs"].append(f"[{time.strftime('%H:%M:%S')}] FATAL: {e}")

        # Refund credit if we never obtained the source audio (yt-dlp / info fetch failure).
        # The user shouldn't be charged for our YouTube download failures.
        owner = job.get("owner_email")
        if (not local_mp3_path
                and not job["files"].get("mp3")
                and owner and owner != ADMIN_EMAIL):
            try:
                con = _db()
                con.execute("UPDATE users SET credits = credits + 1 WHERE email = ?", (owner,))
                con.commit()
                con.close()
                log(f"Credit refunded to {owner} (download-phase failure)")
            except Exception as refund_err:
                log(f"Credit refund failed: {refund_err}")

        save_job(job_id, job)
        job_url = f"{SITE_URL}/job/{job_id}"
        title = (job.get("video_info") or {}).get("title", job.get("source_url", "unknown"))
        send_result_email(job["owner_email"], title, [], job_url,
                          pipeline_error=str(e), log_fn=log)


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET, max_age=30 * 24 * 3600)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/robots.txt", include_in_schema=False)
async def robots():
    return FileResponse("static/robots.txt", media_type="text/plain")

templates = Jinja2Templates(directory="templates")
templates.env.globals["enumerate"] = enumerate

init_db()


# ── Auth helpers ───────────────────────────────────────────────────────────────
def current_user(request: Request) -> Optional[dict]:
    email = request.session.get("email")
    return get_user(email) if email else None

def require_user(request: Request) -> dict:
    user = current_user(request)
    if not user:
        raise HTTPException(status_code=307, headers={"Location": "/login"})
    return user

def can_access_job(user: dict, job: dict) -> bool:
    return user["email"] == job.get("owner_email") or user["email"] == ADMIN_EMAIL

def google_auth_url(state: str) -> str:
    return (
        f"{GOOGLE_AUTH_URL}?client_id={GOOGLE_CLIENT_ID}"
        f"&redirect_uri={OAUTH_REDIRECT_URI}"
        f"&response_type=code&scope=openid%20email%20profile"
        f"&state={state}&access_type=offline&prompt=select_account"
    )


# ── Routes: auth ───────────────────────────────────────────────────────────────
@app.get("/login")
async def login_page(request: Request):
    if current_user(request):
        return RedirectResponse("/")
    state = secrets.token_urlsafe(16)
    request.session["oauth_state"] = state
    next_url = request.query_params.get("next", "/")
    request.session["oauth_next"] = next_url
    return templates.TemplateResponse(request, "login.html", {
        "auth_url": google_auth_url(state),
        "banner": BANNER,
        "error": request.query_params.get("error"),
    })

@app.get("/oauth/callback")
async def oauth_callback(request: Request, code: str = None, error: str = None, state: str = None):
    if error or not code:
        return RedirectResponse(f"/login?error={error or 'cancelled'}")
    try:
        token_resp = http_requests.post(GOOGLE_TOKEN_URL, data={
            "code": code, "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "redirect_uri": OAUTH_REDIRECT_URI, "grant_type": "authorization_code",
        }, timeout=15)
        token_resp.raise_for_status()
        access_token = token_resp.json()["access_token"]
        info = http_requests.get(GOOGLE_USERINFO_URL,
                                  headers={"Authorization": f"Bearer {access_token}"},
                                  timeout=15).json()
        user = get_or_create_user(info["email"], info.get("name", ""), info.get("picture", ""))
        request.session["email"] = user["email"]
    except Exception as e:
        return RedirectResponse(f"/login?error={e}")
    next_url = request.session.get("oauth_next", "/")
    request.session.pop("oauth_next", None)
    return RedirectResponse(next_url)

@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login")


# ── Routes: main ───────────────────────────────────────────────────────────────
BOT_AGENTS = ("facebookexternalhit", "Twitterbot", "LinkedInBot", "WhatsApp", "Slackbot", "TelegramBot")

@app.get("/")
async def index(request: Request):
    user = current_user(request)
    ua = request.headers.get("user-agent", "")
    if not user:
        if any(b in ua for b in BOT_AGENTS):
            return templates.TemplateResponse(request, "login.html", {
                "auth_url": "#", "banner": BANNER, "error": None,
            })
        return RedirectResponse("/login")
    return templates.TemplateResponse(request, "index.html", {
        "user": user, "banner": BANNER,
    })

@app.post("/submit")
async def submit(
    request: Request,
    url: str          = Form(default=""),
    audio: Optional[UploadFile] = File(default=None),
    manual_start: float = Form(0),
    inst_pitch: int   = Form(0),
    vocal_pitch: int  = Form(12),
    speed: float      = Form(1.2),
    inst_gain: float  = Form(1.0),
    vocal_gain: float = Form(0.5),
    meow_gain: float  = Form(0.7),
    full_song: str    = Form("off"),
    chorus_start: float = Form(45.0),
    chorus_dur: float = Form(45.0),
    suno_model: str   = Form("V5_5"),
    meow_count: int   = Form(120),
    suno_style: str   = Form("pop"),
    vocal_gender: str = Form("(none)"),
    style_weight: float = Form(0.0),
    audio_weight: float = Form(0.0),
    weirdness: float  = Form(0.0),
):
    user = current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    if not use_credit(user["email"]):
        return templates.TemplateResponse(request, "index.html", {
            "user": user, "banner": BANNER,
            "error": "No credits remaining. Contact stannor@gmail.com to get more.",
        })

    # Handle file upload or URL
    local_mp3_path = None
    source_description = url
    job_id = str(uuid.uuid4())[:8]

    if audio and audio.filename:
        ext = (os.path.splitext(audio.filename or "")[1] or ".mp3").lower()
        if ext not in _ALLOWED_UPLOAD_EXT:
            ext = ".mp3"
        base_name = sanitize_filename(os.path.splitext(audio.filename or "upload")[0]) or "upload"
        os.makedirs("downloads", exist_ok=True)
        upload_path = os.path.abspath(os.path.join("downloads", f"{base_name}_{job_id}{ext}"))
        content = await audio.read()
        with open(upload_path, "wb") as f:
            f.write(content)
        local_mp3_path = upload_path
        source_description = f"(uploaded: {base_name})"
        url = ""
    elif not url:
        return templates.TemplateResponse(request, "index.html", {
            "user": user, "banner": BANNER,
            "error": "Please provide a YouTube URL or upload an MP3 file.",
        })

    params = {
        "inst_pitch": inst_pitch, "vocal_pitch": vocal_pitch,
        "speed": speed, "inst_gain": inst_gain, "vocal_gain": vocal_gain,
        "meow_gain": meow_gain, "full_song": full_song == "on",
        "chorus_start": chorus_start, "chorus_dur": chorus_dur,
        "manual_start": manual_start, "suno_model": suno_model,
        "meow_count": meow_count, "suno_style": suno_style,
        "vocal_gender": vocal_gender, "style_weight": style_weight,
        "audio_weight": audio_weight, "weirdness": weirdness,
    }
    JOBS[job_id] = {
        "status":      "running",
        "step":        "Starting..." if url else "Starting (uploaded audio)...",
        "logs":        [],
        "files":       {},
        "suno_tracks": [],
        "video_info":  None,
        "source_url":  url,
        "owner_email": user["email"],
        "error":       None,
        "suno_error":  None,
        "params":      params,
    }
    save_job(job_id, JOBS[job_id])
    threading.Thread(
        target=run_pipeline,
        args=(job_id, url, params),
        kwargs={"local_mp3_path": local_mp3_path} if local_mp3_path else {},
        daemon=True,
    ).start()

    if user["email"] != ADMIN_EMAIL and AWS_KEY_ID and AWS_SECRET:
        try:
            boto3.client("ses", region_name="us-east-1",
                         aws_access_key_id=AWS_KEY_ID, aws_secret_access_key=AWS_SECRET).send_email(
                Source=SES_SENDER,
                Destination={"ToAddresses": [ADMIN_EMAIL]},
                Message={
                    "Subject": {"Data": f"Meowify: {user['email']} started a job"},
                    "Body": {"Text": {"Data": f"{user['name']} ({user['email']}) submitted a job.\n\nSource: {source_description}\n\nJob: {SITE_URL}/job/{job_id}"}},
                },
            )
        except Exception as e:
            print(f"Start notification failed: {e}")

    return RedirectResponse(f"/job/{job_id}", status_code=303)


# ── Routes: job ────────────────────────────────────────────────────────────────
@app.get("/job/{job_id}")
async def job_page(job_id: str, request: Request):
    try:
        logger.debug(f"Loading job page for {job_id}")
        user = current_user(request)
        if not user:
            logger.debug(f"No user for job {job_id}, redirecting to login")
            return RedirectResponse(f"/login?next=/job/{job_id}")
        logger.debug(f"User {user['email']} accessing job {job_id}")
        job = JOBS.get(job_id) or load_job(job_id)
        if not job:
            logger.warning(f"Job {job_id} not found")
            raise HTTPException(404, "Job not found")
        logger.debug(f"Job {job_id} loaded: status={job.get('status')}")
        if not can_access_job(user, job):
            logger.warning(f"User {user['email']} cannot access job {job_id}")
            raise HTTPException(403, "Access denied")
        logger.debug(f"Rendering job.html for {job_id}")
        return templates.TemplateResponse(request, "job.html", {
            "user": user, "banner": BANNER,
            "job": job, "job_id": job_id,
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading job {job_id}: {e}", exc_info=True)
        raise HTTPException(500, f"Internal error: {str(e)}")

@app.get("/job/{job_id}/poll")
async def job_poll(job_id: str, request: Request):
    user = current_user(request)
    if not user:
        return HTMLResponse('<p>Session expired. <a href="/login">Log in again</a></p>')
    job = JOBS.get(job_id) or load_job(job_id)
    if not job:
        return HTMLResponse("<p>Job not found (server may have restarted).</p>")
    if not can_access_job(user, job):
        return HTMLResponse("<p>Access denied.</p>")
    return templates.TemplateResponse(request, "_job_status.html", {
        "job": job, "job_id": job_id,
    })


_DEFAULT_PIPELINE_PARAMS = {
    "inst_pitch": 0, "vocal_pitch": 12, "speed": 1.2,
    "inst_gain": 1.0, "vocal_gain": 0.5, "meow_gain": 0.7,
    "full_song": False, "chorus_start": 45.0, "chorus_dur": 45.0,
    "manual_start": 0.0, "suno_model": "V5_5", "meow_count": 120,
    "suno_style": "pop", "vocal_gender": "(none)",
    "style_weight": 0.0, "audio_weight": 0.0, "weirdness": 0.0,
}
_ALLOWED_UPLOAD_EXT = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac"}

@app.post("/job/{job_id}/retry-upload")
async def retry_upload(job_id: str, request: Request, audio: UploadFile = File(...)):
    user = current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    job = JOBS.get(job_id) or load_job(job_id)
    if not job or not can_access_job(user, job):
        raise HTTPException(403)
    if job.get("status") == "running":
        return RedirectResponse(f"/job/{job_id}", status_code=303)

    ext = (os.path.splitext(audio.filename or "")[1] or ".mp3").lower()
    if ext not in _ALLOWED_UPLOAD_EXT:
        ext = ".mp3"
    base_name = sanitize_filename(os.path.splitext(audio.filename or "upload")[0]) or "upload"
    os.makedirs("downloads", exist_ok=True)
    upload_path = os.path.abspath(os.path.join("downloads", f"{base_name}_{job_id}{ext}"))
    content = await audio.read()
    with open(upload_path, "wb") as f:
        f.write(content)

    params = (job.get("params") if isinstance(job.get("params"), dict) else None) or _DEFAULT_PIPELINE_PARAMS.copy()
    JOBS[job_id] = {
        "status":      "running",
        "step":        "Starting (uploaded audio)...",
        "logs":        [],
        "files":       {},
        "suno_tracks": [],
        "video_info":  None,
        "source_url":  job.get("source_url"),
        "owner_email": job.get("owner_email"),
        "error":       None,
        "suno_error":  None,
        "params":      params,
    }
    save_job(job_id, JOBS[job_id])
    threading.Thread(
        target=run_pipeline,
        args=(job_id, job.get("source_url") or "", params),
        kwargs={"local_mp3_path": upload_path},
        daemon=True,
    ).start()
    return RedirectResponse(f"/job/{job_id}", status_code=303)


# ── Routes: files ──────────────────────────────────────────────────────────────
_FILE_LABELS = {
    "mp3":         ("Downloaded MP3",         "audio/mpeg"),
    "chorus_voc":  ("Clip — vocals",          "audio/wav"),
    "chorus_inst": ("Clip — instrumental",    "audio/wav"),
    "meow_local":  ("Local meowified",        "audio/wav"),
    "masked":      ("Attempt 1 — masked",     "audio/wav"),
    "meow_retry":  ("Attempt 2 — remix",      "audio/wav"),
    "heavy_masked":("Attempt 2 — heavy mask", "audio/wav"),
}

@app.get("/files/{job_id}/{key}")
async def serve_file(job_id: str, key: str, request: Request, dl: bool = False):
    user = current_user(request)
    if not user:
        raise HTTPException(403)
    job = JOBS.get(job_id) or load_job(job_id)
    if not job or not can_access_job(user, job):
        raise HTTPException(403)

    # Suno track: key = "suno_0", "suno_1", ...
    if key.startswith("suno_"):
        idx = int(key.split("_")[1])
        tracks = job.get("suno_tracks", [])
        if idx >= len(tracks):
            raise HTTPException(404)
        path = tracks[idx]["path"]
        media_type = "audio/mpeg"
        filename = os.path.basename(path)
    else:
        if key not in _FILE_LABELS:
            raise HTTPException(404)
        path = job["files"].get(key)
        if not path or not os.path.exists(path):
            raise HTTPException(404)
        _, media_type = _FILE_LABELS[key]
        filename = os.path.basename(path)

    headers = {}
    if dl:
        ext = os.path.splitext(filename)[1] or (".mp3" if "mpeg" in media_type else ".wav")
        safe = f"meowify_{job_id}_{key}{ext}"
        headers["Content-Disposition"] = f'attachment; filename="{safe}"'
    return FileResponse(path, media_type=media_type, headers=headers)


# ── Routes: credits via share ──────────────────────────────────────────────────
@app.get("/credits")
async def credits_page(request: Request):
    user = current_user(request)
    if not user:
        return RedirectResponse("/login")
    return templates.TemplateResponse(request, "credits.html", {"user": user})

@app.post("/credits")
async def credits_submit(request: Request, screenshot: UploadFile = File(...)):
    user = current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)

    ext = (os.path.splitext(screenshot.filename or "")[1] or ".png").lower()
    share_id = str(uuid.uuid4())[:12]
    filename = f"{share_id}{ext}"
    os.makedirs(SHARES_DIR, exist_ok=True)
    path = os.path.join(SHARES_DIR, filename)
    content = await screenshot.read()
    with open(path, "wb") as f:
        f.write(content)

    con = _db()
    con.execute("INSERT INTO shares (id, email, screenshot) VALUES (?,?,?)",
                (share_id, user["email"], filename))
    con.commit()
    con.close()

    # Email admin
    approve_url = f"{SITE_URL}/admin/approve-share?id={share_id}"
    view_url    = f"{SITE_URL}/admin/shares/{filename}"
    try:
        client = boto3.client("ses", region_name="us-east-1",
                              aws_access_key_id=AWS_KEY_ID, aws_secret_access_key=AWS_SECRET)
        client.send_email(
            Source=SES_SENDER,
            Destination={"ToAddresses": [ADMIN_EMAIL]},
            Message={
                "Subject": {"Data": f"Meowify: share submission from {user['name'] or user['email']}"},
                "Body": {"Text": {"Data":
                    f"{user['name']} ({user['email']}) submitted a social share screenshot.\n\n"
                    f"View screenshot:\n{view_url}\n\n"
                    f"Approve (+10 credits):\n{approve_url}\n\n-- Meowify"
                }},
            },
        )
    except Exception as e:
        print(f"Share email failed: {e}")

    return templates.TemplateResponse(request, "credits.html", {
        "user": user, "submitted": True,
    })

@app.get("/admin/shares/{filename}")
async def view_share(filename: str, request: Request):
    user = current_user(request)
    if not user or user["email"] != ADMIN_EMAIL:
        raise HTTPException(403)
    path = os.path.join(SHARES_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(404)
    ext = os.path.splitext(filename)[1].lower()
    media_type = "image/png" if ext == ".png" else "image/jpeg" if ext in (".jpg", ".jpeg") else "application/octet-stream"
    return FileResponse(path, media_type=media_type)

@app.get("/admin/approve-share")
async def approve_share(request: Request, id: str):
    user = current_user(request)
    if not user or user["email"] != ADMIN_EMAIL:
        raise HTTPException(403)
    con = _db()
    share = con.execute("SELECT * FROM shares WHERE id=?", (id,)).fetchone()
    if not share:
        con.close()
        raise HTTPException(404, "Share not found")
    if share["approved"]:
        con.close()
        return HTMLResponse("<p>Already approved.</p>")
    con.execute("UPDATE users SET credits=credits+10 WHERE email=?", (share["email"],))
    con.execute("UPDATE shares SET approved=1 WHERE id=?", (id,))
    con.commit()
    con.close()

    # Notify the user
    try:
        client = boto3.client("ses", region_name="us-east-1",
                              aws_access_key_id=AWS_KEY_ID, aws_secret_access_key=AWS_SECRET)
        client.send_email(
            Source=SES_SENDER,
            Destination={"ToAddresses": [share["email"]]},
            Message={
                "Subject": {"Data": "Meowify: you've got 10 more credits!"},
                "Body": {"Text": {"Data":
                    f"Thanks for sharing Meowify!\n\n"
                    f"We've added 10 credits to your account. Go make some cat covers:\n{SITE_URL}\n\n-- Meowify"
                }},
            },
        )
    except Exception as e:
        print(f"Approval email failed: {e}")

    return HTMLResponse(f"<p>✓ Approved! Added 10 credits to {share['email']}. <a href='/metrics'>Back to metrics</a></p>")

# ── Routes: debug & metrics ────────────────────────────────────────────────────
@app.get("/debug/jobs")
async def debug_jobs(request: Request):
    user = current_user(request)
    if not user or user["email"] != ADMIN_EMAIL:
        raise HTTPException(403)
    import json
    return HTMLResponse(f"<pre>{json.dumps({k: {**v, 'files': list(v['files'].keys())} for k, v in JOBS.items()}, indent=2, default=str)}</pre>")

@app.get("/metrics")
async def metrics(request: Request):
    user = current_user(request)
    if not user or user["email"] != ADMIN_EMAIL:
        raise HTTPException(403)
    users = all_users()
    jobs_list = [
        {
            "id":     jid,
            "owner":  j.get("owner_email", ""),
            "title":  (j.get("video_info") or {}).get("title", j.get("source_url", "—")),
            "status": j["status"],
            "step":   j["step"],
            "tracks": len(j.get("suno_tracks", [])),
            "error":  j.get("error") or j.get("suno_error") or "",
        }
        for jid, j in JOBS.items()
    ]
    counts = {"running": 0, "done": 0, "error": 0}
    for j in jobs_list:
        counts[j["status"]] = counts.get(j["status"], 0) + 1
    return templates.TemplateResponse(request, "metrics.html", {
        "user": user,
        "users": users,
        "jobs": list(reversed(jobs_list)),
        "counts": counts,
        "total_users": len(users),
        "total_credits_remaining": sum(u["credits"] for u in users if u["email"] != ADMIN_EMAIL),
        "pending_shares": pending_shares(),
    })

# ── Routes: admin ──────────────────────────────────────────────────────────────
@app.get("/admin/users")
async def admin_users(request: Request):
    user = current_user(request)
    if not user or user["email"] != ADMIN_EMAIL:
        raise HTTPException(403)
    return templates.TemplateResponse(request, "admin_users.html", {
        "user": user, "banner": BANNER,
        "users": all_users(),
    })

@app.post("/admin/credits")
async def admin_add_credits(request: Request, email: str = Form(...), amount: int = Form(5)):
    user = current_user(request)
    if not user or user["email"] != ADMIN_EMAIL:
        raise HTTPException(403)
    con = _db()
    con.execute("UPDATE users SET credits=credits+? WHERE email=?", (amount, email))
    con.commit()
    con.close()
    return RedirectResponse("/admin/users", status_code=303)
