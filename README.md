# Meowify
> meowify.click — github.com/shlomota/meowify

![Meowify](https://cdn-images-1.medium.com/v2/resize:fit:800/1*0yOLabnolhlZDzShFH2vdg.png)

Turn any song into a cat meow cover — powered by Demucs, librosa, and Suno AI.

Live at **[meowify.click](https://meowify.click)**

---

## How it works

### 1. Download
yt-dlp pulls the audio from a URL and converts it to MP3. Only the audio stream is downloaded — no video.

### 2. Separate vocals from instrumental (Demucs)
[Demucs](https://github.com/facebookresearch/demucs) (`htdemucs` model) splits the track into a clean vocal stem and an instrumental stem. This runs locally and takes a couple of minutes on CPU.

### 3. Extract the chorus
Processing a full 3-4 minute song is slow and expensive. Instead we extract a ~30 second section — ideally the first chorus, which is the most recognisable part of the song.

Chorus detection works by computing RMS energy in 1-second windows across the first 3 minutes of audio, then finding the loudest sustained 30-second block between the 30s and 90s mark. If that block is more than 15% louder than the default window (45s–75s), it is used instead, starting 1 second early to catch the chorus pick-up. Otherwise we fall back to the default 45s–75s window.

### 4. Detect melody and synthesise meows (librosa)
librosa's `pyin` algorithm analyses the vocal stem to extract the fundamental frequency (F0) frame by frame. Consecutive voiced frames are grouped into notes, each with a start time, duration, and pitch.

For each detected note, a real cat meow sample is pitch-shifted and time-stretched to match that note's F0 and duration, then placed at the correct position in the timeline. This produces a "meow track" that follows the original melody.

> **Limitation:** `pyin` works well for clean solo vocals but can miss notes or produce inaccurate pitches on busy or reverberant recordings. The result is an approximation of the melody, not a perfect transcription.

### 5. Mix
Three stems are mixed together:
- **Meow track** — synthesised meows following the melody
- **Original vocals** — kept at low volume (default 0.5) to reinforce the melody where the meow synthesis is imperfect
- **Instrumental** — the backing track

Both vocal stems are pitched up one octave by default to sit in a cat-like register. The mix is exported as a WAV.

### 6. Upload to Suno via kie.ai
The locally meowified WAV is uploaded to S3 to get a publicly accessible URL, then passed to the [kie.ai](https://kie.ai) Suno cover endpoint. The lyrics are set to `[Chorus]` followed by "meow" repeated N times (default 120). Suno uses the uploaded audio as a melodic reference and generates a full cover.

> **Content detection:** Suno has built-in fingerprinting that may block well-known songs. If a submission is rejected, try a pitch shift or a different section of the song.

### 7. Result
Generated MP3s are shown with audio players and download buttons. All intermediate stems are also available for download. Users receive an email when their cover is ready.

---

## Stack

- **Backend:** FastAPI + Jinja2 + htmx
- **Auth:** Google OAuth 2.0
- **Database:** SQLite (users, credits, jobs, share approvals)
- **Email:** Amazon SES
- **Storage:** S3 (temporary audio hosting for Suno)
- **Audio:** Demucs, librosa, soundfile, scipy
- **AI:** Suno via kie.ai

---

## Setup

### Requirements
- Python 3.11
- ffmpeg

```bash
brew install ffmpeg        # macOS
```

### Install dependencies

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Cat meow reference sample
Place a WAV file at:

```
cat_samples/separated/htdemucs/cat_meow_ref_trim/vocals_10s.wav
```

Run it through Demucs first to isolate the meow:

```bash
python -m demucs --two-stems vocals -n htdemucs --out cat_samples/separated cat_samples/your_meow.wav
```

### Environment variables
Create a `.env` file in the project root:

```
GOOGLE_CLIENT_ID=...
GOOGLE_CLIENT_SECRET=...
OAUTH_REDIRECT_URI=http://localhost:8503/oauth/callback
SESSION_SECRET=...
KIE_API_KEY=...
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
```

- **Google credentials:** Create an OAuth 2.0 client at [console.cloud.google.com](https://console.cloud.google.com)
- **kie.ai key:** Sign up at [kie.ai](https://kie.ai)
- **AWS credentials:** IAM user with SES send and S3 read/write on one bucket

### Run

```bash
source .venv/bin/activate
uvicorn server:app --host 127.0.0.1 --port 8503 --workers 1
```

> Single worker required — the in-memory job dict lives in the process. Completed jobs are persisted to SQLite and survive restarts.

---

## Project structure

```
server.py               — FastAPI app: routes, auth, job orchestration, email
downloader.py           — Audio download via yt-dlp
meowify_v2.py           — Melody detection, meow synthesis, audio mixing
templates/              — Jinja2 templates (base, login, index, job, metrics, credits)
static/                 — CSS, htmx, OG image, robots.txt
kie_suno_api.yaml       — OpenAPI spec for kie.ai Suno endpoint
requirements.txt
```

---

## Possible next steps

- **Telegram bot** — wrap the pipeline in a bot so users can send a link and get a cover back in minutes
- **Auto-upload to Spotify** — publish generated covers via a distributor like DistroKid
- **Other voices** — swap the reference sample for any voice (dog barks, a baby) — the same pipeline works for any repeating sound
