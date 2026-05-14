# Meowify — Teaching AI to Meow Your Favorite Songs

*A weekend project born from my kids' obsession with meowing*

---

My kids go through phases. Right now, the phase is meowing. Not just occasionally, constantly. At dinner, in the car, instead of answering questions. Full meow conversations. I've had entire exchanges where I ask "did you brush your teeth?" and receive only "meow." It's contagious too. We've all started doing it. Popular songs in our house now get sung entirely in meow, melody intact, lyrics replaced.

So naturally, I started wondering: are there actual meow covers of popular songs?

A quick YouTube search led me down a rabbit hole. I found **Bongo Cat**, an internet meme turned legitimate music creator who's been putting out surprisingly polished meow covers of popular songs. There's a [meow cover of APT by ROSÉ & Bruno Mars](https://www.youtube.com/watch?v=pxISmahJ-4A) and a [meow cover of Billie Eilish's "What Was I Made For"](https://www.youtube.com/watch?v=3-y0p0GL4TI) that are genuinely impressive. The melody is intact, it's recognizable, and it's unmistakably cat.

![Bongo Cat](bongocat.png)

I played them for my kids. They lost their minds.

## How Meow Covers Used to Be Made

It turns out meow covers have existed for a while, but historically they involved a surprising amount of manual effort. Some creators literally hired a singer to meow the melody of a song note by note, matching pitch and timing to the original. Others took a different route: find or transcribe the MIDI file for the song, then use a sample library or synthesizer to play the melody using a cat meow sound.

The MIDI route sounds clean in theory. In practice, the hard part is always the vocals. Instrumental parts are relatively straightforward to transcribe. But converting a sung vocal track to MIDI, extracting the exact pitches and timings a human voice is singing, is notoriously inaccurate. Human voices slide between notes, use vibrato, and are mixed with reverb and compression that makes clean pitch extraction very difficult. Most vocals-to-MIDI tools produce a messy, imprecise result.

What surprised me building Meowify is that Suno handles this gracefully. Even with the noisy, imperfect meow synthesis we generate from `pyin` pitch detection, Suno somehow interprets the melodic intent and produces a clean, musical output. It is tolerant of the noise in a way that a rigid MIDI pipeline is not. That tolerance is what makes the whole approach work.

Then I thought: what if I could make one for any song? Automatically? Paste a YouTube link, get a meow cover back.

That became **Meowify**, a pipeline built on Demucs for vocal separation, librosa for melody detection, and Suno AI to do the actual singing. In cat.

---

## The Plan

The idea is simple in theory:

1. Download a song from YouTube
2. Separate the vocals from the instrumental
3. Detect the melody from the vocals
4. Replace each note with a cat meow pitched to match
5. Feed the result to an AI that sings it back properly

Simple in theory. Wildly fiddly in practice.

---

## Step 1: Getting the Audio

[yt-dlp](https://github.com/yt-dlp/yt-dlp) makes this straightforward. We grab audio-only, no need to download video.

```python
import yt_dlp

def download_youtube_as_mp3(url: str, output_dir: str = "downloads") -> str:
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": f"{output_dir}/%(title)s.%(ext)s",
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return ydl.prepare_filename(info).replace(".webm", ".mp3").replace(".m4a", ".mp3")
```

---

## Step 2: Separating Vocals with Demucs

Facebook Research's [Demucs](https://github.com/facebookresearch/demucs) does vocal separation locally. We use the `htdemucs` model with `--two-stems vocals` which gives us a clean vocal track and a clean instrumental.

```python
import subprocess, sys

subprocess.run([
    sys.executable, "-m", "demucs",
    "--two-stems", "vocals",
    "-n", "htdemucs",
    "--out", "work",
    "downloads/song.mp3"
], check=True)
# Outputs: work/htdemucs/song/vocals.wav
#          work/htdemucs/song/no_vocals.wav
```

This takes a few minutes on CPU. On a MacBook with Apple Silicon it's manageable. The quality is genuinely impressive. Demucs can cleanly pull vocals from heavily produced pop tracks.

---

## Step 3: Finding the Chorus (Without Listening to the Whole Song)

Processing a full 3-4 minute song is slow and expensive at every step downstream. The chorus is almost always the most recognizable 30 seconds, so we extract just that.

The detection is energy-based: compute RMS loudness in 1-second windows across the first 3 minutes, find the loudest sustained 30-second block between 30s and 90s. If it's more than 15% louder than the default (45s-75s), use it, and start 1 second before the detected point to catch the pick-up beat.

```python
import librosa
import numpy as np

def detect_chorus(audio_path: str, default_start: float = 45.0,
                  duration: float = 30.0) -> tuple[float, str]:
    y, sr = librosa.load(audio_path, sr=22050, mono=True, duration=180.0)
    rms = librosa.feature.rms(y=y, frame_length=sr * 2, hop_length=sr)[0]

    w = int(duration)
    def_idx = int(default_start)
    def_e = float(np.mean(rms[def_idx : def_idx + w]))

    best_idx, best_e = def_idx, def_e
    for s in range(30, min(71, int(len(y) / sr - duration))):
        if s + w >= len(rms):
            break
        e = float(np.mean(rms[s : s + w]))
        if e > best_e:
            best_e, best_idx = e, s

    if best_idx != def_idx and best_e > 1.15 * def_e:
        adjusted = max(0.0, float(best_idx) - 1.0)
        return adjusted, f"auto-detected at {best_idx}s"
    return default_start, "default 45s"
```

Not sophisticated, but it works reliably on pop songs where the chorus is almost always the loudest part.

---

## Step 4: Detecting the Melody

This is where things get interesting, and imperfect.

librosa's `pyin` algorithm extracts the fundamental frequency (F0) frame by frame from the vocal stem. We group consecutive voiced frames into notes, each with a start time, duration, and pitch.

```python
def detect_notes(vocals_path: str, sr: int = 44100,
                 min_note_duration: float = 0.03) -> list[dict]:
    y, _ = librosa.load(vocals_path, sr=sr, mono=True)
    hop = 512
    f0, voiced, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C6'),
        sr=sr,
        frame_length=1024,
        hop_length=hop,
    )
    times = librosa.times_like(f0, sr=sr, hop_length=hop)

    notes, in_note, note_f0s = [], False, []
    for i, (t, freq, v) in enumerate(zip(times, f0, voiced)):
        if v and not in_note:
            in_note, note_start, note_f0s = True, i, [freq]
        elif v and in_note:
            note_f0s.append(freq)
        elif not v and in_note:
            in_note = False
            duration = times[i] - times[note_start]
            if duration > min_note_duration and note_f0s:
                notes.append({
                    'start': times[note_start],
                    'end': times[i],
                    'duration': duration,
                    'f0': float(np.median(note_f0s)),
                })
    return notes
```

**The honest caveat:** `pyin` works well for clean solo vocals, but pop productions are rarely clean. Reverb, harmonies, and compression all muddy the F0 detection. The result is an approximation, good enough to be recognizable, not good enough to be perfect. This is fine because Suno picks up the slack.

---

## Step 5: Synthesizing the Meows

We take a reference meow recording, run it through Demucs to isolate it, then for each detected note we pitch-shift and time-stretch the meow to match that note's frequency and duration.

```python
def place_meows(notes: list[dict], meow: np.ndarray,
                meow_f0: float, total_duration: float,
                sr: int = 44100) -> np.ndarray:
    output = np.zeros(int(total_duration * sr), dtype=np.float32)

    for note in notes:
        semitones = 12 * np.log2(note['f0'] / meow_f0)

        stretch_ratio = (len(meow) / sr) / note['duration']
        if 0.1 < stretch_ratio < 10:
            stretched = librosa.effects.time_stretch(meow, rate=stretch_ratio)
        else:
            stretched = meow.copy()

        shifted = librosa.effects.pitch_shift(stretched, sr=sr, n_steps=semitones)

        start_sample = int(note['start'] * sr)
        chunk = shifted[:int(note['duration'] * sr)]
        end = min(start_sample + len(chunk), len(output))
        output[start_sample:end] += chunk[:end - start_sample]

    return output
```

---

## Step 6: Mixing

Three stems go into the final mix:

- **Meow track** — the synthesized meows following the melody
- **Original vocals** — kept quiet (0.5 volume) to fill in where meow synthesis misses
- **Instrumental** — the backing track unchanged

Both vocal stems get pitched up one octave so they sit in a cat-like register.

```python
# Pitch up vocals and meow track by 12 semitones (1 octave)
meow_track = librosa.effects.pitch_shift(meow_track, sr=sr, n_steps=12.0)
voc = librosa.effects.pitch_shift(voc, sr=sr, n_steps=12.0)

n = min(len(meow_track), len(inst), len(voc))
mixed = 1.0 * inst[:n] + 0.5 * voc[:n] + 0.7 * meow_track[:n]
```

Keeping the original vocals in the mix at low volume turned out to be crucial. Without them, the meow synthesis alone isn't accurate enough for Suno to recognize the melody in the next step.

---

## Step 7: Suno AI Sings the Meows

This is the magic step. We hand the meowified audio to Suno AI and tell it to sing "meow" to the melody.

**The catch:** Suno doesn't have a public API. We use [kie.ai](https://kie.ai), a third-party service that wraps Suno's upload-cover endpoint.

We upload our WAV to S3 to get a public URL, then POST to kie.ai with the audio reference and our lyrics, `[Chorus]` followed by "meow" repeated 120 times. Suno treats the audio upload as a melodic template and generates a cover that sings those lyrics to it.

```python
import requests

def suno_submit(upload_url: str, kie_key: str, meow_count: int,
                style: str, title: str, model: str = "V5_5") -> str:
    lyrics = "[Chorus]\n" + "\n".join(["meow"] * meow_count)
    r = requests.post(
        "https://api.kie.ai/api/v1/generate/upload-cover",
        headers={"Authorization": f"Bearer {kie_key}"},
        json={
            "uploadUrl": upload_url,
            "prompt": lyrics,
            "style": style,
            "title": title,
            "customMode": True,
            "instrumental": False,
            "model": model,
            "callBackUrl": "https://api.example.com/callback",
        },
        timeout=30,
    )
    data = r.json()
    return data["data"]["taskId"]
```

Then we poll every 10 seconds until the task completes:

```python
def poll_until_done(task_id: str, kie_key: str) -> list[dict]:
    terminal = {"SUCCESS", "FAILED", "ERROR", "FIRST_SUCCESS"}
    while True:
        r = requests.get(
            "https://api.kie.ai/api/v1/generate/record-info",
            params={"taskId": task_id},
            headers={"Authorization": f"Bearer {kie_key}"},
        )
        data = r.json()["data"]
        if data["status"] in terminal:
            return data["response"]["sunoData"]
        time.sleep(10)
```

**One important note about Suno:** it has content fingerprinting that may identify and block well-known songs. If a submission gets rejected, try processing a different 30-second section or applying a small pitch shift to the audio before uploading. We handle this automatically in the app with a retry on 413 errors.

**On model quality:** Suno V5.5 produces dramatically better results than V4.5. The vocal accuracy, clarity, and the way it follows the melodic reference are all noticeably improved. Always use V5.5.

---

## What It Looks Like Running

Here's an example of the verbose log output from a successful run:

```
[23:41:02] Title: ROSÉ & Bruno Mars - APT. (Official Music Video)
[23:41:02] Duration: 213s
[23:41:04] Downloaded: downloads/ROSE__Bruno_Mars_-_APT_Official_Music_Video.mp3 (4.2 MB)
[23:41:04] Using cached Demucs output
[23:41:06] Chorus: auto-detected at 52s, starting 1s early at 51s (energy ↑22% vs default)
[23:41:06] Clips extracted: chorus_vocals.wav, chorus_instrumental.wav
[23:41:06] [1/6] Loading meow sample...
[23:41:06]        Meow: 0.54s, base F0=523Hz
[23:41:07] [2/6] Detecting notes in vocals...
[23:41:07]        Found 53 notes
[23:41:07] [3/6] Placing meows on melody...
[23:41:34] [4/6] Pitching: instrumental +0st, vocals/meow +12st...
[23:41:51] [5/6] Speeding up everything by 1.2x...
[23:41:55] [6/6] Mixing and saving...
[23:41:56] EQ + noise masking applied
[23:41:58] Uploaded: s3://bucket/meowify/APT_meowified_masked.wav
[23:41:59] Suno task submitted (attempt 1): 73fad7d6647b0af6
[23:42:09] Suno poll (attempt 1): PENDING (10s)
[23:42:19] Suno poll (attempt 1): TEXT_SUCCESS (20s)
[23:43:41] Suno poll (attempt 1): FIRST_SUCCESS (102s)
[23:43:41] Track 1 downloaded: output/APT_suno_1.mp3
```

The whole pipeline from URL to finished cover takes around 2-3 minutes. Suno generates in about a minute, and the local processing (Demucs, pitch detection, mixing) takes another minute or two.

---

## A Note on Copyright

This project sits in genuinely complicated legal territory, and it's worth being honest about that.

Cover songs have always existed in a gray area. In most countries, a human artist can legally record a cover of any song by obtaining a mechanical license and paying royalties to the songwriter. The arrangement is different from the original recording, so it doesn't infringe the sound recording copyright, only the underlying composition copyright, which is handled through the license.

AI changes this calculus significantly. Suno and similar tools have faced major lawsuits from record labels. Sony, Universal, and Warner filed suit against Suno and Udio in 2024, alleging that these models were trained on copyrighted recordings without permission. The core question, whether training an AI on copyrighted music constitutes infringement and whether the output of that AI infringes, is still being litigated.

Suno's own content fingerprinting (the 413 error we work around) is partly a response to this pressure, an attempt to avoid generating material that is too close to existing recordings. The fact that we have to actively disguise our audio to get past it is a small illustration of the larger tension at play.

For a personal project and a bit of fun, meowified covers of popular songs feel harmless and genuinely transformative in a way that seems clearly in the spirit of parody and fair use. But the legal framework for AI-generated music is still being written, and the outcomes of the ongoing lawsuits will matter enormously for the whole space.

---

## The Streamlit App

All of this is wired together in a Streamlit UI. Paste a YouTube URL, hit the button, and the pipeline runs end to end, downloading, separating, detecting the chorus, synthesizing meows, uploading, and returning the Suno-generated cover with audio players and download buttons for every intermediate step.

![Meowify app screenshot](screenshot.png)

The whole thing runs on a laptop. The Suno generation takes 2-4 minutes. Everything else is a couple of minutes on CPU.

---

## What's Next

A few directions this could go:

**Telegram bot.** Wrap the pipeline in a Telegram bot so anyone can send a YouTube link and get a meow cover back in a few minutes, no UI or setup required.

**Auto-upload to Spotify.** Run a nightly job that picks trending songs, meowifies them, and publishes them through a distributor like DistroKid. A novelty cat music channel that auto-publishes covers could be genuinely interesting.

**Other voices.** The pipeline isn't cat-specific. Swap the meow reference sample for any voice, a dog barking, a baby, a specific person, and you get covers in that voice. The `[Chorus]` + repeated-word lyrics trick works for any single repeating sound.

---

My kids have listened to the Meowify outputs on repeat. I'm calling that a success.

The full code is at [github.com/shlomota/meowify](https://github.com/shlomota/meowify).

---

If you'd like to try it out or have ideas for other voices or expansions, let me know in the comments.
