# Meowify
> github.com/shlomota/meowify

![Meowify](https://cdn-images-1.medium.com/v2/resize:fit:800/1*0yOLabnolhlZDzShFH2vdg.png)

Turn any YouTube song into a cat meow cover — powered by Demucs, librosa, and Suno AI.

---

## How it works

### 1. Download
yt-dlp pulls the audio from a YouTube URL and converts it to MP3. Only the audio stream is downloaded — no video.

### 2. Separate vocals from instrumental (Demucs)
[Demucs](https://github.com/facebookresearch/demucs) (`htdemucs` model) splits the track into a clean vocal stem and an instrumental stem. This runs locally and takes a couple of minutes on CPU.

### 3. Extract the chorus
Processing a full 3-4 minute song is slow and expensive. Instead we extract a ~30 second section — ideally the first chorus, which is the most recognisable part of the song.

Chorus detection works by computing RMS energy in 1-second windows across the first 3 minutes of audio, then finding the loudest sustained 30-second block between the 30s and 90s mark. If that block is more than 15% louder than the default window (45s–75s), it is used instead, starting 1 second early to catch the chorus pick-up. Otherwise we fall back to the default 45s–75s window.

### 4. Detect melody and synthesise meows (librosa)
librosa's `pyin` algorithm analyses the vocal stem to extract the fundamental frequency (F0) frame by frame. Consecutive voiced frames are grouped into notes, each with a start time, duration, and pitch.

For each detected note, a real cat meow sample is pitch-shifted and time-stretched to match that note's F0 and duration, then placed at the correct position in the timeline. This produces a "meow track" that follows the original melody.

> **Limitation:** `pyin` works well for clean solo vocals but can miss notes or produce inaccurate pitches on busy or reverberant recordings. The result is an approximation of the melody, not a perfect transcription.

> **Why this is hard historically:** Meow covers have existed for years but typically involved significant manual effort — hiring a vocalist to meow a melody note by note, or transcribing a song to MIDI and replacing the instrument with a meow sample. The MIDI approach breaks down on vocals because vocals-to-MIDI conversion is notoriously inaccurate: human voices slide between notes, use vibrato, and sit in a dense mix. What makes Suno particularly valuable here is that it is tolerant of this noise — it interprets the melodic intent from an imperfect audio reference and produces a clean musical result, which a rigid MIDI pipeline cannot do.

### 5. Mix
Three stems are mixed together:
- **Meow track** — synthesised meows following the melody
- **Original vocals** — kept at low volume (default 0.5) to reinforce the melody where the meow synthesis is imperfect
- **Instrumental** — the backing track

Both vocal stems are pitched up one octave by default to sit in a cat-like register. The mix is exported as a WAV.

> Overlaying the original vocals at reduced volume turns out to be important: without it, Suno in the next step has a harder time following the melody because the meow synthesis alone isn't always accurate enough.

### 6. Upload to Suno via kie.ai
Suno AI does not offer a public API directly. We use [kie.ai](https://kie.ai), a third-party service that wraps Suno's API.

The locally meowified WAV is uploaded to S3 to get a publicly accessible URL, then passed to the kie.ai `/upload-cover` endpoint. The lyrics are set to `[Chorus]` followed by "meow" repeated N times (default 120). Suno uses the uploaded audio as a melodic reference and generates a full cover singing those lyrics to the melody.

> **Content detection:** Suno has built-in fingerprinting that may identify and block well-known songs. If a submission is rejected, try changing the key (pitch shift) or trimming a different section of the song before uploading.

> **Model choice:** V5.5 produces noticeably more accurate and higher-quality results than V4.5. V5.5 is the default.

### 7. Result
The generated MP3s are downloaded and shown in the app with audio players and download buttons. All intermediate stems are also available for download.

---

## Setup

### Requirements
- Python 3.11
- ffmpeg (required by yt-dlp and Demucs)

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
The pipeline needs a reference meow recording to sample from. Place a WAV file at:

```
cat_samples/separated/htdemucs/cat_meow_ref_trim/vocals_10s.wav
```

Any short (5-15s) recording of a single cat meowing works. Run it through Demucs first to isolate the meow from background noise:

```bash
python -m demucs --two-stems vocals -n htdemucs --out cat_samples/separated cat_samples/your_meow.wav
```

### API keys
Create a `.env` file in the project root:

```
KIE_API_KEY=your_kie_ai_key
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
```

- **kie.ai key:** Sign up at [kie.ai](https://kie.ai) and get a key from the API Key page.
- **AWS credentials:** Used only to temporarily host the audio file so Suno can download it. A standard IAM user with S3 read/write on one bucket is sufficient. The file is a few MB and costs fractions of a cent per run.

### Run

```bash
source .venv/bin/activate
streamlit run app.py
```

---

## Project structure

```
app.py              — Streamlit UI and full pipeline orchestration
downloader.py       — YouTube download via yt-dlp
meowify_v2.py       — Melody detection, meow synthesis, and audio mixing
kie_suno_api.yaml   — OpenAPI spec for the kie.ai Suno upload-cover endpoint
requirements.txt
```

---

## Possible next steps

- **Telegram bot** — wrap the pipeline in a Telegram bot so users can send a YouTube link and get a meow cover back in a few minutes, no UI needed.
- **Auto-upload to Spotify** — use a distributor like DistroKid to automatically publish generated covers as a novelty cat music channel.
- **Other voices** — the pipeline is not cat-specific. Swap the reference sample for any voice (dog barks, a baby, a specific person) and the same pipeline produces covers in that voice. "woof" covers. "quack" covers. The `[Chorus]` + repeated-word lyrics trick works for any single repeating sound.
