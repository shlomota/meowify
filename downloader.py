import os
import re
import yt_dlp


def sanitize_filename(name: str) -> str:
    return re.sub(r'[^\w\s-]', '', name).strip().replace(' ', '_')[:80]


COOKIES_FILE = os.path.join(os.path.dirname(__file__), "yt_cookies.txt")

def _base_opts() -> dict:
    opts = {}
    if os.path.exists(COOKIES_FILE):
        opts['cookiefile'] = COOKIES_FILE
    return opts


def get_video_info(url: str) -> dict:
    opts = {**_base_opts(), 'quiet': True, 'no_warnings': True, 'extract_flat': False}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    return {
        'title': info.get('title', 'Unknown'),
        'duration': info.get('duration', 0),
        'thumbnail_url': info.get('thumbnail', ''),
        'uploader': info.get('uploader', ''),
        'id': info.get('id', ''),
    }


def download_youtube_as_mp3(url: str, output_dir: str = "downloads") -> str:
    os.makedirs(output_dir, exist_ok=True)

    info = get_video_info(url)
    safe_name = sanitize_filename(info['title'])
    output_template = os.path.join(output_dir, f"{safe_name}.%(ext)s")

    opts = {
        **_base_opts(),
        'format': 'bestaudio/best',
        'outtmpl': output_template,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': False,
        'no_warnings': False,
    }

    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])

    mp3_path = os.path.join(output_dir, f"{safe_name}.mp3")
    if not os.path.exists(mp3_path):
        # fallback: find any mp3 in output_dir modified recently
        candidates = [
            os.path.join(output_dir, f)
            for f in os.listdir(output_dir)
            if f.endswith('.mp3')
        ]
        if not candidates:
            raise FileNotFoundError(f"MP3 not found in {output_dir} after download")
        mp3_path = max(candidates, key=os.path.getmtime)

    return os.path.abspath(mp3_path)


if __name__ == "__main__":
    url = "https://www.youtube.com/watch?v=ekr2nIex040"
    print(f"Fetching info for: {url}")
    info = get_video_info(url)
    print(f"  Title   : {info['title']}")
    print(f"  Duration: {info['duration']}s")
    print(f"  Thumb   : {info['thumbnail_url']}")

    print("\nDownloading as MP3...")
    path = download_youtube_as_mp3(url)
    print(f"\nSaved to: {path}")
