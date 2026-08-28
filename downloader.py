import json as _json
import os
import re
import time
import urllib.parse
import urllib.request

import requests


def sanitize_filename(name: str) -> str:
    return re.sub(r'[^\w\s-]', '', name).strip().replace(' ', '_')[:80]


def _extract_video_id(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    if 'youtube.com' in parsed.netloc:
        params = urllib.parse.parse_qs(parsed.query)
        return params.get('v', [''])[0]
    elif 'youtu.be' in parsed.netloc:
        return parsed.path.lstrip('/')
    raise ValueError(f"Invalid YouTube URL: {url}")


def get_video_info(url: str) -> dict:
    video_id = _extract_video_id(url)
    try:
        response = requests.get(
            f"https://www.youtube.com/oembed?url=https://youtube.com/watch?v={video_id}&format=json",
            timeout=5
        )
        data = response.json()
        return {
            'title': data.get('title', 'Unknown'),
            'duration': 0,
            'thumbnail_url': data.get('thumbnail_url', ''),
            'uploader': data.get('author_name', ''),
            'id': video_id,
        }
    except Exception as e:
        return {'title': 'Unknown', 'duration': 0, 'thumbnail_url': '', 'uploader': '', 'id': video_id}


def download_youtube_as_mp3(url: str, output_dir: str = "downloads", api_key: str = "") -> str:
    os.makedirs(output_dir, exist_ok=True)

    video_id = _extract_video_id(url)
    if not api_key:
        api_key = os.getenv("RAPIDAPI_KEY", "")
    if not api_key:
        raise RuntimeError("RAPIDAPI_KEY not provided")

    headers = {
        "x-rapidapi-host": "youtube-mp3-audio-video-downloader.p.rapidapi.com",
        "x-rapidapi-key": api_key,
        "Content-Type": "application/json"
    }

    api_url = f"https://youtube-mp3-audio-video-downloader.p.rapidapi.com/get_mp3_download_link/{video_id}?quality=low&wait_until_the_file_is_ready=false"

    download_url = None
    for attempt in range(2):
        try:
            response = requests.get(api_url, headers=headers, timeout=90)
            if response.status_code != 200:
                raise RuntimeError(f"API error: {response.status_code} {response.text}")

            data = response.json()
            download_url = data.get('file') or data.get('reserved_file')
            if not download_url:
                raise RuntimeError(f"No download URL in API response: {data}")
            break
        except requests.exceptions.Timeout as e:
            if attempt == 0:
                continue
            raise RuntimeError(f"API timeout after {attempt + 1} attempts")
        except Exception:
            if attempt == 0:
                continue
            raise

    if not download_url:
        raise RuntimeError("Failed to get download URL from API")

    for attempt in range(60):
        try:
            mp3_response = requests.head(download_url, timeout=5)
            if mp3_response.status_code == 200:
                break
        except Exception:
            pass
        if attempt < 59:
            time.sleep(2)

    info = get_video_info(url)
    safe_name = sanitize_filename(info['title'])
    mp3_path = os.path.join(output_dir, f"{safe_name}.mp3")

    mp3_response = requests.get(download_url, timeout=60)
    if mp3_response.status_code != 200:
        raise RuntimeError(f"Download failed: {mp3_response.status_code}")

    with open(mp3_path, 'wb') as f:
        f.write(mp3_response.content)

    return os.path.abspath(mp3_path)


if __name__ == "__main__":
    url = "https://www.youtube.com/watch?v=ImKzSpGXqOE"
    print(f"Fetching info for: {url}")
    info = get_video_info(url)
    print(f"  Title   : {info['title']}")
    print(f"  Duration: {info['duration']}s")

    print("\nDownloading as MP3...")
    try:
        path = download_youtube_as_mp3(url)
        print(f"\nSaved to: {path}")
    except Exception as e:
        print(f"Error: {e}")
