"""Upload videos to YouTube Meowify channel."""

import os
import subprocess
import tempfile
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload


def get_youtube_service(refresh_token):
    """Get YouTube API service using refresh token."""
    client_id = os.getenv("GOOGLE_CLIENT_ID")
    client_secret = os.getenv("GOOGLE_CLIENT_SECRET")

    if not client_id or not client_secret:
        raise RuntimeError("GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET required")

    creds = Credentials(
        token=None,
        refresh_token=refresh_token,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=client_id,
        client_secret=client_secret
    )
    creds.refresh(Request())
    return build("youtube", "v3", credentials=creds)


def create_video_from_image_and_audio(image_path: str, audio_path: str) -> str:
    """Create MP4 video from static image + audio using ffmpeg.

    Returns: Path to created MP4 file
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    # Create temp MP4 file
    fd, video_path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)

    # Use ffmpeg to create video: loop image for duration of audio
    # -loop 1: loop the image
    # -i image: input image
    # -i audio: input audio
    # -c:v libx264: video codec
    # -c:a aac: audio codec
    # -shortest: stop when shortest input ends
    cmd = [
        "ffmpeg",
        "-loop", "1",
        "-i", image_path,
        "-i", audio_path,
        "-c:v", "libx264",
        "-c:a", "aac",
        "-pix_fmt", "yuv420p",
        "-shortest",
        "-y",  # overwrite
        video_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        if os.path.exists(video_path):
            os.remove(video_path)
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")

    return video_path


def upload_to_youtube(mp3_path: str, title: str, description: str, refresh_token: str,
                     thumbnail_url: str = "", tags: list = None) -> str:
    """Upload video (image + audio) to YouTube.

    Args:
        mp3_path: Path to audio file (MP3 or WAV)
        title: Video title
        description: Video description
        refresh_token: YouTube refresh token
        thumbnail_url: Path to image file (or URL - will use as static image in video)
        tags: Video tags

    Returns: YouTube video URL
    """
    if not refresh_token:
        raise RuntimeError("YOUTUBE_REFRESH_TOKEN not configured")

    if not os.path.exists(mp3_path):
        raise FileNotFoundError(f"Audio not found: {mp3_path}")

    if not thumbnail_url or not os.path.exists(thumbnail_url):
        raise FileNotFoundError(f"Image not found: {thumbnail_url}")

    youtube = get_youtube_service(refresh_token)

    # Create video from image + audio
    video_path = create_video_from_image_and_audio(thumbnail_url, mp3_path)

    try:
        request_body = {
            "snippet": {
                "title": title,
                "description": description,
                "tags": tags or ["meow", "cover", "suno"],
                "categoryId": "10",  # Music category
            },
            "status": {
                "privacyStatus": "public"
            }
        }

        media = MediaFileUpload(video_path, mimetype="video/mp4", resumable=True)

        request = youtube.videos().insert(
            part="snippet,status",
            body=request_body,
            media_body=media
        )

        response = request.execute()
        video_id = response.get("id")

        if not video_id:
            raise RuntimeError(f"YouTube upload failed: {response}")

        return f"https://youtube.com/watch?v={video_id}"

    finally:
        # Clean up temp video file
        if os.path.exists(video_path):
            os.remove(video_path)


if __name__ == "__main__":
    refresh_token = os.getenv("YOUTUBE_REFRESH_TOKEN")
    if not refresh_token:
        print("Error: YOUTUBE_REFRESH_TOKEN not set")
    else:
        print(f"✓ YouTube uploader ready (token exists)")
