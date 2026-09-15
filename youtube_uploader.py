"""Upload videos to YouTube Meowify channel."""

import os
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload


def get_youtube_service(refresh_token):
    """Get YouTube API service using refresh token."""
    creds = Credentials(
        token=None,
        refresh_token=refresh_token,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=os.getenv("GOOGLE_CLIENT_ID"),
        client_secret=os.getenv("GOOGLE_CLIENT_SECRET")
    )
    creds.refresh(Request())
    return build("youtube", "v3", credentials=creds)


def upload_to_youtube(mp3_path: str, title: str, description: str, refresh_token: str,
                     thumbnail_url: str = "", tags: list = None) -> str:
    """Upload MP3 as video to YouTube.

    Returns: YouTube video URL
    """
    if not refresh_token:
        raise RuntimeError("YOUTUBE_REFRESH_TOKEN not configured")

    if not os.path.exists(mp3_path):
        raise FileNotFoundError(f"MP3 not found: {mp3_path}")

    youtube = get_youtube_service(refresh_token)

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

    media = MediaFileUpload(mp3_path, mimetype="audio/mpeg", resumable=True)

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


if __name__ == "__main__":
    refresh_token = os.getenv("YOUTUBE_REFRESH_TOKEN")
    if not refresh_token:
        print("Error: YOUTUBE_REFRESH_TOKEN not set")
    else:
        print(f"✓ YouTube uploader ready (token exists)")
