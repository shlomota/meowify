#!/usr/bin/env python3
"""Get YouTube refresh token for Meowify account."""

from google_auth_oauthlib.flow import InstalledAppFlow

flow = InstalledAppFlow.from_client_secrets_file(
    'youtube_credentials.json',
    scopes=['https://www.googleapis.com/auth/youtube']
)
creds = flow.run_local_server(port=8080)
print(f"\n✓ Authorization successful!\n")
print(f"YOUTUBE_REFRESH_TOKEN={creds.refresh_token}")
print(f"\nAdd this line to your .env file")
