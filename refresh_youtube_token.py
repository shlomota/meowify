#!/usr/bin/env python3
"""Refresh YouTube OAuth token."""
import os
import sys
import json
from pathlib import Path
from google_auth_oauthlib.flow import InstalledAppFlow

# Load env to get client ID/secret
env = {}
for line in Path(".env").read_text().splitlines():
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip()

client_id = env.get("GOOGLE_CLIENT_ID")
client_secret = env.get("GOOGLE_CLIENT_SECRET")

if not client_id or not client_secret:
    print("Error: GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET not in .env")
    sys.exit(1)

print("Getting YouTube authorization...\n")

# Create credentials.json for OAuth flow with explicit redirect URI
creds_json = {
    "installed": {
        "client_id": client_id,
        "client_secret": client_secret,
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "redirect_uris": ["http://localhost:8080/"]
    }
}

with open("google_creds_temp.json", "w") as f:
    json.dump(creds_json, f)

try:
    # Use InstalledAppFlow with explicit redirect_uri
    flow = InstalledAppFlow.from_client_secrets_file(
        'google_creds_temp.json',
        scopes=['https://www.googleapis.com/auth/youtube']
    )

    # Run local server on port 8080
    creds = flow.run_local_server(port=8080, open_browser=True)

    if creds.refresh_token:
        print(f"\n✓ Success!\n")
        print(f"YOUTUBE_REFRESH_TOKEN={creds.refresh_token}\n")
        print(f"Add this to /home/ubuntu/meowify-v2/.env and restart:")
        print(f"  sudo systemctl restart meowify")
    else:
        print("Error: No refresh token obtained")
        sys.exit(1)

except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)

finally:
    # Clean up
    if os.path.exists("google_creds_temp.json"):
        os.remove("google_creds_temp.json")
