#!/usr/bin/env python3
"""Test YouTube OAuth and show what we get back."""
import json
from pathlib import Path
from google_auth_oauthlib.flow import InstalledAppFlow

# Load env
env = {}
for line in Path(".env").read_text().splitlines():
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip()

client_id = env.get("GOOGLE_CLIENT_ID")
client_secret = env.get("GOOGLE_CLIENT_SECRET")

if not client_id or not client_secret:
    print("✗ Missing GOOGLE_CLIENT_ID or GOOGLE_CLIENT_SECRET in .env")
    exit(1)

print(f"Client ID: {client_id}")
print(f"Client Secret: {client_secret[:20]}...\n")

# Create OAuth flow
creds_config = {
    "installed": {
        "client_id": client_id,
        "client_secret": client_secret,
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "redirect_uris": ["http://localhost:8080/"]
    }
}

with open(".google_creds_test.json", "w") as f:
    json.dump(creds_config, f)

try:
    flow = InstalledAppFlow.from_client_secrets_file(
        '.google_creds_test.json',
        scopes=['https://www.googleapis.com/auth/youtube']
    )

    print("Opening browser for authorization...")
    creds = flow.run_local_server(port=8080, open_browser=True)

    print("\n" + "="*60)
    print("Authorization Response:")
    print("="*60)
    print(f"Access Token: {creds.token[:50]}..." if creds.token else "None")
    print(f"Refresh Token: {creds.refresh_token}" if creds.refresh_token else "None")
    print(f"Token Expiry: {creds.expiry}")
    print(f"Scopes: {creds.scopes}")
    print("="*60)

    if creds.refresh_token:
        print(f"\n✓ SUCCESS! Use this in .env:\n")
        print(f"YOUTUBE_REFRESH_TOKEN={creds.refresh_token}\n")
    else:
        print("\n✗ No refresh token received!")
        print("This usually means the app was already authorized.")
        print("You need to revoke access at: https://myaccount.google.com/permissions")
        print("Then try again.\n")

finally:
    import os
    if os.path.exists(".google_creds_test.json"):
        os.remove(".google_creds_test.json")
