#!/bin/bash
set -e

SERVER="ubuntu@carlebot.us"
KEY="$HOME/Downloads/taami.pem"
REMOTE_DIR="/home/ubuntu/meowify"
COOKIE_FILE="yt_cookies.txt"

echo "==> Exporting YouTube cookies from local Chrome profile..."

yt-dlp \
  --cookies-from-browser chrome \
  --cookies "$COOKIE_FILE" \
  "https://www.youtube.com/watch?v=dQw4w9WgXcQ" \
  --skip-download >/dev/null 2>&1

echo "==> Uploading cookies to server..."

scp -i "$KEY" "$COOKIE_FILE" "$SERVER:$REMOTE_DIR/"

echo "==> Testing yt-dlp on remote server..."

ssh -i "$KEY" "$SERVER" << 'EOF'
set -e

pip install -U yt-dlp >/dev/null 2>&1 || true

yt-dlp \
  --cookies /home/ubuntu/meowify/yt_cookies.txt \
  --skip-download \
  "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

echo
echo "SUCCESS: cookies uploaded and yt-dlp authenticated correctly."
EOF
