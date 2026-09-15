#!/usr/bin/env python3
import os
import sys

os.chdir('/home/ubuntu/meowify-v2')
sys.path.insert(0, '/home/ubuntu/meowify-v2')

from downloader import COOKIES_FILE, get_video_info

print(f"Testing with COOKIES_FILE: {COOKIES_FILE}")
print(f"File exists: {os.path.exists(COOKIES_FILE)}")
print(f"File size: {os.path.getsize(COOKIES_FILE) if os.path.exists(COOKIES_FILE) else 'N/A'}")

# Test get_video_info which should use cookies
try:
    url = "https://www.youtube.com/watch?v=beTbA6yOPdw"
    print(f"\nTesting with URL: {url}")
    info = get_video_info(url)
    print(f"✓ Success! Got info: {info['title']}")
except Exception as e:
    print(f"✗ Error: {e}")
