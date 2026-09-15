#!/usr/bin/env python3
import os
import sys
import shutil

os.chdir('/home/ubuntu/meowify-v2')
sys.path.insert(0, '/home/ubuntu/meowify-v2')

from downloader import COOKIES_FILE

yt_dlp_bin = os.path.join(os.path.dirname(sys.executable), "yt-dlp")
if not os.path.exists(yt_dlp_bin):
    yt_dlp_bin = shutil.which("yt-dlp") or "yt-dlp"

url = 'https://www.youtube.com/watch?v=beTbA6yOPdw'

cmd = [yt_dlp_bin]
if os.path.exists(COOKIES_FILE):
    cmd += ["--cookies", COOKIES_FILE]
cmd += [
    "--no-js-runtimes", "--js-runtimes", "node",
    "--remote-components", "ejs:github",
    "--dump-json", "--no-playlist",
    url,
]

print("Command being run:")
print(" ".join(cmd))
print()
print(f"COOKIES_FILE: {COOKIES_FILE}")
print(f"File exists: {os.path.exists(COOKIES_FILE)}")
print(f"File readable: {os.access(COOKIES_FILE, os.R_OK)}")
print()

# Try running it
import subprocess
result = subprocess.run(cmd, capture_output=True, text=True)
print(f"Return code: {result.returncode}")
if result.stderr:
    print(f"Stderr (last 3 lines):")
    for line in result.stderr.strip().split('\n')[-3:]:
        print(f"  {line}")
