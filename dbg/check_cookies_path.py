#!/usr/bin/env python3
import os
import glob

os.chdir('/home/ubuntu/meowify-v2')

from downloader import COOKIES_FILE

print(f"COOKIES_FILE path: {COOKIES_FILE}")
print(f"File exists: {os.path.exists(COOKIES_FILE)}")

# Check what directory downloader.py is in
import downloader
print(f"downloader.py location: {downloader.__file__}")
print(f"Directory: {os.path.dirname(os.path.abspath(downloader.__file__))}")

# List files in that directory
files = glob.glob(os.path.dirname(os.path.abspath(downloader.__file__)) + "/*.txt")
print(f"Text files in that dir: {files}")

# Check if symlink
if os.path.islink(COOKIES_FILE):
    print(f"COOKIES_FILE is a symlink pointing to: {os.readlink(COOKIES_FILE)}")
