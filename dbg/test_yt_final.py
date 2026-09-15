#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/ubuntu/meowify-v2')

from downloader import get_video_info

try:
    result = get_video_info('https://www.youtube.com/watch?v=beTbA6yOPdw')
    print("✓ SUCCESS!")
    print(f"  Title: {result['title']}")
    print(f"  Duration: {result['duration']}s")
    print(f"  Uploader: {result['uploader']}")
except Exception as e:
    print(f"✗ FAILED: {e}")
