#!/usr/bin/env python3
"""Test image editing with KIE.AI on a real YouTube video."""

import os
from image_editor import edit_album_art

# YouTube video ID: aii3fDdZnrM
video_id = "aii3fDdZnrM"
thumbnail_url = f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"

kie_key = os.getenv("KIE_API_KEY")
if not kie_key:
    print("Error: KIE_API_KEY not set")
    exit(1)

print(f"Testing image edit on: {thumbnail_url}")
print("Prompt: Replace people with cartoon cats, add paws and Meowify logo")
print()

try:
    result_url = edit_album_art(thumbnail_url, kie_key)
    print(f"✓ Success! Edited image: {result_url}")
except Exception as e:
    print(f"✗ Error: {e}")
