#!/usr/bin/env python3
"""Test the form submission logic"""

# Test cases
test_cases = [
    {"url": "https://youtube.com/watch?v=123", "audio_filename": None, "should_work": True, "desc": "URL only"},
    {"url": "", "audio_filename": "song.mp3", "should_work": True, "desc": "File only"},
    {"url": "https://youtube.com/watch?v=123", "audio_filename": "song.mp3", "should_work": True, "desc": "Both URL and file"},
    {"url": "", "audio_filename": None, "should_work": False, "desc": "Neither URL nor file"},
    {"url": "", "audio_filename": "", "should_work": False, "desc": "Empty file name"},
]

for test in test_cases:
    url = test["url"]
    audio_filename = test["audio_filename"]

    # Simulate the logic
    local_mp3_path = None
    job_id = "test123"

    if audio_filename:  # simulating: audio and audio.filename
        local_mp3_path = "uploaded_file.mp3"
        url = ""

    if not url and not local_mp3_path:  # elif not url: but local_mp3_path should be set
        result = "ERROR"
    else:
        result = "OK"

    expected = "OK" if test["should_work"] else "ERROR"
    status = "✓" if result == expected else "✗"
    print(f"{status} {test['desc']}: {result} (expected {expected})")
