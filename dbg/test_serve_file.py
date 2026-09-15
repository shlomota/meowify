#!/usr/bin/env python3
"""Test file serving for a job"""
import sqlite3
import json
import os

# Load job from DB
con = sqlite3.connect('users.db')
con.row_factory = sqlite3.Row
row = con.execute("SELECT * FROM jobs WHERE id=?", ('460d0100',)).fetchone()
con.close()

if not row:
    print("Job not found!")
    exit(1)

# Parse the files field
files = json.loads(row['files'])
suno_tracks = json.loads(row['suno_tracks'])

print(f"Job: {row['id']}")
print(f"Status: {row['status']}")
print(f"Owner: {row['owner_email']}")
print()

# Check if suno files exist
print("Suno tracks:")
for i, track in enumerate(suno_tracks):
    path = track['path']
    exists = os.path.exists(path)
    print(f"  [{i}] {path}")
    print(f"       exists: {exists}")
    if not exists:
        # Check if it exists with an absolute path
        abs_path = os.path.join('/home/ubuntu/meowify-v2', path)
        abs_exists = os.path.exists(abs_path)
        print(f"       abs path: {abs_path}")
        print(f"       abs exists: {abs_exists}")

print()
print("Regular files:")
for key in ['mp3', 'chorus_voc', 'chorus_inst', 'meow_local', 'masked']:
    if key in files:
        path = files[key]
        exists = os.path.exists(path)
        print(f"  {key}: {path}")
        print(f"         exists: {exists}")
        if not exists:
            abs_path = os.path.join('/home/ubuntu/meowify-v2', path)
            abs_exists = os.path.exists(abs_path)
            print(f"         abs: {abs_path} -> {abs_exists}")
