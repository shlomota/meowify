#!/usr/bin/env python3
"""Debug 500 error when accessing job page"""
import sys
import os
import traceback

# Change to app directory
os.chdir('/home/ubuntu/meowify-v2')
sys.path.insert(0, '/home/ubuntu/meowify-v2')

try:
    from server import app, load_job, ADMIN_EMAIL
    from fastapi.testclient import TestClient

    client = TestClient(app)

    print("Testing job page access (simulating logged-in user)...")

    # Create a session by calling login page
    response = client.get("/login")
    print(f"Login page status: {response.status_code}")

    # Try to access job without auth
    print("\n1. Accessing /job/460d0100 without auth:")
    response = client.get("/job/460d0100", follow_redirects=False)
    print(f"   Status: {response.status_code}")
    if response.status_code >= 400:
        print(f"   Response: {response.text[:200]}")

    # Try to load job data
    print("\n2. Loading job from database:")
    job = load_job("460d0100")
    if job:
        print(f"   ✓ Job loaded")
        print(f"   Status: {job.get('status')}")
        print(f"   Suno tracks: {len(job.get('suno_tracks', []))}")
        print(f"   Files: {list(job.get('files', {}).keys())}")
    else:
        print("   ✗ Job not found")

except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()
