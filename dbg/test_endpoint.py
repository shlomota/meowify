#!/usr/bin/env python3
"""Test the /job endpoint directly"""
import sys
import json
sys.path.insert(0, '/Users/stannor/PycharmProjects/meowify')

# Setup FastAPI app
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.testclient import TestClient
import os
os.chdir('/Users/stannor/PycharmProjects/meowify')

# Import the server
from server import app, load_job, ADMIN_EMAIL

# Create test client
client = TestClient(app)

# Test fetching job without auth (should redirect)
print("Testing /job/460d0100 without auth...")
response = client.get("/job/460d0100", follow_redirects=False)
print(f"  Status: {response.status_code}")
if response.status_code == 307:
    print(f"  Redirect to: {response.headers.get('location', 'N/A')}")

# Load job from DB to check its state
print("\nLoading job from database...")
job = load_job("460d0100")
if job:
    print(f"  Status: {job.get('status')}")
    print(f"  Owner: {job.get('owner_email')}")
    print(f"  Step: {job.get('step')}")
    print(f"  Files keys: {list(job.get('files', {}).keys())}")
    print(f"  Suno tracks: {len(job.get('suno_tracks', []))}")
else:
    print("  Job not found!")
