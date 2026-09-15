#!/usr/bin/env python3
"""Debug 500 error when accessing job page - simple version"""
import sys
import os
import traceback
import json

# Change to app directory
os.chdir('/home/ubuntu/meowify-v2')
sys.path.insert(0, '/home/ubuntu/meowify-v2')

try:
    from server import load_job, ADMIN_EMAIL, BANNER
    from jinja2 import Environment, FileSystemLoader

    print("Testing job page rendering...")

    # Load job data
    print("\n1. Loading job 460d0100 from database:")
    job = load_job("460d0100")
    if job:
        print(f"   ✓ Job loaded")
        print(f"   Status: {job.get('status')}")
        print(f"   Suno tracks: {len(job.get('suno_tracks', []))}")
        print(f"   Video info: {job.get('video_info', {})}")
    else:
        print("   ✗ Job not found")
        sys.exit(1)

    # Try to render the template
    print("\n2. Testing template rendering:")
    env = Environment(
        loader=FileSystemLoader('templates'),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.globals['enumerate'] = enumerate

    try:
        template = env.get_template('job.html')
        print("   ✓ job.html loaded")
    except Exception as e:
        print(f"   ✗ Failed to load template: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Try to render (will fail without request object, but we can test partial rendering)
    print("\n3. Testing _job_status.html rendering:")
    try:
        status_template = env.get_template('_job_status.html')
        output = status_template.render(job=job, job_id='460d0100')
        print(f"   ✓ Template rendered ({len(output)} chars)")
        # Check if there are any obvious errors in output
        if 'Traceback' in output or 'Error' in output:
            print(f"   ⚠ Possible error in output:")
            print(output[:500])
    except Exception as e:
        print(f"   ✗ Failed to render: {e}")
        traceback.print_exc()

    print("\n✓ All tests passed!")

except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()
