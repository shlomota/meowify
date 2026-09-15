#!/usr/bin/env python3
"""Test template rendering with actual job data"""
import json
import sys
sys.path.insert(0, '/Users/stannor/PycharmProjects/meowify')

from jinja2 import Environment, FileSystemLoader, select_autoescape

# Load the job data from the database query
job_data = {
    'status': 'done',
    'step': 'Complete',
    'error': None,
    'suno_error': None,
    'source_url': 'https://youtu.be/vGfJeW_CcFY',
    'owner_email': 'stannor@gmail.com',
    'files': {
        'mp3': '/home/ubuntu/meowify-v2/downloads/יה_רבון_עלם_-_Ka_Ribon_Olam_-_Shabbat_Song_460d0100.mp3',
        'chorus_start': 52.0,
        'chorus_voc': 'work/htdemucs/יה_רבון_עלם_-_Ka_Ribon_Olam_-_Shabbat_Song_460d0100_clip_52s/vocals.wav',
        'chorus_inst': 'work/htdemucs/יה_רבון_עלם_-_Ka_Ribon_Olam_-_Shabbat_Song_460d0100_clip_52s/no_vocals.wav',
        'meow_local': 'output/יה_רבון_עלם_-_Ka_Ribon_Olam_-_Shabbat_Song_460d0100_meowified.wav',
        'masked': 'output/יה_רבון_עלם_-_Ka_Ribon_Olam_-_Shabbat_Song_460d0100_meowified_masked.wav',
    },
    'suno_tracks': [
        {'path': 'output/יה_רבון_עלם_-_Ka_Ribon_Olam_-_Shabbat_Song_460d0100_suno_1.mp3', 'title': 'יה_רבון_עלם_-_Ka_Ribon_Olam_-_Shabbat_Song_460d0100 (Meow Cover)'},
        {'path': 'output/יה_רבון_עלם_-_Ka_Ribon_Olam_-_Shabbat_Song_460d0100_suno_2.mp3', 'title': 'יה_רבון_עלם_-_Ka_Ribon_Olam_-_Shabbat_Song_460d0100 (Meow Cover)'},
    ],
    'video_info': {
        'title': 'יה_רבון_עלם_-_Ka_Ribon_Olam_-_Shabbat_Song_460d0100',
        'duration': 0,
        'thumbnail_url': '',
        'uploader': '(uploaded)',
        'id': '',
    },
    'logs': [],
}

# Setup Jinja2 environment
env = Environment(
    loader=FileSystemLoader('/Users/stannor/PycharmProjects/meowify/templates'),
    autoescape=select_autoescape(['html', 'xml']),
    trim_blocks=True,
    lstrip_blocks=True,
)
env.globals['enumerate'] = enumerate

try:
    template = env.get_template('job.html')
    print("✓ job.html template loaded successfully")
except Exception as e:
    print(f"✗ Failed to load job.html: {e}")
    sys.exit(1)

try:
    # Try to render with just the job data (simulating what would be passed)
    # Note: we can't actually render without the full request context,
    # but we can test the _job_status partial
    status_template = env.get_template('_job_status.html')
    print("✓ _job_status.html template loaded successfully")

    # Try rendering the status template
    output = status_template.render(job=job_data, job_id='460d0100')
    print("✓ _job_status.html rendered successfully")
    print(f"  Output length: {len(output)} chars")

except Exception as e:
    print(f"✗ Failed to render: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nAll template checks passed!")
