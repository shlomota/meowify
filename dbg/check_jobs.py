#!/usr/bin/env python3
import sqlite3
import json
con = sqlite3.connect('users.db')
con.row_factory = sqlite3.Row
# Get all completed jobs
rows = con.execute('SELECT id, status, owner_email, video_info FROM jobs WHERE status=? ORDER BY created_at DESC LIMIT 10', ('done',)).fetchall()
print(f"Found {len(rows)} completed jobs:")
for row in rows:
    print(f"  {row['id']}: {row['owner_email']}")
con.close()
