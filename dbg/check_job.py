#!/usr/bin/env python3
import sqlite3
con = sqlite3.connect('users.db')
con.row_factory = sqlite3.Row
job = con.execute('SELECT id, status FROM jobs WHERE id=?', ('4175c4df',)).fetchone()
if job:
    print(f"Job found: {job['id']} - {job['status']}")
else:
    print('Job 4175c4df not found')
    jobs = con.execute('SELECT id FROM jobs ORDER BY created_at DESC LIMIT 5').fetchall()
    print(f'Recent jobs: {[j[0] for j in jobs]}')
con.close()
