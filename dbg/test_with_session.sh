#!/bin/bash
# Test accessing the job page with a session cookie

HOST="http://127.0.0.1:8503"
JOB_ID="460d0100"
TEMP_COOKIES="/tmp/cookies.txt"

echo "Testing job page with simulated session..."
echo

# First, get a login page to see what it looks like
echo "1. Getting login page..."
curl -s -c "$TEMP_COOKIES" "$HOST/login" | head -20

echo
echo "2. Simulating logged-in user (creating fake session)..."
# This will fail because we can't actually log in via curl, but we can try to create a session
# For now, let's just test with the cookies from login
curl -s -b "$TEMP_COOKIES" "$HOST/job/$JOB_ID" 2>&1 | head -100
