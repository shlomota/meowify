#!/bin/bash
# Test accessing the job page

HOST="http://127.0.0.1:8503"
JOB_ID="460d0100"

echo "Testing job page access..."
echo

# Test 1: Without auth (should redirect)
echo "1. Accessing job without auth:"
RESPONSE=$(curl -s -w "\n%{http_code}" -L "$HOST/job/$JOB_ID" 2>&1)
HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$RESPONSE" | head -n-1)
echo "HTTP Status: $HTTP_CODE"
if [ $HTTP_CODE -ge 500 ]; then
  echo "ERROR: Server error!"
  echo "$BODY" | head -20
fi

echo
echo "2. Checking if /login is accessible:"
RESPONSE=$(curl -s -w "\n%{http_code}" "$HOST/login" 2>&1)
HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
echo "HTTP Status: $HTTP_CODE"

echo
echo "3. Checking if static files are accessible:"
RESPONSE=$(curl -s -w "\n%{http_code}" "$HOST/static/style.css" 2>&1)
HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
echo "style.css HTTP Status: $HTTP_CODE"
