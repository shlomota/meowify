"""
Standalone Google OAuth test page — run with:
    source .venv/bin/activate
    streamlit run oauth_test.py --server.port 8504

Does NOT touch app.py. Just verifies the OAuth flow works and shows
the user info (email, name, picture) we'll get in the real integration.
"""

import os
import secrets
import requests
import streamlit as st
from pathlib import Path


# ── Config ────────────────────────────────────────────────────────────────────
def load_env() -> dict:
    env = {}
    p = Path(".env")
    if p.exists():
        for line in p.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env

env = load_env()
CLIENT_ID     = env.get("GOOGLE_CLIENT_ID", "")
CLIENT_SECRET = env.get("GOOGLE_CLIENT_SECRET", "")
REDIRECT_URI  = "http://localhost:8504"

GOOGLE_AUTH_URL     = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL    = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v3/userinfo"
SCOPES = "openid email profile"

# ── Page ──────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="OAuth Test", page_icon="🔐")
st.title("Google OAuth Test")

if not CLIENT_ID or not CLIENT_SECRET:
    st.error("Add GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET to your .env file first.")
    st.stop()

# ── Already logged in ─────────────────────────────────────────────────────────
if "user" in st.session_state:
    user = st.session_state.user
    c1, c2 = st.columns([1, 4])
    with c1:
        if user.get("picture"):
            st.image(user["picture"], width=80)
    with c2:
        st.success("OAuth flow worked!")
        st.write(f"**Name:** {user.get('name')}")
        st.write(f"**Email:** {user.get('email')}")
        st.write(f"**Google ID:** {user.get('sub')}")
    with st.expander("Full token response"):
        st.json(user)
    if st.button("Logout / test again"):
        del st.session_state["user"]
        st.query_params.clear()
        st.rerun()
    st.stop()

# ── OAuth callback ─────────────────────────────────────────────────────────────
code  = st.query_params.get("code", "")
error = st.query_params.get("error", "")

if error:
    st.error(f"Google returned an error: {error}")

elif code:
    with st.spinner("Exchanging code for token..."):
        token_resp = requests.post(GOOGLE_TOKEN_URL, data={
            "code":          code,
            "client_id":     CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "redirect_uri":  REDIRECT_URI,
            "grant_type":    "authorization_code",
        }, timeout=15)

    if token_resp.status_code != 200:
        st.error(f"Token exchange failed ({token_resp.status_code}): {token_resp.text}")
        st.stop()

    access_token = token_resp.json().get("access_token")
    user_resp = requests.get(
        GOOGLE_USERINFO_URL,
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=15,
    )
    st.session_state.user = user_resp.json()
    st.query_params.clear()
    st.rerun()

# ── Login button ──────────────────────────────────────────────────────────────
else:
    state = secrets.token_urlsafe(16)
    params = (
        f"?client_id={CLIENT_ID}"
        f"&redirect_uri={REDIRECT_URI}"
        f"&response_type=code"
        f"&scope={SCOPES.replace(' ', '%20')}"
        f"&state={state}"
        f"&access_type=offline"
        f"&prompt=consent"
    )
    st.write("Click below to test the Google OAuth login flow.")
    st.link_button("🔐 Login with Google", GOOGLE_AUTH_URL + params)
