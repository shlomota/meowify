"""Edit album art with OpenAI image API."""

import os
import requests
import json
import base64
import tempfile
import uuid


def edit_album_art(image_url: str, openai_key: str) -> str:
    """Edit album art: replace people with cartoon cats, add logo.

    Returns: File path to edited image (PNG)
    """
    if not openai_key:
        raise RuntimeError("OPENAI_API_KEY not configured")

    # Call OpenAI image edit API with gpt-image-2.5-flare
    headers = {
        "Authorization": f"Bearer {openai_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "gpt-image-2.5-flare",
        "images": [
            {
                "image_url": image_url
            }
        ],
        "prompt": "Replace any people or characters in this album art with cute cartoon cats. Add playful cat paws in the corners and a small 'Meowify' logo. Keep the original art style and colors.",
        "n": 1,
        "size": "1024x1024"
    }

    response = requests.post(
        "https://api.openai.com/v1/images/edits",
        headers=headers,
        json=payload,
        timeout=60
    )

    if response.status_code != 200:
        raise RuntimeError(f"OpenAI API error: {response.status_code} {response.text}")

    data = response.json()

    if "error" in data:
        raise RuntimeError(f"OpenAI error: {data['error'].get('message', 'Unknown error')}")

    image_data = data.get("data", [{}])[0]

    # Handle base64 response (b64_json)
    if "b64_json" in image_data:
        image_bytes = base64.b64decode(image_data["b64_json"])
    elif "url" in image_data:
        resp = requests.get(image_data["url"], timeout=10)
        image_bytes = resp.content
    else:
        raise RuntimeError(f"No image data in response: {data}")

    # Save to temp file
    tmp_dir = tempfile.gettempdir()
    file_path = os.path.join(tmp_dir, f"meowify_cover_{uuid.uuid4().hex[:8]}.png")

    with open(file_path, "wb") as f:
        f.write(image_bytes)

    return file_path


if __name__ == "__main__":
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        print("✓ Image editor ready (OPENAI_API_KEY exists)")
    else:
        print("Error: OPENAI_API_KEY not set")
