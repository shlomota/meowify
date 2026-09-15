"""Edit album art with KIE.AI (ChatGPT image API)."""

import requests
import time


def edit_album_art(image_url: str, kie_key: str, timeout: int = 600) -> str:
    """Edit album art: replace people with cartoon cats, add logo.

    Returns: URL to edited image
    """
    if not kie_key:
        raise RuntimeError("KIE_API_KEY not configured")

    payload = {
        "model": "gpt-image-2-5-flare-image-to-image",
        "input": {
            "prompt": "Replace any people or characters in this album art with cute cartoon cats. Add playful cat paws in the corners and a small 'Meowify' logo. Keep the original art style and colors as much as possible.",
            "input_urls": [image_url],
            "aspect_ratio": "auto",
            "resolution": "2K"
        }
    }

    headers = {
        "Authorization": f"Bearer {kie_key}",
        "Content-Type": "application/json"
    }

    # Submit job
    response = requests.post(
        "https://api.kie.ai/api/v1/jobs/createTask",
        json=payload,
        headers=headers,
        timeout=30
    )

    if response.status_code != 200:
        raise RuntimeError(f"KIE API error: {response.status_code} {response.text}")

    response_data = response.json()

    if response_data.get("code") != 200:
        raise RuntimeError(f"KIE API error: {response_data.get('msg')}")

    data = response_data.get("data", {})
    job_id = data.get("taskId") or data.get("recordId")
    download_url = data.get("output_url")

    if not job_id:
        raise RuntimeError(f"No job ID in response: {data}")

    # If URL already available, return it
    if download_url:
        return download_url

    # Poll for completion
    start = time.time()
    while time.time() - start < timeout:
        time.sleep(3)

        status_response = requests.get(
            f"https://api.kie.ai/api/v1/jobs/getRecord/{job_id}",
            headers=headers,
            timeout=10
        )

        if status_response.status_code == 200:
            status_data = status_response.json()

            if status_data.get("code") == 200:
                result = status_data.get("data", {})
                output_url = result.get("output_url")

                if output_url:
                    return output_url

                status = result.get("status")
                if status in ["failed", "error"]:
                    raise RuntimeError(f"Image edit failed: {result}")
            else:
                raise RuntimeError(f"KIE API error: {status_data.get('msg')}")

    raise RuntimeError(f"Image edit timeout after {timeout}s")


if __name__ == "__main__":
    import os
    kie_key = os.getenv("KIE_API_KEY")
    if kie_key:
        print("✓ Image editor ready (KIE_API_KEY exists)")
    else:
        print("Error: KIE_API_KEY not set")
