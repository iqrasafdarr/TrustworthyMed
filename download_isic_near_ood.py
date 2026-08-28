"""
Download near-OOD dermoscopic images from the ISIC Archive for TrustworthyMed.

Pulls images from diagnosis categories NOT in HAM10000's 7 classes:
  - seborrheic keratosis
  - solar lentigo (a.k.a. lentigo simplex / actinic lentigo naming varies by dataset version)

Saves them into: data/raw/isic_near_ood/<diagnosis>/<isic_id>.jpg

Uses the public ISIC Archive API (no auth needed for public images):
  https://api.isic-archive.com/api/v2/images/

Run from your project root:
  python download_isic_near_ood.py
"""

import os
import time
import requests

# ==================== CONFIG ====================
OUTPUT_DIR = "data/raw/isic_near_ood"
TARGET_DIAGNOSES = [
    "seborrheic keratosis",
    "solar lentigo",
]
IMAGES_PER_DIAGNOSIS = 100   # adjust: more images = tighter CI on your near-OOD AUROC
API_BASE = "https://api.isic-archive.com/api/v2/images/"
PAGE_SIZE = 100              # ISIC API page size
REQUEST_DELAY = 0.3          # seconds between requests, be polite to the API


def fetch_images_for_diagnosis(diagnosis, limit):
    """
    Query the ISIC API for images matching a diagnosis, paging until `limit` reached.
    Returns a list of dicts with at least 'isic_id' and a downloadable file url.
    """
    collected = []
    next_url = API_BASE
    params = {
        "query": f'diagnosis:"{diagnosis}"',
        "limit": PAGE_SIZE,
    }

    while next_url and len(collected) < limit:
        resp = requests.get(next_url, params=params if next_url == API_BASE else None, timeout=30)
        if resp.status_code != 200:
            print(f"  WARNING: request failed ({resp.status_code}) for '{diagnosis}': {resp.text[:200]}")
            break

        data = resp.json()
        results = data.get("results", [])
        if not results:
            print(f"  No (more) results found for diagnosis='{diagnosis}'.")
            break

        for item in results:
            if len(collected) >= limit:
                break
            collected.append(item)

        next_url = data.get("next")  # ISIC API returns a full next-page URL or None
        time.sleep(REQUEST_DELAY)

    return collected


def download_image(item, save_path):
    """
    Download a single image file given its ISIC API metadata item.
    ISIC v2 API item structure includes a 'files' dict with a 'full' -> 'url' key.
    """
    try:
        url = item["files"]["full"]["url"]
    except (KeyError, TypeError):
        print(f"  WARNING: no downloadable URL found for item {item.get('isic_id', '?')}, skipping.")
        return False

    resp = requests.get(url, timeout=60)
    if resp.status_code != 200:
        print(f"  WARNING: failed to download {url} ({resp.status_code})")
        return False

    with open(save_path, "wb") as f:
        f.write(resp.content)
    return True


def main():
    if not os.path.isdir(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

    total_downloaded = 0

    for diagnosis in TARGET_DIAGNOSES:
        print(f"\n{'='*60}")
        print(f"Fetching metadata for: {diagnosis}")
        print(f"{'='*60}")

        items = fetch_images_for_diagnosis(diagnosis, IMAGES_PER_DIAGNOSIS)
        print(f"  Found {len(items)} candidate images for '{diagnosis}'.")

        if len(items) == 0:
            print(f"  WARNING: zero images found for '{diagnosis}'. "
                  f"Double check the diagnosis string matches ISIC's exact taxonomy "
                  f"(check https://api.isic-archive.com/api/v2/images/?query=... in a browser first).")
            continue

        diag_dir = os.path.join(OUTPUT_DIR, diagnosis.replace(" ", "_"))
        os.makedirs(diag_dir, exist_ok=True)

        downloaded_this_diag = 0
        for item in items:
            isic_id = item.get("isic_id", f"unknown_{downloaded_this_diag}")
            save_path = os.path.join(diag_dir, f"{isic_id}.jpg")

            if os.path.exists(save_path):
                downloaded_this_diag += 1
                continue  # already downloaded, don't re-fetch

            ok = download_image(item, save_path)
            if ok:
                downloaded_this_diag += 1
                total_downloaded += 1
            time.sleep(REQUEST_DELAY)

        print(f"  Downloaded {downloaded_this_diag} images for '{diagnosis}' -> {diag_dir}")

    print(f"\n{'='*60}")
    print(f"DONE. Total new images downloaded this run: {total_downloaded}")
    print(f"All near-OOD images are under: {OUTPUT_DIR}")
    print(f"{'='*60}")

    if total_downloaded == 0:
        print("\nWARNING: zero images downloaded. Check your internet connection, the API endpoint, "
              "and that TARGET_DIAGNOSES match ISIC's exact diagnosis naming before relying on this.")


if __name__ == "__main__":
    main()