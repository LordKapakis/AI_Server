from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import os
import re
from urllib.parse import urljoin

BASE_URL = "https://docs.unity3d.com/ScriptReference/"
SAVE_FOLDER = "unity_docs_full"
os.makedirs(SAVE_FOLDER, exist_ok=True)

CATEGORIES = ("UnityEngine", "UnityEditor", "Unity.", "Other", "Accessibility")

def keep_link(href: str) -> bool:
    if not href or not href.endswith(".html"):
        return False
    if href.startswith("http"):
        return False
    if "Manual/" in href:
        return False
    if "index.html" in href:
        return False
    return any(term in href for term in CATEGORIES)

def clean_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    # Try API docs container first
    main = soup.select_one("div#content-container") or soup.select_one("div#content-wrap") or soup

    # Remove nav/headers/footers/forms
    for sel in ["nav", "header", "footer", "form", "script", "style", ".footer-wrapper"]:
        for tag in main.select(sel):
            tag.decompose()

    text = main.get_text(separator="\n").strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text

def safe_filename(href: str) -> str:
    name = href.replace("/", "_").replace("\\", "_")
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    return name.replace(".html", ".txt")

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto(BASE_URL, wait_until="networkidle")

    anchors = page.query_selector_all("a")
    raw_links = [a.get_attribute("href") for a in anchors if keep_link(a.get_attribute("href"))]
    links = sorted(set(raw_links))
    print(f"Found {len(links)} Unity API pages")

    for href in links:
        url = urljoin(BASE_URL, href)
        try:
            print(f"Scraping {url}")
            page.goto(url, wait_until="networkidle")

            # Don't force a selector, just grab the HTML
            html = page.content()
            text = clean_text(html)

            if len(text) < 300:
                print(f"⚠️ Skipping {href} (too short)")
                continue

            out_path = os.path.join(SAVE_FOLDER, safe_filename(href))
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"✅ Saved {out_path}")

        except Exception as e:
            print(f"❌ Failed {url}: {e}")

    browser.close()
