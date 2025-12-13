import requests
from bs4 import BeautifulSoup
import os

BASE_URL = "https://docs.unity3d.com/ScriptReference/"
INDEX_URL = BASE_URL + "30_search.html"   # entrypoint that has the full tree
SAVE_FOLDER = "unity_docs"
os.makedirs(SAVE_FOLDER, exist_ok=True)

def clean_text(html):
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator="\n").strip()

# Step 1: Get the search/tree page (contains all links in sidebar)
resp = requests.get(INDEX_URL)
resp.raise_for_status()
soup = BeautifulSoup(resp.text, "html.parser")

# Step 2: Find all UnityEngine links
links = []
for a in soup.find_all("a", href=True):
    href = a["href"]
    if href.startswith("UnityEngine") and href.endswith(".html"):
        if href == "UnityEngine.html":  # skip the root
            continue
        links.append(href)

links = sorted(list(set(links)))  # deduplicate + sort
print(f"Found {len(links)} UnityEngine pages")

# Step 3: Download each page
for link in links:
    url = BASE_URL + link
    try:
        print(f"Downloading {url}")
        page = requests.get(url)
        page.raise_for_status()

        text = clean_text(page.text)
        filename = os.path.join(SAVE_FOLDER, link.replace(".html", ".txt"))
        with open(filename, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"✅ Saved {filename}")
    except Exception as e:
        print(f"❌ Failed {url}: {e}")
