import os
import requests
from bs4 import BeautifulSoup
import urllib.parse
import shutil
import time

BRAND_QUERIES = [
    "Nike logo", "Nike swoosh", "Nike brand logo",
    "Nike logo transparent", "Nike sign",
    "Nike emblem", "Nike icon", "Nike vector logo",
    "Nike white logo", "Nike black logo",
    "Nike sports logo", "Nike retro logo", "Nike logo design",
    "Nike logo art", "Nike logo wallpaper",
    "Nike logo 3d", "Nike logo illustration", "Nike logo background",
    "Nike logo print", "Nike logo image"
]

NEGATIVE_QUERIES = [
    "random images", "nature photos", "city pictures",
    "abstract backgrounds", "landscape photos",
    "animals pictures", "food photography", "architecture",
    "space photography", "sports photography",
    "ocean photos", "mountain photos", "forest photography",
    "street photography", "travel photos",
    "texture images", "sky photography", "art background",
    "colorful photos", "minimalistic backgrounds"
]

NUM_IMAGES_PER_QUERY = 40
OUTPUT_DIR = "dataset"

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def download_image(url, path):
    try:
        r = requests.get(url, stream=True, timeout=5)
        if r.status_code == 200:
            with open(path, 'wb') as f:
                r.raw.decode_content = True
                shutil.copyfileobj(r.raw, f)
    except Exception as e:
        print(f"Error downloading {url}: {e}")

def search_images(query, num_images=40):
    print(f"\nПошук зображень для: {query}")
    headers = {"User-Agent": "Mozilla/5.0"}
    query_string = urllib.parse.quote(query)
    image_urls = []
    offset = 0

    while len(image_urls) < num_images:
        url = f"https://www.bing.com/images/async?q={query_string}&first={offset}&count=100"
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")

        for img in soup.find_all("a", class_="iusc"):
            try:
                m = img.get("m")
                if m:
                    m_json = eval(m)
                    img_url = m_json.get("murl")
                    if img_url and img_url not in image_urls:
                        image_urls.append(img_url)
            except Exception:
                continue
            if len(image_urls) >= num_images:
                break

        offset += 100
        time.sleep(1)
        if offset > num_images * 3:
            break

    print(f"Знайдено {len(image_urls)} зображень для '{query}'")
    return image_urls

def main():
    create_dir(OUTPUT_DIR)

    brand_dir = os.path.join(OUTPUT_DIR, "positive")
    neg_dir = os.path.join(OUTPUT_DIR, "negative")
    create_dir(brand_dir)
    create_dir(neg_dir)

    idx = 0
    for query in BRAND_QUERIES:
        brand_urls = search_images(query, NUM_IMAGES_PER_QUERY)
        for url in brand_urls:
            path = os.path.join(brand_dir, f"{idx}.jpg")
            download_image(url, path)
            idx += 1

    idx = 0
    for query in NEGATIVE_QUERIES:
        neg_urls = search_images(query, NUM_IMAGES_PER_QUERY)
        for url in neg_urls:
            path = os.path.join(neg_dir, f"{idx}.jpg")
            download_image(url, path)
            idx += 1

    print(f"\nЗавантаження датасету завершено! Дані збережено в '{OUTPUT_DIR}'.")

if __name__ == "__main__":
    main()
