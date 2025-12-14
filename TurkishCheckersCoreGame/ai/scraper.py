import requests
from bs4 import BeautifulSoup
import os
import time
import random

# --- AYARLAR ---
TARGET_PLAYERS = ["dfp7345g", "redkid", "qahwachi", "gsk3655g", "mahmuthoca", "thorxx", "zokeytli"]
GAME_TYPE = "tu"  # URL'deki 'g=' parametresi ve dosya ismindeki önek
SAVE_FOLDER = "raw_games"

# Siteye Chrome tarayıcı gibi görüneceğiz
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# Profil URL yapısı (Geçmiş sayfası: sk=2)
BASE_PROFILE_URL = "https://www.playok.com/tr/stat.phtml?u={}&g={}&sk=2"
BASE_DOMAIN = "https://www.playok.com"

if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)


def download_txt(url):
    try:
        # Direkt txt linkine istek atıyoruz
        response = requests.get(url, headers=HEADERS)
        if response.status_code == 200:
            return response.text
    except Exception as e:
        print(f"Hata (İndirme): {e}")
    return None


def process_profile(username):
    print(f"\n--- Hedef Oyuncu: {username} Taranıyor ---")

    url = BASE_PROFILE_URL.format(username, GAME_TYPE)

    try:
        response = requests.get(url, headers=HEADERS)
        if response.status_code != 200:
            print(f"Siteye girilemedi! Kod: {response.status_code}")
            return
    except Exception as e:
        print(f"Bağlantı hatası: {e}")
        return

    soup = BeautifulSoup(response.text, 'html.parser')

    # Tüm linkleri al
    links = soup.find_all('a', href=True)

    found_count = 0
    new_download_count = 0

    for link in links:
        href = link['href']
        link_text = link.text.strip().lower()  # Linkin görünen adı "txt" mi?

        # Linkin içinde "/p/?g=" geçiyor mu ve yazısı "txt" mi?
        # Senin attığın örnek: <a href="/p/?g=tu24480646.txt">txt</a>
        if "/p/?g=" in href and "txt" in link_text:
            try:
                # ID'yi temizleyip alalım
                # href örneği: /p/?g=tu24480646.txt
                # Eşittirden sonrasını al -> tu24480646.txt
                raw_filename = href.split("=")[-1]

                # "tu" ve ".txt" kısımlarını atıp sadece numarayı alalım
                game_id = raw_filename.replace(".txt", "").replace(GAME_TYPE, "")

                # Dosya ismi
                filename = f"{SAVE_FOLDER}/game_{game_id}.txt"

                found_count += 1

                # Dosya zaten var mı?
                if os.path.exists(filename):
                    continue

                # Yoksa indir
                print(f">> İndiriliyor: {game_id}")

                # İndirme linki relative (/p/...) olduğu için başına domain ekliyoruz
                download_url = BASE_DOMAIN + href

                content = download_txt(download_url)

                if content:
                    with open(filename, "w", encoding="utf-8") as f:
                        f.write(content)
                    new_download_count += 1

                    # Nezaket beklemesi
                    time.sleep(1)

            except Exception as e:
                print(f"Link işleme hatası: {e}")
                continue

    print(f"Bitti: {username} için {found_count} oyun bulundu, {new_download_count} tanesi indirildi.")


# --- ANA ÇALIŞTIRMA ---
for player in TARGET_PLAYERS:
    process_profile(player)
    time.sleep(random.uniform(2, 4))