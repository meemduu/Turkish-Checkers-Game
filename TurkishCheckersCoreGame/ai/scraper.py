import requests
from bs4 import BeautifulSoup
import os
import time
import random

# --- AYARLAR ---
# Hedef oyuncu listeni buraya sağlam doldur dayı.
#cebeci060
TARGET_PLAYERS = ["dfp7345g", "redkid", "qahwachi", "gsk3655g", "mahmuthoca", "thorxx", "zokeytli", "asafbey", "fmu", "neco8866", "khak", "arapkadri", "sansiro", "hocakeskin"]

GAME_TYPE = "tu"
SAVE_FOLDER = "raw_games"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

BASE_DOMAIN = "https://www.playok.com"

# DÜZELTME BURADA: &pg değil &page yaptık!
BASE_PROFILE_URL = "https://www.playok.com/tr/stat.phtml?u={}&g={}&sk=2&page={}"

if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)


def download_txt(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code == 200:
            return response.text
    except:
        return None


def process_player_history(username):
    print(f"\n--- {username} İÇİN GERÇEK TARİH KAZISI ---")

    # PlayOK sayfalama mantığına göre 1'den başlatalım
    page_num = 1
    total_downloaded_for_player = 0
    consecutive_empty_pages = 0

    while True:
        current_url = BASE_PROFILE_URL.format(username, GAME_TYPE, page_num)
        print(f"  >> {username} - Sayfa {page_num} taranıyor...")

        try:
            response = requests.get(current_url, headers=HEADERS, timeout=10)
            if response.status_code != 200:
                print("    ! Sayfaya erişilemedi.")
                break
        except:
            print("    ! Bağlantı hatası.")
            break

        soup = BeautifulSoup(response.text, 'html.parser')
        links = soup.find_all('a', href=True)

        games_found_on_page = 0
        new_downloads_on_page = 0

        for link in links:
            href = link['href']
            text = link.text.strip().lower()

            # Link txt mi?
            if "/p/?g=" in href and "txt" in text:
                games_found_on_page += 1

                try:
                    raw_filename = href.split("=")[-1]
                    game_id = raw_filename.replace(".txt", "").replace(GAME_TYPE, "")
                    filename = f"{SAVE_FOLDER}/game_{game_id}.txt"

                    # ZATEN VARSA
                    if os.path.exists(filename):
                        continue

                        # YOKSA İNDİR
                    download_url = BASE_DOMAIN + href
                    content = download_txt(download_url)

                    if content:
                        with open(filename, "w", encoding="utf-8") as f:
                            f.write(content)
                        total_downloaded_for_player += 1
                        new_downloads_on_page += 1

                except:
                    continue

        # --- KONTROL ---
        if games_found_on_page == 0:
            print(f"    ! Sayfa {page_num} boş (veya oyun linki yok). Bitiyor.")
            consecutive_empty_pages += 1
        else:
            print(f"    + Sayfada {games_found_on_page} oyun var. {new_downloads_on_page} tanesi YENİ indirildi.")
            consecutive_empty_pages = 0

        # 2 boş sayfa gelirse bitir (Garanti olsun)
        if consecutive_empty_pages >= 2:
            print(f"--- {username} tamamlandı. Toplam {total_downloaded_for_player} yeni oyun cepte. ---")
            break

        page_num += 1
        time.sleep(random.uniform(1.0, 2.0))  # Nezaket beklemesi


# --- ANA ÇALIŞTIRMA ---
for player in TARGET_PLAYERS:
    process_player_history(player)
    time.sleep(3)