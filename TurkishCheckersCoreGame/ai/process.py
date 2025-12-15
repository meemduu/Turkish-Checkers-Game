import os
import re
import json

# --- AYARLAR ---
SOURCE_FOLDER = "raw_games"
OUTPUT_FILE = "processed_games.json"
MIN_AVG_ELO = 1600  # Bu puanın altındaki maçları kaale alma


def parse_game_file(content):
    """
    Bir oyun dosyasının içeriğini alır, etiketleri ve hamleleri ayıklar.
    """
    data = {}

    # --- 1. Metadata (Etiketler) Çıkarma ---
    # Regex ile köşeli parantez içindeki bilgileri alıyoruz
    headers = {m.group(1): m.group(2) for m in re.finditer(r'\[(\w+) "(.*?)"\]', content)}

    # ELO puanlarını sayıya çevirelim (Hata olursa 0 varsayalım)
    try:
        w_elo = int(headers.get("WhiteElo", 0))
        b_elo = int(headers.get("BlackElo", 0))
    except:
        w_elo, b_elo = 0, 0

    avg_elo = (w_elo + b_elo) / 2

    # ELO Filtresi: Eğer maç kalitesizse None dön
    if avg_elo < MIN_AVG_ELO:
        return None

    data["white_elo"] = w_elo
    data["black_elo"] = b_elo
    data["result"] = headers.get("Result", "*")
    data["winner"] = headers.get("White") if data["result"] == "1-0" else (
        headers.get("Black") if data["result"] == "0-1" else "Draw")

    # --- 2. Hamleleri Temizleme ---
    # Köşeli parantezli etiketlerden sonraki kısım hamlelerdir.
    # Genelde boş bir satırdan sonra başlar.

    # Önce tüm etiketleri siliyoruz
    moves_text = re.sub(r'\[.*?\]', '', content, flags=re.DOTALL)

    # Satır sonlarını boşlukla değiştir
    moves_text = moves_text.replace("\n", " ")

    # "1. ", "2. " gibi hamle numaralarını sil (Regex: Sayı + Nokta + Boşluk)
    moves_text = re.sub(r'\d+\.', '', moves_text)

    # Oyun sonu sonucunu (1-0 veya 0-1) sil
    moves_text = moves_text.replace("1-0", "").replace("0-1", "").replace("1/2-1/2", "")

    # Fazla boşlukları temizle ve listeye çevir
    # moves_text.split() boşluklara göre böler
    move_list = moves_text.split()

    data["moves"] = move_list

    return data


# --- ANA İŞLEM ---
all_processed_games = []
files = [f for f in os.listdir(SOURCE_FOLDER) if f.endswith(".txt")]

print(f"Toplam {len(files)} dosya işleniyor...")

for filename in files:
    path = os.path.join(SOURCE_FOLDER, filename)

    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        game_data = parse_game_file(content)

        if game_data:  # Eğer filtreye takılmadıysa
            all_processed_games.append(game_data)

    except Exception as e:
        print(f"Hata ({filename}): {e}")

# Sonucu Kaydet
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(all_processed_games, f, indent=4)

print(f"\nİşlem Tamam dayı!")
print(f"Toplam {len(files)} oyundan {len(all_processed_games)} tanesi kriterlere uydu.")
print(f"Temiz veriler '{OUTPUT_FILE}' dosyasına kaydedildi.")