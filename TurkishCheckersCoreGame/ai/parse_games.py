import os
import re
import json

# --- AYARLAR ---
RAW_GAMES_DIR = "raw_games"
OUTPUT_FILE = "processed_games.json"


def parse_pgn(text):
    """
    PlayOK PGN formatından hamleleri çeker.
    Örn: "1. c3-d4 f6-e5 2. ..." formatını temizler.
    """
    # Metadata kısımlarını (köşeli parantezli yerler) at
    text = re.sub(r'\[.*?\]', '', text)

    # Hamle numaralarını (1. 2. gibi) at
    text = re.sub(r'\d+\.', '', text)

    # Fazla boşlukları temizle
    text = text.strip()

    # Hamleleri ayır (PlayOK hamleleri boşlukla ayrılır)
    moves = text.split()

    # Sadece geçerli hamle formatına uyanları al (örn: c3-d4 veya a1xb2)
    # Basit regex: harf-sayı-ayıraç-harf-sayı
    clean_moves = []
    for m in moves:
        if re.match(r'[a-h][1-8][\-x][a-h][1-8]', m):
            clean_moves.append(m)

    return clean_moves


def main():
    print(f"'{RAW_GAMES_DIR}' klasöründeki oyunlar taranıyor...")

    all_games = []
    files = os.listdir(RAW_GAMES_DIR)

    for filename in files:
        if not filename.endswith(".txt"):
            continue

        filepath = os.path.join(RAW_GAMES_DIR, filename)

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            moves = parse_pgn(content)

            # Eğer hamle bulabildiysek listeye ekle
            if len(moves) > 5:  # 5 hamleden kısa oyunları kaale alma (erken terk vs.)
                all_games.append({"moves": moves, "result": "*"})  # Sonucu şimdilik önemsemiyoruz

        except Exception as e:
            print(f"Hata ({filename}): {e}")

    print(f"Toplam {len(all_games)} oyun başarıyla ayıklandı.")

    # JSON olarak kaydet
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_games, f, indent=None)  # indent=None dosya boyutunu küçültür

    print(f"Dosya oluşturuldu: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()