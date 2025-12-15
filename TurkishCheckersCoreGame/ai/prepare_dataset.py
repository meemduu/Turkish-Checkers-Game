import json
import numpy as np
import data_processor as dp
import os

# --- AYARLAR ---
INPUT_FILE = "processed_games.json"
OUTPUT_FILE = "training_data.npy"


def create_dataset():
    if not os.path.exists(INPUT_FILE):
        print(f"Hata: {INPUT_FILE} bulunamadı!")
        return

    print("Veri seti yükleniyor...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        games = json.load(f)

    X = []  # Girdiler (Tahta Pozisyonları)
    y = []  # Çıktılar (Yapılan Hamleler - Şimdilik basitleştirilmiş)

    # NOT: Yapay zekaya "Hangi hamleyi yapayım?" diye sormak için
    # Çıktı (y) formatını belirlememiz lazım.
    # Şimdilik sadece "Pozisyonu" kaydedeceğiz. İleride "Policy Network" için hamle indekslemesi ekleyeceğiz.
    # Bu aşamada amacımız veriyi board formatına çevirebilmek.

    processed_count = 0

    for game in games:
        board = dp.init_board()
        moves = game["moves"]

        # Beyaz başlar
        turn = 1  # 1: Beyaz, -1: Siyah

        for move in moves:
            # Mevcut tahta durumunu kaydet (Yapay Zeka bunu görecek)
            # Sadece Beyaz'ın hamlelerini öğrensin istiyorsak filtreleyebiliriz.
            # Amaç genel oyun öğrenmekse hepsini alalım.

            # Tahtayı kopyala (Referans hatası olmasın)
            current_board_state = board.copy()

            # --- VERİ TOPLAMA NOKTASI ---
            # Burada X'e board durumunu, y'ye ise 'move' bilgisini eklememiz lazım.
            # Şimdilik sadece Board durumlarını biriktiriyoruz ki akışı görelim.
            X.append(dp.encode_board(current_board_state))

            # Tahtada hamleyi oynat
            board = dp.apply_move(board, move, turn)

            # Sırayı değiştir
            turn *= -1

        processed_count += 1
        if processed_count % 50 == 0:
            print(f"{processed_count} oyun işlendi...")

    print(f"\nToplam {len(X)} pozisyon (hamle) kaydedildi.")

    # NumPy dizisine çevir ve kaydet
    X_array = np.array(X, dtype=np.int8)
    np.save(OUTPUT_FILE, X_array)
    print(f"Veri seti '{OUTPUT_FILE}' olarak kaydedildi. Boyut: {X_array.shape}")


if __name__ == "__main__":
    create_dataset()