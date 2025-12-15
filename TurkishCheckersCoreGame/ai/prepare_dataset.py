import json
import numpy as np
import data_processor as dp
import os

# --- AYARLAR ---
INPUT_FILE = "processed_games.json"
OUTPUT_DATA_FILE = "training_data.npy"  # Sorular (X)
OUTPUT_LABEL_FILE = "training_labels.npy"  # Cevaplar (y)
MOVE_DICT_FILE = "move_dictionary.json"  # Sözlük (Hangi ID hangi hamle?)


def create_dataset():
    if not os.path.exists(INPUT_FILE):
        print(f"Hata: {INPUT_FILE} bulunamadı!")
        return

    print("1. Aşama: Tüm benzersiz hamleler toplanıyor (Sözlük oluşturuluyor)...")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        games = json.load(f)

    # Tüm hamleleri bir kümede (set) toplayalım ki tekrar edenler elensin
    unique_moves = set()
    for game in games:
        for move in game["moves"]:
            unique_moves.add(move)

    # Hamleleri alfabetik sıralayıp ID verelim
    sorted_moves = sorted(list(unique_moves))
    move_to_id = {move: i for i, move in enumerate(sorted_moves)}

    # Sözlüğü kaydedelim (İleride AI oynarken lazım olacak)
    with open(MOVE_DICT_FILE, "w", encoding="utf-8") as f:
        json.dump(move_to_id, f, indent=4)

    print(f">> Sözlük oluşturuldu! Toplam {len(unique_moves)} farklı hamle çeşidi var.")

    print("\n2. Aşama: Veri seti (X ve y) hazırlanıyor...")

    X = []  # Tahta Durumları
    y = []  # Yapılması Gereken Hamle ID'si

    processed_count = 0

    for game in games:
        board = dp.init_board()
        moves = game["moves"]
        turn = 1  # 1: Beyaz, -1: Siyah

        for move in moves:
            # Sadece hamle sözlüğünde olanları işleyelim (Garanti olsun)
            if move not in move_to_id:
                continue

            # X: Şu anki tahta
            X.append(dp.encode_board(board.copy()))

            # y: Bu tahtada yapılması gereken hamlenin ID'si
            move_id = move_to_id[move]
            y.append(move_id)

            # Tahtayı güncelle
            board = dp.apply_move(board, move, turn)
            turn *= -1

        processed_count += 1
        if processed_count % 50 == 0:
            print(f"{processed_count} oyun işlendi...")

    # NumPy dizisine çevir
    X_array = np.array(X, dtype=np.int8)
    y_array = np.array(y, dtype=np.int16)  # ID'ler için int16 yeterli

    # Kaydet
    np.save(OUTPUT_DATA_FILE, X_array)
    np.save(OUTPUT_LABEL_FILE, y_array)

    print(f"\nİşlem Tamam dayı!")
    print(f"Eğitim Verisi (X): {X_array.shape} -> {OUTPUT_DATA_FILE}")
    print(f"Etiketler (y): {y_array.shape} -> {OUTPUT_LABEL_FILE}")
    print(f"Hamle Sözlüğü: {MOVE_DICT_FILE}")


if __name__ == "__main__":
    create_dataset()