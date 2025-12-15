import numpy as np

# --- AYARLAR ---
BOARD_SIZE = 8


# Tahta Temsili:
# 0: Boş
# 1: Beyaz Piyon
# 2: Beyaz Dama
# -1: Siyah Piyon
# -2: Siyah Dama
# (Şimdilik sadece piyon varmış gibi basit başlıyoruz, dama mantığını model öğrensin)

def init_board():
    """
    Türk Daması başlangıç pozisyonunu oluşturur.
    2. ve 3. satırlar Beyaz (1), 5. ve 6. satırlar Siyah (-1).
    """
    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)

    # Beyazlar (Satır 1 ve 2 - İndeks 1 ve 2)
    board[1, :] = 1
    board[2, :] = 1

    # Siyahlar (Satır 5 ve 6 - İndeks 5 ve 6)
    board[5, :] = -1
    board[6, :] = -1

    return board


def parse_square(sq_str):
    """
    'a1' gibi bir metni (row, col) koordinatına çevirir.
    PlayOK'ta 'a1' sol alt köşedir.
    Matrisimizde: board[7][0] -> a1 olmalı.

    a -> 0, h -> 7
    1 -> 7, 8 -> 0 (Ters çeviriyoruz ki matrisle uyumlu olsun)
    """
    col_map = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}

    try:
        col = col_map[sq_str[0]]
        row = 8 - int(sq_str[1:])  # '1' -> 7 (En alt satır), '8' -> 0 (En üst)
        return row, col
    except:
        return None


def apply_move(board, move_str, player_color):
    """
    PlayOK formatındaki hamleyi tahtada oynatır.
    Örn: 'h3-h4' veya 'e8xe3xb3'
    """
    # Hamle parçalarını ayır (hem '-' hem 'x' ayırıcı olabilir)
    # Regex yerine basit replace kullanalım
    clean_str = move_str.replace("x", "-")
    steps = clean_str.split("-")

    # Başlangıç karesi
    start_sq = parse_square(steps[0])
    if not start_sq: return board

    curr_r, curr_c = start_sq

    # Hareket eden taşı al
    piece = board[curr_r, curr_c]

    # Eğer o karede taş yoksa (veri hatası olabilir), işlem yapma
    if piece == 0:
        return board

    # Taşı başlangıçtan kaldır
    board[curr_r, curr_c] = 0

    # Zincirleme hamleleri tek tek işle
    for next_step_str in steps[1:]:
        next_sq = parse_square(next_step_str)
        if not next_sq: continue

        next_r, next_c = next_sq

        # --- YEME MANTIĞI ---
        # Eğer arada taş varsa onu kaldırmamız lazım.
        # Türk damasında hareket dikeydir/yataydır.

        # Yönü bul
        r_step = 0
        if next_r > curr_r:
            r_step = 1
        elif next_r < curr_r:
            r_step = -1

        c_step = 0
        if next_c > curr_c:
            c_step = 1
        elif next_c < curr_c:
            c_step = -1

        # Aradaki kareleri tarayıp düşman taşlarını temizle
        # (Basit bir yaklaşım: Başlangıç ve bitiş arasındaki her şeyi sil)
        temp_r, temp_c = curr_r + r_step, curr_c + c_step
        while (temp_r != next_r) or (temp_c != next_c):
            board[temp_r, temp_c] = 0  # Yenen taşı kaldır
            temp_r += r_step
            temp_c += c_step

            # Sonsuz döngü koruması (Hatalı veri için)
            if not (0 <= temp_r < 8 and 0 <= temp_c < 8): break

        # Yeni konuma geç
        curr_r, curr_c = next_r, next_c

    # Taşı son konuma koy
    board[curr_r, curr_c] = piece

    # Dama olma kontrolü (Basit versiyon)
    if piece == 1 and curr_r == 0: board[curr_r, curr_c] = 2  # Beyaz Dama
    if piece == -1 and curr_r == 7: board[curr_r, curr_c] = -2  # Siyah Dama

    return board


def encode_board(board):
    """
    Yapay zeka için tahtayı 0-1 formatına çevirir (One-Hot Encoding benzeri).
    Çıktı Şekli: (8, 8, 1) -> Basit tutuyoruz, tek kanal (Kendi taşın pozitif, rakip negatif)
    """
    # Yapay zeka veriyi normalize sever. -2 ile 2 arasını -1 ile 1 arasına sıkıştırabiliriz
    # Veya direkt olduğu gibi verebiliriz. Şimdilik olduğu gibi verelim.
    return board.reshape(8, 8, 1)