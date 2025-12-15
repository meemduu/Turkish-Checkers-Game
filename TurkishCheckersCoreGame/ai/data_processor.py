import numpy as np


class HeadlessBoard:
    def __init__(self):
        # 8x8 Matris: 0=Boş, 1=Beyaz, -1=Siyah, 2=Beyaz Dama, -2=Siyah Dama
        self.board = np.zeros((8, 8), dtype=int)
        self.setup_board()

    def setup_board(self):
        # Siyahlar (Üstte: Satır 1 ve 2)
        self.board[1:3, :] = -1
        # Beyazlar (Altta: Satır 5 ve 6)
        self.board[5:7, :] = 1

    def move_piece(self, start_pos, end_pos):
        r1, c1 = start_pos
        r2, c2 = end_pos

        piece = self.board[r1, c1]
        self.board[r1, c1] = 0  # Eski yeri boşalt
        self.board[r2, c2] = piece  # Yeni yere koy

        # Dama Olma Kontrolü (Basit mantık)
        if piece == 1 and r2 == 0:
            self.board[r2, c2] = 2  # Beyaz Dama
        elif piece == -1 and r2 == 7:
            self.board[r2, c2] = -2  # Siyah Dama

        # Eğer arada taş yendiyse onu kaldır
        # Basit yeme: Aradaki fark 2 ise (Yoz taşlar için genelde)
        # Veya Dama yemiştir. Ortalamasını alıp silelim.
        # Not: PlayOK notasyonunda her adım tek tek veriliyor, o yüzden
        # sadece başlangıç ve bitiş arasındaki karelere bakmak yeterli.

        dr = r2 - r1
        dc = c2 - c1

        # Yön belirle
        step_r = 0 if dr == 0 else int(dr / abs(dr))
        step_c = 0 if dc == 0 else int(dc / abs(dc))

        # Aradaki kareleri tarayıp temizle (Yeme işlemi)
        cur_r, cur_c = r1 + step_r, c1 + step_c
        while (cur_r, cur_c) != (r2, c2):
            self.board[cur_r, cur_c] = 0
            cur_r += step_r
            cur_c += step_c

    def print_board(self):
        # Konsolda görmek için basit çizim
        print(self.board)
        print("-" * 20)


def parse_coordinate(coord_str):
    """
    Örnek: 'h3' -> (row, col)
    PlayOK: 1 en alt, 8 en üst.
    Bizim Matris: 0 en üst, 7 en alt.
    """
    col_map = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}

    col_char = coord_str[0]
    row_num = int(coord_str[1:])

    col = col_map[col_char]
    row = 8 - row_num  # Dönüşüm: 8->0, 1->7

    return (row, col)


def process_game_text(game_text):
    board = HeadlessBoard()
    dataset = []  # (Board_Durumu, Hamle_Metni) ikililerini saklayacağız

    # Metni temizle ve hamleleri ayır
    # 1. h3-h4 gibi numaraları temizlememiz lazım.
    import re

    # Satır satır oku
    lines = game_text.strip().split('\n')
    moves = []

    for line in lines:
        if line.startswith('['): continue  # Metadata'yı geç
        if not line: continue

        # "1. h3-h4 a6-a5" formatını parçala
        # Regex ile sadece hamleleri çekelim (h3-h4 veya h5xh3 gibi)
        found = re.findall(r'([a-h][1-8][\-x][a-h][1-8](?:[x][a-h][1-8])*)', line)
        moves.extend(found)

    print(f"Toplam {len(moves)} hamle bulundu.")

    for move_str in moves:
        # Mevcut tahta durumunu kopyala (AI'a input olacak)
        current_state = board.board.copy()
        dataset.append((current_state, move_str))

        # Hamleyi işle
        if 'x' in move_str:  # Yeme işlemi (Zincirleme olabilir: d8xd5xa5)
            steps = move_str.split('x')
            # Zincirleme hamleleri adım adım oynat
            for i in range(len(steps) - 1):
                start = parse_coordinate(steps[i])
                end = parse_coordinate(steps[i + 1])
                board.move_piece(start, end)
        else:  # Normal hamle (h3-h4)
            start_str, end_str = move_str.split('-')
            start = parse_coordinate(start_str)
            end = parse_coordinate(end_str)
            board.move_piece(start, end)

    return dataset


# --- TEST KISMI ---
if __name__ == "__main__":
    # Senin attığın oyun verisi
    sample_game = """
1. h3-h4 a6-a5 2. b3-b4 g6-g5 3. f3-f4 e6-e5 4. e3-f3 e5-f5 5. h2-h3 d6-d5 6.
b4-a4 a7-a6 7. e2-e3 e7-e6 8. e3-e4 d7-d6 9. b2-b3 g7-g6 10. b3-b4 h7-g7 11.
h4-g4 h6-h5 12. a2-b2 a5-b5 13. b2-b3 c6-c5 14. e4-d4 d5-e5 15. h3-h4 h5xh3 16.
d4-e4 e5xe3 17. f2-e2 e3xe1 18. f4-e4 e1xe5 19. f3-e3 e5xe2xh2 20. b4-c4 h3xf3
21. c4xc6xc8 g5xg3 22. d3-e3 f3xd3xd1 23. c8-d8 h2xb2xb4 24.
d8xd5xa5xa7xe7xe5xg5xg1xb1xb5xb8 1-0
    """

    print("Oyun işleniyor...")
    data = process_game_text(sample_game)

    print("\n--- Sonuç Örneği (Son Hamle Sonrası Tahta) ---")
    # Son durumdaki tahtayı yazdıralım
    # 24. hamledeki o efsane temizlikten sonra tahta boşalmış olmalı
    final_board = HeadlessBoard()
    for state, move in data:
        # Sadece son durumu görmek için tekrar oynatmak yerine
        # Parser'ın içindeki board nesnesini dışarı almadık,
        # ama state'ler kaydedildi.
        pass

    print(f"İşlenen hamle sayısı: {len(data)}")
    print("Veri başarıyla matrise çevrildi!")

    # İlk hamlenin matris halini görelim
    print("\nBaşlangıç Matrisi (İlk input):")
    print(data[0][0])
    print(f"\nYapılan Hamle: {data[0][1]}")