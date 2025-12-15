import pygame
import torch
import torch.nn as nn
import numpy as np
import json
import os
import sys
import random

# --- YOL AYARLARI ---
sys.path.append(os.path.join(os.path.dirname(__file__), 'ai'))

from dama.constants import WIDTH, HEIGHT, SQUARE_SIZE, WHITE, BLACK, ROWS, COLS
from dama.game import Game

# --- KULLANICI AYARLARI ---
# BURADAN SEÇ: AI hangi renk olsun?
# BLACK -> AI Siyah (Sen Beyaz, taşların aşağıda)
# WHITE -> AI Beyaz (Sen Siyah, taşların aşağıda - EKRAN DÖNER)
AI_PLAYER = BLACK

FPS = 60
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "ai", "dama_ai_model.pth")
DICT_PATH = os.path.join(BASE_DIR, "ai", "move_dictionary.json")

WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption(f"Dama AI: AI ({'BEYAZ' if AI_PLAYER == WHITE else 'SİYAH'}) vs İNSAN")


# --- MODEL MİMARİSİ ---
class DamaCNN(nn.Module):
    def __init__(self, num_classes):
        super(DamaCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=0)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 6 * 6, 256)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 128)
        self.out = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.conv3(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        return self.out(x)


# --- YARDIMCI FONKSİYONLAR ---
def load_ai():
    print("Yapay Zeka yükleniyor...")
    if not os.path.exists(DICT_PATH):
        print("Sözlük yok!")
        return None, None, None

    with open(DICT_PATH, "r") as f:
        move_to_id = json.load(f)
    id_to_move = {v: k for k, v in move_to_id.items()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"AI Motoru: {device}")

    model = DamaCNN(len(move_to_id)).to(device)

    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            model.eval()
            print(">> Beyin yüklendi!")
        except:
            print(">> Model yüklenemedi, rastgele oynayacak.")
    else:
        print(">> Model dosyası yok.")

    return model, id_to_move, device


def board_to_matrix(game_board, ai_color):
    """
    AI Beyaz ise tahtayı mantıksal olarak çevirir (Ayna Taktiği).
    """
    matrix = np.zeros((8, 8), dtype=np.int8)

    for row in range(ROWS):
        for col in range(COLS):
            piece = game_board.get_piece(row, col)
            if piece != 0:
                val = 1 if piece.color == WHITE else -1
                if piece.king: val *= 2
                matrix[row, col] = val

    if ai_color == WHITE:
        matrix = matrix * -1
        # .copy() ÖNEMLİ: PyTorch negatif stride hatasını önler
        matrix = np.rot90(matrix, 2).copy()

    return matrix.reshape(8, 8, 1)


def playok_to_coords(move_str, ai_color):
    """
    AI'nın hamlelerini koordinata çevirir. AI Beyaz ise koordinatı düzeltir.
    """
    cols_map = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}

    clean_str = move_str.replace("x", "-")
    parts = clean_str.split("-")
    if len(parts) < 2: return None, None

    try:
        s_col = cols_map[parts[0][0]]
        s_row = 8 - int(parts[0][1:])
        e_col = cols_map[parts[1][0]]
        e_row = 8 - int(parts[1][1:])

        # AI Beyaz ise o ters dünyada yaşıyor, koordinatları gerçeğe çevir
        if ai_color == WHITE:
            s_row, s_col = 7 - s_row, 7 - s_col
            e_row, e_col = 7 - e_row, 7 - e_col

        return (s_row, s_col), (e_row, e_col)
    except:
        return None, None


def get_row_col_from_mouse(pos, rotate_screen):
    """
    Mouse koordinatlarını alır.
    Eğer ekran dönükse (rotate_screen=True), mouse'u da tersine çevirir.
    """
    x, y = pos

    # EKRAN DÖNÜKSE MOUSE'U DA DÖNDÜR
    if rotate_screen:
        x = WIDTH - x
        y = HEIGHT - y

    row = y // SQUARE_SIZE
    col = x // SQUARE_SIZE
    return row, col


def main():
    run = True
    clock = pygame.time.Clock()
    game = Game(WIN)

    model, id_to_move, device = load_ai()

    # Eğer AI Beyaz ise, sen Siyahsın demektir -> Ekranı döndüreceğiz.
    ROTATE_SCREEN = (AI_PLAYER == WHITE)

    print("\n" + "=" * 40)
    if AI_PLAYER == WHITE:
        print("MOD: Sen SİYAHSIN, AI BEYAZ.")
        print("Ekran senin için döndürüldü.")
    elif AI_PLAYER == BLACK:
        print("MOD: Sen BEYAZSIN, AI SİYAH.")
    print("=" * 40 + "\n")

    while run:
        clock.tick(FPS)

        if game.winner() is not None:
            print(f"KAZANAN: {game.winner()}")
            run = False

        # --- AI HAMLESİ ---
        if (game.turn == AI_PLAYER) or (AI_PLAYER is None):
            pygame.time.wait(600)

            # AI için tahtayı hazırla
            current_turn_color = game.turn
            matrix = board_to_matrix(game.get_board(), current_turn_color)

            ai_input = np.transpose(matrix, (2, 0, 1))
            tensor_board = torch.tensor(ai_input, dtype=torch.float32).unsqueeze(0).to(device)

            found_move = False

            with torch.no_grad():
                outputs = model(tensor_board)
                probs, indices = torch.topk(outputs, 100)
                candidates = indices.cpu().numpy()[0]

            valid_moves = game.get_all_valid_moves(current_turn_color)
            valid_moves = game.filter_max_capture_moves(valid_moves)

            for move_id in candidates:
                move_str = id_to_move.get(move_id)
                if not move_str: continue

                start, end = playok_to_coords(move_str, current_turn_color)
                if not start: continue

                piece = game.board.get_piece(start[0], start[1])
                if piece != 0 and piece in valid_moves:
                    if end in valid_moves[piece]:
                        print(f"AI ({'Beyaz' if current_turn_color == WHITE else 'Siyah'}) Oynadı: {move_str}")
                        game.select(start[0], start[1])
                        game.select(end[0], end[1])
                        found_move = True
                        break

            if not found_move:
                print("AI hamle bulamadı, rastgele atıyor...")
                if valid_moves:
                    p = random.choice(list(valid_moves.keys()))
                    m = random.choice(list(valid_moves[p].keys()))
                    game.select(p.row, p.col)
                    game.select(m[0], m[1])

        # --- İNSAN HAMLESİ ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                if AI_PLAYER is None: continue

                if game.turn != AI_PLAYER:
                    pos = pygame.mouse.get_pos()
                    # Mouse koordinatını ROTATE durumuna göre alıyoruz
                    row, col = get_row_col_from_mouse(pos, ROTATE_SCREEN)
                    game.select(row, col)

        # --- ÇİZİM İŞLEMLERİ (MANUEL UPDATE) ---
        # Game class'ındaki update'i çağırmıyoruz, çünkü araya girmemiz lazım.
        game.board.draw(WIN)
        game.draw_valid_moves(game.valid_moves)

        # EĞER EKRAN DÖNECEKSE:
        if ROTATE_SCREEN:
            # Mevcut ekranı al, 180 derece çevir ve tekrar bas
            rotated_surface = pygame.transform.rotate(WIN, 180)
            WIN.blit(rotated_surface, (0, 0))

        pygame.display.update()

    pygame.quit()


if __name__ == "__main__":
    main()