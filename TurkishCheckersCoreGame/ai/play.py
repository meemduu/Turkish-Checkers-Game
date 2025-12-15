import torch
import torch.nn as nn
import numpy as np
import json
import os
import data_processor as dp  # Senin yazdığın processor

# --- AYARLAR ---
MODEL_PATH = "dama_ai_model.pth"
DICT_PATH = "move_dictionary.json"


# --- MODEL MİMARİSİ (Train ile aynı olmalı) ---
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
def load_stuff():
    # 1. Sözlüğü Yükle ve Ters Çevir (ID -> Hamle İsmi)
    with open(DICT_PATH, "r") as f:
        move_to_id = json.load(f)

    # { "a1-a2": 0 }  --> { 0: "a1-a2" }
    id_to_move = {v: k for k, v in move_to_id.items()}

    # 2. Modeli Yükle
    num_classes = len(move_to_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DamaCNN(num_classes).to(device)

    # Eğitilmiş ağırlıkları yükle (map_location cpu önemli)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()  # Modeli 'Sınav Modu'na al (Dropout'u kapatır)

    return model, move_to_id, id_to_move, device


def print_board_visual(board):
    print("\n   a b c d e f g h")
    print("  -----------------")
    for r in range(8):
        row_str = f"{8 - r}| "
        for c in range(8):
            val = board[r, c]
            if val == 1:
                symbol = "⚪"  # Beyaz
            elif val == 2:
                symbol = "WB"  # Beyaz Dama
            elif val == -1:
                symbol = "⚫"  # Siyah
            elif val == -2:
                symbol = "BB"  # Siyah Dama
            else:
                symbol = " ."
            row_str += symbol + " "
        print(row_str)
    print("\n")


def get_ai_move(model, board, id_to_move, device):
    # Tahtayı modelin anlayacağı formata çevir
    # (8,8,1) -> (1,8,8) -> (1,1,8,8) [Batch boyutu ekle]
    encoded = dp.encode_board(board)  # (8,8,1)

    # Transpose: (8,8,1) -> (1,8,8)
    encoded = np.transpose(encoded, (2, 0, 1))

    # Tensor yap ve batch boyutu ekle
    tensor_board = torch.tensor(encoded, dtype=torch.float32).unsqueeze(0).to(device)

    # Tahmin et
    with torch.no_grad():
        outputs = model(tensor_board)

        # En yüksek puanı alan hamleyi seç (Argmax)
        # İleride buraya 'geçerli hamle kontrolü' eklenebilir.
        _, predicted_id = torch.max(outputs, 1)
        move_id = predicted_id.item()

    return id_to_move.get(move_id, None)


# --- OYUN DÖNGÜSÜ ---
def play():
    model, move_to_id, id_to_move, device = load_stuff()
    board = dp.init_board()

    print("--- DAMA AI ARENA ---")
    print("Sen: BEYAZ (⚪) | AI: SİYAH (⚫)")
    print("Hamle formatı: 'c3-d4' veya 'a1xa2' (yeme)")

    while True:
        print_board_visual(board)

        # --- SENİN SIRAN (BEYAZ) ---
        user_move = input("Senin Hamlen (Çıkış için 'q'): ").strip().lower()
        if user_move == 'q': break

        # Hamleyi uygula
        # (Burada hamle geçerli mi diye kontrol etmiyoruz, kullanıcıya güveniyoruz şimdilik)
        try:
            board = dp.apply_move(board, user_move, 1)
        except Exception as e:
            print(f"Hata: Geçersiz hamle formatı! ({e})")
            continue

        print_board_visual(board)
        print("Yapay Zeka düşünüyor...")

        # --- AI SIRASI (SİYAH) ---
        ai_move_str = get_ai_move(model, board, id_to_move, device)

        print(f">> AI Oynadı: {ai_move_str}")

        if ai_move_str:
            board = dp.apply_move(board, ai_move_str, -1)
        else:
            print("AI pas geçti veya hamle bulamadı!")


if __name__ == "__main__":
    play()