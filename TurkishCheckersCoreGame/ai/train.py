import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os

# --- AYARLAR ---
DATA_FILE = "training_data.npy"
LABEL_FILE = "training_labels.npy"
DICT_FILE = "move_dictionary.json"
MODEL_SAVE_PATH = "dama_ai_model.pth"  # PyTorch uzantısı .pth olur

BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.001

# --- 1. Cihaz Seçimi (Varsa GPU, Yoksa CPU) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan Cihaz: {device}")


# --- 2. Veri Seti Sınıfı ---
class DamaDataset(Dataset):
    def __init__(self):
        # Verileri yükle
        self.X = np.load(DATA_FILE)
        self.y = np.load(LABEL_FILE)

        # PyTorch kanal yapısı (Channel, Height, Width) ister.
        # Bizim verimiz (Height, Width, Channel) -> (8, 8, 1)
        # Bunu (1, 8, 8) formatına çevirmemiz lazım.
        self.X = np.transpose(self.X, (0, 3, 1, 2))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Veriyi Tensor'a çevir (float32 formatında)
        board = torch.tensor(self.X[idx], dtype=torch.float32)
        label = torch.tensor(self.y[idx], dtype=torch.long)
        return board, label


# --- 3. CNN Modeli Mimarisi ---
class DamaCNN(nn.Module):
    def __init__(self, num_classes):
        super(DamaCNN, self).__init__()

        # Katman 1: Görme
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        # Katman 2: Detaylandırma
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        # Katman 3: Yapıyı Anlama
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=0)
        # Padding=0 olduğu için boyut 8x8'den 6x6'ya düşer

        # Karar Katmanları (Flatten + Dense)
        self.flatten = nn.Flatten()

        # 128 kanal * 6x6 boyut = 4608 giriş
        self.fc1 = nn.Linear(128 * 6 * 6, 256)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 128)
        self.out = nn.Linear(128, num_classes)  # Çıktı katmanı

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.conv3(x))

        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))

        # Softmax'i burada kullanmıyoruz, CrossEntropyLoss içinde gizli zaten
        x = self.out(x)
        return x


# --- ANA EĞİTİM DÖNGÜSÜ ---
if __name__ == "__main__":

    if not os.path.exists(DATA_FILE):
        print("Dosyalar yok! prepare_dataset.py çalıştır önce.")
        exit()

    # Hamle sayısını (sınıf sayısını) bul
    with open(DICT_FILE, "r") as f:
        move_dict = json.load(f)
    num_classes = len(move_dict)
    print(f"Toplam Hamle Sınıfı: {num_classes}")

    # Veri setini hazırla
    dataset = DamaDataset()
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Modeli oluştur ve GPU'ya (varsa) gönder
    model = DamaCNN(num_classes).to(device)

    # Hata hesaplayıcı ve Optimizasyoncu (Öğretmen)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ... (Kodun üst kısımleri aynı kalsın) ...

    print(f"\n--- EĞİTİM BAŞLIYOR ({EPOCHS} Tur) ---\n")
    print(f"Toplam Veri Sayısı: {len(dataset)}")

    # Toplam adım sayısını hesaplayalım
    total_steps = len(train_loader)

    for epoch in range(EPOCHS):
        running_loss = 0.0
        correct = 0
        total = 0

        model.train()

        for i, (boards, labels) in enumerate(train_loader):
            boards, labels = boards.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(boards)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # İstatistik
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # --- DEĞİŞİKLİK BURADA: HER 100 ADIMDA BİR BİLGİ VER ---
            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch + 1}/{EPOCHS}] -> Adım [{i + 1}/{total_steps}] | Anlık Hata: {loss.item():.4f}")

        # Tur sonu genel raporu
        acc = 100 * correct / total
        avg_loss = running_loss / len(train_loader)
        print(f"\n>>> Epoch [{epoch + 1}/{EPOCHS}] TAMAMLANDI! Ort. Hata: {avg_loss:.4f} | Doğruluk: %{acc:.2f}\n")

    # ... (Kaydetme kısmı aynı kalsın) ...

    # Modeli kaydet
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\nModel başarıyla '{MODEL_SAVE_PATH}' olarak kaydedildi!")