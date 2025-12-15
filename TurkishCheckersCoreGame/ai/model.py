import torch
import torch.nn as nn
import torch.nn.functional as F


class DamaNet(nn.Module):
    def __init__(self):
        super(DamaNet, self).__init__()

        # --- GÖRME KATMANI (Convolutional Layers) ---
        # 1. Katman: Tahtayı tarar.
        # in_channels=1: Çünkü tahtamız tek katmanlı (sadece sayılar var, RGB renkli değil)
        # out_channels=32: 32 farklı özellik/desen arayacak (köşe sıkışması, ikili yeme pozisyonu vb.)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)

        # 2. Katman: Bulduğu desenleri birleştirir.
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        # 3. Katman: Daha karmaşık stratejiler geliştirir.
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)

        # --- KARAR KATMANI (Fully Connected Layers) ---
        # flatten: 8x8'lik haritayı düz bir çizgiye çevirir.
        # 8 * 8 * 128 (boyutlar)
        self.fc1 = nn.Linear(8 * 8 * 128, 512)
        self.fc2 = nn.Linear(512, 128)

        # ÇIKIŞ KATMANI
        # Şimdilik 1 çıktı veriyoruz: Bu tahta durumu ne kadar iyi? (Value Network)
        # Pozitifse Beyaz kazanır, Negatifse Siyah kazanır.
        self.fc_value = nn.Linear(128, 1)

    def forward(self, x):
        # x: Tahta Verisi (Batch_Size, 1, 8, 8)

        # Görme işlemleri + Aktivasyon (ReLU = Negatif değerleri at, pozitiflere odaklan)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Düzleştirme (Matristen Vektöre)
        x = x.view(-1, 8 * 8 * 128)

        # Karar Verme
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Sonuç (Value)
        value = torch.tanh(self.fc_value(x))  # -1 ile 1 arası sonuç verir

        return value


if __name__ == "__main__":
    # TEST KISMI
    # Rastgele bir tahta verip model çalışıyor mu bakalım.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model şu cihazda test ediliyor: {device}")

    model = DamaNet().to(device)

    # Rastgele bir tahta oluştur (1 oyun, 1 kanal, 8 satır, 8 sütun)
    dummy_input = torch.randn(1, 1, 8, 8).to(device)

    output = model(dummy_input)
    print(f"Model Çıktısı (Tahmin Skoru): {output.item()}")
    print("Beyin başarıyla oluşturuldu!")