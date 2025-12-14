import requests
import os
import time
import re


def get_elo_from_content(content):
    """
    Metin içinden WhiteElo ve BlackElo değerlerini bulur.
    """
    try:
        # Regex ile "WhiteElo" ve "BlackElo" etiketlerini arıyoruz
        white_match = re.search(r'\[WhiteElo "(\d+)"\]', content)
        black_match = re.search(r'\[BlackElo "(\d+)"\]', content)

        white_elo = int(white_match.group(1)) if white_match else 0
        black_elo = int(black_match.group(1)) if black_match else 0

        return white_elo, black_elo
    except:
        return 0, 0


def download_elite_games(start_id, count=50, min_elo=1800, save_dir="raw_games"):
    """
    Sadece belirtilen ELO üzerindeki kaliteli oyunları indirir.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print(f"--- ELIT İndirme Başlıyor: {start_id} ID'sinden itibaren ---")
    print(f"--- Hedef: ELO Ortalaması {min_elo}+ olan {count} oyun ---")

    saved_count = 0
    current_id = start_id
    checked_count = 0

    # Hedeflenen sayıya ulaşana kadar veya çok fazla (hedefin 10 katı) deneyene kadar devam et
    while saved_count < count and checked_count < count * 10:
        game_id = f"tu{current_id}"
        url = f"https://www.playok.com/p/?g={game_id}.txt"

        try:
            response = requests.get(url, timeout=5)

            if response.status_code == 200 and len(response.text) > 50:
                content = response.text

                # 1. Kontrol: Türk Daması mı?
                if '[GameType "30' in content:

                    # 2. Kontrol: ELO Yüksek mi?
                    w_elo, b_elo = get_elo_from_content(content)
                    avg_elo = (w_elo + b_elo) / 2

                    if avg_elo >= min_elo:
                        filename = f"{save_dir}/{game_id}_elo{int(avg_elo)}.txt"
                        with open(filename, "w", encoding="utf-8") as f:
                            f.write(content)

                        print(f"[+] KALİTELİ OYUN: {game_id} (Ort. ELO: {avg_elo}) -> Kaydedildi.")
                        saved_count += 1
                    else:
                        # ELO düşükse kaydetme ama ekrana bilgi ver
                        print(f"[-] Düşük ELO: {game_id} ({avg_elo}) -> Pas geçildi.")
                else:
                    # Türk daması değilse sessizce geç veya belirt
                    pass
            else:
                print(f"[!] {game_id} boş veya yok.")

        except Exception as e:
            print(f"[x] Hata: {e}")

        time.sleep(0.3)  # IP ban yememek için bekleme süresi
        current_id += 1  # ID'yi arttır (Geleceğe git)
        # current_id -= 1 # ID'yi azalt (Geçmişe git - İstersen bunu aç, üsttekini kapa)
        checked_count += 1

    print(f"\n--- İşlem Tamamlandı ---")
    print(f"Taranan ID sayısı: {checked_count}")
    print(f"Kaydedilen ELIT oyun sayısı: {saved_count}")


if __name__ == "__main__":
    # Başlangıç ID'sini senin dediğin yere çektik.
    # count=50: Bize şimdilik 50 tane sağlam maç yeter, boru hattını test edeceğiz.
    # min_elo=1700: Çıtayı biraz yüksek tutalım, ustaları izlesin.
    download_elite_games(start_id=24000000, count=50, min_elo=1700)