# dataset_23: CICIoT2023 (Kişi 1)

- Kaynak: https://www.unb.ca/cic/datasets/iotdataset-2023.html (Merged CSV sürümü)
- Kullanılan dosyalar: `Merged01.csv` … `Merged05.csv` (her biri ~140 MB, toplam 3.579.609 satır)
- Dosyalar Git'e girmez. Colab'da Drive'a koyun: `MyDrive/iot-data/dataset_23/`

Veri kontrolü (5 dosya, 18.10.2026):

| | satır |
|---|---|
| Ham | 3.579.609 |
| Tekrarlar silindikten sonra | 2.207.417 |
| inf içeren satır | 81 |

- 1.372.192 tekrarın yaklaşık 529 bini **dosyalar arası**. Bu yüzden tekrarlar birleştirmeden sonra silinir.
- `Tot size` sütunu `AVG` ile birebir aynı, `Variance` de `Std²`'ye eşit. İkisi de atılır.
- En nadir ince etiketler (tekil): UPLOADING_ATTACK 112, RECON-PINGSWEEP 183, BACKDOOR_MALWARE 234, XSS 326.

## Test sırasında bulunanlar (18.10.2026)

**1. `Number` bir veri toplama izi.** DDoS, DoS ve Mirai satırlarında `Number` ≈ 100, diğer bütün sınıflarda ≈ 10. Veri seti flood saldırıları 100 paketlik, geri kalanları 10 paketlik pencerelerle üretmiş. SHAP'ta neredeyse her sınıfın en önemli özelliği olarak çıkıyor.

Hızlı ablation (etiket başına 3.000 satır, 3-fold, LGBM test macro-F1):

| Varyant | macro-F1 |
|---|---|
| Tüm özellikler | 0,7155 |
| `Number` atılmış | 0,7146 |
| `Number` atılmış + çakışan satırlar atılmış | 0,7165 |

`Number`'ı atmak skoru değiştirmiyor, ama bu iz ortadan kalkıyor. Asıl deney `--drop-features Number` ile çalıştırılır.

**2. 192.066 satır çakışıyor (aynı özellikler, farklı etiket).** Bu satırlar 88.665 grup oluşturuyor; grupların 72.479'u DoS ile DDoS arasında. Pencere bazlı özellikler saldırgan sayısını görmediği için aynı flood'un DoS ve DDoS sürümleri ayırt edilemiyor. DoS ile DDoS arasındaki karışıklığın kaynağı bu; makalede sınırlama olarak yazılmalı. DSS açısından ikisinin aksiyonu aynı (BLOCK), bu yüzden karar sonucunu etkilemiyor. `--drop-conflicts` seçeneği bu satırları atar, ama zor örnekleri attığı için skoru yapay olarak yükseltir.

## Resmi sonuç ve duyarlılık analizi (18.10.2026)

Ekip protokolüne göre (Edge-IIoTset ile aynı) **çakışan satırlar atılarak** üretilen sonuç resmidir: `outputs/dataset_23/`. Çakışmaları tutan ilk çalıştırma duyarlılık analizi olarak `outputs_ablation/dataset_23_with_conflicts/` altında durur (Git'e girmez).

Tam deney (etiket başına en fazla 40.000 satır, 5-fold OOF, tuning, `Number` atılmış), test macro-F1:

| Model | Çakışmalar dahil | Çakışmalar atılmış (resmi) |
|---|---|---|
| RF (birincil) | 0,706 | 0,706 |
| XGB | 0,661 | 0,704 |
| LGBM | 0,675 | 0,678 |
| DT | 0,629 | 0,633 |
| LR | 0,551 | 0,552 |

- Birincil modelin skoru değişmiyor; yani çakışmaları atmak sonucu şişirmiyor. RF'de DoS F1 0,668 → 0,705, DDoS 0,837 → 0,870 oluyor.
- XGB'deki artış çakışmalardan değil, hiperparametre aramasının farklı ayar seçmesinden geliyor (200 → 400 ağaç, öğrenme oranı 0,05 → 0,2). Arama kararsız; iyileştirilmesi gerekiyor.
- DoS ve DDoS tek sınıf sayılırsa birleşik sınıfın F1'i 1,000 (çakışmalar dahil sürümde ölçüldü). Model flood saldırılarını kusursuz tespit ediyor; ayıramadığı şey yalnızca saldırgan sayısı.
- BRUTEFORCE testte yalnızca 202, WEB 378 örnek; bu sınıfların F1 değerleri çalıştırmalar arasında birkaç puan oynayabilir.
