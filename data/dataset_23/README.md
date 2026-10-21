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

## İyileştirmeler sonrası nihai sonuç (19.10.2026)

Değişiklikler: LR yakınsama (max_iter 5000), tuning 30 deneme / 200 bin satır, XGB ve LGBM'de **early stopping** (ağaç sayısı aranmıyor; eğitim verisinin %10'u doğrulama; durma kriteri **ağırlıksız** doğrulama kaybı), DSS tüm test setinde ve ciddiyet grubuna göre karar.

Test macro-F1 (147.629 satır), %95 bootstrap güven aralığı:

| Model | Macro-F1 | %95 aralık | Ağaç sayısı |
|---|---|---|---|
| **DSS** (RF+XGB+LGBM) | **0,712** | 0,702 – 0,722 | |
| **XGB** (birincil, OOF 0,714) | 0,707 | 0,697 – 0,717 | 1.275 (early stopping) |
| RF | 0,706 | 0,696 – 0,716 | 200 |
| LGBM | 0,694 | 0,685 – 0,702 | 88 (early stopping) |
| DT | 0,644 | 0,635 – 0,651 | |
| LR | 0,552 | 0,549 – 0,556 | |

- DSS, XGB ve RF'nin güven aralıkları çakışıyor: aralarındaki farklar istatistiksel olarak anlamlı değil. LGBM, DT ve LR açıkça geride.
- **Early stopping teşhisi:** Sınıf ağırlıklı doğrulama kaybı 64. turda en düşüğe iniyor (nadir sınıfların birkaç düzine örneği hızla ezberleniyor), macro-F1 ise ~400. tura kadar artıyor. Ağırlıksız kayıp macro-F1 ile aynı noktada en düşük olduğu için durma kriteri olarak o kullanıldı. Ağırlıklı kriterle XGB 64 turda duruyor ve 0,667 alıyordu.
- **NIDS:** saldırı tespit oranı %98,5; yanlış alarm oranı XGB %17,5, DSS %17,7, RF %19,2.
- **DSS kararları:** Ciddiyet grubuna göre karar sayesinde DoS, DDoS ve Mirai'nin %100'ü BLOCK alıyor (önceden DDoS'un ~üçte biri LOG_MONITOR'a düşüyordu).
- **En zayıf sınıflar** hâlâ BRUTEFORCE (0,31; aralık 0,24–0,38) ve WEB (0,28; 0,25–0,33). İkisi de çoğunlukla RECON sanılıyor; az örnek yüzünden aralıklar geniş.

## Görülmemiş saldırı deneyi: protokol ve başarı ölçütleri (sonuçlardan ÖNCE belirlendi, 21.10.2026)

> Aile tanımları, Backdoor kategorisi ve yöntem sırası ekip olarak kararlaştırıldı (21.10.2026, aşağıda). Kısayol koşulları (A/C/E) görülmemiş saldırı sonuçlarından ÖNCE, aşağıdaki gerekçelerle seçildi. Sonuçlar görüldükten sonra bu bölüm değiştirilmez; değişiklik gerekirse tarihli bir not olarak eklenir ve gerekçesi yazılır.

**Protokol** (Edge-IIoTset deneyiyle aynı; `data/dnn_edgeiiot/README.md`):

- Ekip bölmesi aynen kullanılır: 80/20, ince etikete göre katmanlı, seed 42; test kümesi asıl deneyle aynı (147.629 satır).
- **Aileler (6):** FLOOD (DoS + DDoS), MIRAI, RECON, SPOOFING, WEB, BRUTEFORCE. DoS ve DDoS tek aile sayılır, çünkü pencere özellikleri saldırgan sayısını göremiyor: çakışan satır gruplarının %82'si bu ikisi arasında ve birleşik F1 1,000. Ayrı tutulsaydı "görülmemiş DDoS"un DoS sanılması bir tespit başarısızlığı gibi görünürdü.
- Her turda bir aile eğitimden tamamen çıkarılır; kalan eğitim verisi %90 eğitim / %10 kalibrasyon. Eşik yalnızca kalibrasyondaki **bilinen** satırlarla seçilir (bilinenlerin %95'i kabul). Çıkarılan aileden hiçbir bilgi kullanılmaz.
- **Koşullar (A/C/E):** Bütün koşullarda `Number` (veri toplama izi) ve çakışan etiketli satırlar atılmıştır; satırlar her koşulda aynıdır, yalnızca sütunlar değişir. A: tüm 36 özellik. **C (ana koşul):** `Time_To_Live` çıkarılmış. E: ayrıca `IAT`, `Rate` ve `Header_Length` çıkarılmış. Seçimin gerekçesi aşağıda ("Kısayol adayları"). Duyarlılık kontrolü olarak bir kez `--keep-conflicts`.
- **Skorlar:** msp, entropi, topluluk anlaşmazlığı (RF: ağaçlar; TabM: k üye), enerji (TabM), kNN (5 komşu, QuantileTransformer uzayı). XGB ve LGBM'in ağaçları sırayla eğitildiği için bağımsız üye sayılmaz; anlaşmazlık skoru onlarda kullanılmaz.
- **Metrikler:** AUROC, AUPR, FPR95, kalibrasyon eşiğinde yakalama ve yanlış red oranı, open-set macro-F1, bilinen sınıflarda macro-F1 ve:
  - **Sessiz BENIGN kaçışı:** görülmemiş saldırının kabul edilip BENIGN sanılması (en tehlikeli hata).
  - **Güvenli hata oranı** (bu veri setine eklenen): görülmemiş saldırının bilinmeyen diye işaretlenmeden **başka bir saldırı** sanılıp DSS'te yine BLOCK ya da ALERT alması.
- Sonuçlar 3 eğitim tohumuyla (42, 1, 2), ortalama ± standart sapma olarak; her ailenin sonucu ayrı raporlanır.

**Kısayol adayları ve koşulların seçimi** (Edge-IIoTset'teki üç adım: tek özellik testi, SHAP, ablation):

1. **Tek özellik sızıntı testi** (her sütun tek başına, derinlik 8 karar ağacı, kategori başına en fazla 5.000 satır, 3-fold): tek başına sınıfı ele veren sütun yok. En güçlüler `Tot sum` 0,46, `Max` 0,46, `AVG` 0,42, `Header_Length` 0,39; tüm özellikler birlikte 0,63. `IAT` tek başına 0,30: Kaggle'daki "CICIoT2026" kopyasında görülen IAT sızıntısı bu Merged sürümde yok; değerler gerçekçi (flood'larda mikrosaniye, normal trafikte milisaniye). Çıktı: `outputs/dataset_23/ablation/single_feature_leakage.csv`.
2. **Kayda özgü olabilecek sütunlar** (değer dağılımları ve SHAP):
   - **`Time_To_Live`** (C ve E): flood saldırılarında (DDoS, DoS, Mirai) satırların yarısından fazlasında TTL tam 64, normal trafikte 59–217 arasına yayılıyor. TTL gönderen makinenin işletim sistemini ve ağdaki konumunu gösterir; saldırıyı değil, laboratuvardaki saldırgan makineyi tanıtır. SHAP'ta BENIGN, BRUTEFORCE ve WEB'in en önemli dört özelliğinden biri. Edge-IIoTset'teki `tcp.seq`/`tcp.ack`'ın karşılığı olarak C'de çıkarılır.
   - **`Header_Length`** (E): Mirai'de neredeyse hep 0 (medyan 0,24); büyük olasılıkla özellik çıkarma aracının GRE paketlerini ayrıştıramamasından kaynaklanan bir ölçüm izi.
   - **`IAT` ve `Rate`** (E): neredeyse aynı bilgi (Spearman −0,996). Gerçek zamanlama davranışı taşıyorlar, ama değerleri laboratuvarın bant genişliğine bağlı ve IAT veri setinin başka sürümlerinde sızıntı yapmıştı. Gerçek ağda anlamlı oldukları için C'de kalır, en sert koşulda çıkarılır.
   - Paket boyutu istatistikleri (`Tot sum`, `AVG`, `Max`, `Min`, `Std`), protokol göstergeleri ve bayrak sayıları saldırı davranışının kendisidir; hiçbir koşulda çıkarılmaz.
3. **Ablation** (kapalı küme, aynı satırlar ve aynı bölme, sabit ayarlar; RF 200 ağaç, DT derinlik 30; çıktı `outputs/dataset_23/ablation/ablation_macro_f1.csv`):

| Koşul | Çıkarılan | RF test macro-F1 | DT | Edge-IIoTset RF (karşılaştırma) |
| --- | --- | --- | --- | --- |
| A: tam | – | 0,706 | 0,644 | 0,993 |
| C: ana | `Time_To_Live` | 0,701 (−0,005) | 0,644 | 0,824 (−0,169) |
| E: en sert | + `IAT`, `Rate`, `Header_Length` | 0,681 (−0,025) | 0,635 | 0,618 (−0,376) |

**Bulgu:** CICIoT2023'te kapalı küme başarısı kısayol adaylarına neredeyse hiç dayanmıyor; en sert koşulda bile kayıp 0,025. E'de en çok etkilenen sınıflar WEB (0,28 → 0,23) ve BRUTEFORCE (0,31 → 0,26). Edge-IIoTset'te aynı işlem başarıyı 0,99'dan 0,62'ye düşürüyordu. Bu nedenle CICIoT2023, ekibin yönteminin kısayol olmadan nasıl çalıştığını gösteren veri setidir.

**Hipotezler** (yönleri önceden yazılır; sonuç ne çıkarsa raporlanır):

| | Hipotez | Dayanak |
| --- | --- | --- |
| H1 | Topluluk anlaşmazlığı, msp ve enerjiden yüksek AUROC verir (6 ailenin en az 4'ünde). | Edge-IIoTset'te kısayolsuz koşullarda msp ve enerji rastgeleye yaklaştı. |
| H2 | En zor aileler WEB ve BRUTEFORCE olur: en iyi skorla bile AUROC < 0,75 ve sessiz BENIGN kaçışı diğer ailelerden yüksek. | Bu iki sınıf bilinen haldeyken bile çoğunlukla RECON sanılıyor (F1 0,28 ve 0,31). |
| H3 | Görülmemiş MIRAI çoğunlukla FLOOD'a atanır; sessiz BENIGN kaçışı %5'in altında kalır (güvenli hata). | MIRAI de bir flood; ciddiyeti aynı. |
| H4 | Görülmemiş FLOOD için AUROC düşük olabilir (MIRAI'ye benzerlik), ama sessiz BENIGN kaçışı %1'in altında kalır. | Flood trafiği normal trafikten çok farklı. |
| H5 | Bilinmeyen tespiti A'dan E'ye Edge-IIoTset'tekinden çok daha az düşer (en iyi skorun ortalama AUROC'u A ile E arasında en fazla 0,05 fark). | Kapalı küme ablation'ında A → E kaybı 0,025; Edge-IIoTset'te 0,376. |

**Başarı ölçütleri:**

Sıra (ekip kararı): önce temel skorlar A/C/E'de (TabM ve en iyi ağaç modeli: msp, entropi, enerji, anlaşmazlık, kNN); sonra CP-TabM v3 iç provayla (`cp_tune`); ekibin iki sinyalli karar kuralı (anlaşmazlık alarm verir, emin olma derecesi BENIGN kararlarını denetler) netleşince o da eklenir. FP-TabM Edge-IIoTset'te durduruldu, çalıştırılmaz.

1. **Önerilen yöntem** (CP-TabM v3), ana koşul C'de 6 ailenin ortalamasıyla en iyi temel skorla (RF ve TabM skorları ile kNN dahil) karşılaştırılır. Ayarlar Edge-IIoTset'teki gibi yalnızca iç provayla seçilir: her dış aile için, eğitimdeki diğer ailelerden biri geçici olarak "bilinmeyen" yapılır; seçilen ayar dış aile üzerinde bir kez çalıştırılır. Test ailesine bakılarak ayar yapılmaz.

| Sonuç | Ölçüt |
| --- | --- |
| **Açık başarı** | AUROC'ta en iyi temel skordan en az +0,05; sessiz BENIGN kaçışı yaklaşık yarıya; bilinen sınıflarda macro-F1 kaybı sade TabM'e göre en fazla 0,01; fark tohumlar arası sapmadan büyük |
| **Kısmi başarı** | Ortalamada +0,01–0,05, ama WEB ve BRUTEFORCE'ta ya da sessiz BENIGN kaçışında belirgin iyileşme |
| **Başarısız** | Temel skorlarla aynı veya daha kötü; ya da tespit artarken bilinen başarı belirgin düşüyor. Sonuç ve olası nedenleri olduğu gibi raporlanır. |

2. **DSS'e bilinmeyen katmanı** ("bilinmeyen → ANALYST_REVIEW, asla ALLOW"):

| Sonuç | Ölçüt |
| --- | --- |
| **Başarı** | ALLOW edilen görülmemiş saldırı oranı katmansız DSS'e göre en az yarıya iner; normal trafikte ANALYST_REVIEW'a giden oran en fazla 10 puan artar |
| **Kısmi başarı** | ALLOW edilen görülmemiş saldırı azalır ama yarıya inmez, ya da analist yükü 10 puandan fazla artar |
| **Başarısız** | ALLOW edilen görülmemiş saldırı belirgin şekilde azalmaz |

**Not (21.10.2026):** Koşullar ve ölçütler yazıldıktan sonra ortak araç (PR #6) `--quick` ile (etiket başına 3.000 satır, RF) üç koşulda tesisat denemesinden geçirildi: sütun adları doğrulandı, FLOOD ailesi iki kategoriyle çalışıyor. Bu deneme küçük örneklemli olduğu için sonuçları yorumlanmadı; koşullar ve ölçütler değiştirilmedi.

**Ekip kararları (21.10.2026, Gamze Serin ile):**

- **FLOOD tek aile.** Edge-IIoTset'te DoS kategorisi yok; oradaki DDOS ailesi buradaki FLOOD ile karşılaştırılır.
- **Doğrudan karşılaştırma ortak ailelerde:** FLOOD/DDOS, RECON, WEB, BRUTEFORCE. Yalnız bu veri setinde olan MIRAI ve SPOOFING (Edge-IIoTset'te MITM satırları sütun kayması yüzünden atıldı) ile yalnız Edge-IIoTset'te olan MALWARE veri setine özel raporlanır.
- **Karşılaştırmada dikkat:** Bilinen sınıf sayısı farklı (burada 8 kategori ve 6 aile, FLOOD iki kategori; Edge-IIoTset'te 6 kategori ve 5 aile). Veri setlerinin zorluğu da çok farklı (kapalı küme macro-F1 0,71 ve 0,99). Bu yüzden ham AUROC'lar değil göreli bulgular karşılaştırılır: kısayollar çıkınca ne kadar düşüyor, anlaşmazlık msp'den iyi mi, CP-TabM sade TabM'e göre nerede. Küçük ailelerde (testte BRUTEFORCE 202, WEB 378 satır) her sonuç tohumlar arası sapma ya da güven aralığıyla verilir.
- **Backdoor her veri setinde kendi yazarlarının gruplamasında kalır:** burada WEB (CICIoT2023'te web üzerinden taşınıyor), Edge-IIoTset'te MALWARE. WEB karşılaştırmasında bu fark not edilir.
- **Yöntem:** FP-TabM önerilmiyor. 3 tohumla sade TabM'in üye anlaşmazlığı (ort. AUROC 0,862 ± 0,001, Edge-IIoTset koşul C) CP-TabM v3'ten (0,805 ± 0,004) iyi; CP-TabM'in emin olma derecesi skoru ise sessiz BENIGN kaçışını %2,5'ten %0,2'ye indiriyor.
- **Ortak LOFO aracı:** `src/open_set_lofo.py` ve `src/common/open_set_registry.py` (PR #6, birleşti). Bu veri setinin kaydı orada: yükleyici `model_ml.load_data`'yı sarmalar, `Number` ve çakışmaları atar; `Number` veride kalırsa deney durur. Aileler ve A/C/E koşulları bu README'deki tanımlarla aynı.
- **Sıra (ekip kararı, Gamze Serin'in 21.10.2026 önerisi):** (1) TabM'in kapalı küme deneyine 6. model olarak eklenmesi, (2) görülmemiş saldırı deneyi RF + TabM, A / C / E, (3) tohum 1 ve 2, (4) ablation'ın ortak formatta (sabit RF/DT ayarları, bootstrap aralıkları) raporlanması, (5) CP-TabM iç provası ve iki sinyalli karar kuralı.
