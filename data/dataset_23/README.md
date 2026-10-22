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

**Başarı ölçütleri** (güncellendi 21.10.2026, görülmemiş saldırı sonuçlarından ÖNCE; aşağıdaki nota bakın):

Sıra: (1) ablation'ın ortak formatta raporlanması, (2) LOFO RF, A / C / E, (3) TabM'in kapalı küme deneyine eklenmesi (PR #7), (4) LOFO TabM, A / C / E, (5) tohum 1 ve 2, (6) iki sinyalli kural (PR #8) ve CP-TabM ayar aktarımı, (7) DSS ortak ölçütü.

1. **Temel skorlar** (msp, entropi, enerji, anlaşmazlık, kNN; RF ve TabM): başarı iddiası yoktur; H1–H5 hipotezleriyle değerlendirilir.

2. **Ana yöntem: iki sinyalli kural** (TabM). TabM üyelerinin anlaşmazlığı alarm verir; alarm vermeyen ve BENIGN denen kararlar ikinci bir skorla denetlenir. **Kontrol skoru önceden tek seçilir: TabM enerjisi** (ana); msp yalnızca ikincil olarak raporlanır ve başarı kararında kullanılmaz ve normal trafiğin önceden belirlenen bir **analist bütçesi** kadarı (q ∈ {%0,5, %1, %2, %5, %10}) incelemeye ayrılır. Eşikler yalnızca kalibrasyondaki bilinen satırlarla seçilir. **Ortak ölçüt (Edge-IIoTset ile aynı): ALLOW edilen görülmemiş saldırı oranı** (sessiz BENIGN kaçışı), ana koşul C'de, 6 ailenin ortalaması, 3 tohum (`two_signal.py --model-seed 1` ve `2` ile; Edge-IIoTset de 3 tohumla tamamlanır). **Ana bütçe q = %5; ikincil bütçe q = %1** (Edge-IIoTset'teki ana ölçüt, doğrudan karşılaştırma için).

| Sonuç | Ölçüt |
| --- | --- |
| **Başarı** | q = %5'te ALLOW edilen görülmemiş saldırı oranı, yalnız alarm kuralına göre en az yarıya iner ve fark tohumlar arası sapmadan büyüktür; gerçekleşen inceleme oranı bütçeye yakın kalır (en fazla 1 puan fark) |
| **Kısmi başarı** | Oran azalır ama yarıya inmez; ya da yarıya iniş yalnızca q = %10'da görülür |
| **Başarısız** | Oran belirgin şekilde azalmaz; ya da gerçekleşen inceleme oranı bütçeyi belirgin şekilde aşar |

BRUTEFORCE ve WEB ayrıca raporlanır (H2). Kural sınıf tahminini değiştirmez; bilinen sınıflardaki macro-F1 aynı kalır.

3. **CP-TabM: ayar aktarımı.** Edge-IIoTset'in iç provasında seçilen kör oranı (0,0625) bu veri setinde yeniden ayarlanmadan, C koşulunda bir kez çalıştırılır. Sonuç dosyasında `blind_frac_source: ayar aktarımı` yazar. Başarı iddiası yoktur; sade TabM ile karşılaştırma tanımlayıcı olarak raporlanır. (Gerekçe: Edge-IIoTset'te CP-TabM sade TabM'i geçmedi ve iki sinyalli kuralda TabM'in kendi skorları aynı işi gördü; aktarılmış ayar, yöntemin başka bir veri setine genellenmesinin de temiz bir testidir.)

**Güncelleme notu (21.10.2026, sonuçlardan önce):** İlk sürümde 1. ölçüt "CP-TabM v3, iç provayla seçilen ayarla, en iyi temel skordan en az +0,05 AUROC" idi; 2. ölçüt "DSS'e bilinmeyen katmanı: ALLOW edilen görülmemiş saldırı en az yarıya, analist yükü en fazla 10 puan artış" idi. Edge-IIoTset'te iki sinyalli kuralın sonucu (CP-TabM'e gerek olmadığı) üzerine ekip kararıyla ana yöntem iki sinyalli kural, CP-TabM ayar aktarımı oldu; DSS ölçütü iki sinyalli kuralın ortak ölçütüyle birleştirildi ve analist yükü sabit bütçe olarak tanımlandı. Bu değişiklik CICIoT2023'te hiçbir görülmemiş saldırı sonucu (tesisat denemesi dışında) görülmeden yapıldı. Ana bütçe %5, **Edge-IIoTset'in iki sinyalli kural sonucu bilinerek** seçildi (orada sessiz kaçış %2,8'den %0,65–0,90'a iniyordu); CICIoT2023'ün sonucu bilinmiyordu. Edge-IIoTset'teki ana bütçe olan %1 karşılaştırma için ikincil ölçüt olarak raporlanır. Kontrol skoru (TabM enerjisi) da sonuçlardan önce tek olarak seçildi (Gamze Serin'in önerisi).

**Not (21.10.2026):** Koşullar ve ölçütler yazıldıktan sonra ortak araç (PR #6) `--quick` ile (etiket başına 3.000 satır, RF) üç koşulda tesisat denemesinden geçirildi: sütun adları doğrulandı, FLOOD ailesi iki kategoriyle çalışıyor. Bu deneme küçük örneklemli olduğu için sonuçları yorumlanmadı; koşullar ve ölçütler değiştirilmedi.

**Ekip kararları (21.10.2026, Gamze Serin ile):**

- **FLOOD tek aile.** Edge-IIoTset'te DoS kategorisi yok; oradaki DDOS ailesi buradaki FLOOD ile karşılaştırılır.
- **Doğrudan karşılaştırma ortak ailelerde:** FLOOD/DDOS, RECON, WEB, BRUTEFORCE. Yalnız bu veri setinde olan MIRAI ve SPOOFING (Edge-IIoTset'te MITM satırları sütun kayması yüzünden atıldı) ile yalnız Edge-IIoTset'te olan MALWARE veri setine özel raporlanır.
- **Karşılaştırmada dikkat:** Bilinen sınıf sayısı farklı (burada 8 kategori ve 6 aile, FLOOD iki kategori; Edge-IIoTset'te 6 kategori ve 5 aile). Veri setlerinin zorluğu da çok farklı (kapalı küme macro-F1 0,71 ve 0,99). Bu yüzden ham AUROC'lar değil göreli bulgular karşılaştırılır: kısayollar çıkınca ne kadar düşüyor, anlaşmazlık msp'den iyi mi, CP-TabM sade TabM'e göre nerede. Küçük ailelerde (testte BRUTEFORCE 202, WEB 378 satır) her sonuç tohumlar arası sapma ya da güven aralığıyla verilir.
- **Backdoor her veri setinde kendi yazarlarının gruplamasında kalır:** burada WEB (CICIoT2023'te web üzerinden taşınıyor), Edge-IIoTset'te MALWARE. WEB karşılaştırmasında bu fark not edilir.
- **Yöntem:** FP-TabM önerilmiyor; ana yöntem TabM + iki sinyalli kural. 3 tohumla sade TabM'in üye anlaşmazlığı (ort. AUROC 0,862 ± 0,001, Edge-IIoTset koşul C) CP-TabM v3'ten (0,805 ± 0,004) iyi; CP-TabM'in emin olma derecesi skoru ise sessiz BENIGN kaçışını %2,5'ten %0,2'ye indiriyor.
- **Ortak LOFO aracı:** `src/open_set_lofo.py` ve `src/common/open_set_registry.py` (PR #6, birleşti). Bu veri setinin kaydı orada: yükleyici `model_ml.load_data`'yı sarmalar, `Number` ve çakışmaları atar; `Number` veride kalırsa deney durur. Aileler ve A/C/E koşulları bu README'deki tanımlarla aynı.
- **Sıra:** yukarıdaki "Başarı ölçütleri" bölümünde. CP-TabM'in tam iç provası yerine ayar aktarımı (Gamze Serin'in önerisi, 21.10.2026).

## TabM, kapalı küme (21.10.2026)

TabM (Gorishniy vd., ICLR 2025; ekibin ortak `model_ml.py` bağlantısı, PR #7) resmi varsayılanlarla ve early stopping ile eğitildi; hiperparametre araması yok (Edge-IIoTset ile aynı). Diğer beş model önbellekten okundu, değişmedi. GPU: RTX 5060, eğitim 253 s.

| Model | Test macro-F1 | %95 aralık |
| --- | --- | --- |
| DSS birleşik tahmini (RF+XGB+LGBM) | 0,712 | 0,702 – 0,722 |
| XGB | 0,707 | 0,697 – 0,717 |
| RF | 0,706 | 0,696 – 0,716 |
| LGBM | 0,694 | 0,685 – 0,702 |
| **TabM** | **0,663** | **0,658 – 0,669** |
| DT | 0,644 | 0,635 – 0,651 |
| LR | 0,552 | 0,549 – 0,556 |

- **TabM ağaç modellerinin ~0,04 gerisinde** ve güven aralıkları çakışmıyor. Edge-IIoTset'te fark binde 3,5'ti. Fark en çok zor sınıflarda: WEB 0,18 (XGB 0,28), BRUTEFORCE 0,18 (XGB 0,31), RECON 0,74 (0,82). DDOS'ta ağaçlarla aynı (0,87).
- OOF macro-F1 0,654; DSS oylama kuralının (en iyinin en fazla 0,05 altı) dışında kalıyor, DSS'in oylayanları değişmedi (RF, XGB, LGBM).
- Early stopping: fold modelleri 12–26. epoch'ta, son model 6. epoch'ta durdu (ağırlıksız doğrulama kaybı, sabır 8).
- **Görülmemiş saldırı deneyi için not:** TabM, ölçütlerde belirlendiği gibi resmi varsayılanlarla kullanılır; sonuçlardan sonra ayar yapılmaz. TabM'li sonuçlar yorumlanırken bu veri setinde sınıflandırıcı olarak ağaç modellerinden zayıf olduğu, özellikle WEB ve BRUTEFORCE'ta, göz önünde tutulur.

## Görülmemiş saldırı deneyi: sonuçlar ve ölçütlerle değerlendirme (22.10.2026)

> Yukarıdaki protokol, hipotezler ve başarı ölçütleri değiştirilmeden uygulandı. Bu bölüm sonuçlardan SONRA yazıldı; ölçütler sonuçlara göre yeniden yorumlanmadı.

**Çalıştırılanlar** (hepsi ortak araçlarla, ekip bölmesi seed 42):

- Ablation, ortak format: `src/ablation_dnn_edgeiiot.py --dataset dataset_23` → `outputs/dataset_23/ablation/ablation.json`
- LOFO: RF ve TabM, A / C / E (tohum 42); koşul C'de tohum 1 ve 2 (`--model-seed`, dosya eki `_seed1`, `_seed2`); CP-TabM, C, ayar aktarımı (kör oranı 0,0625) → `outputs/dataset_23/open_set/`
- İki sinyalli kural: koşul C, tohum 42, 1, 2 → `outputs/dataset_23/two_signal/` (satır dosyaları `.npz` yeniden üretilebilir, depoya girmez)

### Ablation, ortak format

Ortak araç sabit ayarlarla çalışır (RF 100 ağaç, derinlik 40, `max_features` 0,5; DT derinlik 20; sınıf ağırlığı yok; 300 bootstrap). Bu yüzden değerler yukarıdaki ilk tablodan biraz düşük; ekip karşılaştırmasında bu tablo esas alınır.

| Koşul | RF test macro-F1 [%95 GA] | DT | En güçlü tek sütun |
| --- | --- | --- | --- |
| A: tam (36) | 0,695 [0,683 – 0,706] | 0,633 | `Max` 0,46, `Tot sum` 0,45 |
| C: −`Time_To_Live` (35) | 0,691 (−0,004) | 0,628 | |
| E: + −`IAT`, `Rate`, `Header_Length` (32) | 0,670 (−0,025) | 0,629 | |

Bulgu aynı: kısayol adayları çıkınca kayıp en fazla 0,025. **Not:** E'de, kalan 32 sütunda farklı kategoriden birebir aynı olan satır oranı %0,03'ten %32'ye çıkıyor (A ve C'de çakışmalar zaten atılmıştı; E'de sütun azalınca yeniden oluşuyor). Bunun büyük kısmı DoS ↔ DDoS arasında (DDoS ince etiketlerinin %81–90'ı, DoS'un %53–77'si), yani aynı FLOOD ailesinin içinde. Ama BENIGN satırlarının %26'sı ve RECON, WEB, BRUTEFORCE etiketlerinin %4–18'i de başka kategoriyle çakışıyor. `IAT`/`Rate` sızıntı değil, satırları ayıran gerçek bilgi taşıyor; E bu yüzden "kısayolsuz" değil "bilgisi azaltılmış" bir koşul olarak yorumlanır.

### LOFO, koşul C (3 tohum ortalaması ± sapma, eşik: kalibrasyonda bilinenlerin %95'i)

| Görülmemiş aile | RF anlaşmazlık AUROC | RF sessiz BENIGN | TabM en iyi skor AUROC | TabM sessiz BENIGN (o skorla) | kNN AUROC (tohum 42) |
| --- | --- | --- | --- | --- | --- |
| FLOOD | 0,857 ± 0,006 | %0,2 | anlaşmazlık 0,770 ± 0,025 | %0 | **0,977** |
| MIRAI | 0,960 ± 0,010 | %0 | anlaşmazlık 0,959 ± 0,037 | %0 | 0,912 |
| RECON | 0,921 ± 0,000 | **%30,0** | enerji **0,978 ± 0,007** | **%4,6** | 0,829 |
| SPOOFING | 0,950 ± 0,001 | %10,2 | enerji 0,972 ± 0,002 | %9,3 | 0,810 |
| WEB | 0,910 ± 0,003 | %5,6 | enerji 0,950 ± 0,004 | %12,9 | 0,753 |
| BRUTEFORCE | 0,922 ± 0,003 | %9,4 | anlaşmazlık 0,945 ± 0,005 | %20,8 | 0,799 |

- Tohumlar arası sapma küçük (AUROC'ta çoğunlukla < 0,01); sonuçlar tohuma bağlı değil.
- RF ve TabM birbirini tamamlıyor: TabM enerjisi RECON'daki sessiz kaçışı %30'dan %5'e indiriyor; WEB ve BRUTEFORCE'ta ise RF daha az sessiz kaçış veriyor.
- TabM'in msp, entropi ve enerjisi görülmemiş MIRAI'de rastgele düzeyde (AUROC 0,52 – 0,56); MIRAI'yi yalnız üye anlaşmazlığı yakalıyor. FLOOD'da da TabM skorları zayıf (≤ 0,77); en iyisi kNN.

### Hipotezler

| | Sonuç | Kanıt |
| --- | --- | --- |
| H1 | **Kısmen desteklendi** | RF: anlaşmazlık msp'den 6/6 ailede yüksek (ortalama +0,08). TabM: anlaşmazlık msp ve enerjiden yalnız 3/6 ailede (FLOOD, MIRAI, BRUTEFORCE) yüksek; RECON, SPOOFING, WEB'de enerji daha iyi. Edge-IIoTset'in aksine burada enerji çökmüyor. |
| H2 | **Desteklenmedi** | WEB ve BRUTEFORCE'ta en iyi AUROC 0,95 ve 0,95 (< 0,75 beklenmişti). Sessiz kaçış TabM'de yüksek (%13, %21), ama RF'de en yüksek kaçış RECON'da (%30). Kapalı kümedeki zorluk (F1 0,28 / 0,31) bilinmeyen tespitine taşınmıyor. |
| H3 | **Desteklendi** | Görülmemiş MIRAI'nin %99,9'u DDOS/DOS'a atanıyor (güvenli hata, RF tohum 42); sessiz BENIGN kaçışı her modelde ve tohumda %0. |
| H4 | **Desteklendi** (gerekçesi kısmen değil) | FLOOD için RF/TabM skorlarının AUROC'u görece düşük (0,86 / 0,77), sessiz kaçış ≤ %0,2. Ancak görülmemiş FLOOD MIRAI'ye değil çoğunlukla RECON'a atanıyor (RF: RECON %56, MIRAI %27). |
| H5 | **Desteklendi** | Aile başına en iyi skorun (kNN dahil) ortalama AUROC'u, tohum 42: RF A 0,940 → E 0,951; TabM A 0,967 → E 0,965. Fark ≤ 0,011 (sınır 0,05). Edge-IIoTset'teki çöküş burada yok. |

### Başarı ölçütü: iki sinyalli kural (ana yöntem)

Koşul C, 6 ailenin ortalaması, 3 tohum; kontrol skoru **TabM enerjisi** (önceden seçilen). Değer: ALLOW edilen görülmemiş saldırı oranı (sessiz BENIGN kaçışı).

| Bütçe q | Yalnız alarm | İki sinyalli | Oran | Gerçekleşen inceleme |
| --- | --- | --- | --- | --- |
| %1 (ikincil, Edge-IIoTset ana ölçütü) | %12,8 ± 2,0 | %12,4 ± 1,9 | 0,97 | %0,73 |
| **%5 (ana)** | **%12,8 ± 2,0** | **%10,8 ± 1,7** | **0,85** | **%3,6** |
| %10 | %12,8 ± 2,0 | %9,2 ± 1,2 | 0,72 | %7,0 |

Tohum başına (q = %5): 14,1 → 11,9; 13,8 → 11,8; 10,5 → 8,9. Aile başına (q = %5, 3 tohum ort.): RECON 23,6 → 19,8; WEB 20,1 → 17,0; BRUTEFORCE 20,8 → 17,2; SPOOFING 12,4 → 11,1; FLOOD ve MIRAI 0 → 0. msp ile (ikincil) sonuç benzer: %10,5.

**Karar: kısmi başarı.** Oran her tohumda azalıyor (ortalama −2,0 puan, tohumlar arası sapma kadar ama her tohumda aynı yönde), fakat hiçbir bütçede yarıya inmiyor. Gerçekleşen inceleme oranı bütçenin altında (q = %5'te %3,6; ölçütteki "en fazla 1 puan fark"tan 1,4 puan aşağıda, yani analist bütçeyi aşmıyor. Fark tanımdan geliyor: `two_signal.py` eşiği q'yu yalnız alarm almamış ve BENIGN denmiş gerçek BENIGN satırlarına uygular, inceleme oranını ise **bütün** BENIGN test satırlarına böler; bu satırların ~%72'si kontrole gelir ve 0,72 × %5 ≈ %3,6. Edge-IIoTset'te de aynı tanım geçerli). Kural sınıf tahminini değiştirmez; bilinen sınıflardaki macro-F1 aynıdır.

**Yorum:** Kalan kaçış neredeyse tamamen RECON, WEB ve BRUTEFORCE'tan geliyor; bu aileler normal trafiğe benziyor ve BENIGN sanılan satırları enerji de normalden ayıramıyor. Not: TabM enerjisi *alarm* skoru olarak RECON'da sessiz kaçışı %4,6'ya indiriyor (LOFO tablosu); iki sinyalli kuralda alarm üye anlaşmazlığından verildiği için bu güç kullanılmıyor. Bu, kuralın kurgusunun (önceden sabitlendi) bir sonucu olarak raporlanır, sonradan değiştirilmez.

### CP-TabM, ayar aktarımı (betimleyici, başarı iddiası yok)

Koşul C, tohum 42, aile başına en iyi skor: ortalama AUROC CP-TabM 0,936, sade TabM 0,919; sessiz kaçış %7,2 ve %7,4. Farkın çoğu FLOOD'dan geliyor (0,866 / 0,744); diğer ailelerde iki model benzer. Aktarılan ayar bu veri setinde çalışıyor, ama sade TabM'e belirgin bir üstünlük yok (Edge-IIoTset'teki sonuçla tutarlı).

### Henüz yapılmayanlar

- A ve E koşulları yalnız tohum 42 ile çalıştı (3 tohum ana koşul C'de; protokoldeki "3 tohum" ifadesi ana koşul için uygulandı). A/E'deki tek tohumlu farklar, C'deki tohum sapmasıyla (AUROC'ta çoğunlukla < 0,01) birlikte okunur.

- `--keep-conflicts` duyarlılık kontrolü (protokolde bir kez öngörüldü).
- Güvenli hata oranı yalnız atama dağılımı olarak raporlandı (`unknown_assigned_to`); DSS kararına (BLOCK/ALERT) çevrilmiş hali henüz hesaplanmadı.
