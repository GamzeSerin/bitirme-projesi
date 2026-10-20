# dnn_edgeiiot: Edge-IIoTset (Kişi 3)

- Kaynak (resmî, yazar hesabı): https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot
- Kullanılan dosya: `Edge-IIoTset dataset/Selected dataset for ML and DL/DNN-EdgeIIoT-dataset.csv` (1,22 GB, 2.219.201 satır, 63 sütun). Arşivin geri kalanı (PCAP'ler, ham CSV'ler, ML dosyası) kullanılmaz.
- Lisans: CC BY-NC-SA 4.0 (akademik kullanım serbest, atıf zorunlu). Dosya Git'e girmez.
- Atıf: M. A. Ferrag, O. Friha, D. Hamouda, L. Maglaras, H. Janicke, "Edge-IIoTset: A New Comprehensive Realistic Cyber Security Dataset of IoT and IIoT Applications for Centralized and Federated Learning," *IEEE Access*, 2022. doi:10.1109/ACCESS.2022.3165809
- Çalışma ortamı: Kaggle Notebooks. Veri not defterine girdi olarak bağlanır, indirilmez (`notebooks/dnn_edgeiiot_kaggle.ipynb`).

Ortam: endüstriyel / edge IoT test düzeneği (7 katman, 10'dan fazla sensör türü, Modbus TCP, MQTT). Satırlar **paket düzeyinde**dir (CICIoT2023 pencere/akış düzeyindedir); bu yüzden özellikler diğer veri setleriyle eşleşmez.

## Etiketler ve ortak kategoriler

| Ortak kategori | Edge-IIoTset etiketleri | Ham satır |
| --- | --- | --- |
| BENIGN | Normal | 1.615.643 |
| DDOS | DDoS_UDP, DDoS_ICMP, DDoS_TCP, DDoS_HTTP | 337.977 |
| WEB | SQL_injection, Uploading, XSS | 104.752 |
| RECON | Vulnerability_scanner, Port_Scanning, Fingerprinting | 73.675 |
| BRUTEFORCE | Password | 50.153 |
| MALWARE | Backdoor, Ransomware | 35.787 |
| SPOOFING | MITM | 1.214 |

Eşleme `src/common/taxonomy.py` içindeki `EDGEIIOT_MAP`'tedir. Not: CICIoT2023'te `BACKDOOR_MALWARE` resmi gruplamada WEB'e gider; Edge-IIoTset'te Backdoor makaledeki "Malware" tehdidine ait olduğu için MALWARE'e eşlendi.

## Veri denetimi (18.10.2026, Aşama 0 not defteri)

**1. Kimlik, zaman ve ham içerik sütunları atılır.** 19 metin sütunu: `frame.time`, IP adresleri (`ip.src_host`, `ip.dst_host`, `arp.*.proto_ipv4`), ham içerik (`tcp.payload`, `http.file_data`, `http.request.full_uri`, `http.request.uri.query`, `tcp.options`, `mqtt.msg`), `tcp.srcport` ve az değerli kategorik alanlar (`http.request.method`, `mqtt.protoname`, `mqtt.topic` vb.). 4 sütun sabit (`icmp.unused`, `http.tls_port`, `dns.qry.type`, `mqtt.msg_decoded_as`).

**2. Sayısal ama kimlik gibi davranan 4 sütun atılır:** `tcp.ack_raw` (ham onay numarası, her bağlantıda rastgele başlar), `tcp.checksum`, `icmp.checksum`, `udp.stream` (kayıt içindeki akış numarası). Kalan: **34 özellik**.

**3. Tek-özellik sızıntı testi** (derinlik 8 karar ağacı, sınıf başına 5.000 satır): tek başına en güçlü özellik `tcp.seq` (macro-F1 0,425), ardından `tcp.ack` 0,421, `tcp.dstport` 0,388, `tcp.len` 0,371; tüm özellikler birlikte 0,538. CICIoT2023'teki `IAT` gibi tek başına sınıfı ele veren bir sütun yok. `tcp.dstport`, `udp.port`, `tcp.seq`, `tcp.ack` kısayol adayı: ablation `--drop-features tcp.dstport,udp.port,tcp.seq,tcp.ack`.

**4-5. Tekrarlar ve etiket çakışması: satır muhasebesi** (19.10.2026'da netleştirildi)

Tanımlar (hepsi 34 özellik üzerinden):
- **Tekrar:** özellikleri VE etiketi birebir aynı olan satırın ikinci ve sonraki kopyaları. Bunlar hatalı kayıt olmak zorunda değildir; sel saldırılarında aynı başlığa sahip binlerce meşru paket vardır. Silinme nedeni: aynı vektörün kopyaları hem eğitime hem teste düşüp skoru şişirmesin.
- **Etiket çakışması:** aynı özellik vektörünün birden fazla farklı ince etiketle görünmesi. Model bu satırları ayırt edemez; etiketler paket bazında değil kayıt dosyası bazında verildiği için saldırı kayıtlarına karışan sıradan paketler saldırı etiketi almıştır. Çakışan grubun tüm satırları atılır (`--drop-conflicts`).

Sayılar işlem sırasına bağlıdır; iki sıra da aynı veri üzerinde doğrudur:

| Sıra | Adım | Kalan satır | Atılan |
| --- | --- | --- | --- |
| Asıl deney (`model_dnn_edgeiiot.py`) | Ham | 2.219.201 | |
| | Tekrarlar silindi | 688.640 | 1.530.561 (**%69,0**) |
| | Çakışan satırlar atıldı | 675.675 | 12.965 (%1,9) |
| Aşama 0 not defteri | Ham | 2.219.201 | |
| | Çakışan satırlar atıldı (kopyalar dahil sayıldı) | 2.022.583 | 196.618 |
| | Tekrarlar silindi | 675.656 | 1.346.927 (ham verinin %60,7'si) |

Raporlanacak tekrar oranı asıl deneydeki **%69,0**'dır. Aşama 0'daki "%61", çakışmalar önce atıldığı için daha düşük çıkmıştır. 196.618 ile 12.965 aynı olguyu iki farklı sırada sayar: ilki her kopyayı ayrı sayar, ikincisi tekrarlar silindikten sonraki tekil satırları (iki sıranın sonunda kalan satır sayısı 19 satır farkla aynıdır).

Kimlik gibi davranan 4 sütun (38 özellik) varken çakışma (tekrarlar silinmeden) yalnızca 21.486 satırdı: o sütunlar modelin ayırt edilemeyen paketleri "ezberlemesini" sağlıyordu.

Hangi etiketler çakışıyor, sıraya göre değişir:
- **Kopyalar dahil sayılınca (Aşama 0)** en çok MITM etkileniyor (1.214 satırın 821'i): ARP paketleri normal ARP trafiğinden ayırt edilemiyor.
- **Tekil satırlar sayılınca (asıl deney, 12.965 satır):** Password 3.273, Normal 2.751, Uploading 1.990, SQL_injection 1.716, XSS 1.152, DDoS_HTTP 985, Ransomware 579, Vulnerability_scanner 430; MITM yalnızca 1. Yani tekil düzeyde çakışma çoğunlukla HTTP/TCP tabanlı saldırılarla normal trafik arasında: aynı bağlantıdaki el sıkışma ve onay paketleri saldırıya özgü değil.

**6. Temizlik sonrası nadir sınıflar:** MITM 357, Fingerprinting 181 satır.

**7. Sütunları kaymış satırlar: DDoS_UDP ve MITM (19.10.2026).** Bu iki saldırının satırlarında sütunlar kaymış:
- **DDoS_UDP:** `frame.time`'da saldırganın (sahte) IP'si, `ip.src_host`'ta hedef IP, `ip.dst_host`'ta 0 var; UDP flood portu `udp.port` yerine `tcp.srcport`'ta. En azından baştaki sütunlar bir sola kaymış.
- **MITM (ARP zehirleme):** `frame.time`'da "6.0", ARP alanlarının hepsi sıfır, bazı satırlarda `udp.port`/`tcp.srcport`'ta mDNS metni (`_ipp._tcp.local`).

Kayma yayımlanan **DNN, ML ve ham "Attack traffic" CSV'lerinin hepsinde** aynı, dolayısıyla düzgün bir kaynak yok. Sütun sayısı (63) ve etiketler yerinde olduğu için kaymanın nerede bittiği tek satırdan çıkarılamıyor; tahminle hizalamak güvenilir değil. Diğer 13 türde ve Normal'de `frame.time` her satırda geçerli bir zaman damgası.

Sonuç: model bu iki türü saldırının kendisinden değil, "neredeyse tamamı sıfır, düzeni bozuk satır" kalıbından tanıyordu (MITM her koşulda, kısayolsuz modelde bile F1 = 1,0). **Hizasız satırlar `frame.time` biçimine bakılarak bulunur ve atılır** (`misaligned_mask`, ham veride DDoS_UDP 121.568 + MITM 1.214 = 122.782 satır, %5,53). DDOS kategorisi üç alt türle (HTTP, ICMP, TCP) devam eder; **SPOOFING kategorisi bu veri setinde yoktur.** Duyarlılık analizi için `--keep-misaligned`. Düzeltme yolu olarak ileride MITM'in PCAP kaydından yeniden özellik çıkarılabilir.


## Sonuçlar (19.10.2026, temiz veri, asıl deney)

Ayarlar: `--drop-conflicts --tune --shap --gpu`; hizasız satırlar atıldı; etiket başına en fazla 40.000 satır; stratified 80/20 (seed=42); 5-fold OOF; 30 deneme / 200 bin satırlık ayar araması (Kişi 1 ile aynı protokol); tahminler **tüm test kümesinde** (59.302 satır). Kaggle, GPU (T4). 18.10.2026 sonuçları hizasız satırlar veride iken alınmıştı ve geçersizdir (yerlerini bu bölüm aldı).

**Veri hazırlama:** ham 2.219.201 → hizasız 122.782 satır atıldı → tekrarlar silindikten sonra 676.748 → çakışan 12.963 satır atıldı → örneklem **296.506 satır**, 34 özellik, **6 kategori** (SPOOFING yok). Eğitim 237.204, test 59.302.

**Test kümesinin kullanımı:** hiperparametre araması yalnızca eğitim kümesinin alt örnekleminde 3-fold CV ile yapılır, DSS güvenilirlik ağırlıkları eğitim kümesinin 5-fold OOF tahminlerinden gelir. Test kümesi yalnızca son değerlendirmede kullanılır.

| Model | OOF macro-F1 | Test macro-F1 | Eğitim (s) | Tahmin (ms/1k) | Seçilen ayarlar |
| --- | --- | --- | --- | --- | --- |
| **RF** (birincil) | 0,9941 | **0,9940** | 24,6 | 5,0 | 200 ağaç, derinlik 40, max_features 0,5, min_samples_leaf 2 |
| DT | 0,9938 | 0,9938 | 0,7 | 0,1 | sınırsız derinlik, gini, min_samples_leaf 2 |
| XGB | 0,9919 | 0,9924 | 8,3 (GPU) | 2,7 | derinlik 4, lr 0,2; early stopping 603 ağaç |
| TabM | 0,9915 | 0,9905 | 465 (GPU) | 20,5 | k=32, PLE gömmeleri, ayar araması yok; early stopping 37 epoch |
| LGBM | 0,9911 | 0,9902 | 12,8 | 25,9 | 63 yaprak, lr 0,05; early stopping 145 ağaç |
| LR | 0,7453 | 0,7453 | 43,7 | 0,6 | C = 1 |

RF test F1 (sınıf): BENIGN 0,9996 · RECON 0,9967 · BRUTEFORCE 0,9899 · **WEB 0,9852** · MALWARE 0,9993 · DDOS 0,9930. İnce etiket recall'ı 0,99'un altında olanlar: DDoS_HTTP 0,952, Fingerprinting 0,889.

**DSS (tüm test kümesi):** oylayanlar DT, RF, XGB, LGBM, TabM (LR elendi); macro-F1 **0,9941**. İkili NIDS bakışı: tespit oranı %100,0, yanlış alarm %0,00, kaçırılan saldırı 2 (yaklaşık 51 bin saldırıdan). Aksiyonlar: BLOCK 27.119, ALERT 23.393, ALLOW 7.991, LOG_MONITOR 775, ANALYST_REVIEW 24; **ALLOW edilen saldırı: 1**.

**Bulgular:**

1. **Aşırı uyum yok:** OOF ve test skorları her modelde çok yakın.
2. **TabM**, ayar araması olmadan LGBM ile aynı seviyede, RF'nin binde 3,5 gerisinde. Ekibin durma kuralı (ağırlıksız doğrulama kaybı) TabM'de de doğrulandı: kayba ve doğrulama macro-F1'ine göre en iyi epoch'lar arasında macro-F1 farkı en fazla ~0,001.
3. **En zayıf kategori WEB** (0,985): DDoS_HTTP (recall 0,952) WEB ile karışıyor; içerik sütunları atıldığı için model yalnızca başlık alanlarını görüyor.
4. **SHAP (RF):** her sınıfta en önemli üç özellik neredeyse aynı: `tcp.ack`, `tcp.dstport`, `tcp.seq` (DDOS'ta `tcp.flags`). Sınıfa özgü bir açıklama yok; bkz. ablation ve zaman bölmesi.

## Kısayol özellik ablation deneyi (19.10.2026, temiz veri)

Kod: `src/ablation_dnn_edgeiiot.py`, çıktı: `outputs/dnn_edgeiiot/ablation/`. Protokol: asıl deneydeki satırlar ve aynı 80/20 bölme (seed=42); RF ve DT sabit hiperparametrelerle; yalnızca sütunlar çıkarılır; %95 güven aralıkları 300 bootstrap.

| Senaryo | Çıkarılan | RF test macro-F1 (%95 GA) | DT | Fark (RF) | Başka kategoriyle aynı vektöre düşen satır |
| --- | --- | --- | --- | --- | --- |
| A: Tam | – | 0,9932 (0,9925–0,9939) | 0,9932 | | %0 |
| B: Portsuz | `tcp.dstport`, `udp.port` | 0,9570 (0,9554–0,9586) | 0,9561 | −0,036 | %13,4 |
| C: Sıra/onay yok | `tcp.seq`, `tcp.ack` | 0,8239 (0,8206–0,8270) | 0,8533 | −0,169 | %22,2 |
| D: Sayaç/zaman yok | `mbtcp.trans_id`, `icmp.transmit_timestamp` | 0,9932 (0,9925–0,9938) | 0,9931 | 0,000 | %0 |
| E: Hepsi | 6 sütun | **0,6175** (0,6133–0,6215) | 0,6169 | **−0,376** | **%59,2** |

**Bulgular:**

1. **Yüksek skorun büyük kısmı dört alandan geliyor.** Port ve TCP sıra/onay numaraları çıkınca macro-F1 0,993'ten 0,618'e iniyor; satırların %59'u başka bir kategorideki bir satırla birebir aynı hâle geliyor.
2. **En büyük katkı `tcp.seq`/`tcp.ack`'tan (−0,169), sonra portlardan (−0,036).** Bunlar Wireshark'ın göreli değerleri (bağlantı içindeki konum); kimlik sızıntısı değil, saldırı aracına ve kayıt düzenine bağlı **bağlama özgü kısayol**.
3. **E'de en çok çöken sınıflar:** BRUTEFORCE (F1 0,35), WEB (0,40), MALWARE (0,64), BENIGN (0,70; Normal recall 0,57 → çok sayıda yanlış alarm). İnce etiketlerde XSS (recall 0,20), Uploading (0,22), SQL_injection (0,31), DDoS_HTTP (0,37). Bu saldırıların paketleri aynı HTTP/TCP bağlantılarının sıradan paketleriyle aynı başlığa sahip (Backdoor %98, Ransomware %100, Password %84, Uploading %86 çakışma).
4. **Şüpheli iki sütun temiz:** `mbtcp.trans_id` (22 farklı değer) ve `icmp.transmit_timestamp` (84) tek başına şans düzeyinde (macro-F1 0,11); çıkarmak sonucu değiştirmiyor (D).
5. **Temizlik bulguyu güçlendirdi:** hizasız satırlar veride iken E'de RF 0,675 idi; MITM'in her koşulda F1 = 1,0 alması skoru yapay olarak yükseltiyordu.

## Zaman bazlı bölme kontrolü (19.10.2026, temiz veri)

Kod: `src/split_check_dnn_edgeiiot.py`, çıktı: `outputs/dnn_edgeiiot/split_check/`. Aynı satırlar ve sabit hiperparametreler. **Rastgele:** ince etikete göre stratified 80/20, her seed farklı bölme. **Zaman:** her ince etiketin kayıt sırasına (CSV satır sırası) göre ilk %80'i eğitim, son %20'si test. `frame.time`'da tarih yok (yalnızca günün saati); kayıt sırası ile saat sırası 11 etikette tamamen aynı (Spearman ρ = 1,0), Normal (0,88) ve Ransomware (0,11) gece yarısını geçtiği için satır sırası kullanılır. `frame.time` hiçbir zaman özellik değildir. 5 seed.

| Koşul | Model | Rastgele | Zaman | Fark | Yanlış alarm (rastgele → zaman) | Kaçırılan saldırı (zaman) |
| --- | --- | --- | --- | --- | --- | --- |
| A (34 özellik) | RF | 0,9937 ± 0,0002 | **0,9237 ± 0,0129** | −0,070 | %0,01 → **%3,9** | %0,00 |
| A (34 özellik) | DT | 0,9934 ± 0,0002 | **0,8943 ± 0,0008** | −0,099 | %0,03 → **%4,0** | %0,00 |
| E (kısayolsuz) | RF | 0,6157 ± 0,0014 | 0,6204 ± 0,0001 | +0,005 | %43,1 → %42,0 | %1,2 |
| E (kısayolsuz) | DT | 0,6149 ± 0,0014 | 0,6191 ± 0,0000 | +0,004 | %43,1 → %42,1 | %1,4 |

**Bulgular:**

1. **Rastgele bölme iyimser.** Geçmişle eğitilip "gelecekteki" trafikte denenince 34 özellikli modelin macro-F1'i 0,07–0,10 düşüyor, yanlış alarm oranı yüzlerce kat artıyor (%0,01 → %3,9).
2. **Kaybın tamamı kısayol alanlarından.** Kısayolsuz model zaman bölmesinde kaybetmiyor. Port ve TCP sıra/onay değerleri kayıt ilerledikçe değişiyor; model geçmişteki değerleri ezberlemiş. Ablation ile bağımsız iki kanıt aynı sonucu veriyor.
3. **Güvenlik açısından:** iki koşulda da saldırılar neredeyse hiç kaçırılmıyor; sorun yanlış alarm.

**Karar:** Ana deneyler A (34 özellik) ile yapılır (literatür ve ekip ile karşılaştırılabilirlik); E "kısayolsuz" sağlamlık koşulu olarak bilinmeyen saldırı deneylerinde de raporlanır. Sınırlılık: veri paket düzeyinde; akış düzeyinde özellikler yalnızca arşivdeki ham kayıtlardan çıkarılabilir.

## Maliyet analizi (19.10.2026)

Kod: `src/cost_dnn_edgeiiot.py`, çıktı: `outputs/dnn_edgeiiot/cost/`. Asıl deneyin modelleri yeniden eğitilmeden yüklendi; **hepsi CPU tek çekirdekte** ölçüldü (adil karşılaştırma; GPU'suz edge cihazı için yaklaşık gösterge), XGB ve TabM için ayrıca GPU (T4). Toplu tahmin: 10.000 test satırı, 3 tekrarın medyanı; tek paket: 300 tekil tahminin medyanı.

| Model | Test F1 | Boyut | Eğitim | Toplu (ms / 1.000) | Saniyede paket (1 çekirdek) | Tek paket (ms) | GPU toplu (ms / 1.000) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DT | 0,9938 | 0,4 MB | 0,7 s | 0,17 | ~5,8 milyon | 0,9 | – |
| RF | 0,9940 | 53,2 MB | 24,6 s | 15,8 | ~63 bin | 12,0 | – |
| XGB | 0,9924 | 5,1 MB | 8,3 s (GPU) | 45,9 | ~22 bin | 2,2 | 3,0 |
| LGBM | 0,9902 | 6,1 MB | 12,8 s | 78,8 | ~13 bin | 0,8 | – |
| TabM | 0,9905 | 3,0 MB | 465 s (GPU) | 689 | ~1,5 bin | 10,7 | 20,5 |

**DSS oylayıcıları** (5 modelin 31 alt kümesi; seçim yalnızca OOF macro-F1 ile, test yalnızca raporlama):

| Oylayıcılar | OOF | Test | ALLOW edilen saldırı | Yanlış alarm | Toplu (ms / 1.000) |
| --- | --- | --- | --- | --- | --- |
| Şu anki kural (DT+RF+XGB+LGBM+TabM) | 0,9941 | 0,9941 | 1 | %0,00 | 830 |
| **DT+RF+XGB** (OOF'ta en iyi) | **0,9944** | **0,9946** | 1 | %0,00 | **62** |
| DT (önceden konan "OOF'ta en iyiye ≤0,001 yakın, en hızlı" kuralı) | 0,9938 | 0,9938 | 2 | %0,04 | 0,2 |

**Bulgular:**

1. **DT neredeyse aynı başarıyı ~100 kat daha ucuza veriyor**; kısayol bulgusuyla tutarlı (birkaç alanla karar verilebiliyor).
2. **TabM CPU'da en pahalı model** (RF'nin ~40 katı), GPU'da 34 kat hızlanıyor; model boyutu ise en küçüklerden (3 MB). GPU'suz edge cihazları için bu ödünleşim CP-TabM tartışmasında yazılmalı.
3. **DSS için öneri: DT+RF+XGB.** OOF ve testte en iyi, şu anki kuraldan 13 kat ucuz; farklar binde 1'in altında olduğu için doğru ifade "başarı ve güvenlik değişmeden maliyet 13 kat düşüyor". TabM'in rolü sınıflandırma oylaması değil, görülmemiş saldırı tespiti. Ortak DSS kuralını değiştirdiği için ekip kararı gerekiyor.
4. Sınırlılık: ölçümler Kaggle'da; gerçek bir edge cihazında ölçülmedi.

## Açıklama Doğrulama: SHAP ve LIME (20.10.2026)

Kod: `src/xai_faithfulness_dnn_edgeiiot.py` (özet grafik: `src/xai_faithfulness_summary_dnn_edgeiiot.py`), çıktı: `outputs/dnn_edgeiiot/xai_faithfulness/`. Soru: SHAP ve LIME'ın "önemli" dediği özellikler, modelden gerçekten çıkarıldığında başarıyı düşürenlerle aynı mı? Açıklamalar her alarmda analiste gösterileceği için doğru olmaları gerekir.

**Yöntem.** Temiz veri, tam koşul (A), RF ve DT (ablation ayarları, test macro-F1 0,9932 / 0,9932). Gerçek etki iki şekilde ölçüldü: **drop-one** (her özellik çıkarılıp model yeniden eğitilir, macro-F1 kaybı) ve **permütasyon** (özellik karıştırılır, kayıp). Açıklama yöntemleri: **SHAP** TreeExplainer (sınıf başına 100 test satırı, |değer| ortalaması) ve **LIME** (sınıf başına 50 satır, tahmin edilen sınıf için |ağırlık|). Uyum Spearman/Kendall ve ilk-k Jaccard ile; LIME kararlılığı 30 satırın 3 farklı tohumla yeniden açıklanmasıyla ölçüldü. LIME'ın ayara bağlı olup olmadığını görmek için RF'de iki ek koşu yapıldı (5000 örnek; ayrıklaştırma kapalı; drop-one değerleri ilk koşudan alındı).

![Açıklama doğrulama özeti](../../outputs/dnn_edgeiiot/xai_faithfulness/fig_xai_faithfulness_summary_dnn_edgeiiot_RF_DT.png)

| Spearman (ilk-5 Jaccard) | DT | RF | RF, LIME 5000 | RF, LIME ayrıklaştırmasız |
| --- | --- | --- | --- | --- |
| SHAP ~ permütasyon | **0,89 (1,00)** | **0,63 (0,67)** | – | – |
| SHAP ~ drop-one | 0,57 (0,67) | 0,37 (0,67) | – | – |
| LIME ~ permütasyon | 0,40 (0,43) | 0,44 (0,67) | 0,37 (0,43) | 0,60 (0,43) |
| LIME ~ drop-one | 0,13 (0,43) | 0,17 (0,43) | 0,02 (0,43) | 0,37 (0,67) |
| SHAP ~ LIME | 0,40 | 0,75 | 0,55 | 0,88 |
| LIME kararlılığı: ilk-5 Jaccard ort. (en kötü) | 0,33 (0,00) | 0,33 (0,00) | 0,38 (0,11) | 0,67 (0,11) |

**Bulgular:**

1. **SHAP iki modelde de gerçek etkiyle en uyumlu yöntem** ve deterministik (aynı satır her zaman aynı açıklamayı alır). DSS'de analiste gösterilecek açıklamalar için SHAP seçildi.
2. **LIME yapılandırmaya duyarlı.** Varsayılan (ayrıklaştırmalı) LIME bu veride kararsız (tekrarlarda ilk 5 özelliğin ancak üçte biri korunuyor) ve modelin kullanmadığı alanları (`mqtt.*`, `dns.*`) öne çıkarabiliyor. Örnek sayısını 5000'e çıkarmak düzeltmiyor; **ayrıklaştırmayı kapatmak** düzeltiyor (kararlılık 0,67, SHAP ile uyum 0,88). Olası sebep: çoğu sütun neredeyse hep 0, çeyrek sınırları anlamsızlaşıyor.
3. **İyi ayarlı LIME bile satır bazında kararsız** (en kötü satırda ilk 5'ten yalnızca 1 ortak; ağırlık sıralaması uyumu 0,40). Bu yüzden LIME tamamlayıcı olarak raporlanır, birincil açıklama SHAP'tır.
4. **Drop-one, birbirinin yerini tutan özellikleri küçük gösterir** (`tcp.seq`/`tcp.ack`; RF'de başka ağaçlar telafi ediyor). Bu yüzden permütasyon ikinci referans olarak kullanıldı; SHAP'ın drop-one uyumunun daha düşük olması bu kör noktadan.
5. **Kısayol bulgusu dört yöntemde de aynı:** ilk üç özellik `tcp.ack`, `tcp.dstport`, `tcp.seq` (ablation ve zaman bölmesi bulgularıyla tutarlı).
6. Sınırlılık: yalnızca tam koşulda (A) ve ağaç modelleriyle yapıldı; kısayolsuz koşul (E) ve TabM açıklamaları sonraki adım.

## Görülmemiş saldırı deneyi: temel skorlar (20.10.2026)

Kod: `src/open_set_lofo_dnn_edgeiiot.py` (özet grafik: `src/open_set_summary_dnn_edgeiiot.py`), çıktı: `outputs/dnn_edgeiiot/open_set/`. Toplantı kararıyla deney her kişinin kendi dalında yürütülüyor.

**Protokol (leave-one-family-out).** Ekip bölmesi (80/20, ince etikete göre katmanlı, seed 42) aynen kullanılır. Her turda bir saldırı ailesi (DDOS, RECON, WEB, BRUTEFORCE, MALWARE) eğitimden tamamen çıkarılır; kalan eğitim verisi %90 eğitim / %10 kalibrasyon. Test kümesi asıl deneyle aynıdır; çıkarılan ailenin satırları "bilinmeyen" sayılır. **Eşik yalnızca kalibrasyondaki bilinen satırlarla seçilir** (bilinenlerin %95'i kabul); bilinmeyen aileden hiçbir bilgi kullanılmaz. Koşullar ablation'daki gibidir ve satırlar her koşulda aynıdır (yalnızca sütunlar değişir): **A** tüm 34 özellik, **C** `tcp.seq`/`tcp.ack` çıkarılmış, **E** port, seq/ack ve şüpheli kimlik alanları çıkarılmış.

**Skorlar** (büyük = bilinmeyen): `msp` (1 − en yüksek olasılık), `entropy`, `disagree` (topluluk üyeleri arası karşılıklı bilgi; RF'de ağaçlar), `knn` (bilinen eğitim verisinde 5 en yakın komşuya uzaklık, modelden bağımsız). TabM'de ayrıca `energy` ve 32 üyenin anlaşmazlığı (sonraki çalıştırma).

![Görülmemiş saldırı özeti](../../outputs/dnn_edgeiiot/open_set/fig_open_set_summary_dnn_edgeiiot_RF.png)

| 5 ailenin ortalaması | A: tüm özellikler | C: seq/ack yok | E: kısayolsuz |
| --- | --- | --- | --- |
| Bilinen sınıflarda macro-F1 (RF) | 0,99 | 0,86 | 0,64 |
| AUROC: RF msp / entropy / disagree | 0,96 / 0,96 / 0,96 | 0,69 / 0,70 / 0,73 | 0,61 / 0,61 / **0,71** |
| AUROC: kNN | **0,98** | **0,88** | 0,67 |
| Yakalanan bilinmeyen (kalibrasyon eşiği, en iyi skor) | %97 | %54 (kNN) | %43 (kNN) |
| Sessizce BENIGN'e kaçan (msp / en iyi) | %0 / %0 | %7 / %6 | %13 / %4 |

**Bulgular:**

1. **Tam veride görülmemiş aileler kolayca yakalanıyor** (AUROC 0,96–0,98) ama bu da kısayol alanlarından: seq/ack çıkınca standart skorlar 0,69–0,73'e, tüm kısayollar çıkınca 0,61'e (neredeyse rastgele) düşüyor. E'de görülmemiş DDOS'a model bilinenlerden daha emin (AUROC 0,36).
2. **En tehlikeli hata, bilinmeyen saldırının BENIGN sanılması**, kısayollar çıkınca ortaya çıkıyor: E'de görülmemiş MALWARE'in %32'si sessizce "normal trafik" olarak geçiyor (msp).
3. **Topluluk anlaşmazlığı RF'nin en iyi skoru** (C'de 0,73, E'de 0,71; msp 0,69 / 0,61) ve BENIGN kaçışını en çok azaltan model skoru. CP-TabM'in dayandığı varsayım (üyelerin anlaşmazlığı bilinmeyeni gösterir) için ön kanıt.
4. **Modelden bağımsız kNN uzaklığı A ve C'de en güçlü temel çizgi** (C'de 0,88); E'de zayıflıyor. CP-TabM'in geçmesi gereken çıta bu.
5. **Ana değerlendirme koşulu olarak C öneriliyor.** `tcp.seq`/`tcp.ack` Wireshark'ın göreli sayaçlarıdır (ezber), hedef port ise gerçek ağlarda anlamlı bir bilgidir. A iyimser üst sınır, E en sert dayanıklılık testi olarak raporlanır. E'de bilinen sınıflar bile ayrılamadığı (F1 0,64) için orada bilinmeyen tespitinin tavanı düşüktür.
6. Not: msp ile E'de "yakalanan %0" görünmesi hata değil; bilinenlerin kendisi çok belirsiz olduğundan %95 eşiği en yüksek belirsizlik değerine oturuyor ve hiçbir bilinmeyen onu aşamıyor.

### Başarı ölçütleri (sonuçlardan ÖNCE belirlendi, 20.10.2026)

CP-TabM, ana koşul **C**'de, 5 ailenin ortalamasıyla, **en iyi temel skorla** (RF ve TabM skorları ile kNN dahil; şu an kNN, AUROC 0,88) karşılaştırılır. Tüm ayarlar yalnızca kalibrasyon verisiyle seçilir; beş ailenin hepsi raporlanır; sonuçlar en az 3 tohumla verilir.

| Sonuç | Ölçüt |
| --- | --- |
| **Açık başarı** | AUROC'ta en iyi temel skordan **en az +0,05**, sessiz BENIGN kaçışında belirgin düşüş (yaklaşık yarıya), bilinen sınıflarda macro-F1 kaybı sade TabM'e göre **en fazla 0,01**; fark tohumlar arası sapmadan büyük |
| **Kısmi başarı** | Ortalamada küçük fark (+0,01–0,05) ama en zor ailelerde (WEB, MALWARE, BRUTEFORCE) ya da sessiz BENIGN kaçışında belirgin iyileşme |
| **Başarısız** | Temel skorlarla aynı veya daha kötü; ya da tespit artarken bilinen başarı belirgin düşüyor. Bu durumda sonuç ve olası nedenleri olduğu gibi raporlanır |

Bileşen testleri: sınıflara bölünmüş üyeler, enerji farkı kaybı ve imza tutarlılığı skoru teker teker çıkarılır; ölçülebilir katkısı olmayan bileşen raporda belirtilir.

## Sütun hizası ve değer aralığı denetimi (19.10.2026)

Kod: `src/audit_dnn_edgeiiot.py`, çıktı: `outputs/dnn_edgeiiot/audit/audit.json`. Ham verideki 15 etiketin her biri için 6 kontrol (zaman damgası, IP, port 0–65535, 0/1 bayrak, ARP alanları, sayısal sütunlarda metin). İhlal yalnızca DDoS_UDP (zaman %100) ve MITM'de (zaman %100, port %30, bayrak %19); Normal'de %0,11 bayrak ihlali (önemsiz); diğer 12 saldırı türünde %0. Bu iki tür bulgu 7 gereği atıldı.

## Çalıştırma

```bash
pip install tabm rtdl_num_embeddings   # yalnızca TabM için (torch gerekir)
# Hızlı deneme
python src/model_dnn_edgeiiot.py --data-file <DNN-EdgeIIoT-dataset.csv> --quick --drop-conflicts \
    --out-dir outputs_quick/dnn_edgeiiot --work-dir work/quick_dnn_edgeiiot
# Asıl deney (6 model, GPU önerilir)
python src/model_dnn_edgeiiot.py --data-file <csv> --drop-conflicts --tune --shap --gpu
# Kontroller
python src/ablation_dnn_edgeiiot.py    --data-file <csv>
python src/split_check_dnn_edgeiiot.py --data-file <csv>
python src/audit_dnn_edgeiiot.py       --data-file <csv>
python src/cost_dnn_edgeiiot.py        --data-file <csv> --work-dir <asıl deneyin work dizini>
pip install lime shap
python src/xai_faithfulness_dnn_edgeiiot.py --data-file <csv> --model RF   # ve --model DT
python src/xai_faithfulness_dnn_edgeiiot.py --data-file <csv> --model RF --tag A_tam_nodisc --lime-no-discretize \
    --drop-one-from outputs/dnn_edgeiiot/xai_faithfulness/faithfulness_RF_A_tam.json
python src/xai_faithfulness_summary_dnn_edgeiiot.py
# Görülmemiş saldırı deneyi (koşullar: A_tam, C_seq_ack_yok, E_hepsi_cikarildi)
python src/open_set_lofo_dnn_edgeiiot.py --data-file <csv> --models RF,TabM --condition C_seq_ack_yok
python src/open_set_summary_dnn_edgeiiot.py
```

Çalıştırma kayıtları: `outputs/dnn_edgeiiot/logs/`.
