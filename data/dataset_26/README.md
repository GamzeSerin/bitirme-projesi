# dataset_26: Edge-IIoTset (Kişi 3)

- Kaynak (resmî, yazar hesabı): https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot
- Kullanılan dosya: `Edge-IIoTset dataset/Selected dataset for ML and DL/DNN-EdgeIIoT-dataset.csv` (1,22 GB, 2.219.201 satır, 63 sütun). Arşivin geri kalanı (PCAP'ler, ham CSV'ler, ML dosyası) kullanılmaz.
- Lisans: CC BY-NC-SA 4.0 (akademik kullanım serbest, atıf zorunlu). Dosya Git'e girmez.
- Atıf: M. A. Ferrag, O. Friha, D. Hamouda, L. Maglaras, H. Janicke, "Edge-IIoTset: A New Comprehensive Realistic Cyber Security Dataset of IoT and IIoT Applications for Centralized and Federated Learning," *IEEE Access*, 2022. doi:10.1109/ACCESS.2022.3165809
- Çalışma ortamı: Kaggle Notebooks. Veri not defterine girdi olarak bağlanır, indirilmez (`notebooks/dataset_26_kaggle.ipynb`).

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
| Asıl deney (`model_other.py`) | Ham | 2.219.201 | |
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

## Sonuçlar ve bulgular (18.10.2026, asıl deney)

Ayarlar: `--drop-conflicts --tune --shap`, etiket başına en fazla 40.000 satır, stratified 80/20 (seed=42), 5-fold OOF, Kaggle CPU.

**Veri hazırlama (model_other.py):** ham 2.219.201 → tekrarlar silindikten sonra 688.640 (%69,0 tekrar) → etiket çakışması 12.965 satır (atıldı) → 675.675 → etiket başına üst sınırla örneklem 308.396 satır, 34 özellik, 7 kategori. Sayıların Aşama 0 ile ilişkisi yukarıdaki satır muhasebesi tablosunda.

**Test kümesinin kullanımı:** hiperparametre araması yalnızca eğitim kümesinin alt örnekleminde 3-fold CV ile yapılır, DSS güvenilirlik ağırlıkları eğitim kümesinin 5-fold OOF tahminlerinden gelir. Test kümesi yalnızca son değerlendirmede kullanılır.

| Model | OOF macro-F1 | Test macro-F1 | Eğitim (s) | Tahmin (ms/1k) |
| --- | --- | --- | --- | --- |
| **RF** (birincil) | 0,9951 | 0,9953 | 20,5 | 3,9 |
| DT | 0,9943 | 0,9942 | 1,1 | 0,1 |
| XGB | 0,9934 | 0,9930 | 94,6 | 73,7 |
| LGBM | 0,9923 | 0,9921 | 19,1 | 49,3 |
| LR | 0,7774 | 0,7757 | 29,8 | 1,4 |

Seçilen hiperparametreler: DT derinlik 20, entropy; RF 100 ağaç, derinlik 40, max_features 0,5; XGB 800 ağaç, derinlik 8, lr 0,05; LGBM 63 yaprak, 200 ağaç, lr 0,05.

**Bulgular:**

1. **Overfit yok:** OOF ve test skorları her modelde binde 1'den az farklı.
2. **En zayıf kategori WEB** (RF test F1 0,9856). İnce etiket recall'ında DDoS_HTTP 0,953: HTTP sel saldırıları büyük olasılıkla WEB ile karışıyor (karışıklık matrisiyle doğrulanmalı). WEB saldırılarının ayırt edici bilgisi paket içeriğinde; içerik sütunları sızıntı riski nedeniyle atıldığı için model yalnızca başlık alanlarını görüyor.
3. **En zor alt etiket Fingerprinting** (recall 0,917; temizlik sonrası 181 satır, testte ~36).
4. **SPOOFING (MITM) F1 = 1,0:** çakışan (normal ARP'den ayırt edilemeyen) satırlar atıldıktan sonra kalan 357 satır kolay ayrılıyor; az örnekle raporlanmalı.
5. **SHAP kısayol uyarısı:** DSS açıklamalarında en önemli özellikler neredeyse her zaman `tcp.ack`, `tcp.dstport`, `tcp.seq`. Model büyük ölçüde hedef port ve bağlantı içindeki konuma dayanıyor olabilir. Yapılacak ablation: `--drop-features tcp.dstport,udp.port,tcp.seq,tcp.ack`. Görülmemiş saldırı tespiti açısından önemli (aynı portu kullanan yeni bir saldırı bilinen saldırı sanılabilir).
6. **DSS:** test örnekleminde (12.071 satır) DSS macro-F1 0,9969; oylayanlar DT, RF, XGB, LGBM (LR elendi). ALLOW edilen saldırı: 0; ANALYST_REVIEW: 3.

## Çalıştırma

```bash
# Hızlı deneme
python src/model_other.py --data-file <DNN-EdgeIIoT-dataset.csv> --quick --drop-conflicts \
    --out-dir outputs_quick/dataset_26 --work-dir work/quick_26
# Asıl deney
python src/model_other.py --data-file <DNN-EdgeIIoT-dataset.csv> --drop-conflicts --tune --shap --gpu
```
