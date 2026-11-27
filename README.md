# IoT Güvenliği: Çok Veri Setli Karşılaştırma ve DSS

Üç kişi, üç farklı IoT veri seti. Herkes kendi veri setinde en iyi algoritmaları
eğitir. Sonuçlar ortak bir formatta `outputs/` altına yazılır ve `main.py` bunları
güvenilirlik ağırlıklı bir Karar Destek Sistemi'nde (DSS) birleştirir.

```
iot-security-project/
├── data/                      # Ham veri: Git'e GİRMEZ (yerelde ya da Drive'da durur)
│   ├── dataset_23/            # Kişi 1: CICIoT2023
│   ├── dataset_25/            # Kişi 2
│   └── dnn_edgeiiot/          # Kişi 3: Edge-IIoTset
├── src/
│   ├── common/
│   │   ├── taxonomy.py        # Ortak saldırı kategorileri + ciddiyet (herkes aynısını kullanır)
│   │   ├── contract.py        # Ortak çıktı formatı, metrikler, ortam bilgisi, doğrulama
│   │   └── report.py          # Analiz tabloları ve makale grafikleri
│   ├── model_ml.py            # Kişi 1: dataset_23, DT/RF/XGB/LGBM/LR + SHAP
│   ├── model_dl.py            # Kişi 2: dataset_25 (şablon)
│   ├── model_dnn_edgeiiot.py  # Kişi 3: Edge-IIoTset, model_ml.py akışını yeniden kullanır
│   └── ablation_dnn_edgeiiot.py # Kişi 3: Edge-IIoTset kısayol özellik ablation deneyi
├── notebooks/
│   ├── analysis.ipynb         # ⭐ Tablolar + grafikler (yalnızca outputs/ okur, veri gerekmez)
│   ├── dataset_23_colab.ipynb # Colab'da eğitim şablonu (güçlü bilgisayarı olmayanlar için)
│   └── dnn_edgeiiot_kaggle.ipynb # Kişi 3: Edge-IIoTset, Kaggle'da eğitim
├── outputs/                   # Herkesin çıktı JSON'ları: Git'e GİRER
│   └── <dataset_id>/{manifest,metrics,reliability}.json, test_predictions.csv.gz, figures/ (SHAP)
├── figures/                   # analysis.ipynb'nin ürettiği makale grafikleri (PNG) ve tablolar
├── main.py                    # ⭐ Birleştirme noktası (DSS)
└── .github/workflows/validate.yml   # Her PR'da çıktı formatını kontrol eder
```

## Ortak kurallar (herkes uyar)

1. **Etiketler** `src/common/taxonomy.py` içindeki ortak kategorilere eşlenir (BENIGN, DDOS, DOS, RECON, WEB, BRUTEFORCE, SPOOFING, MALWARE, MIRAI). Yeni kategori gerekiyorsa önce PR ile oraya eklenir.
2. **Temizlik split'ten önce yapılır:** inf/NaN silinir, tekrarlar dosyalar birleştirildikten sonra silinir, **aynı özellik vektörüne sahip ama etiketi farklı (çakışan) satırların hepsi atılır** (etiket gürültüsü). Kimlik gibi davranan ya da veri toplama izi olan sütunlar (ör. CICIoT2023'te `Number`) atılır ve README'de gerekçesiyle yazılır. Çakışmaları tutan sürüm duyarlılık analizi olarak `outputs_ablation/` altında saklanabilir.
3. **Stratified 80/20 split, `seed=42`.** Test seti kilitlenir.
4. **Ön işleme, dengesizlik yönetimi ve hiperparametre araması** yalnızca eğitim verisinde, CV içinde yapılır.
5. **DSS ağırlıkları OOF (cross-validation) tahminlerinden** hesaplanır. Test yalnızca en sonda raporlama için kullanılır.
6. **Her model `predict_proba` vermeli.** SVM için `CalibratedClassifierCV` kullanılır.
7. **Ana metrik macro-F1.** Ek olarak sınıf bazında precision, recall, F1 ve FPR ile süre raporlanır.
8. **Çıktılar `contract.write_outputs(...)` ile yazılır.** Fonksiyon formatı kendisi doğrular.

## DSS nasıl çalışıyor?

Veri setlerinin özellikleri farklı olduğu için bir kayıt başka ortamın modeline verilemez. Bu yüzden DSS **bağlama göre yönlendirmeli** çalışır:

1. Trafik hangi ortamdan geldiyse o ortamın modellerine gider.
2. Ortam içinde modeller sınıf bazında ağırlıklı oylanır: `skor(c) = Σ_m w(m,c)·p_m(c)`. Burada `w(m,c)`, m modelinin c sınıfındaki OOF F1 değeridir ve sınıf başına normalize edilir. Oylamaya yalnızca OOF macro-F1'i en iyi modelin en fazla 0,05 altında kalan modeller girer (`VOTER_MARGIN`). Bu kural deneyden önce sabitlendi; test sonucuna bakılarak değiştirilmez.
3. Güven ve ciddiyete göre bir aksiyon önerilir: ALLOW, LOG_MONITOR, ALERT, BLOCK veya ANALYST_REVIEW. Her kararın yanında SHAP'tan gelen en önemli 3 özellik de verilir.

## Docker (DSS, doğrulama, demo)

Eğitim Kaggle/Colab'da yapılır. Docker, `outputs/` klasörünü okuyan DSS'i her bilgisayarda aynı şekilde çalıştırmak içindir (veri gerekmez):

```bash
docker compose run --rm validate                      # sözleşme kontrolü (CI ile aynı)
docker compose run --rm dss                           # DSS raporu -> outputs/dss_report.json
DEMO_DATASET=dnn_edgeiiot docker compose run --rm demo  # örnek kararlar ve açıklamalar
```

Yerelde eğitimi tekrar üretmek için (veri `./data/<dataset_id>/` altında olmalı):

```bash
docker compose --profile train run --rm train-23
docker compose --profile train run --rm train-dnn_edgeiiot
```

`Dockerfile` iki hedef içerir: `dss` (yalnızca numpy, pandas, scikit-learn) ve `train` (`requirements.txt`'in tamamı).

## Çalışma düzeni (GitHub)

- `main` dalı korunur. Herkes kendi dalında çalışır (`dataset_23`, `dataset_25`, `dnn_edgeiiot`) ve Pull Request ile birleştirir.
- Ortak dosyaları (`src/common/*`, `main.py`) değiştiren PR'ı herkes onaylar.
- Ham veri ve modeller Git'e girmez (`.gitignore`). Herkes veriyi kendi `data/<dataset_id>/` klasöründe tutar; Colab kullanan Drive'a koyar.
- GitHub Actions her PR'da `python main.py --validate-only` çalıştırır. Format bozuksa PR kırmızı olur.

## Görülmemiş saldırı deneyi: ortak araç (LOFO)

Herkes kendi dalında aynı protokolü çalıştırır; sonuçlar böylece veri setleri arasında karşılaştırılabilir.

- **Betik:** `src/open_set_lofo.py --dataset <dataset_id>` (veri setinden bağımsız). İç provayla yöntem ayarı: `src/cp_tune_dnn_edgeiiot.py --dataset <dataset_id>` (adı tarihsel; her veri setinde çalışır). İki sinyalli kural: `src/two_signal.py --dataset <dataset_id>`. Kısayol ablation'ı (kapalı küme, aynı A/C/E koşulları, sabit RF/DT ayarları, bootstrap aralıkları): `src/ablation_dnn_edgeiiot.py --dataset <dataset_id>` (adı tarihsel). Maliyet analizi (model başına gecikme/boyut, DSS oylayıcı alt kümeleri; asıl deneyin kayıtlı modelleriyle, yeniden eğitim yok): `src/cost_dnn_edgeiiot.py --dataset <dataset_id> --run-dir work/<dataset_id>/<çalıştırma>` (adı tarihsel).
- **Veri setine özel her şey tek yerde:** `src/common/open_set_registry.py`. Her kayıt şunları içerir:
  - yükleyici: `(data_file, cap, seed) -> X, fine, category, stats`,
  - aileler: `{aile: [kategoriler]}`; bir aile birden fazla kategori kapsayabilir (CICIoT2023'te FLOOD = DOS + DDOS),
  - koşullar: A tam, C ana koşul, E en sert; her koşulda çıkarılacak sütunlar.

  Kayıtta olup veride olmayan bir sütun deneyi açık bir hatayla durdurur.
- **Protokol (değişmez):**
  - ekip bölmesi 80/20, ince etikete göre katmanlı, seed 42,
  - eğitimin %10'u kalibrasyon,
  - eşik yalnızca bilinen kalibrasyon verisinden seçilir (bilinenlerin %95'i kabul),
  - koşul sütunları veri yüklendikten sonra çıkarılır (her koşulda satırlar aynı),
  - RF ve DT **bütün veri setlerinde aynı sabit ayarlarla** eğitilir (`ablation_dnn_edgeiiot.build_models`: RF 100 ağaç, derinlik 40, max_features 0,5; DT derinlik 20, entropy). Ayarlar Edge-IIoTset'in asıl deneyinde seçildi; her veri setinde yeniden aranmaz ki skor farkı ayardan değil veriden gelsin.
- **Ölçütler:** AUROC, AUPR, FPR95, yakalanan, yanlış reddedilen bilinen, sessiz BENIGN kaçışı, open-F1, bilinen F1.
- **Doğrudan karşılaştırılan aileler** (`COMPARABLE`): FLOOD~DDOS, RECON, WEB, BRUTEFORCE. MIRAI, SPOOFING ve MALWARE veri setine özeldir.
- **Backdoor:** her veri seti kendi yazarlarının gruplamasını korur (CICIoT2023: WEB, Edge-IIoTset: MALWARE).

```bash
python src/open_set_lofo.py --dataset dataset_23   --data-file <Merged*.csv klasörü> --models RF,TabM --condition C_ttl_yok
python src/open_set_lofo.py --dataset dnn_edgeiiot --data-file <DNN-EdgeIIoT-dataset.csv> --models RF,TabM --condition C_seq_ack_yok
python src/ablation_dnn_edgeiiot.py --dataset dataset_23 --data-file <klasör>    # kısayol ablation'ı (~10 dk, CPU)
python src/two_signal.py --dataset dataset_23 --data-file <klasör>               # iki sinyalli kural (aşağıda)
# önce her komutu --quick ile deneyin (~1 dk)
```

### İki sinyalli karar kuralı (DSS'in bilinmeyen katmanı)

`src/two_signal.py --dataset <dataset_id>`: görülmemiş bir saldırının sessizce "normal trafik" diye geçirilmesini (ALLOW edilen görülmemiş saldırı) ölçer.

1. **Alarm:** TabM üye anlaşmazlığı bilinen kalibrasyonun %95 eşiğini aşarsa "bilinmeyen".
2. **BENIGN kontrolü:** alarm yoksa ve tahmin BENIGN ise, güven skoru eşiği aşan satır ALLOW edilmez, analiste gider. Eşik, kalibrasyondaki gerçek BENIGN trafiğin yalnızca **q** kadarını (analist bütçesi; %0,5–10) işaretleyecek şekilde seçilir.
3. **Ortak DSS ölçütü:** `silent_benign` = ALLOW edilen görülmemiş saldırı oranı; her zaman bütçeyle (`benign_review`) birlikte raporlanır.

Karşılaştırılan kontrol skorları: TabM'in kendi skorları (anlaşmazlık, MSP, enerji), CP-TabM BENIGN uzmanları ve kNN.

**CP-TabM ayarı:** Edge-IIoTset'te iç provayla seçildi (`--cp-config`). Diğer veri setlerine **ayar aktarımı** ile taşınır: kör oranı 0,0625 (Edge-IIoTset iç provasında 5 ailenin 4'ünde seçilen değer), hedef veride ayar yapılmaz. JSON'daki `blind_frac_source` alanı bunu kaydeder.

Edge-IIoTset sonucu ve önceden yazılan ölçütler: `data/dnn_edgeiiot/README.md` ("İki sinyalli karar kuralı").

### Ortak bulgu: gerçek DSS'te bilinmeyen katmanı (25.10.2026)

Aynı bekçi (TabM: alarm = üye anlaşmazlığı, kontrol = enerji) iki veri setinde **gerçek DSS'in** kararına uygulandı (`src/dss_guard.py --dataset <dataset_id>`; oylayanlar asıl DSS'le aynı ve sabit, ayarlar asıl deneyden, ağırlıklar 5-fold OOF). Alarm kademesi KEŞİF'tir (sonuçlardan sonra tanımlandı; `src/alarm_tiers.py`, yeni eğitim yok). Koşul C, tohum 42, ailelerin ortalaması:

| | Edge-IIoTset (5 aile) | CICIoT2023 (6 aile) |
| --- | --- | --- |
| Oylayanlar | DT, RF, XGB, LGBM, TabM | RF, XGB, LGBM |
| Görülmemiş saldırı ALLOW: yalnız DSS → alarm → alarm + kontrol (q = %5) | %2,5 → %2,1 → **%0,8** | %6,0 → %4,4 → **%4,1** |
| Katkının çoğu | enerji kontrolünden | TabM alarmından |
| Normal trafikte analiste giden: yalnız DSS | %1,6 | %6,4 |
| … alarm + kontrol, q = %5, tek kademe | %8,1 | %23,3 |
| … **skor kademesi** (alarm %95–99 → LOG_MONITOR, %99+ → analist) | **%5,6** | **%11,1** |
| Görülmemiş flood'un analiste gitmesi: tek → skor → ciddiyet kademesi | %49 → %46,5 → %3,2 (DDOS) | %57 → %55 → %38 (FLOOD) |

- **Kural gerçek DSS'e taşınabiliyor:** kontrol TabM'in ya da DSS'in kararına uygulansa sonuç aynı.
- **Gücü veriye bağlı:** saldırıların normalden ayırt edilebildiği veride (Edge-IIoTset) düşük bedelle ~3 kat azalma; ayırt edilemediği veride (CICIoT2023) küçük kazanç, yüksek yük (normal trafik TabM için de belirsiz, alarm sık çalıyor).
- **Skor kademesi** iki veri setinde de ALLOW'u değiştirmeden analist yükünü belirgin azaltıyor (LOG_MONITOR'a giden normal trafik artıyor; bu bir kayıt, insan incelemesi değil).
- **Ciddiyet kademesi önerilmez:** görülmemiş ağır saldırılar sık sık RECON sanıldığı için insan incelemesinden kaçıyor (iki veri setinde de).
- Ayrıntılar: `data/dnn_edgeiiot/README.md` ve `data/dataset_23/README.md` ("Gerçek DSS'te BENIGN kontrolü"). Sınırlılık: tek tohum; test kümesinde saldırı oranı ~%95, gerçek ağda normal trafik çoğunlukta olduğu için analist yükü mutlak sayıda çok daha büyük.

### Bekçinin açıklanması ve derin öğrenme karşılaştırması (ortak araçlar)

Edge-IIoTset'te yapılan analizler (planlar ve sonuçlar: `data/dnn_edgeiiot/README.md`); hepsi `--dataset` ile çalışır.

| Analiz | Betik | Önkoşul |
| --- | --- | --- |
| Bilinmeyen Kararının Açıklanması (XAI-2): bekçinin kararına permütasyon Shapley + silme eğrisi sadakati | `src/xai_unknown_decision.py` | `dss_guard.py` satır dosyaları |
| Kaçışların Karşı-Olgusal Açıklaması (XAI-3) + eş kontrolü | `src/xai_escape_counterfactual.py` | XAI-2 (kayıtlı TabM, Shapley) |
| Derin öğrenme karşılaştırması: TabM / MLP / MC-dropout | `src/two_signal.py --classifier MLP\|MCDropout`, `src/dl_compare.py` | TabM'in two_signal koşuları (3 tohum) |
| Tahmin süresi | `src/inference_time.py` | XAI-2 (kayıtlı TabM) |
| Literatür karşılaştırması: OpenMax, Mahalanobis, otokodlayıcı (tek başına ve BENIGN kontrolü olarak, 3 tohum) | `src/open_set_baselines.py` | yok (GPU önerilir) |
| Etiket karışıklığı payı: ana koşulda başka etiketle birebir aynı vektöre düşen satırlar | `src/label_noise_check.py` | yok (CPU) |
| Zaman bölmesinde iki sinyal kuralı (3 tohum) | `src/two_signal.py --split time --skip-cp` | yükleyicinin satır sırası kayıt sırası olmalı (önce kontrol) |
| Zaman bölmesinde yeniden kalibrasyon: model aynı, eşikler test penceresinin ilk %25'inden (3 tohum) | `src/two_signal.py --split time --skip-cp --recal-frac 0.25` | zaman bölmesiyle aynı |

Hepsini sırayla (her adımda önce `--quick`) tek komutla. Varsayılan adımlar `xai2,dl,xai3,timing,baselines,noise`; zaman
bölmesi (`time`) yalnız satır sırası kayıt sırasıysa `--steps time` ile ayrıca eklenir:

```bash
python src/xai_dl_suite.py --dataset dataset_23 --data-file data/dataset_23 \
    --run-dir work/dataset_23/<asıl deney> --params-from run --families RECON,SPOOFING,WEB,BRUTEFORCE
```

## Nasıl çalıştırılır?

**1. Kurulum (bir kez):**

```bash
pip install -r requirements.txt
```

**2. Hızlı deneme (~2 dk):** Kodun uçtan uca çalıştığını görmek için. Sonuçlar `outputs_quick/` klasörüne gider, Git'e girmez.

```bash
python src/model_ml.py --quick --shap --out-dir outputs_quick/dataset_23 --work-dir work/quick
```

**3. Asıl deney (~1-2 saat):** Makaledeki sonuçlar. `outputs/dataset_23/` klasörüne yazar. Yarıda kesilirse aynı komutu tekrar çalıştır; biten modeller `work/` klasöründen okunur.

```bash
python src/model_ml.py --tune --shap --gpu
```

**4. DSS ve sözleşme kontrolü:**

```bash
python main.py
```

**5. Grafikler ve tablolar:** `notebooks/analysis.ipynb` dosyasını VS Code'da aç ve *Run All*'a bas. Colab'da GitHub'dan açılabilir; veri gerekmez. Grafikler `figures/` klasörüne kaydedilir.

**Colab ile eğitim:** Bilgisayarı yetersiz olan ekip üyeleri `notebooks/dataset_23_colab.ipynb` dosyasını şablon olarak kullanabilir (veri Drive'da).
