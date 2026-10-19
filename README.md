# IoT Güvenliği: Çok Veri Setli Karşılaştırma ve DSS

Üç kişi, üç farklı IoT veri seti. Herkes kendi veri setinde en iyi algoritmaları
eğitir. Sonuçlar ortak bir formatta `outputs/` altına yazılır ve `main.py` bunları
güvenilirlik ağırlıklı bir Karar Destek Sistemi'nde (DSS) birleştirir.

```
iot-security-project/
├── data/                      # Ham veri: Git'e GİRMEZ (yerelde ya da Drive'da durur)
│   ├── dataset_23/            # Kişi 1: CICIoT2023
│   ├── dataset_25/            # Kişi 2
│   └── dataset_26/            # Kişi 3: Edge-IIoTset
├── src/
│   ├── common/
│   │   ├── taxonomy.py        # Ortak saldırı kategorileri + ciddiyet (herkes aynısını kullanır)
│   │   ├── contract.py        # Ortak çıktı formatı, metrikler, ortam bilgisi, doğrulama
│   │   └── report.py          # Analiz tabloları ve makale grafikleri
│   ├── model_ml.py            # Kişi 1: dataset_23, DT/RF/XGB/LGBM/LR + SHAP
│   ├── model_dl.py            # Kişi 2: dataset_25 (şablon)
│   └── model_other.py         # Kişi 3: dataset_26 (Edge-IIoTset), model_ml.py akışını yeniden kullanır
├── notebooks/
│   ├── analysis.ipynb         # ⭐ Tablolar + grafikler (yalnızca outputs/ okur, veri gerekmez)
│   └── dataset_23_colab.ipynb # Colab'da eğitim şablonu (güçlü bilgisayarı olmayanlar için)
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

## Çalışma düzeni (GitHub)

- `main` dalı korunur. Herkes kendi dalında çalışır (`dataset_23`, `dataset_25`, `dataset_26`) ve Pull Request ile birleştirir.
- Ortak dosyaları (`src/common/*`, `main.py`) değiştiren PR'ı herkes onaylar.
- Ham veri ve modeller Git'e girmez (`.gitignore`). Herkes veriyi kendi `data/<dataset_id>/` klasöründe tutar; Colab kullanan Drive'a koyar.
- GitHub Actions her PR'da `python main.py --validate-only` çalıştırır. Format bozuksa PR kırmızı olur.

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
