"""TabM (Gorishniy vd., ICLR 2025) için scikit-learn uyumlu sınıflandırıcı.

Ekip protokolündeki diğer modellerle aynı arayüz: fit(X, y), predict_proba(X), classes_.
Böylece model_ml.run_model (OOF, test, süre ölçümü) ve DSS değişmeden kullanılır.

TabM, ağırlıklarının çoğunu paylaşan k adet MLP'den oluşan verimli bir topluluktur
(BatchEnsemble). Her üye ayrı tahmin üretir; sınıflandırmada olasılıkların ortalaması
alınır. Üye bazlı olasılıklar (predict_member_proba) bilinmeyen saldırı deneyi ve
CP-TabM için ayrıca dışarı verilir.

Eğitim, paketin resmi örneğini izler (yandex-research/tabm, example.ipynb):
  - QuantileTransformer (normal çıktı), yalnızca eğitim verisine uydurulur
  - PiecewiseLinearEmbeddings (48 bin, d_embedding 16, version B)
  - TabM.make varsayılanları (k=32, d_block 512, dropout 0,1), AdamW lr 2e-3, wd 3e-4,
    gradyan kırpma 1,0, tüm üyeler aynı batch'leri görür
Ekip protokolüne uyarlamalar:
  - Sınıf ağırlıklı (balanced) çapraz entropi, diğer modellerdeki class_weight ile aynı
  - Early stopping: VERİLEN eğitim verisinin %10'u (stratified) doğrulama; durma ölçütü
    AĞIRLIKSIZ doğrulama kaybı (Kişi 1'in boosting bulgusu, bkz. model_ml.EarlyStopping)
  - Eğitim kümesinde sabit olan sütunlar model içinde atılır (PLE sabit sütun kabul etmez)

Kurulum (Kaggle): pip install tabm rtdl_num_embeddings   (torch Kaggle'da hazır)
"""

import copy
import math
import time
import warnings

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.utils.class_weight import compute_class_weight


class TabMClassifier(ClassifierMixin, BaseEstimator):
    VERSION = 1  # davranış değişince önbelleği geçersiz kılar (model_ml.model_key)

    def __init__(self, k=32, n_blocks=None, d_block=512, dropout=0.1, lr=2e-3,
                 weight_decay=3e-4, batch_size=512, max_epochs=60, patience=8,
                 n_bins=48, d_embedding=16, val_size=0.1, balanced=True,
                 device="auto", random_state=42, verbose=True):
        self.k = k
        self.n_blocks = n_blocks
        self.d_block = d_block
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.n_bins = n_bins
        self.d_embedding = d_embedding
        self.val_size = val_size
        self.balanced = balanced
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

    # ------------------------------------------------------------ yardımcılar
    def _torch_device(self):
        import torch
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def _transform(self, X):
        """Sütun bazında: QuantileTransformer; QT'nin sabite sıkıştırdığı (neredeyse hep 0
        olan) sütunlarda standartlaştırılmış ham değer. Yalnızca eğitime uydurulmuş parametreler."""
        Z = self.qt_.transform(X).astype(np.float32)
        Z[:, self.std_cols_] = (X[:, self.std_cols_] - self.mu_) / self.sd_
        return Z[:, self.final_cols_]

    def _prep(self, X):
        return self._transform(np.asarray(X, dtype=np.float32)[:, self.keep_cols_])

    def _logits(self, Xt, batch=8192):
        """(n, k, C) logit; değerlendirme modunda, batch batch."""
        import torch
        dev = next(self.model_.parameters()).device
        self.model_.eval()
        out = []
        with torch.inference_mode():
            for i in range(0, len(Xt), batch):
                xb = torch.as_tensor(Xt[i:i + batch], device=dev)
                out.append(self.model_(xb).float().cpu())
        return torch.cat(out).numpy()

    # ------------------------------------------------------------ eğitim
    def fit(self, X, y, sample_weight=None):
        import rtdl_num_embeddings
        import tabm
        import torch
        import torch.nn.functional as F

        rng = np.random.default_rng(self.random_state)
        torch.manual_seed(self.random_state)
        dev = self._torch_device()
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        n_classes = len(self.classes_)

        self.keep_cols_ = np.where(np.ptp(X, axis=0) > 0)[0]
        X = X[:, self.keep_cols_]
        Xtr, Xva, ytr, yva = train_test_split(X, y_idx, test_size=self.val_size, stratify=y_idx,
                                              random_state=self.random_state)

        noise = rng.normal(0.0, 1e-5, Xtr.shape).astype(np.float32)
        self.qt_ = QuantileTransformer(n_quantiles=max(min(len(Xtr) // 30, 1000), 10),
                                       output_distribution="normal", subsample=10**9,
                                       random_state=self.random_state).fit(Xtr + noise)
        # Satırların çok küçük bir kısmında dolu sütunlar (ör. icmp.transmit_timestamp, %0,15)
        # QT'de sabite sıkışır; bunlar için standartlaştırılmış ham değer kullanılır.
        Zq = self.qt_.transform(Xtr)
        self.std_cols_ = np.where(np.ptp(Zq, axis=0) <= 1e-6)[0]
        self.mu_ = Xtr[:, self.std_cols_].mean(axis=0)
        self.sd_ = Xtr[:, self.std_cols_].std(axis=0) + 1e-6
        self.final_cols_ = np.arange(Xtr.shape[1])
        Z = self._transform(Xtr)
        # Eğitim bölümünde hâlâ sabit kalan sütun (dolu satırların hepsi doğrulamaya düştüyse) atılır
        self.final_cols_ = np.where(np.ptp(Z, axis=0) > 1e-6)[0]
        Xtr, Xva = Z[:, self.final_cols_], self._transform(Xva)
        self.n_features_used_ = int(len(self.final_cols_))
        if self.verbose:
            print(f"    TabM: {self.n_features_used_} özellik "
                  f"({len(self.std_cols_)} seyrek sütun standartlaştırıldı, "
                  f"{X.shape[1] - self.n_features_used_} sabit sütun atıldı)")

        with warnings.catch_warnings():
            # 0/1 bayraklar gibi iki değerli sütunlarda tek bin oluşur; beklenen durum
            warnings.filterwarnings("ignore", message=".*just two bin edges.*")
            bins = rtdl_num_embeddings.compute_bins(torch.as_tensor(Xtr),
                                                    n_bins=min(self.n_bins, len(Xtr) - 1))
            emb = rtdl_num_embeddings.PiecewiseLinearEmbeddings(
                bins, d_embedding=self.d_embedding, activation=False, version="B")
        kwargs = dict(n_num_features=Xtr.shape[1], d_out=n_classes, num_embeddings=emb,
                      k=self.k, d_block=self.d_block, dropout=self.dropout)
        if self.n_blocks is not None:
            kwargs["n_blocks"] = self.n_blocks
        model = tabm.TabM.make(**kwargs).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        w = (compute_class_weight("balanced", classes=np.arange(n_classes), y=ytr)
             if self.balanced else np.ones(n_classes))
        w_t = torch.as_tensor(w, dtype=torch.float32, device=dev)
        Xtr_t = torch.as_tensor(Xtr, device=dev)
        ytr_t = torch.as_tensor(ytr, dtype=torch.long, device=dev)
        k = model.backbone.k

        best = (math.inf, -1, None)
        bad = 0
        self.history_ = []
        t0 = time.perf_counter()
        for epoch in range(self.max_epochs):
            model.train()
            perm = torch.randperm(len(Xtr_t), device=dev)
            for idx in perm.split(self.batch_size):
                opt.zero_grad()
                logits = model(Xtr_t[idx])                       # (b, k, C)
                loss = F.cross_entropy(logits.flatten(0, 1), ytr_t[idx].repeat_interleave(k),
                                       weight=w_t)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            self.model_ = model
            proba_va = torch.softmax(torch.as_tensor(self._logits(Xva)), -1).mean(1).numpy()
            val_loss = float(-np.log(np.clip(proba_va[np.arange(len(yva)), yva], 1e-12, None)).mean())
            # Yalnızca izleme: durma kararı val_loss'a göre; macro-F1 eğrisiyle karşılaştırmak için
            val_f1 = float(f1_score(yva, proba_va.argmax(1), average="macro"))
            self.history_.append({"epoch": epoch, "val_loss": round(val_loss, 5),
                                  "val_macro_f1": round(val_f1, 5)})
            if val_loss < best[0] - 1e-5:
                best = (val_loss, epoch, copy.deepcopy(model.state_dict()))
                bad = 0
            else:
                bad += 1
            if self.verbose:
                print(f"    TabM epoch {epoch:3d}  val_loss={val_loss:.4f}  val_macroF1={val_f1:.4f}"
                      f"{'  *' if bad == 0 else ''}  ({time.perf_counter() - t0:.0f} s)")
            if bad > self.patience:
                break
        model.load_state_dict(best[2])
        self.model_ = model
        self.best_epoch_ = best[1]
        self.best_iteration_ = best[1] + 1   # run_model bunu "early stopping" parametresi olarak kaydeder
        f1s = [h["val_macro_f1"] for h in self.history_]
        self.best_epoch_by_f1_ = int(np.argmax(f1s))
        if self.verbose:
            print(f"    TabM durdu: en iyi epoch val_loss'a göre {self.best_epoch_} "
                  f"(macro-F1 {f1s[self.best_epoch_]:.4f}), macro-F1'e göre {self.best_epoch_by_f1_} "
                  f"(macro-F1 {max(f1s):.4f})")
        return self

    # ------------------------------------------------------------ tahmin
    def predict_member_proba(self, X):
        """(n, k, C): her TabM üyesinin sınıf olasılıkları (bilinmeyen skorları için)."""
        import torch
        return torch.softmax(torch.as_tensor(self._logits(self._prep(X))), -1).numpy()

    def predict_proba(self, X):
        return self.predict_member_proba(X).mean(axis=1)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    # joblib ile kaydedip GPU'suz ortamda açılabilsin
    def __getstate__(self):
        state = self.__dict__.copy()
        if "model_" in state:
            state["model_"] = copy.deepcopy(state["model_"]).cpu()
        return state
