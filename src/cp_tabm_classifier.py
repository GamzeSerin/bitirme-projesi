"""CP-TabM: sınıf-bölümlü TabM ile görülmemiş saldırı tespiti (önerilen yöntem).

Fikir. TabM'in k üyesi (BatchEnsemble) aynı gövdeyi paylaşır ama ayrı tahmin üretir. Sade TabM'de
üyelerin anlaşmazlığı görülmemiş saldırının en güçlü model içi sinyaliydi (bkz. open_set_lofo
sonuçları). CP-TabM bu anlaşmazlığı rastlantıya bırakmaz, eğitimle YAPILANDIRIR:

1. Körleme matrisi A (k x C). Her bilinen sınıf, üyelerin bir kısmına (blind_frac, ör. %25) "kör"
   bırakılır; atama dengelidir, her üye en az iki sınıf görür ve her sınıfın körleme deseni
   (imzası) farklıdır. Her üye böylece kendi gördüğü sınıflar için bir "uzman" olur.
2. Maskeli kayıp. Üye m yalnızca gördüğü sınıflarda çapraz entropi öğrenir (kör sınıfların
   logitleri maskelenir). Ek olarak üye bazlı enerji sınırı (Liu vd., NeurIPS 2020'den uyarlama):
   üye, gördüğü sınıfların örneklerinde düşük enerji (E <= m_in), kör olduğu sınıfların
   örneklerinde yüksek enerji (E >= m_out) üretmeye zorlanır. Böylece her üye, eğitimde
   "görmediği" sınıfları reddetmeyi öğrenir: görülmemiş saldırı provası.
3. İmza tutarlılığı skoru U(x). Bilinen sınıf c'den bir örnekte, c'yi gören üyeler kabul eder,
   c'ye kör üyeler reddeder: üyelerin red vektörü r(x), c'nin imzasıyla (1 - A[:, c]) örtüşür.
   Görülmemiş bir saldırı hiçbir sınıfın imzasına uymaz (çoğu üye reddeder ya da desen karışır).
   U(x) = min_c ort_m |r_m(x) - (1 - A[m, c])|   (büyük = bilinmeyen)
   Red değeri r_m, üyenin doğrulama verisindeki kabul/red enerji seviyelerine göre 0-1'e ölçeklenir.

Açıklanabilirlik. U(x) hangi sınıfın imzasına ne kadar yakın olunduğunu (sınıf başına uzaklık)
ve hangi uzman grubunun reddettiğini doğrudan verir; explain() bunu döndürür. Özellik düzeyinde
açıklama (SHAP) U(x) üzerinde ayrıca hesaplanabilir.

2. sürüm skorları (20.10.2026; ilk sonuçlarda imza skoru, bilinen bir sınıfa çok benzeyen
bilinmeyenlerde (DDOS ~ RECON) o sınıfın imzasını taşıdığı için zayıf kaldı):
   expert_disagree   tahmin edilen sınıfı GÖREN üyeler arası anlaşmazlık; kör üyelerin tasarım gereği
                     farklı oyu dışarıda kalır, sade TabM'in doğal anlaşmazlık sinyali geri gelir
   expert_energy_z   o uzmanların enerjisinin, aynı sınıfın gerçek örneklerindeki (iç doğrulama)
                     seviyesine göre sapması: "uzmanlar her zamankinden ne kadar az emin"
   İkisi de eğitimi değiştirmez; yalnızca iç doğrulama verisiyle ölçeklenir.

3. sürüm (20.10.2026; 2. sürümde iki skor farklı ailelerde güçlü çıktı: anlaşmazlık genel tespitte,
emin olma derecesi sessiz BENIGN kaçışını önlemede):
   comb_max / comb_mean  her iki skor, iç doğrulamadaki bilinenlere göre yüzdeliğe çevrilip en
                         büyüğü / ortalaması alınır. Hangisinin ve kaç kör üyenin kullanılacağı
                         cp_tune_dnn_edgeiiot.py ile (iç içe LOFO, test ailesi kullanılmadan) seçilir.

Bileşen testleri (open_set_lofo_dnn_edgeiiot.py):
  TabM          körleme yok, enerji kaybı yok (sade TabM; anlaşmazlık skoru)
  CPTabM-noE    körleme var, enerji kaybı yok (lam_energy = 0)
  CPTabM        körleme + enerji kaybı
  ve her modelde imza skoru ile msp / disagree / energy karşılaştırılır.

En yakın öncül: Vyas vd. (ECCV 2018), sınıfları dışarıda bırakan AYRI ağlardan oluşan topluluk.
Burada ayrı ağlar yerine tek TabM'in paylaşılan gövdeli üyeleri kullanılır (k kat daha ucuz) ve
karar, üyelerin ortak red imzasıyla verilir.
"""

import numpy as np

from tabm_classifier import TabMClassifier

COMBINE = ("expert_disagree", "expert_energy_z")  # birleştirilen iki tamamlayıcı skor
NEG = -1e4  # maskeli logit (float32'de güvenli; -inf logsumexp'te NaN üretebilir)


def blind_matrix(k, n_classes, blind_frac, seed, min_seen=2, max_tries=200):
    """(k, C) bool 'görür' matrisi. Her sınıf r = round(blind_frac * k) üyeye kördür; kör yük
    üyelere dengeli dağıtılır; her üye >= min_seen sınıf görür; sınıf imzaları birbirinden farklıdır."""
    rng = np.random.default_rng(seed)
    r = int(round(blind_frac * k))
    if r <= 0:
        return np.ones((k, n_classes), dtype=bool)
    r = min(r, k - 1)
    for _ in range(max_tries):
        sees = np.ones((k, n_classes), dtype=bool)
        load = np.zeros(k)
        ok = True
        for c in rng.permutation(n_classes):
            cand = [m for m in rng.permutation(k) if sees[m].sum() - 1 >= min_seen]
            if len(cand) < r:
                ok = False
                break
            cand.sort(key=lambda m: load[m])  # en az kör yükü olan üyeler önce (dengeli)
            pick = cand[:r]
            sees[pick, c] = False
            load[pick] += 1
        if ok and len({tuple(col) for col in sees.T}) == n_classes:
            return sees
    raise RuntimeError("Körleme matrisi kurulamadı; blind_frac'ı düşürün")


class CPTabMClassifier(TabMClassifier):
    VERSION = 1

    def __init__(self, blind_frac=0.25, lam_energy=0.1, m_in=-6.0, m_out=-1.0, k=32, n_blocks=None,
                 d_block=512, dropout=0.1, lr=2e-3, weight_decay=3e-4, batch_size=512, max_epochs=60,
                 patience=8, n_bins=48, d_embedding=16, val_size=0.1, balanced=True, device="auto",
                 random_state=42, verbose=True, feature_frac=None):
        super().__init__(feature_frac=feature_frac, k=k, n_blocks=n_blocks, d_block=d_block, dropout=dropout, lr=lr,
                         weight_decay=weight_decay, batch_size=batch_size, max_epochs=max_epochs,
                         patience=patience, n_bins=n_bins, d_embedding=d_embedding, val_size=val_size,
                         balanced=balanced, device=device, random_state=random_state, verbose=verbose)
        self.blind_frac = blind_frac
        self.lam_energy = lam_energy
        self.m_in = m_in
        self.m_out = m_out

    # ------------------------------------------------------------ eğitim kancaları
    def _setup(self, n_classes, k, device, y_train):
        import torch
        self.A_ = blind_matrix(k, n_classes, self.blind_frac, self.random_state)
        self.mask_t_ = torch.as_tensor(self.A_, device=device)
        if self.verbose:
            print(f"    CP-TabM: {k} üye, sınıf başına {int((~self.A_).sum(0)[0])} kör üye, "
                  f"üye başına {self.A_.sum(1).min()}-{self.A_.sum(1).max()} sınıf görülüyor, "
                  f"lam_energy={self.lam_energy}, m_in={self.m_in}, m_out={self.m_out}")

    def _train_loss(self, logits, y, w):
        import torch
        import torch.nn.functional as F
        b, k, C = logits.shape
        mask = self.mask_t_                                          # (k, C)
        ml = logits.masked_fill(~mask[None], NEG)
        sees = mask[:, y].T                                          # (b, k): üye örneğin sınıfını görüyor mu
        ce = F.cross_entropy(ml.flatten(0, 1), y.repeat_interleave(k), weight=w,
                             reduction="none").view(b, k)
        wy = w[y][:, None].expand(b, k)
        loss = (ce * sees).sum() / (wy * sees).sum().clamp_min(1e-12)
        if self.lam_energy > 0:
            E = -torch.logsumexp(ml, dim=-1)                         # (b, k) üye enerjisi (gördüğü sınıflar)
            e_in = F.relu(E - self.m_in).pow(2)
            e_out = F.relu(self.m_out - E).pow(2)
            n_in, n_out = sees.sum().clamp_min(1), (~sees).sum().clamp_min(1)
            loss = loss + self.lam_energy * ((e_in * sees).sum() / n_in + (e_out * ~sees).sum() / n_out)
        return loss

    # ------------------------------------------------------------ olasılıklar (numpy)
    def _masked(self, logits):
        return np.where(self.A_[None], logits, NEG)

    def _member_proba_np(self, logits):
        return self._softmax_np(self._masked(logits))

    def _proba_np(self, logits):
        """Sınıf c'nin olasılığı yalnızca c'yi gören üyelerin ortalaması; satırlar normalize edilir."""
        Pm = self._member_proba_np(logits)                           # (n, k, C), kör sınıflar 0
        P = (Pm * self.A_[None]).sum(1) / self.A_.sum(0)[None]
        return P / P.sum(1, keepdims=True)

    def _energy_np(self, logits):
        from scipy.special import logsumexp
        return -logsumexp(self._masked(logits), axis=-1)             # (n, k)

    def _post_fit(self, Xva, yva):
        """Üye başına kabul / red enerji seviyeleri (iç doğrulama verisinden; test hiç kullanılmaz)."""
        E = self._energy_np(self._logits(Xva))
        sees = self.A_[:, yva].T                                     # (n, k)
        k = E.shape[1]
        self.e_acc_ = np.array([np.median(E[sees[:, m], m]) if sees[:, m].any() else self.m_in
                                for m in range(k)])
        self.e_rej_ = np.array([np.median(E[~sees[:, m], m]) if (~sees[:, m]).any() else self.m_out
                                for m in range(k)])
        gap = self.e_rej_ - self.e_acc_
        self.energy_gap_ = float(np.median(gap))
        # Sınıfa özgü "emin olma" seviyesi: gerçek c örneklerinde, c'yi gören uzmanların ort. enerjisi
        Eexp = self._expert_energy(E, yva)
        C = self.A_.shape[1]
        self.cls_mu_ = np.array([np.median(Eexp[yva == c]) if (yva == c).any() else 0.0 for c in range(C)])
        self.cls_sd_ = np.array([max(np.subtract(*np.percentile(Eexp[yva == c], [75, 25])) / 1.349, 1e-3)
                                 if (yva == c).sum() > 3 else 1.0 for c in range(C)])
        # Birleştirme için referans: iç doğrulamadaki (yalnızca bilinen) örneklerin skor dağılımı
        _, raw = self._scores_from_logits(self._logits(Xva))
        self.ref_ = {k: np.sort(raw[k]) for k in COMBINE}
        if self.verbose:
            print(f"    CP-TabM doğrulama enerjisi: kabul medyanı {np.median(self.e_acc_):.2f}, "
                  f"red medyanı {np.median(self.e_rej_):.2f}, üye ayrımı (medyan) {self.energy_gap_:.2f}"
                  f"{'  ! bazı üyeler reddi öğrenmedi' if (gap <= 0.1).any() else ''}")

    def _expert_energy(self, E, cls):
        """Her örnek için, verilen sınıfı gören üyelerin (uzmanların) ortalama enerjisi."""
        S = self.A_[:, cls].T                                        # (n, k)
        return (E * S).sum(1) / S.sum(1)

    def _expert_disagreement(self, Pm, cls):
        """Yalnızca verilen sınıfı gören üyeler arasında karşılıklı bilgi (kör üyelerin yapay
        anlaşmazlığı dışarıda kalır; sade TabM'deki doğal anlaşmazlık sinyalini geri kazanır)."""
        S = self.A_[:, cls].T.astype(float)[:, :, None]             # (n, k, 1)
        n_exp = S.sum(1)                                             # (n, 1)
        Pbar = (Pm * S).sum(1) / n_exp                               # (n, C)

        def H(p):
            return -(p * np.log(np.clip(p, 1e-12, None))).sum(-1)
        return H(Pbar) - (H(Pm) * S[:, :, 0]).sum(1) / n_exp[:, 0]

    def _rejection(self, E):
        span = np.maximum(self.e_rej_ - self.e_acc_, 1e-3)
        return np.clip((E - self.e_acc_[None]) / span[None], 0.0, 1.0)  # 0 kabul, 1 red

    def _signature_dist(self, R):
        """(n, C): red vektörünün her sınıf imzasına ortalama mutlak uzaklığı."""
        S = (~self.A_).astype(float)                                 # (k, C), 1 = reddetmeli
        return np.abs(R[:, :, None] - S[None]).mean(axis=1)

    # ------------------------------------------------------------ bilinmeyen skorları
    def _pct(self, name, x):
        """Skoru, iç doğrulamadaki bilinenlerin yüzde kaçından daha şüpheli olduğuna çevirir (0-1)."""
        ref = self.ref_[name]
        return np.searchsorted(ref, x, side="right") / len(ref)

    def unknown_scores(self, X):
        """Dönen: P (n, C) sınıf olasılıkları, skor sözlüğü (büyük = bilinmeyen)."""
        P, scores = self._scores_from_logits(self.predict_logits(X))
        pct = np.stack([self._pct(k, scores[k]) for k in COMBINE])  # (2, n)
        # 3. sürüm: iki tamamlayıcı skorun birleşimi (uzmanlar anlaşamıyor YA DA her zamankinden az emin)
        scores["comb_max"] = pct.max(axis=0)
        scores["comb_mean"] = pct.mean(axis=0)
        return P, scores

    def _scores_from_logits(self, logits):
        Pm = self._member_proba_np(logits)
        P = self._proba_np(logits)
        E = self._energy_np(logits)
        D = self._signature_dist(self._rejection(E))
        pred = P.argmax(1)

        def H(p):
            return -(p * np.log(np.clip(p, 1e-12, None))).sum(-1)
        seen_pred = self.A_[:, pred].T                               # tahmin edilen sınıfı gören üyeler
        e_exp = self._expert_energy(E, pred)
        z = (e_exp - self.cls_mu_[pred]) / self.cls_sd_[pred]   # uzmanlar her zamankinden ne kadar az emin
        dis = self._expert_disagreement(Pm, pred)
        scores = {
            "signature": D.min(axis=1),
            "msp": 1.0 - P.max(axis=1),
            "disagree": H(Pm.mean(1)) - H(Pm).mean(1),
            "energy": (E * seen_pred).sum(1) / seen_pred.sum(1),
            # 2. sürüm skorları (aynı eğitim; yalnızca karar kuralı)
            "expert_disagree": dis,
            "expert_energy_z": z,
        }
        return P, scores

    def class_energy_z(self, X, cls):
        """Belirli bir sınıfın uzmanlarının, o sınıfın gerçek örneklerine göre ne kadar az emin olduğu
        (z; büyük = her zamankinden az emin). İki sinyalli karar kuralında BENIGN kararlarının
        kontrolü için: tahmin hangi modelden gelirse gelsin, BENIGN uzmanlarına sorulur."""
        E = self._energy_np(self.predict_logits(X))
        cls_arr = np.full(len(E), int(cls))
        return (self._expert_energy(E, cls_arr) - self.cls_mu_[cls]) / self.cls_sd_[cls]

    def explain(self, X, class_names=None, top=3):
        """Her örnek için: tahmin, imza skoru, en yakın sınıf imzaları ve reddeden uzman grupları."""
        names = list(class_names) if class_names is not None else [str(c) for c in self.classes_]
        logits = self.predict_logits(X)
        P = self._proba_np(logits)
        R = self._rejection(self._energy_np(logits))
        D = self._signature_dist(R)
        _, raw = self._scores_from_logits(logits)
        pd_ = self._pct("expert_disagree", raw["expert_disagree"])
        pz = self._pct("expert_energy_z", raw["expert_energy_z"])
        out = []
        for i in range(len(P)):
            order = np.argsort(D[i])[:top]
            experts = {names[c]: round(float(1 - R[i, self.A_[:, c]].mean()), 3) for c in range(len(names))}
            out.append({
                "pred": names[int(P[i].argmax())], "prob": round(float(P[i].max()), 3),
                "signature_score": round(float(D[i].min()), 3),
                "nearest_signatures": [(names[c], round(float(D[i, c]), 3)) for c in order],
                "expert_acceptance": experts,   # sınıf c'yi gören üyelerin ortalama kabul oranı
                # Tahmin edilen sınıfın uzmanları: bilinenlerin yüzde kaçından daha çok anlaşamıyor /
                # yüzde kaçından daha az emin (1'e yakın = bilinenlerde nadir görülen şüphe)
                "experts_disagree_pct": round(float(pd_[i]), 3),
                "experts_unsure_pct": round(float(pz[i]), 3),
            })
        return out
