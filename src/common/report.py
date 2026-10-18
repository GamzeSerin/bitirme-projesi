"""Analiz tabloları ve makale grafikleri.

Yalnızca outputs/<dataset_id>/ dosyalarını okur; ham veriye ihtiyaç duymaz.
notebooks/analysis.ipynb bu fonksiyonları çağırır.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix, f1_score

# --- Renkler (dataviz referans paleti, açık tema) --------------------------------
SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
# Kategorik slotlar sabit sırayla; bir model rengini her grafikte korur.
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
KNOWN_MODELS = ["DT", "RF", "XGB", "LGBM", "LR"]
DSS_COLOR = INK_2
BLUES = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
# DSS aksiyonları durum renkleri (yalnızca durum için ayrılmış)
ACTION_COLORS = {"ALLOW": "#0ca30c", "LOG_MONITOR": "#fab219", "ALERT": "#ec835a",
                 "BLOCK": "#d03b3b", "ANALYST_REVIEW": MUTED}
ACTION_ORDER = ["ALLOW", "LOG_MONITOR", "ANALYST_REVIEW", "ALERT", "BLOCK"]


def setup_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK_2, "text.color": INK,
        "xtick.color": INK_2, "ytick.color": INK_2, "axes.titlesize": 12,
        "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.8,
        "font.size": 10, "legend.frameon": False,
    })


def model_colors(models):
    """Bilinen modeller sabit slotlarını alır, diğerleri sıradaki boş slotları."""
    colors, free = {}, [c for i, c in enumerate(SLOTS) if i >= len(KNOWN_MODELS)]
    for m in models:
        if m in KNOWN_MODELS:
            colors[m] = SLOTS[KNOWN_MODELS.index(m)]
        else:
            colors[m] = free.pop(0) if free else MUTED
    return colors


def save(fig, fig_dir, name):
    os.makedirs(fig_dir, exist_ok=True)
    path = os.path.join(fig_dir, f"{name}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    return path


# --- Tablolar --------------------------------------------------------------------
def model_summary(envs):
    rows = []
    for eid, env in envs.items():
        for m in env.models:
            met = env.metrics[m]
            rows.append({
                "dataset": eid, "model": m,
                "OOF macro-F1": met["oof"]["macro_f1"],
                "Test macro-F1": met["test"]["macro_f1"],
                "Test accuracy": met["test"]["accuracy"],
                "Test weighted-F1": met["test"]["weighted_f1"],
                "Eğitim (s)": met.get("timing", {}).get("fit_seconds"),
                "Tahmin (ms/1k)": met.get("timing", {}).get("predict_ms_per_1k"),
                "Birincil": m == env.manifest["primary_model"],
                "DSS'te oy": m in env.voters,
            })
    return pd.DataFrame(rows).set_index(["dataset", "model"])


def per_class_f1(env, split="test"):
    return pd.DataFrame({m: {c: env.metrics[m][split]["per_class"][c]["f1"] for c in env.classes}
                         for m in env.models})


def best_model_table(env):
    """Saldırı tipi × en iyi model. Seçim OOF'a göre; test F1 raporlanır."""
    oof, test = per_class_f1(env, "oof"), per_class_f1(env, "test")
    rows = []
    for c in env.classes:
        ranked = oof.loc[c].sort_values(ascending=False)
        best, second = ranked.index[0], ranked.index[1] if len(ranked) > 1 else None
        rows.append({"sınıf": c, "en iyi model (OOF)": best,
                     "OOF F1": round(ranked.iloc[0], 4),
                     "test F1": round(test.loc[c, best], 4),
                     "ikinci": second,
                     "fark (OOF)": round(ranked.iloc[0] - ranked.iloc[1], 4) if second else None,
                     "test FPR": env.metrics[best]["test"]["per_class"][c]["fpr"]})
    return pd.DataFrame(rows).set_index("sınıf")


def test_frame(env):
    df = pd.read_csv(os.path.join(env.dir, "test_predictions.csv.gz"))
    y = np.array([env.classes.index(c) for c in df["y_true"]])
    return df, y


def nids_binary(env, dss_predict):
    """İkili NIDS bakışı: saldırı var mı? (BENIGN dışı her şey saldırı).

    Not: test_predictions varsayılan olarak tüm test setidir (sınıf oranları sınırlanmış); tespit oranı
    ve FPR sınıf oranından bağımsızdır, ama 'alarm precision' gerçek ağdaki
    saldırı oranına göre değişir.
    """
    df, y = test_frame(env)
    benign = env.classes.index("BENIGN") if "BENIGN" in env.classes else None
    if benign is None:
        return None
    preds = {m: env.probas_from_frame(df, [m])[0].argmax(1) for m in env.models}
    dss = dss_predict(env, env.probas_from_frame(df))
    preds["DSS"] = np.array([env.classes.index(d["predicted"]) for d in dss])
    is_attack = y != benign
    rows = []
    for name, p in preds.items():
        alarm = p != benign
        tp, fn = (alarm & is_attack).sum(), (~alarm & is_attack).sum()
        fp, tn = (alarm & ~is_attack).sum(), (~alarm & ~is_attack).sum()
        rows.append({"model": name,
                     "Tespit oranı (DR)": tp / (tp + fn),
                     "Yanlış alarm (FPR)": fp / (fp + tn),
                     "Kaçırılan saldırı": int(fn),
                     "Binary F1": f1_score(is_attack, alarm)})
    return pd.DataFrame(rows).set_index("model").round(4)


# --- Grafikler -------------------------------------------------------------------
def _hbar_labels(ax, bars, fmt="{:.3f}"):
    for b in bars:
        ax.text(b.get_width() + 0.005, b.get_y() + b.get_height() / 2, fmt.format(b.get_width()),
                va="center", fontsize=9, color=INK_2)


def plot_model_comparison(envs, fig_dir, sample_scores):
    """Her veri seti için bir panel: modeller ve DSS, AYNI test satırlarında macro-F1.

    sample_scores[eid] = main.evaluate_environment(...)["macro_f1_on_test_sample"]
    """
    n = len(envs)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 3.6), squeeze=False)
    for ax, (eid, env) in zip(axes[0], envs.items()):
        palette = model_colors(env.models)
        names = [m for m in sample_scores[eid] if m in env.models] + ["DSS"]
        vals = [sample_scores[eid][m] for m in names[:-1]] + [sample_scores[eid]["DSS (ağırlıklı)"]]
        colors = [palette[m] for m in names[:-1]] + [DSS_COLOR]
        order = np.argsort(vals)
        bars = ax.barh([names[i] for i in order], [vals[i] for i in order],
                       color=[colors[i] for i in order], height=0.62)
        _hbar_labels(ax, bars)
        ax.set_xlim(0, 1.08)
        ax.set_title(f"{eid} (DSS oylayanlar: {', '.join(env.voters)})", fontsize=11)
        ax.set_xlabel("Macro-F1 (test_predictions: modeller ve DSS aynı satırlarda)")
        ax.grid(axis="x")
        ax.set_axisbelow(True)
    fig.tight_layout()
    return fig, save(fig, fig_dir, "fig_model_comparison")


def plot_per_class_heatmap(env, fig_dir):
    f1 = per_class_f1(env, "test")
    fig, ax = plt.subplots(figsize=(1.2 * len(f1.columns) + 2.5, 0.48 * len(f1) + 1.2))
    im = ax.imshow(f1.values, cmap=BLUES, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(f1.columns)), f1.columns)
    ax.set_yticks(range(len(f1.index)), f1.index)
    best = f1.values.argmax(axis=1)
    for i in range(f1.shape[0]):
        for j in range(f1.shape[1]):
            v = f1.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if v > 0.6 else INK,
                    fontweight="bold" if j == best[i] else "normal")
    ax.spines[:].set_visible(False)
    ax.set_title(f"{env.id}: sınıf bazında test F1 (kalın = satırın en iyisi)")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="F1")
    fig.tight_layout()
    return fig, save(fig, fig_dir, f"fig_per_class_f1_{env.id}")


def plot_confusion(env, fig_dir, model=None):
    model = model or env.manifest["primary_model"]
    cm = np.array(env.metrics[model]["test"]["confusion_matrix"], dtype=float)
    cmn = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    k = len(env.classes)
    fig, ax = plt.subplots(figsize=(0.75 * k + 2.5, 0.7 * k + 1.5))
    im = ax.imshow(cmn, cmap=BLUES, vmin=0, vmax=1)
    ax.set_xticks(range(k), env.classes, rotation=40, ha="right")
    ax.set_yticks(range(k), env.classes)
    for i in range(k):
        for j in range(k):
            if cmn[i, j] >= 0.01:
                ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if cmn[i, j] > 0.6 else INK)
    ax.set_xlabel("Tahmin")
    ax.set_ylabel("Gerçek")
    ax.spines[:].set_visible(False)
    ax.set_title(f"{env.id} · {model}: normalize confusion matrix (test)")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="Satır oranı")
    fig.tight_layout()
    return fig, save(fig, fig_dir, f"fig_confusion_{env.id}_{model}")


def plot_fine_label_recall(env, fig_dir, model=None):
    model = model or env.manifest["primary_model"]
    rec = env.metrics[model].get("test_fine_label_recall")
    if not rec:
        return None, None
    s = pd.Series(rec).sort_values()
    fig, ax = plt.subplots(figsize=(7, 0.26 * len(s) + 1))
    bars = ax.barh(s.index, s.values, color=SLOTS[0], height=0.62)
    _hbar_labels(ax, bars, "{:.2f}")
    ax.set_xlim(0, 1.1)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_xlabel("Doğru kategoriye atanan oran (recall)")
    ax.set_title(f"{env.id} · {model}: ince etiket bazında recall")
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig, save(fig, fig_dir, f"fig_fine_label_recall_{env.id}")


def plot_dss_actions(env, dss_predict, fig_dir):
    """Gerçek sınıf başına DSS aksiyon dağılımı (%100 yığılmış çubuk)."""
    df, _ = test_frame(env)
    acts = pd.DataFrame({"true": df["y_true"],
                         "action": [d["action"] for d in dss_predict(env, env.probas_from_frame(df))]})
    tab = pd.crosstab(acts["true"], acts["action"], normalize="index")
    tab = tab.reindex(index=env.classes[::-1], columns=[a for a in ACTION_ORDER if a in tab.columns],
                      fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(tab) + 1.4))
    left = np.zeros(len(tab))
    for a in tab.columns:
        ax.barh(tab.index, tab[a], left=left, color=ACTION_COLORS[a], label=a, height=0.62,
                edgecolor=SURFACE, linewidth=1.5)
        left += tab[a].values
    ax.set_xlim(0, 1)
    ax.set_xlabel("Oran")
    ax.set_title(f"{env.id}: gerçek sınıfa göre DSS aksiyonları")
    ax.legend(ncol=len(tab.columns), loc="upper center", bbox_to_anchor=(0.5, -0.18), fontsize=9)
    fig.tight_layout()
    return fig, save(fig, fig_dir, f"fig_dss_actions_{env.id}")


def plot_cross_dataset(envs, fig_dir):
    """Birincil modellerin sınıf bazında F1'i, veri setleri yan yana (ortak taksonomi)."""
    if len(envs) < 2:
        return None, None
    data = {eid: per_class_f1(env, "test")[env.manifest["primary_model"]]
            for eid, env in envs.items()}
    tab = pd.DataFrame(data)
    fig, ax = plt.subplots(figsize=(1.6 * len(tab.columns) + 2.5, 0.48 * len(tab) + 1.2))
    vals = tab.values
    im = ax.imshow(np.nan_to_num(vals, nan=0), cmap=BLUES, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(tab.columns)),
                  [f"{e}\n({envs[e].manifest['primary_model']})" for e in tab.columns])
    ax.set_yticks(range(len(tab.index)), tab.index)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            ax.text(j, i, "—" if np.isnan(vals[i, j]) else f"{vals[i, j]:.2f}", ha="center",
                    va="center", fontsize=9,
                    color="white" if (not np.isnan(vals[i, j]) and vals[i, j] > 0.6) else INK)
    ax.spines[:].set_visible(False)
    ax.set_title("Veri setleri arası: birincil model sınıf bazında test F1")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="F1")
    fig.tight_layout()
    return fig, save(fig, fig_dir, "fig_cross_dataset_f1")


# --- Güven aralıkları ------------------------------------------------------------
def _f1_from_confusion(cm):
    tp = np.diag(cm).astype(float)
    prec = tp / np.clip(cm.sum(axis=0), 1, None)
    rec = tp / np.clip(cm.sum(axis=1), 1, None)
    return np.where(prec + rec > 0, 2 * prec * rec / np.clip(prec + rec, 1e-12, None), 0.0)


def bootstrap_ci(env, dss_predict, n_boot=500, seed=42, alpha=0.05):
    """Test satırlarını yerine koyarak yeniden örnekler; macro-F1 ve sınıf F1'i için
    %95 güven aralığı döner. İki modelin aralıkları çakışıyorsa farkları anlamlı sayılmaz.

    Dönüş: (macro_df, per_class_df)  per_class_df birincil model ve DSS içindir.
    """
    df, y = test_frame(env)
    k = len(env.classes)
    preds = {m: env.probas_from_frame(df, [m])[0].argmax(1) for m in env.models}
    preds["DSS"] = np.array([env.classes.index(d["predicted"])
                             for d in dss_predict(env, env.probas_from_frame(df))])
    rng = np.random.default_rng(seed)
    n = len(y)
    boots = {m: np.empty((n_boot, k)) for m in preds}
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        for m, p in preds.items():
            cm = np.bincount(yb * k + p[idx], minlength=k * k).reshape(k, k)
            boots[m][b] = _f1_from_confusion(cm)
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    macro_rows, class_rows = [], []
    for m, p in preds.items():
        point = _f1_from_confusion(np.bincount(y * k + p, minlength=k * k).reshape(k, k))
        mb = boots[m].mean(axis=1)
        macro_rows.append({"model": m, "macro-F1": point.mean(),
                           "%95 alt": np.percentile(mb, lo), "%95 üst": np.percentile(mb, hi)})
        if m in (env.manifest["primary_model"], "DSS"):
            for i, c in enumerate(env.classes):
                class_rows.append({"model": m, "sınıf": c, "F1": point[i],
                                   "%95 alt": np.percentile(boots[m][:, i], lo),
                                   "%95 üst": np.percentile(boots[m][:, i], hi),
                                   "destek": int((y == i).sum())})
    macro = pd.DataFrame(macro_rows).set_index("model").sort_values("macro-F1", ascending=False)
    per_class = pd.DataFrame(class_rows).set_index(["model", "sınıf"])
    return macro.round(4), per_class.round(4)
