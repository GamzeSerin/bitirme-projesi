"""GEÇİCİ: dataset_23 LOFO duyarlılık kontrolü, çakışan etiketli satırlar KORUNARAK.

Ortak LOFO aracında (open_set_lofo.py) bu seçenek yok; kayıttaki yükleyici çakışmaları her zaman atar.
Bu betik ortak dosyaları değiştirmeden, yalnız bu çalıştırmada dataset_23 yükleyicisini
drop_conflicts=False ile değiştirir ve open_set_lofo.py'yi aynı argümanlarla çalıştırır.
Ortak araca --keep-conflicts seçeneği eklenince bu dosya kaldırılır.

Kullanım (sonuçlar README'deki duyarlılık bölümünde):
  python src/lofo_keep_conflicts.py --dataset dataset_23 --data-file data/dataset_23 \
      --condition C_ttl_yok --models RF --tag-suffix _keep_conflicts
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import open_set_lofo  # noqa: E402
from common import open_set_registry as registry  # noqa: E402

DATASET = "dataset_23"


def _ciciot_loader_keep(data_file, cap, seed):
    from model_ml import load_data
    X, fine, category, stats = load_data(data_file, cap, seed, drop_features=registry.CICIOT_PROTOCOL_DROP,
                                         drop_conflicts=False)
    leaked = [c for c in registry.CICIOT_PROTOCOL_DROP if c in X.columns]
    if leaked:
        raise SystemExit(f"CICIoT2023: protokol gereği atılması gereken sütun(lar) veride: {leaked}")
    return X, fine, category, dict(stats, sensitivity="keep_conflicts (çakışan etiketli satırlar korundu)")


if __name__ == "__main__":
    spec = registry.DATASETS.get(DATASET)
    # kayıt yapısı değişirse sessizce yanlış çalışmasın
    if spec is None or spec.get("loader") is not registry._ciciot_loader or not hasattr(registry, "CICIOT_PROTOCOL_DROP"):
        raise SystemExit("Ortak kayıt (open_set_registry.py) beklenen yapıda değil; bu geçici betiği güncelleyin.")
    if "--dataset" in sys.argv and sys.argv[sys.argv.index("--dataset") + 1] != DATASET:
        raise SystemExit(f"Bu betik yalnız {DATASET} içindir.")
    spec["loader"] = _ciciot_loader_keep
    open_set_lofo.main()
