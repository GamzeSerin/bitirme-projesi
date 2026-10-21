"""Geriye uyumluluk: Edge-IIoTset için eski komut.

Asıl kod ekip ortak aracı open_set_lofo.py'dedir; bu dosya --dataset dnn_edgeiiot ile onu çağırır
ve eski içe aktarmaları (FAMILIES, evaluate, model_scores, ...) korur.
  python src/open_set_lofo_dnn_edgeiiot.py --data-file <csv> ...   ==   python src/open_set_lofo.py --dataset dnn_edgeiiot ...
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from open_set_lofo import (FAMILIES, OUT_DIR, UNKNOWN, KNNScore, entropy, evaluate,  # noqa: F401
                           main, model_scores, plot, rf_members, scores_from_members)

if __name__ == "__main__":
    if "--dataset" not in sys.argv:
        sys.argv[1:1] = ["--dataset", "dnn_edgeiiot"]
    main()
