import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from ..metrics import A2ACalculator, C2CCoverage, MoJoCalculator, TurboMQ


def evaluate_clustering(y_true, labels, df_deps=None, node_names=None):
    y_true = np.asarray(y_true, dtype=int)
    preds, _ = pd.factorize(np.asarray(labels))
    row = {
        "mojofm": MoJoCalculator(preds, y_true, mode="array").mojofm(),
        "a2a": A2ACalculator(preds, y_true, mode="array").a2a(),
        "c2c_cvg_33": C2CCoverage(preds, y_true, mode="array").c2c_cvg(threshold=0.33),
        "c2c_cvg_50": C2CCoverage(preds, y_true, mode="array").c2c_cvg(threshold=0.50),
        "c2c_cvg_66": C2CCoverage(preds, y_true, mode="array").c2c_cvg(threshold=0.66),
        "c2c_cvg_80": C2CCoverage(preds, y_true, mode="array").c2c_cvg(threshold=0.80),
        "ari": adjusted_rand_score(y_true, preds),
    }
    if df_deps is not None and node_names is not None:
        pred_list = preds.tolist()
        row["normalized_turbomq"] = TurboMQ(
            df_deps=df_deps,
            labels=(list(map(str, node_names)), pred_list),
            normalized=True,
        ).score()
        row["turbomq"] = TurboMQ(
            df_deps=df_deps,
            labels=(list(map(str, node_names)), pred_list),
            normalized=False,
        ).score()
    return row
