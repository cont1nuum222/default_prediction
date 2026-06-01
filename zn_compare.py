import numpy as np
import pandas as pd
import gc
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    confusion_matrix,
    accuracy_score,
)

from pipeline import build_dataset, X_cols, cat_features, MODEL_PATH, EPS

DATA_DIR = r"D:\Desktop\project\data_parquet"
REPORT_PATH = "zn_compare_report.txt"

ZN_OKVED_MIN = 5
ZN_OKVED_MAX = 33
ZN_REVENUE_MIN = 800_000
ZN_LOG_REVENUE_MIN = float(np.log1p(ZN_REVENUE_MIN))

full_data = build_dataset(DATA_DIR)
test_df = full_data.loc[full_data["year"] > 2020].copy()
del full_data
gc.collect()

y_test = test_df["target_default"].astype(np.int8).to_numpy()
log_revenue = test_df["log_revenue"].to_numpy()
okved_num = pd.to_numeric(test_df["okved_short"], errors="coerce").to_numpy()

model = CatBoostClassifier()
model.load_model(MODEL_PATH)
proba = model.predict_proba(Pool(test_df[X_cols], cat_features=cat_features))[:, 1]
del test_df
gc.collect()

zn_filter = (okved_num >= ZN_OKVED_MIN) & (okved_num <= ZN_OKVED_MAX) & (log_revenue >= ZN_LOG_REVENUE_MIN)
y_zn = y_test[zn_filter]
p_zn = proba[zn_filter]

out = [
    f"Фильтр Жукова-Никулина (ОКВЭД {ZN_OKVED_MIN}-{ZN_OKVED_MAX}, выручка >= {ZN_REVENUE_MIN:,})",
    f"n={len(y_zn):,}, base rate {y_zn.mean() * 100:.2f}%",
    f"PR-AUC = {average_precision_score(y_zn, p_zn):.4f}",
]
for thr in (0.5, 0.3):
    pred = (p_zn >= thr).astype(int)
    acc = accuracy_score(y_zn, pred)
    p = precision_score(y_zn, pred, zero_division=0)
    r = recall_score(y_zn, pred, zero_division=0)
    f1 = 2 * p * r / (p + r + EPS)
    cm = confusion_matrix(y_zn, pred)
    out += [
        "",
        f"--- порог {thr:.1f} ---",
        f"  accuracy={acc:.4f}  precision={p:.4f}  recall={r:.4f}  F1={f1:.4f}",
        f"  TN={cm[0, 0]:,}  FP={cm[0, 1]:,}  FN={cm[1, 0]:,}  TP={cm[1, 1]:,}",
    ]

text = "\n".join(out)
print(text)
with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write(text + "\n")
