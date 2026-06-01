import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    confusion_matrix,
)
from scipy.stats import ks_2samp

EPS = 1e-6
MODEL_PATH = "model.cbm"

proba = np.load("cb_default_proba.npy")
y_test = np.load("y_test_default.npy")
mask = np.isfinite(proba)
proba = proba[mask]
y_test = y_test[mask]

n = len(y_test)
base_rate = float(y_test.mean())
auc = roc_auc_score(y_test, proba)
ks = ks_2samp(proba[y_test == 1], proba[y_test == 0]).statistic
pr_auc = average_precision_score(y_test, proba)

print(f"Метрики итоговой модели ({MODEL_PATH})")
print(f"Тест 2021-2022: {n:,} наблюдений, дефолтов {int(y_test.sum()):,} ({base_rate * 100:.2f}%)")
print()
print(f"  ROC-AUC = {auc:.4f}")
print(f"  KS      = {ks:.4f}")
print(f"  PR-AUC  = {pr_auc:.4f}  (lift {pr_auc / base_rate:.2f}x)")

precs, recs, thrs = precision_recall_curve(y_test, proba)
f1s = 2 * precs * recs / (precs + recs + EPS)
thr_f1 = float(thrs[int(np.argmax(f1s[:-1]))])

idx_p70 = np.where(precs[:-1] >= 0.70)[0]
thr_p70 = float(thrs[idx_p70[np.argmax(recs[idx_p70])]])

idx_p85 = np.where(precs[:-1] >= 0.85)[0]
thr_p85 = float(thrs[idx_p85[np.argmax(recs[idx_p85])]])

print("\nПороги классификации")
for name, thr in [
    ("F1-оптимум", thr_f1),
    ("Высокая точность (P>=0.70)", thr_p70),
    ("Сверхточный (P>=0.85)", thr_p85),
]:
    pred = (proba >= thr).astype(int)
    p = precision_score(y_test, pred, zero_division=0)
    r = recall_score(y_test, pred, zero_division=0)
    f1 = 2 * p * r / (p + r + EPS)
    cm = confusion_matrix(y_test, pred)
    print(f"\n{name} (порог = {thr:.4f})")
    print(f"  precision = {p:.4f}   recall = {r:.4f}   F1 = {f1:.4f}")
    print(f"  TN={cm[0, 0]:>8,}  FP={cm[0, 1]:>8,}")
    print(f"  FN={cm[1, 0]:>8,}  TP={cm[1, 1]:>8,}")

print("\nLift в топ-K% по риск-скору")
order = np.argsort(proba)[::-1]
for k in (0.005, 0.01, 0.02, 0.05, 0.10):
    top_n = max(int(len(proba) * k), 1)
    top = float(y_test[order[:top_n]].mean())
    print(f"  top-{k * 100:>4.1f}%: precision={top:.4f}  lift={top / base_rate:.2f}x")

print("\nТоп-20 признаков по важности")
model = CatBoostClassifier()
model.load_model(MODEL_PATH)
imp = pd.DataFrame({
    "feature": model.feature_names_,
    "importance": model.get_feature_importance(),
}).sort_values("importance", ascending=False).reset_index(drop=True)
print(imp.head(20).to_string(index=False))
