import numpy as np
import pandas as pd
import gc
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    confusion_matrix,
)
from scipy.stats import ks_2samp

from pipeline import build_dataset, X_cols, cat_features, MODEL_PATH, EPS

DATA_DIR = r"D:\Desktop\project\data_parquet"

full_data = build_dataset(DATA_DIR)

train_mask = full_data["year"] <= 2020
test_mask = full_data["year"] > 2020

X_train = full_data.loc[train_mask, X_cols]
y_train = full_data.loc[train_mask, "target_default"].astype(np.int8).to_numpy()
X_test = full_data.loc[test_mask, X_cols]
y_test = full_data.loc[test_mask, "target_default"].astype(np.int8).to_numpy()
y_test_dissolution = full_data.loc[test_mask, "target_dissolution"].astype(np.int8).to_numpy()
zscore_baselines_test = {
    "altman_z": -full_data.loc[test_mask, "altman_z"].values,
    "saifullin_r": -full_data.loc[test_mask, "saifullin_r"].values,
    "igea_r": -full_data.loc[test_mask, "igea_r"].values,
}
del full_data
gc.collect()

print(f"   train: {len(X_train):,} строк, positives {int(y_train.sum()):,} ({y_train.mean() * 100:.2f}%)")
print(f"   test:  {len(X_test):,} строк, positives {int(y_test.sum()):,} ({y_test.mean() * 100:.2f}%)")

train_pool = Pool(X_train, y_train, cat_features=cat_features)
del X_train, y_train
gc.collect()
test_pool = Pool(X_test, y_test, cat_features=cat_features)
del X_test
gc.collect()

model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=7,
    l2_leaf_reg=3.0,
    border_count=64,
    boosting_type="Plain",
    loss_function="Logloss",
    eval_metric="AUC",
    cat_features=cat_features,
    random_seed=42,
    thread_count=4,
    used_ram_limit="6gb",
    verbose=200,
)

model.fit(train_pool)
del train_pool
gc.collect()

proba = model.predict_proba(test_pool)[:, 1]
del test_pool
gc.collect()

auc = roc_auc_score(y_test, proba)
pr_auc = average_precision_score(y_test, proba)
model.save_model(MODEL_PATH)
np.save("cb_default_proba.npy", proba)
np.save("y_test_default.npy", y_test)
np.save("y_test_dissolution.npy", y_test_dissolution)


def banking_metrics(y_true, scores, name=""):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    finite = np.isfinite(scores)
    y_true = y_true[finite]
    scores = scores[finite]
    auc_v = roc_auc_score(y_true, scores)
    gini = 2 * auc_v - 1
    ks_stat, _ = ks_2samp(scores[y_true == 1], scores[y_true == 0])
    return {"name": name, "AUC": auc_v, "Gini": gini, "KS": ks_stat}


print(f"""
Сравнение моделей на тесте (target = балансовый дефолт)
  test rows: {len(y_test):,}, positives: {int(y_test.sum()):,} ({y_test.mean() * 100:.2f}%)
""")

results = [
    banking_metrics(y_test, zscore_baselines_test["altman_z"], "Altman Z'"),
    banking_metrics(y_test, zscore_baselines_test["saifullin_r"], "Сайфуллин-Кадыков"),
    banking_metrics(y_test, zscore_baselines_test["igea_r"], "Давыдова-Беликов"),
    banking_metrics(y_test, proba, "CatBoost"),
]

print(f"  {'Модель':<26} {'AUC':>8} {'Gini':>8} {'KS':>8}")
for r in results:
    print(f"  {r['name']:<26} {r['AUC']:>8.4f} {r['Gini']:>8.4f} {r['KS']:>8.4f}")

print(f"\nROC-AUC: {auc:.4f}")
print(f"PR-AUC:  {pr_auc:.4f} (base rate {y_test.mean() * 100:.2f}%)")

precs, recs, thrs = precision_recall_curve(y_test, proba)
f1s = 2 * precs * recs / (precs + recs + EPS)
thr_f1 = float(thrs[int(np.argmax(f1s[:-1]))])

idx_p70 = np.where(precs[:-1] >= 0.70)[0]
thr_p70 = float(thrs[idx_p70[np.argmax(recs[idx_p70])]])

idx_p85 = np.where(precs[:-1] >= 0.85)[0]
thr_p85 = float(thrs[idx_p85[np.argmax(recs[idx_p85])]])

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
    print(f"\n{name} (порог {thr:.4f})")
    print(f"  precision = {p:.4f}   recall = {r:.4f}   F1 = {f1:.4f}")
    print(f"  TN={cm[0, 0]:>8,}  FP={cm[0, 1]:>8,}")
    print(f"  FN={cm[1, 0]:>8,}  TP={cm[1, 1]:>8,}")

print("\nLift в топ-K% риска")
base_rate = float(y_test.mean())
order = np.argsort(proba)[::-1]
for top_k in (0.005, 0.01, 0.02, 0.05, 0.10):
    top_n = max(int(len(proba) * top_k), 1)
    top_rate = float(y_test[order[:top_n]].mean())
    print(f"  top-{top_k * 100:>4.1f}%: precision={top_rate:.4f}, lift={top_rate / base_rate:.2f}x")

feat_imp = pd.DataFrame({
    "feature": X_cols,
    "importance": model.get_feature_importance(),
}).sort_values("importance", ascending=False).reset_index(drop=True)
print("\nТоп-20 признаков по важности")
print(feat_imp.head(20).to_string(index=False))
