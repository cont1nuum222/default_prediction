import pandas as pd
import numpy as np
import gc
import warnings
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.preprocessing import StandardScaler
from scipy.stats import ks_2samp

from pipeline import build_dataset, X_cols, cat_features

warnings.filterwarnings("ignore")

DATA_DIR = r"D:\Desktop\project\data_parquet"
PROD_MODEL_PATH = "model_prod.cbm"
REPORT_PATH = "benchmark_report.txt"
RANDOM_SEED = 42

TREE_TRAIN_CAP = 500_000
GBM_ITERS = 1000
GBM_LR = 0.05
GBM_DEPTH = 7

full_data = build_dataset(DATA_DIR)

num_cols = [c for c in X_cols if c not in cat_features]

train_mask = (full_data["year"] <= 2020).values
test_mask = (full_data["year"] > 2020).values

y_train = full_data.loc[train_mask, "target_default"].astype(np.int8).to_numpy()
y_test = full_data.loc[test_mask, "target_default"].astype(np.int8).to_numpy()

X_train_cb = full_data.loc[train_mask, X_cols].copy()
X_test_cb = full_data.loc[test_mask, X_cols].copy()

X_train_num = full_data.loc[train_mask, num_cols].to_numpy(dtype=np.float32)
X_test_num = full_data.loc[test_mask, num_cols].to_numpy(dtype=np.float32)

reg_cat = pd.Categorical(full_data.loc[train_mask, "region"])
okv_cat = pd.Categorical(full_data.loc[train_mask, "okved_short"])
region_codes_tr = reg_cat.codes.astype(np.int32) + 1
okved_codes_tr = okv_cat.codes.astype(np.int32) + 1
region_codes_te = pd.Categorical(full_data.loc[test_mask, "region"],
                                 categories=reg_cat.categories).codes.astype(np.int32) + 1
okved_codes_te = pd.Categorical(full_data.loc[test_mask, "okved_short"],
                                categories=okv_cat.categories).codes.astype(np.int32) + 1

del full_data
gc.collect()

base_rate_test = float(y_test.mean())
n_train, n_test = len(y_train), len(y_test)

train_medians = np.nanmedian(X_train_num, axis=0)
train_medians = np.where(np.isnan(train_medians), train_medians).astype(np.float32)


def impute(X):
    out = X.copy()
    idx = np.where(np.isnan(out))
    out[idx] = np.take(train_medians, idx[1])
    return out


out_file = open(REPORT_PATH, "w", encoding="utf-8")


def log(line=""):
    print(line)
    out_file.write(line + "\n")


def evaluate(name, proba, note=""):
    auc = roc_auc_score(y_test, proba)
    ks = ks_2samp(proba[y_test == 1], proba[y_test == 0]).statistic
    pr_auc = average_precision_score(y_test, proba)
    order = np.argsort(proba)[::-1]
    n1 = max(int(len(proba) * 0.01), 1)
    lift1 = float(y_test[order[:n1]].mean()) / base_rate_test
    log(f"  {name:<30} ROC-AUC={auc:.4f}  KS={ks:.4f}  PR-AUC={pr_auc:.4f}  "
        f"Lift@1%={lift1:5.2f}{('  ' + note) if note else ''}")
    return {"name": name, "AUC": auc, "KS": ks, "PR_AUC": pr_auc,
            "lift1": lift1, "note": note}


results = []

log(f"""Сравнение алгоритмов: балансовый дефолт через 2 года (out-of-time)

Train (≤2020): {n_train:,} строк, Test (2021-2022): {n_test:,} строк
Base rate (test): {base_rate_test * 100:.2f}%
Признаков: {len(num_cols)}, для CatBoost {len(X_cols)} (+2 нативные категории)
Бюджет бустингов: {GBM_ITERS} деревьев, lr={GBM_LR}, depth≈{GBM_DEPTH}
RAM-cap для RandomForest/ExtraTrees: train ≤ {TREE_TRAIN_CAP:,}

Результаты:
""")

rng = np.random.default_rng(RANDOM_SEED)
if n_train > TREE_TRAIN_CAP:
    sub_idx = rng.choice(n_train, size=TREE_TRAIN_CAP, replace=False)
else:
    sub_idx = np.arange(n_train)

# 1. LogisticRegression
Xtr = impute(X_train_num)
scaler = StandardScaler().fit(Xtr)
Xtr = scaler.transform(Xtr).astype(np.float32)
clf = LogisticRegression(max_iter=1000, C=1.0, n_jobs=-1, solver="lbfgs")
clf.fit(Xtr, y_train)
del Xtr
gc.collect()
Xte = scaler.transform(impute(X_test_num)).astype(np.float32)
proba = clf.predict_proba(Xte)[:, 1]
del Xte
gc.collect()
results.append(evaluate("Логистическая регрессия", proba,
                        note="scaled+imputed, числовой набор"))

# 2. DecisionTree
Xtr = impute(X_train_num)
clf = DecisionTreeClassifier(max_depth=10, min_samples_leaf=200, random_state=RANDOM_SEED)
clf.fit(Xtr, y_train)
del Xtr
gc.collect()
proba = clf.predict_proba(impute(X_test_num))[:, 1]
results.append(evaluate("Дерево решений", proba,
                        note="одиночное дерево"))

# 3. RandomForest
Xtr = impute(X_train_num[sub_idx])
clf = RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_leaf=50,
                             n_jobs=-1, random_state=RANDOM_SEED)
clf.fit(Xtr, y_train[sub_idx])
del Xtr
gc.collect()
proba = clf.predict_proba(impute(X_test_num))[:, 1]
results.append(evaluate("Случайный лес", proba,
                        note=f"train подвыборка {len(sub_idx):,}"))

# 4. ExtraTrees
Xtr = impute(X_train_num[sub_idx])
clf = ExtraTreesClassifier(n_estimators=300, max_depth=None, min_samples_leaf=50,
                           n_jobs=-1, random_state=RANDOM_SEED)
clf.fit(Xtr, y_train[sub_idx])
del Xtr
gc.collect()
proba = clf.predict_proba(impute(X_test_num))[:, 1]
results.append(evaluate("Extra Trees", proba,
                        note=f"train подвыборка {len(sub_idx):,}"))

# 5. HistGradientBoosting (нативные NaN)
clf = HistGradientBoostingClassifier(max_iter=GBM_ITERS, learning_rate=GBM_LR,
                                     max_depth=GBM_DEPTH, l2_regularization=1.0,
                                     random_state=RANDOM_SEED)
clf.fit(X_train_num, y_train)
proba = clf.predict_proba(X_test_num)[:, 1]
results.append(evaluate("HistGradientBoosting", proba,
                        note="sklearn, нативные NaN"))

# 6. LightGBM (числовой набор + вариант с нативными категориями)
clf = lgb.LGBMClassifier(n_estimators=GBM_ITERS, learning_rate=GBM_LR,
                         num_leaves=2 ** GBM_DEPTH, max_depth=GBM_DEPTH,
                         subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                         random_state=RANDOM_SEED, verbose=-1)
clf.fit(X_train_num, y_train)
proba = clf.predict_proba(X_test_num)[:, 1]
results.append(evaluate("LightGBM", proba,
                        note="нативные NaN, числовой набор"))

Xtr_lgb = np.column_stack([X_train_num, region_codes_tr, okved_codes_tr]).astype(np.float32)
Xte_lgb = np.column_stack([X_test_num, region_codes_te, okved_codes_te]).astype(np.float32)
cat_idx = [Xtr_lgb.shape[1] - 2, Xtr_lgb.shape[1] - 1]
clf = lgb.LGBMClassifier(n_estimators=GBM_ITERS, learning_rate=GBM_LR,
                         num_leaves=2 ** GBM_DEPTH, max_depth=GBM_DEPTH,
                         subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                         random_state=RANDOM_SEED, verbose=-1)
clf.fit(Xtr_lgb, y_train, categorical_feature=cat_idx)
proba = clf.predict_proba(Xte_lgb)[:, 1]
del Xtr_lgb, Xte_lgb
gc.collect()
results.append(evaluate("LightGBM (нативные категории)", proba,
                        note="region/okved как категории"))

# 7. CatBoost (тот же бюджет, нативные категории)
train_pool = Pool(X_train_cb, y_train, cat_features=cat_features)
test_pool = Pool(X_test_cb, cat_features=cat_features)
clf = CatBoostClassifier(iterations=GBM_ITERS, learning_rate=GBM_LR, depth=GBM_DEPTH,
                         l2_leaf_reg=3.0, border_count=64, boosting_type="Plain",
                         loss_function="Logloss", random_seed=RANDOM_SEED, verbose=False)
clf.fit(train_pool)
proba = clf.predict_proba(test_pool)[:, 1]
del train_pool, test_pool
gc.collect()
results.append(evaluate("CatBoost", proba,
                        note="нативные категории, ordered TS"))

# 8. CatBoost production (артефакт обучен отдельным запуском)
prod = CatBoostClassifier()
prod.load_model(PROD_MODEL_PATH)
test_pool = Pool(X_test_cb, cat_features=cat_features)
proba = prod.predict_proba(test_pool)[:, 1]
del test_pool
gc.collect()
results.append(evaluate("CatBoost (production)", proba,
                        note=f"загружен {PROD_MODEL_PATH}, только predict"))

log()
log("Итог по ROC-AUC (test 2021-2022):")
log()
log(f"  {'Алгоритм':<30} {'ROC-AUC':>8} {'KS':>8} {'PR-AUC':>9} {'Lift@1%':>9}")
for r in sorted(results, key=lambda x: x["AUC"], reverse=True):
    log(f"  {r['name']:<30} {r['AUC']:>8.4f} {r['KS']:>8.4f} {r['PR_AUC']:>9.4f} "
        f"{r['lift1']:>9.2f}")

out_file.close()
