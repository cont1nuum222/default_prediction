import pandas as pd
import numpy as np
import gc
import warnings
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import ks_2samp

from pipeline import build_dataset, X_cols, cat_features

warnings.filterwarnings("ignore")

DATA_DIR = r"D:\Desktop\project\data_parquet"
REPORT_PATH = "sensitivity_report.txt"
RANDOM_SEED = 42

BASELINE = dict(
    iterations=1000,
    learning_rate=0.05,
    depth=7,
    l2_leaf_reg=3.0,
    border_count=64,
    boosting_type="Plain",
    loss_function="Logloss",
    random_seed=RANDOM_SEED,
)

SWEEP_LR = [0.02, 0.03, 0.05, 0.08, 0.10]
SWEEP_DEPTH = [4, 6, 7, 8, 10]
SWEEP_L2 = [1.0, 3.0, 5.0, 10.0]
SWEEP_BORDER = [32, 64, 128, 254]
SWEEP_ITERS = [300, 600, 1000, 1500]

RU_NAMES = {
    "learning_rate": "Скорость обучения",
    "depth": "Глубина деревьев",
    "l2_leaf_reg": "L2-регуляризация листьев",
    "border_count": "Число порогов квантования",
    "iterations": "Число итераций",
}

full_data = build_dataset(DATA_DIR, te_train_cutoff_year=2016)

inner_mask = (full_data["year"] <= 2016).values
val_mask = ((full_data["year"] >= 2017) & (full_data["year"] <= 2020)).values
test_mask = (full_data["year"] > 2020).values

X_inner = full_data.loc[inner_mask, X_cols].copy()
y_inner = full_data.loc[inner_mask, "target_default"].astype(np.int8).to_numpy()
X_val = full_data.loc[val_mask, X_cols].copy()
y_val = full_data.loc[val_mask, "target_default"].astype(np.int8).to_numpy()
X_test = full_data.loc[test_mask, X_cols].copy()
y_test = full_data.loc[test_mask, "target_default"].astype(np.int8).to_numpy()
del full_data
gc.collect()

inner_pool = Pool(X_inner, y_inner, cat_features=cat_features)
val_pool = Pool(X_val, cat_features=cat_features)

out_file = open(REPORT_PATH, "w", encoding="utf-8")


def log(line=""):
    print(line)
    out_file.write(line + "\n")


def metrics_on_val(proba):
    auc = roc_auc_score(y_val, proba)
    ks = ks_2samp(proba[y_val == 1], proba[y_val == 0]).statistic
    pr = average_precision_score(y_val, proba)
    return auc, ks, pr


all_val_aucs = []


def fit_eval(params):
    model = CatBoostClassifier(verbose=False, **params)
    model.fit(inner_pool)
    proba = model.predict_proba(val_pool)[:, 1]
    auc, ks, pr = metrics_on_val(proba)
    all_val_aucs.append(auc)
    return auc, ks, pr


spread_by_param = {}


def run_ofat(param_name, values, baseline):
    log()
    log(f"{RU_NAMES[param_name]}:")
    log(f"  {'Значение':>12}  {'ROC-AUC':>8}  {'KS':>8}  {'PR-AUC':>8}")
    aucs = []
    for v in values:
        params = dict(baseline)
        params[param_name] = v
        auc, ks, pr = fit_eval(params)
        aucs.append(auc)
        mark = "  <- baseline" if v == baseline.get(param_name) else ""
        log(f"  {str(v):>12}  {auc:>8.4f}  {ks:>8.4f}  {pr:>8.4f}{mark}")
    spread = max(aucs) - min(aucs)
    spread_by_param[param_name] = spread
    log(f"  разброс ROC-AUC: {spread:.4f} ({spread * 100:.2f} п.п.)")


baseline_str = ", ".join(f"{k}={v}" for k, v in BASELINE.items()
                         if k not in ("loss_function", "random_seed"))
log(f"""Чувствительность CatBoost к гиперпараметрам (OFAT на внутренней валидации)

inner-train (≤2016): {len(y_inner):,} строк, base rate {y_inner.mean() * 100:.2f}%
validation (2017-2020): {len(y_val):,} строк, base rate {y_val.mean() * 100:.2f}%
test (2021-2022, не трогаем при подборе): {len(y_test):,} строк

Baseline: {baseline_str}
Метрики ниже на валидации 2017-2020. Тест используется один раз в конце.""")

run_ofat("learning_rate", SWEEP_LR, BASELINE)
run_ofat("depth", SWEEP_DEPTH, BASELINE)
run_ofat("l2_leaf_reg", SWEEP_L2, BASELINE)
run_ofat("border_count", SWEEP_BORDER, BASELINE)

log()
log(f"{RU_NAMES['iterations']}:")
log(f"  {'Значение':>12}  {'ROC-AUC':>8}  {'KS':>8}  {'PR-AUC':>8}")
big = CatBoostClassifier(verbose=False, **{**BASELINE, "iterations": max(SWEEP_ITERS)})
big.fit(inner_pool)
iter_aucs = []
for n in SWEEP_ITERS:
    proba = big.predict_proba(val_pool, ntree_end=n)[:, 1]
    auc, ks, pr = metrics_on_val(proba)
    iter_aucs.append(auc)
    all_val_aucs.append(auc)
    mark = "  <- baseline" if n == BASELINE["iterations"] else ""
    log(f"  {n:>12}  {auc:>8.4f}  {ks:>8.4f}  {pr:>8.4f}{mark}")
spread_iter = max(iter_aucs) - min(iter_aucs)
spread_by_param["iterations"] = spread_iter
log(f"  разброс ROC-AUC: {spread_iter:.4f} ({spread_iter * 100:.2f} п.п.)")
del big
gc.collect()

log()
log("Сводка чувствительности по параметрам:")
log()
log(f"  {'параметр':<26}  {'разброс ROC-AUC':>15}  {'в п.п.':>8}")
for k, v in sorted(spread_by_param.items(), key=lambda x: x[1], reverse=True):
    log(f"  {RU_NAMES[k]:<26}  {v:>15.4f}  {v * 100:>7.2f}")
amin, amax = min(all_val_aucs), max(all_val_aucs)
amean, astd = float(np.mean(all_val_aucs)), float(np.std(all_val_aucs))
log()
log("По всем испытанным конфигам:")
log(f"  ROC-AUC: min={amin:.4f}  max={amax:.4f}  размах={amax - amin:.4f} "
    f"({(amax - amin) * 100:.2f} п.п.)")
log(f"  ROC-AUC: mean={amean:.4f}  std={astd:.4f}")

log()
log("Перенос baseline на полный train (≤2020):")
X_full = pd.concat([X_inner, X_val], ignore_index=True)
y_full = np.concatenate([y_inner, y_val])
full_pool = Pool(X_full, y_full, cat_features=cat_features)
test_pool = Pool(X_test, cat_features=cat_features)
final = CatBoostClassifier(verbose=False, **BASELINE)
final.fit(full_pool)
proba_test = final.predict_proba(test_pool)[:, 1]
auc_t = roc_auc_score(y_test, proba_test)
ks_t = ks_2samp(proba_test[y_test == 1], proba_test[y_test == 0]).statistic
pr_t = average_precision_score(y_test, proba_test)
log(f"  train (≤2020): {len(y_full):,}, test (2021-2022): {len(y_test):,} "
    f"(base rate {y_test.mean() * 100:.2f}%)")
log(f"  ROC-AUC={auc_t:.4f}  KS={ks_t:.4f}  PR-AUC={pr_t:.4f}")

out_file.close()
