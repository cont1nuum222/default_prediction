import os
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score, average_precision_score

from pipeline import build_dataset, X_cols, cat_features, MODEL_PATH

DATA_DIR = r"D:\Desktop\project\data_parquet"
SCORES_PATH = "test_scores.npz"
REPORT_PATH = "bootstrap_report.txt"
B = 1000
SEED = 42


def compute_scores():
    full_data = build_dataset(DATA_DIR)
    test_df = full_data.loc[full_data["year"] > 2020].copy()
    del full_data

    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)
    proba = model.predict_proba(Pool(test_df[X_cols], cat_features=cat_features))[:, 1]

    np.savez(SCORES_PATH,
             y=test_df["target_default"].astype(np.int8).to_numpy(),
             proba=proba,
             debt_ratio=test_df["debt_ratio"].to_numpy(),
             altman=-test_df["altman_z"].to_numpy(),
             altman2=-test_df["altman_z2"].to_numpy(),
             saif=-test_df["saifullin_r"].to_numpy(),
             igea=-test_df["igea_r"].to_numpy())


if not os.path.exists(SCORES_PATH):
    compute_scores()

data = np.load(SCORES_PATH)
y_full = data["y"].astype(np.int8)
model_score = data["proba"].astype(float)
debt_ratio = data["debt_ratio"].astype(float)
classics_full = {
    "Altman Z'": data["altman"].astype(float),
    "Altman Z''": data["altman2"].astype(float),
    "Сайфуллин-Кадыков": data["saif"].astype(float),
    "Давыдова-Беликов": data["igea"].astype(float),
}


def bootstrap_ci(y, score, rng):
    mask = np.isfinite(score)
    y, s = y[mask], score[mask]
    n = len(y)
    auc_boot = np.empty(B)
    pr_boot = np.empty(B)
    for i in range(B):
        idx = rng.integers(0, n, n)
        y_b = y[idx]
        auc_boot[i] = roc_auc_score(y_b, s[idx])
        pr_boot[i] = average_precision_score(y_b, s[idx])
    return (n,
            roc_auc_score(y, s), np.percentile(auc_boot, [2.5, 97.5]),
            average_precision_score(y, s), np.percentile(pr_boot, [2.5, 97.5]))


def paired_diff(y, score_model, score_classic, rng):
    mask = np.isfinite(score_model) & np.isfinite(score_classic)
    y, a, c = y[mask], score_model[mask], score_classic[mask]
    n = len(y)
    delta_auc = np.empty(B)
    delta_pr = np.empty(B)
    for i in range(B):
        idx = rng.integers(0, n, n)
        y_b = y[idx]
        delta_auc[i] = roc_auc_score(y_b, a[idx]) - roc_auc_score(y_b, c[idx])
        delta_pr[i] = average_precision_score(y_b, a[idx]) - average_precision_score(y_b, c[idx])
    return (n,
            roc_auc_score(y, a) - roc_auc_score(y, c),
            np.percentile(delta_auc, [2.5, 97.5]),
            float((delta_auc <= 0).mean()),
            average_precision_score(y, a) - average_precision_score(y, c),
            np.percentile(delta_pr, [2.5, 97.5]),
            float((delta_pr <= 0).mean()))


def format_p(p):
    return f"<{1.0 / B:.3f}" if p == 0 else f"={p:.4f}"


def report(y, model_score, baselines, classics, title):
    rng = np.random.default_rng(SEED)
    scores = {"Модель (CatBoost)": model_score, **baselines, **classics}
    n_total = len(y)
    base_pct = y.mean() * 100
    header = (f"{title}: n={n_total:,}, base rate {base_pct:.2f}%"
              if "(" in title
              else f"{title} (n={n_total:,}, base rate {base_pct:.2f}%)")
    lines = [
        header,
        "",
        f"{'Модель':<30} {'n':>8}  {'ROC-AUC':<25}PR-AUC",
    ]
    aucs = {}
    for name, score in scores.items():
        n, auc, auc_ci, pr, pr_ci = bootstrap_ci(y, score, rng)
        aucs[name] = auc
        lines.append(f"{name:<30} {n:>8,}  {auc:.4f} [{auc_ci[0]:.4f}, {auc_ci[1]:.4f}]  "
                     f"{pr:.4f} [{pr_ci[0]:.4f}, {pr_ci[1]:.4f}]")
    best = max(classics, key=lambda k: aucs[k])
    n, d_auc, d_auc_ci, p_auc, d_pr, d_pr_ci, p_pr = paired_diff(y, model_score, classics[best], rng)
    lines += [
        "",
        f"Разница с {best} (n={n:,}):",
        f"  ROC-AUC: {d_auc:+.4f} [{d_auc_ci[0]:+.4f}, {d_auc_ci[1]:+.4f}]  p{format_p(p_auc)}",
        f"  PR-AUC: {d_pr:+.4f} [{d_pr_ci[0]:+.4f}, {d_pr_ci[1]:+.4f}]  p{format_p(p_pr)}",
    ]
    return lines


baselines = {"Наивный предиктор (debt_ratio)": debt_ratio}
lines = report(y_full, model_score, baselines, classics_full, "Полный тест 2021-2022")

healthy = debt_ratio < 0.9
lines.append("")
lines += report(y_full[healthy], model_score[healthy],
                {k: s[healthy] for k, s in baselines.items()},
                {k: s[healthy] for k, s in classics_full.items()},
                "Здоровое ядро (debt_ratio < 0.9)")

text = "\n".join(lines)
print(text)
with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write(text + "\n")
