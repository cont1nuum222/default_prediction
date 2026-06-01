import matplotlib.pyplot as plt
import numpy as np
from catboost import CatBoostClassifier

LABELS = {
    "region_okved_te": "Регион×ОКВЭД",
    "year_okved_te": "Год×ОКВЭД",
    "industry_density": "Плотность дефолтов",
    "debt_ratio": "Долг/активы",
    "debt_ratio_rel": "Долг/активы отн.",
    "equity_ratio": "Капитал/активы",
    "equity_ratio_rel": "Капитал/активы отн.",
    "equity_ratio_prev": "Капитал/активы −1",
    "roa": "ROA",
    "roa_prev": "ROA −1",
    "roa_2y_avg": "ROA ср. 2 года",
    "roa_rel": "ROA отн.",
    "roa_x_drain": "ROA × отток",
    "log_assets": "log(активы)",
    "log_revenue": "log(выручка)",
    "log_equity": "log(капитал)",
    "log_profit": "log(прибыль)",
    "log_st_debt": "log(кор. долг)",
    "log_drain": "log(отток)",
    "current_ratio": "Тек. ликвидность",
    "working_capital": "Раб. капитал",
    "working_capital_rel": "Раб. капитал отн.",
    "asset_turnover": "Оборач. активов",
    "asset_turnover_rel": "Оборач. отн.",
    "op_margin": "Опер. маржа",
    "asset_drain_1y": "Отток активов 1г",
    "asset_drain_2y": "Отток активов 2г",
    "rev_growth_1y": "Рост выручки 1г",
    "rev_growth_2y": "Рост выручки 2г",
    "drain_accel": "Ускорение оттока",
    "profit_growth": "Рост прибыли",
    "equity_drain": "Отток капитала",
    "debt_growth": "Рост долга",
    "delta_roa": "Δ ROA",
    "delta_equity_ratio": "Δ капитал/активы",
    "delta_log_assets": "Δ log(активы)",
    "leverage_size": "Долг × размер",
    "altman_z": "Altman Z",
    "altman_distress": "Altman: distress",
    "altman_grey": "Altman: серая",
    "saifullin_r": "Сайфуллин R",
    "saifullin_distress": "Сайфуллин: distr",
    "igea_r": "Иркутская R",
    "igea_distress": "Иркутская: distr",
    "industry_avg_lifespan": "Срок жизни отрасли",
    "relative_age": "Отн. возраст",
    "lifespan_anomaly": "Аномалия возраста",
    "years_since_first_report": "Лет с 1-го отчёта",
    "history_size_so_far": "Размер истории",
    "is_new_company": "Новая компания",
    "is_zombie": "is_zombie",
    "neg_equity": "Отриц. капитал",
    "neg_profit": "Отриц. прибыль",
    "zero_revenue": "Нулевая выручка",
    "no_okved": "Без ОКВЭД",
    "no_history": "Без истории",
    "no_history_2y": "Без истории 2г",
    "risk_score": "Risk score",
    "region": "Регион",
    "okved_short": "ОКВЭД",
    "year_num": "Год",
}

model = CatBoostClassifier()
model.load_model("model.cbm")

importance = np.array(model.get_feature_importance())
names = list(model.feature_names_)
idx = np.argsort(importance)[::-1][:10]

top_names = [LABELS.get(names[i], names[i]) for i in idx][::-1]
top_vals = importance[idx][::-1]

fig, ax = plt.subplots(figsize=(7, 5))
y = np.arange(len(top_names))
bars = ax.barh(y, top_vals, color="#1F3A5F", edgecolor="none")
ax.set_yticks(y)
ax.set_yticklabels(top_names, fontsize=11)
ax.set_xlabel("Важность", fontsize=10, color="#444")

vmax = float(top_vals.max())
for bar, v in zip(bars, top_vals):
    ax.text(v + vmax * 0.01, bar.get_y() + bar.get_height() / 2,
            f"{v:.1f}", va="center", fontsize=9, color="#555")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#888")
ax.spines["bottom"].set_color("#888")
ax.tick_params(colors="#444")
ax.set_xlim(0, vmax * 1.18)

plt.tight_layout()
plt.savefig("feature_importance_full.png", dpi=200, bbox_inches="tight")
