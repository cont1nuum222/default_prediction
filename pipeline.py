import pandas as pd
import numpy as np
import glob
import os
import gc
from tqdm import tqdm

EPS = 1e-6
MODEL_PATH = "model.cbm"
MIN_ASSETS = 1_000

HISTORY_LINES = [
    "line_1100", "line_1200", "line_1300", "line_1400", "line_1500",
    "line_1600", "line_2110", "line_2120", "line_2200", "line_2400",
]

MACRO_BY_YEAR = {
    2011: (8.00, 4.3, 29.4, 109), 2012: (8.00, 4.0, 31.1, 110),
    2013: (5.50, 1.8, 31.8, 108), 2014: (8.50, 0.7, 38.4, 98),
    2015: (12.5, -2.0, 60.9, 51), 2016: (10.5, 0.2, 67.0, 41),
    2017: (8.75, 1.8, 58.3, 53), 2018: (7.50, 2.8, 62.7, 70),
    2019: (7.00, 2.2, 64.7, 64), 2020: (5.25, -2.7, 72.1, 41),
    2021: (6.50, 5.6, 73.6, 69), 2022: (12.0, -1.2, 68.5, 75),
    2023: (10.5, 3.6, 85.0, 63), 2024: (16.5, 4.0, 92.5, 70),
}
CRISIS_YEARS = {2014, 2020, 2022}
POST_CRISIS_YEARS = {2015, 2021, 2023}

X_cols = [
    "log_assets", "log_revenue", "log_equity", "log_profit", "log_st_debt", "roa",
    "debt_ratio", "equity_ratio", "working_capital", "current_ratio", "asset_turnover",
    "op_margin", "asset_drain_1y", "asset_drain_2y", "rev_growth_1y", "rev_growth_2y",
    "drain_accel", "log_drain", "profit_growth", "equity_drain", "debt_growth",
    "leverage_size", "roa_x_drain", "roa_rel", "equity_ratio_rel", "debt_ratio_rel",
    "working_capital_rel", "asset_turnover_rel", "region_okved_te", "year_okved_te",
    "history_size_so_far", "years_since_first_report", "is_new_company",
    "industry_avg_lifespan", "relative_age", "lifespan_anomaly", "roa_prev",
    "roa_2y_avg", "equity_ratio_prev", "delta_roa", "delta_equity_ratio",
    "delta_log_assets", "altman_z", "altman_distress", "altman_grey", "saifullin_r",
    "saifullin_distress", "igea_r", "igea_distress", "industry_density", "is_zombie",
    "neg_equity", "neg_profit", "zero_revenue", "no_okved", "no_history",
    "no_history_2y", "risk_score", "region", "okved_short", "year_num",
]
cat_features = ["region", "okved_short"]

RATIO_COLS = [
    "roa", "debt_ratio", "equity_ratio", "working_capital", "current_ratio",
    "asset_turnover", "op_margin", "roe", "asset_drain_1y", "asset_drain_2y",
    "rev_growth_1y", "rev_growth_2y", "drain_accel", "log_drain", "profit_growth",
    "profit_chg", "equity_drain", "debt_growth", "leverage_size", "roa_x_drain",
    "delta_roa", "delta_equity_ratio", "delta_debt_ratio", "delta_log_assets",
]


def build_history_df(files):
    chunks = []
    cols = ["inn", "year"] + HISTORY_LINES
    for f in tqdm(files, desc="Indexing"):
        try:
            df = pd.read_parquet(f, columns=[c for c in cols if c])
            if "line_1600" not in df.columns:
                continue
            df = df[df["line_1600"] >= MIN_ASSETS]
            if df.empty:
                continue
            df["inn"] = pd.to_numeric(df["inn"], errors="coerce").astype("Int64")
            df = df.dropna(subset=["inn"])
            df["inn"] = df["inn"].astype("int64")
            df["year"] = df["year"].astype("int16")
            for c in HISTORY_LINES:
                df[c] = df[c].astype("float32") if c in df.columns else np.float32(np.nan)
            chunks.append(df[["inn", "year"] + HISTORY_LINES])
        except Exception:
            continue
    history = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()
    history = history.drop_duplicates(subset=["inn", "year"], keep="last")
    history = history.set_index(["inn", "year"]).sort_index()
    print(f"   history rows: {len(history):,}")
    return history


def lookup_lag(history, inn_arr, year_arr, lag):
    keys = pd.MultiIndex.from_arrays([inn_arr, year_arr - lag])
    sub = history.reindex(keys)
    return [sub[c].to_numpy(dtype=np.float32, na_value=np.nan) for c in HISTORY_LINES]


def lookup_default_status(history, inn_arr, year_arr, horizon=2):
    keys = pd.MultiIndex.from_arrays([inn_arr, year_arr + horizon])
    sub = history.reindex(keys)
    L14 = sub["line_1400"].to_numpy(dtype=np.float32, na_value=np.nan)
    L15 = sub["line_1500"].to_numpy(dtype=np.float32, na_value=np.nan)
    L16 = sub["line_1600"].to_numpy(dtype=np.float32, na_value=np.nan)
    liabilities = np.where(np.isnan(L14), 0.0, L14) + np.where(np.isnan(L15), 0.0, L15)
    has_data = ~np.isnan(L16)
    return np.where(has_data, (liabilities > L16).astype(np.float32), np.nan)


def build_company_age_stats(history):
    flat = history.reset_index()[["inn", "year"]].sort_values(["inn", "year"])
    first_year = flat.groupby("inn")["year"].min().astype("int16")
    flat["size_so_far"] = flat.groupby("inn").cumcount() + 1
    size_lookup = flat.set_index(["inn", "year"])["size_so_far"].astype("int16")
    return first_year, size_lookup


def build_company_meta(files, batch_size=200):
    okved_acc = None
    diss_acc = None

    def reduce(chunks):
        full = pd.concat(chunks, ignore_index=True)
        ok = (full.dropna(subset=["okved2"]).drop_duplicates("inn", keep="first")
              .set_index("inn")["okved2"])
        ds = (full.dropna(subset=["dissolution_year"]).groupby("inn")["dissolution_year"]
              .min().astype("int16"))
        del full
        gc.collect()
        return ok, ds

    chunks = []
    for f in tqdm(files, desc="Meta"):
        try:
            df = pd.read_parquet(f, columns=["inn", "okved", "dissolution_date"])
            df["inn"] = pd.to_numeric(df["inn"], errors="coerce").astype("Int64")
            df = df.dropna(subset=["inn"])
            df["inn"] = df["inn"].astype("int64")
            okved2 = df["okved"].astype(str).str.split(".").str[0]
            okved2 = okved2.where(okved2.notna() & (okved2 != "nan") & (okved2 != "") & (okved2 != "None"))
            df["okved2"] = okved2
            df["dissolution_year"] = pd.to_datetime(df["dissolution_date"], errors="coerce").dt.year.astype("Int16")
            chunks.append(df[["inn", "okved2", "dissolution_year"]].drop_duplicates("inn"))
            del df, okved2
        except Exception:
            continue
        if len(chunks) >= batch_size:
            ok, ds = reduce(chunks)
            chunks = []
            if okved_acc is None:
                okved_acc, diss_acc = ok, ds
            else:
                okved_acc = okved_acc.combine_first(ok)
                diss_acc = pd.concat([diss_acc, ds]).groupby(level=0).min().astype("int16")
            del ok, ds
            gc.collect()
    if chunks:
        ok, ds = reduce(chunks)
        if okved_acc is None:
            okved_acc, diss_acc = ok, ds
        else:
            okved_acc = okved_acc.combine_first(ok)
            diss_acc = pd.concat([diss_acc, ds]).groupby(level=0).min().astype("int16")
    okved_acc.index.name = "inn"
    diss_acc.index.name = "inn"
    okved_acc = okved_acc.rename("okved2")
    diss_acc = diss_acc.rename("dissolution_year")
    print(f"   meta: {len(okved_acc):,} okved, {len(diss_acc):,} dissolved")
    return okved_acc, diss_acc


def compute_industry_lifespan(first_year, okved_lookup, diss_lookup):
    df = pd.DataFrame({"first_year": first_year, "dissolution_year": diss_lookup,
                       "okved2": okved_lookup}).dropna()
    df["lifespan"] = (df["dissolution_year"] - df["first_year"]).astype("float32")
    df = df[df["lifespan"] > 0]
    by_okved = df.groupby("okved2")["lifespan"].mean().astype("float32")
    overall = float(df["lifespan"].mean()) if len(df) else 5.0
    print(f"   industries: {len(by_okved):,}, global avg = {overall:.2f}")
    return by_okved, overall


def signed_log(x):
    return np.sign(x) * np.log1p(np.abs(x))


def create_ultimate_dataset(files, history, age_stats, lifespan_stats):
    chunks = []
    first_year, size_lookup = age_stats
    okved_lifespan, global_lifespan = lifespan_stats
    n_ok = n_err = 0

    for f in tqdm(files, desc="Processing"):
        try:
            df = pd.read_parquet(f)
            if "line_1600" not in df.columns or df.empty:
                continue
            df = df[df["line_1600"] >= MIN_ASSETS].copy()
            if df.empty:
                continue

            df["dissolution_date"] = pd.to_datetime(df["dissolution_date"], errors="coerce")
            death_year = df["dissolution_date"].dt.year
            alive = death_year.isna() | (death_year > df["year"])
            df = df.loc[alive].copy()
            if df.empty:
                continue
            death_year = death_year.loc[df.index]
            df["target_dissolution"] = 0
            dies_in_2y = death_year.notna() & (death_year > df["year"]) & (death_year <= df["year"] + 2)
            df.loc[dies_in_2y, "target_dissolution"] = 1

            df["inn"] = pd.to_numeric(df["inn"], errors="coerce").astype("Int64")
            df = df.dropna(subset=["inn"])
            df["inn"] = df["inn"].astype("int64")
            inn_arr = df["inn"].to_numpy()
            yr_arr = df["year"].to_numpy().astype(np.int16)

            df["target_default"] = lookup_default_status(history, inn_arr, yr_arr, horizon=2)
            l1 = lookup_lag(history, inn_arr, yr_arr, lag=1)
            l2 = lookup_lag(history, inn_arr, yr_arr, lag=2)
            (L11_l1, L12_l1, L13_l1, L14_l1, L15_l1, L16_l1, L21_l1, _, L22_l1, L24_l1) = l1
            _, _, _, _, _, L16_l2, L21_l2, _, _, L24_l2 = l2

            def get(c, fill0=False):
                if c in df.columns:
                    s = df[c].astype("float32")
                    return s.fillna(0).to_numpy() if fill0 else s.to_numpy()
                return np.zeros(len(df), dtype=np.float32)

            L11 = get("line_1100", fill0=True); L12 = get("line_1200", fill0=True)
            L13 = get("line_1300"); L14 = get("line_1400", fill0=True)
            L15 = get("line_1500", fill0=True); L16 = df["line_1600"].astype("float32").to_numpy()
            L21 = df["line_2110"].astype("float32").to_numpy(); L2120 = get("line_2120", fill0=True)
            L22 = get("line_2200", fill0=True); L24 = get("line_2400", fill0=True)

            df["asset_drain_1y"] = (L16 / (L16_l1 + EPS) - 1).astype("float32")
            df["asset_drain_2y"] = (L16 / (L16_l2 + EPS) - 1).astype("float32")
            df["rev_growth_1y"] = (L21 / (np.abs(L21_l1) + EPS) - 1).astype("float32")
            df["rev_growth_2y"] = (L21 / (np.abs(L21_l2) + EPS) - 1).astype("float32")
            df["drain_accel"] = (df["asset_drain_1y"] - (L16_l1 / (L16_l2 + EPS) - 1)).astype("float32")
            df["log_drain"] = signed_log(df["asset_drain_1y"]).astype("float32")
            df["profit_growth"] = (L24 / (np.abs(L24_l1) + EPS) - 1).astype("float32")
            df["profit_chg"] = signed_log(L24 - L24_l1).astype("float32")
            df["equity_drain"] = (L13 / (np.abs(L13_l1) + EPS) - 1).astype("float32")
            df["debt_growth"] = (L15 / (L15_l1 + EPS) - 1).astype("float32")
            df["no_history"] = np.isnan(L16_l1).astype(np.int8)
            df["no_history_2y"] = np.isnan(L16_l2).astype(np.int8)

            df["log_assets"] = signed_log(L16).astype("float32")
            df["log_revenue"] = signed_log(L21).astype("float32")
            df["log_equity"] = signed_log(L13).astype("float32")
            df["log_profit"] = signed_log(L24).astype("float32")
            df["log_st_debt"] = signed_log(L15).astype("float32")

            df["roa"] = (L24 / (L16 + EPS)).astype("float32")
            df["debt_ratio"] = ((L14 + L15) / (L16 + EPS)).astype("float32")
            df["equity_ratio"] = (L13 / (L16 + EPS)).astype("float32")
            df["working_capital"] = ((L12 - L15) / (L16 + EPS)).astype("float32")
            df["current_ratio"] = (L12 / (L15 + EPS)).astype("float32")
            df["asset_turnover"] = (L21 / (L16 + EPS)).astype("float32")
            df["op_margin"] = (L22 / (np.abs(L21) + EPS)).astype("float32")
            df["roe"] = (L24 / (np.abs(L13) + EPS)).astype("float32")

            x1 = (L12 - L15) / (L16 + EPS)
            x2 = L13 / (L16 + EPS)
            x3 = L22 / (L16 + EPS)
            x4 = L13 / (L14 + L15 + EPS)
            x5 = L21 / (L16 + EPS)
            df["altman_z"] = (0.717 * x1 + 0.847 * x2 + 3.107 * x3
                              + 0.420 * x4 + 0.998 * x5).astype("float32")
            df["altman_distress"] = (df["altman_z"] < 1.23).astype(np.int8)
            df["altman_grey"] = ((df["altman_z"] >= 1.23) & (df["altman_z"] < 2.9)).astype(np.int8)

            kosos = (L13 - L11) / (L12 + EPS)
            ktl = L12 / (L15 + EPS)
            koa = L21 / (L16 + EPS)
            km = L24 / (np.abs(L21) + EPS)
            ksk = L24 / (np.abs(L13) + EPS)
            df["saifullin_r"] = (2.0 * kosos + 0.1 * ktl + 0.08 * koa
                                 + 0.45 * km + ksk).astype("float32")
            df["saifullin_distress"] = (df["saifullin_r"] < 1.0).astype(np.int8)

            k1 = (L12 - L15) / (L16 + EPS)
            k2 = L24 / (L13 + EPS)
            k3 = L21 / (L16 + EPS)
            k4 = L24 / (np.abs(L2120) + EPS)
            df["igea_r"] = (8.38 * k1 + k2 + 0.054 * k3 + 0.63 * k4).astype("float32")
            df["igea_distress"] = (df["igea_r"] < 0.32).astype(np.int8)

            df["leverage_size"] = (df["log_assets"] * df["debt_ratio"]).astype("float32")
            df["roa_x_drain"] = (df["roa"] * df["asset_drain_1y"]).astype("float32")
            df["is_zombie"] = ((L13 < 0) & (L24 < 0)).astype(np.int8)
            df["neg_equity"] = (L13 < 0).astype(np.int8)
            df["neg_profit"] = (L24 < 0).astype(np.int8)
            df["zero_revenue"] = (L21 <= 0).astype(np.int8)
            df["no_okved"] = (df["okved"].isna() | (df["okved"] == "")).astype(np.int8)
            df["debt_over_eq"] = (df["debt_ratio"] > 0.95).astype(np.int8)
            df["risk_score"] = (df["is_zombie"] + df["neg_equity"] + df["neg_profit"]
                                + df["zero_revenue"] + df["no_okved"] + df["debt_over_eq"]).astype(np.int8)

            keys = pd.MultiIndex.from_arrays([inn_arr, yr_arr])
            df["history_size_so_far"] = size_lookup.reindex(keys).fillna(1).astype("int16").values
            df["years_since_first_report"] = (df["year"] - df["inn"].map(first_year)).fillna(0).astype("int16")
            df["is_new_company"] = (df["years_since_first_report"] == 0).astype(np.int8)
            okved2_now = df["okved"].astype(str).str.split(".").str[0]
            ind_life = okved2_now.map(okved_lifespan).fillna(global_lifespan).astype("float32").values
            df["industry_avg_lifespan"] = ind_life
            df["relative_age"] = (df["years_since_first_report"].astype("float32").values / (ind_life + EPS)).astype("float32")
            df["lifespan_anomaly"] = (df["years_since_first_report"].astype("float32").values - ind_life).astype("float32")

            roa_prev = (L24_l1 / (L16_l1 + EPS)).astype("float32")
            equity_ratio_prev = (L13_l1 / (L16_l1 + EPS)).astype("float32")
            debt_ratio_prev = (L15_l1 / (L16_l1 + EPS)).astype("float32")
            df["roa_prev"] = roa_prev
            df["roa_2y_avg"] = ((df["roa"].values + roa_prev) / 2).astype("float32")
            df["equity_ratio_prev"] = equity_ratio_prev
            df["delta_roa"] = (df["roa"].values - roa_prev).astype("float32")
            df["delta_equity_ratio"] = (df["equity_ratio"].values - equity_ratio_prev).astype("float32")
            df["delta_debt_ratio"] = (df["debt_ratio"].values - debt_ratio_prev).astype("float32")
            df["delta_log_assets"] = (df["log_assets"].values - signed_log(L16_l1)).astype("float32")

            df["okved_short"] = df["okved"].astype(str).str.split(".").str[0]
            df["region"] = df["region"].astype(str)
            df["region_okved"] = (df["region"] + "_" + df["okved_short"]).astype(str)
            df["year_okved"] = (df["year"].astype(str) + "_" + df["okved_short"]).astype(str)
            df["year_num"] = df["year"].astype("float32")

            keep = [
                "log_assets", "log_revenue", "log_equity", "log_profit", "log_st_debt",
                "roa", "debt_ratio", "equity_ratio", "working_capital", "current_ratio",
                "asset_turnover", "op_margin", "roe", "asset_drain_1y", "asset_drain_2y",
                "rev_growth_1y", "rev_growth_2y", "drain_accel", "log_drain",
                "profit_growth", "profit_chg", "equity_drain", "debt_growth",
                "no_history", "no_history_2y", "leverage_size", "roa_x_drain",
                "is_zombie", "neg_equity", "neg_profit", "zero_revenue", "no_okved",
                "debt_over_eq", "risk_score", "history_size_so_far",
                "years_since_first_report", "is_new_company", "industry_avg_lifespan",
                "relative_age", "lifespan_anomaly", "roa_prev", "roa_2y_avg",
                "equity_ratio_prev", "delta_roa", "delta_equity_ratio",
                "delta_debt_ratio", "delta_log_assets", "altman_z", "altman_distress",
                "altman_grey", "saifullin_r", "saifullin_distress", "igea_r",
                "igea_distress", "region", "okved_short", "region_okved", "year_okved",
                "year_num", "year", "target_default", "target_dissolution",
            ]

            df = df.dropna(subset=["target_default"])
            if df.empty:
                continue
            df["target_default"] = df["target_default"].astype(np.int8)
            sample_mask = (df["inn"].values % 100) < 20
            df_sampled = df[sample_mask]
            if df_sampled.empty:
                continue
            chunks.append(df_sampled[keep])
            n_ok += 1
        except Exception:
            n_err += 1
            continue

    print(f"   processed: ok={n_ok}, errors={n_err}")
    return pd.concat(chunks, ignore_index=True)


def crisis_age(y, crisis_years=CRISIS_YEARS):
    past = [c for c in crisis_years if c <= y]
    return (y - max(past)) if past else 99


def target_encode(col, train_df, all_df, gm, smoothing=30.0, target="target_default"):
    agg = train_df.groupby(col)[target].agg(["count", "mean"])
    sm = (agg["count"] * agg["mean"] + smoothing * gm) / (agg["count"] + smoothing)
    return all_df[col].map(sm.to_dict()).fillna(gm).astype("float32")


def lagged_def_rate(group_cols, name, train_df, all_df, gm, lag=2, smoothing=30, target="target_default"):
    g = train_df.groupby(group_cols + ["year"])[target].agg(["count", "mean"])
    rate = ((g["count"] * g["mean"] + smoothing * gm) / (g["count"] + smoothing)).astype("float32")
    rate.index = rate.index.set_levels(rate.index.levels[-1] + lag, level=-1)
    keys = pd.MultiIndex.from_arrays([all_df[c].values for c in group_cols] + [all_df["year"].values])
    all_df[name] = rate.reindex(keys).fillna(gm).astype("float32").values


def build_dataset(data_dir, te_train_cutoff_year=2020):
    """ETL и конструирование признаков. Возвращает full_data, готовый к делению
    на train/test по year. te_train_cutoff_year задаёт верхнюю границу train
    для безутечного TE и lagged-rate (2020 для финальной модели, 2016 для
    sensitivity).
    """
    all_files = glob.glob(os.path.join(data_dir, "**", "*.parquet"), recursive=True)
    history = build_history_df(all_files)
    age_stats = build_company_age_stats(history)
    okved_lookup, diss_lookup = build_company_meta(all_files)
    lifespan_stats = compute_industry_lifespan(age_stats[0], okved_lookup, diss_lookup)
    del okved_lookup, diss_lookup
    gc.collect()

    full_data = create_ultimate_dataset(all_files, history, age_stats, lifespan_stats)
    del history, age_stats, lifespan_stats
    gc.collect()

    print(f"Датасет: {len(full_data):,} строк, "
          f"дефолтов {int(full_data['target_default'].sum()):,} "
          f"({full_data['target_default'].mean() * 100:.2f}%)")

    density = full_data.groupby(["region", "okved_short"], sort=False).size().rename("_density")
    full_data = full_data.merge(density, left_on=["region", "okved_short"], right_index=True, how="left")
    full_data["industry_density"] = np.log1p(full_data["_density"]).astype("float32")
    full_data = full_data.drop(columns=["_density"])
    del density
    gc.collect()

    yr = full_data["year"].astype(int)
    full_data["macro_key_rate"] = yr.map({y: v[0] for y, v in MACRO_BY_YEAR.items()}).fillna(8.0).astype("float32")
    full_data["macro_gdp"] = yr.map({y: v[1] for y, v in MACRO_BY_YEAR.items()}).fillna(2.0).astype("float32")
    full_data["macro_usd"] = yr.map({y: v[2] for y, v in MACRO_BY_YEAR.items()}).fillna(70.0).astype("float32")
    full_data["macro_oil"] = yr.map({y: v[3] for y, v in MACRO_BY_YEAR.items()}).fillna(70.0).astype("float32")
    full_data["is_crisis_year"] = yr.isin(CRISIS_YEARS).astype(np.int8)
    full_data["is_post_crisis"] = yr.isin(POST_CRISIS_YEARS).astype(np.int8)
    full_data["years_since_crisis"] = yr.map({int(y): int(crisis_age(int(y))) for y in yr.unique()}).astype(np.int8)
    del yr
    gc.collect()

    for c in RATIO_COLS:
        lo, hi = full_data[c].quantile([0.001, 0.999])
        full_data[c] = full_data[c].clip(lo, hi).astype("float32")

    ind_group = full_data.groupby(["year", "okved_short"], sort=False)
    for c in ["roa", "equity_ratio", "debt_ratio", "working_capital", "asset_turnover"]:
        med = ind_group[c].transform("median")
        full_data[f"{c}_rel"] = (full_data[c] - med).astype("float32")
    del ind_group
    gc.collect()

    train_part = full_data[full_data["year"] <= te_train_cutoff_year]
    global_pos_rate = float(train_part["target_default"].mean())

    full_data["region_okved_te"] = target_encode("region_okved", train_part, full_data, global_pos_rate, smoothing=20)
    full_data["year_okved_te"] = target_encode("year_okved", train_part, full_data, global_pos_rate, smoothing=15)
    full_data = full_data.drop(columns=["year_okved"])

    lagged_def_rate(["region"], "region_def_rate_lag2", train_part, full_data, global_pos_rate, smoothing=30)
    lagged_def_rate(["okved_short"], "okved_def_rate_lag2", train_part, full_data, global_pos_rate, smoothing=30)
    lagged_def_rate(["region", "okved_short"], "region_okved_def_rate_lag2", train_part, full_data, global_pos_rate, smoothing=15)
    del train_part
    gc.collect()

    for c in cat_features:
        full_data[c] = full_data[c].astype(str)

    return full_data
