# select_hmc_targets.py

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class SelectionConfig:
    input_csv: str = "toi560_wide_grid_chi2.csv"
    output_csv: str = "toi560_wide_hmc_targets.csv"

    # Basic filters
    min_delta_chi2: float = 10.0
    min_mass_mearth: float = 0.05
    max_mass_mearth: float = 80.0

    # Eccentricity filters
    max_eb: float = 0.30
    max_ed: float = 0.30
    max_ec: float = 0.30

    # These must match the widths used in the grid search
    fixed_P_width: float = 0.15
    fixed_hk_width: float = 0.02
    fixed_tic_width: float = 0.10

    # Edge filter: reject if best-fit is too close to the bounds edge
    max_Pd_edge_frac: float = 0.9
    max_hk_edge_frac: float = 0.9
    max_ticd_edge_frac: float = 0.9

    # Remove near-duplicate solutions in Pd_best
    Pd_bin_width: float = 0.01

    # Number of final HMC targets
    top_n: int = 10

    # Optional phidot hard cut.
    # If None, abs(phidot) is used only for ranking.
    max_abs_phidot: float | None = None


CFG = SelectionConfig()


def load_and_prepare_grid_results(cfg):
    df = pd.read_csv(cfg.input_csv)

    required_cols = [
        "search_type",
        "Pd",
        "ed",
        "omega_deg",
        "phase_or_phi_deg",
        "hd",
        "kd",
        "tic_d",
        "chi2_base",
        "chi2_3p",
        "delta_chi2",
        "success",
        "delta_bd_2_1",
        "delta_dc_3_2",
        "phidot_rad_per_day",
        "mb_mearth",
        "md_mearth",
        "mc_mearth",
        "Pb_best",
        "Pd_best",
        "Pc_best",
    ]

    # Strongly recommended extra columns
    recommended_cols = [
        "hb_best",
        "hd_best",
        "hc_best",
        "kb_best",
        "kd_best",
        "kc_best",
        "tic_d_best",
    ]

    missing_required = [c for c in required_cols if c not in df.columns]
    if missing_required:
        raise ValueError(
            f"Missing required columns in {cfg.input_csv}: {missing_required}"
        )

    missing_recommended = [c for c in recommended_cols if c not in df.columns]
    if missing_recommended:
        raise ValueError(
            "To check eccentricities and h/k/tic_d edge sticking, "
            f"the CSV must include these columns: {missing_recommended}\n"
            "Add hb_best, hd_best, hc_best, kb_best, kd_best, kc_best, "
            "tic_d_best to the grid-search output first."
        )

    # Normalize success column
    if df["success"].dtype == bool:
        df["success_bool"] = df["success"]
    else:
        df["success_bool"] = (
            df["success"]
            .astype(str)
            .str.lower()
            .isin(["true", "1", "yes", "ok"])
        )

    numeric_cols = required_cols.copy()
    numeric_cols.remove("search_type")
    numeric_cols.remove("success")
    numeric_cols += recommended_cols

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=numeric_cols).copy()

    # Period edge check
    df["Pd_shift"] = df["Pd_best"] - df["Pd"]
    df["Pd_edge_frac"] = np.abs(df["Pd_shift"]) / cfg.fixed_P_width

    # h/k edge checks for d
    df["hd_shift"] = df["hd_best"] - df["hd"]
    df["kd_shift"] = df["kd_best"] - df["kd"]

    df["hd_edge_frac"] = np.abs(df["hd_shift"]) / cfg.fixed_hk_width
    df["kd_edge_frac"] = np.abs(df["kd_shift"]) / cfg.fixed_hk_width
    df["hk_edge_frac_max"] = df[["hd_edge_frac", "kd_edge_frac"]].max(axis=1)

    # tic_d edge check
    df["tic_d_shift"] = df["tic_d_best"] - df["tic_d"]
    df["tic_d_edge_frac"] = np.abs(df["tic_d_shift"]) / cfg.fixed_tic_width

    # Eccentricities from optimized h/k
    df["eb_best"] = np.sqrt(df["hb_best"] ** 2 + df["kb_best"] ** 2)
    df["ed_best"] = np.sqrt(df["hd_best"] ** 2 + df["kd_best"] ** 2)
    df["ec_best"] = np.sqrt(df["hc_best"] ** 2 + df["kc_best"] ** 2)

    # Resonance diagnostics
    df["abs_phidot_rad_per_day"] = np.abs(df["phidot_rad_per_day"])
    df["abs_delta_bd_2_1"] = np.abs(df["delta_bd_2_1"])
    df["abs_delta_dc_3_2"] = np.abs(df["delta_dc_3_2"])

    return df


def apply_hmc_selection_filters(df, cfg):
    selected = df[
        (df["success_bool"])
        & (df["delta_chi2"] > cfg.min_delta_chi2)

        # mass filters
        & (df["mb_mearth"] > cfg.min_mass_mearth)
        & (df["mb_mearth"] < cfg.max_mass_mearth)
        & (df["md_mearth"] > cfg.min_mass_mearth)
        & (df["md_mearth"] < cfg.max_mass_mearth)
        & (df["mc_mearth"] > cfg.min_mass_mearth)
        & (df["mc_mearth"] < cfg.max_mass_mearth)

        # eccentricity filters
        & (df["eb_best"] < cfg.max_eb)
        & (df["ed_best"] < cfg.max_ed)
        & (df["ec_best"] < cfg.max_ec)

        # bounds-edge filters
        & (df["Pd_edge_frac"] < cfg.max_Pd_edge_frac)
        & (df["hd_edge_frac"] < cfg.max_hk_edge_frac)
        & (df["kd_edge_frac"] < cfg.max_hk_edge_frac)
        & (df["tic_d_edge_frac"] < cfg.max_ticd_edge_frac)
    ].copy()

    if cfg.max_abs_phidot is not None:
        selected = selected[
            selected["abs_phidot_rad_per_day"] < cfg.max_abs_phidot
        ].copy()

    return selected


def remove_near_duplicate_Pd_solutions(df, cfg):
    """
    Keep only the best delta_chi2 solution in each Pd_best bin.
    This avoids sending many near-identical period solutions to HMC.
    """
    df = df.copy()

    df["Pd_best_bin"] = (
        np.round(df["Pd_best"] / cfg.Pd_bin_width) * cfg.Pd_bin_width
    )

    deduped = (
        df.sort_values("delta_chi2", ascending=False)
        .groupby("Pd_best_bin", as_index=False)
        .head(1)
        .copy()
    )

    return deduped


def rank_candidates(df, cfg):
    """
    Ranking priority:
      1. larger delta_chi2
      2. smaller abs(phidot_rad_per_day)
      3. smaller max edge fraction
    """
    df = df.copy()

    df["max_edge_frac"] = df[
        [
            "Pd_edge_frac",
            "hd_edge_frac",
            "kd_edge_frac",
            "tic_d_edge_frac",
        ]
    ].max(axis=1)

    ranked = df.sort_values(
        ["delta_chi2", "abs_phidot_rad_per_day", "max_edge_frac"],
        ascending=[False, True, True],
    ).copy()

    ranked["hmc_rank"] = np.arange(1, len(ranked) + 1)

    return ranked.head(cfg.top_n).copy()


def main(cfg):
    df = load_and_prepare_grid_results(cfg)

    print("Total usable grid rows:", len(df))

    selected = apply_hmc_selection_filters(df, cfg)
    print("After physical/numerical filters:", len(selected))

    deduped = remove_near_duplicate_Pd_solutions(selected, cfg)
    print("After Pd_best de-duplication:", len(deduped))

    targets = rank_candidates(deduped, cfg)
    print("Final HMC targets:", len(targets))

    display_cols = [
        "hmc_rank",
        "search_type",

        "Pd",
        "Pd_best",
        "Pd_shift",
        "Pd_edge_frac",

        "ed",
        "omega_deg",
        "phase_or_phi_deg",

        "chi2_base",
        "chi2_3p",
        "delta_chi2",

        "mb_mearth",
        "md_mearth",
        "mc_mearth",

        "eb_best",
        "ed_best",
        "ec_best",

        "hd",
        "hd_best",
        "hd_shift",
        "hd_edge_frac",

        "kd",
        "kd_best",
        "kd_shift",
        "kd_edge_frac",

        "tic_d",
        "tic_d_best",
        "tic_d_shift",
        "tic_d_edge_frac",

        "hk_edge_frac_max",
        "max_edge_frac",

        "delta_bd_2_1",
        "delta_dc_3_2",
        "phidot_rad_per_day",
        "abs_phidot_rad_per_day",

        "Pd_best_bin",
        "success",
    ]

    display_cols = [c for c in display_cols if c in targets.columns]

    print("\nSelected HMC targets:")
    print(targets[display_cols].to_string(index=False))

    targets.to_csv(cfg.output_csv, index=False)
    print(f"\nSaved: {cfg.output_csv}")


if __name__ == "__main__":
    main(CFG)