import jax, os
jax_version = jax.__version__
major, minor, patch = (int(x) for x in jax_version.split(".")[:3])
if (major, minor, patch) >= (0, 4, 32):
    print(f"JAX version: {jax_version}")
    os.environ["XLA_FLAGS"] = "--xla_cpu_use_thunk_runtime=false"
import numpy as np
import jax.numpy as jnp

import pandas as pd
import matplotlib.pyplot as plt

from dataclasses import dataclass
from itertools import product
from pathlib import Path

from jax import config, random
import numpyro, jax
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_value

from jnkepler.jaxttv import JaxTTV
from jnkepler.jaxttv.infer import (
    ttv_default_parameter_bounds,
    ttv_optim_curve_fit,
)

config.update("jax_enable_x64", True)
numpyro.set_platform("cpu")
num_chains = 4
numpyro.set_host_device_count(num_chains)
print("# jax device count:", jax.local_device_count())


M_EARTH_IN_MSUN = 3.0034893e-6
DEG = np.pi / 180.0


@dataclass
class Config:
    transit_file: str = "toi560_transit_times.txt"
    output_csv: str = "toi560_wide_grid_chi2.csv"

    Pb0: float = 6.3980661
    Pc0: float = 18.8805

    # jnkepler optimization settings
    dt: float = 0.2
    t_margin: float = 6.17
    transit_time_method: str = "fast"

    # Whether to use JAX Jacobian inside ttv_optim_curve_fit.
    # Start with False for robustness.
    use_jax_jacobian: bool = False

    # Search region
    run_wide: bool = True
    run_chain: bool = False

    wide_P_min: float = 7.0
    wide_P_max: float = 18.0
    wide_NP: int = 30

    #12.639
    chain_P_min: float = 12.10
    chain_P_max: float = 13.10
    chain_NP: int = 30

    ed_grid: tuple = (0.0, 0.02, 0.05)
    omega_grid_deg: tuple = (0.0, 90.0, 180.0, 270.0)
    phase_grid_deg: tuple = (0.0, 90.0, 180.0, 270.0)

    # Default bounds for b/c/d
    dtic: float = 0.20
    dp_frac: float = 0.01
    emax: float = 0.20
    mmin: float = 0.05 * M_EARTH_IN_MSUN
    mmax: float = 80.0 * M_EARTH_IN_MSUN

    # To fix d grid parameters, use very narrow bounds.
    # ここの値は変えて良い！wide→0.15, chain→0.003
    fixed_P_width: float = 0.15
    fixed_hk_width: float = 0.02
    fixed_tic_width: float = 0.1

    save_every: int = 20


CFG = Config()


# ============================================================
# Data
# ============================================================

def load_transit_times(path):
    """
    Expected columns:
        tnum, tc, tcerr, dnum, planum

    planum=1 -> TOI-560 b
    planum=2 -> TOI-560 c
    """
    df = pd.read_csv(
        path,
        header=None,
        names=["tnum", "tc", "tcerr", "dnum", "planum"],
        comment="#",
        skip_blank_lines=True,
    )

    for col in ["tnum", "tc", "tcerr", "dnum", "planum"]:
        df[col] = pd.to_numeric(df[col], errors="raise")

    df_b = df[df["planum"].astype(int) == 1].sort_values("tc")
    df_c = df[df["planum"].astype(int) == 2].sort_values("tc")

    if len(df_b) == 0:
        raise ValueError("planum=1, TOI-560 b, が見つかりません。")
    if len(df_c) == 0:
        raise ValueError("planum=2, TOI-560 c, が見つかりません。")

    tcobs = [
        df_b["tc"].to_numpy(float),
        df_c["tc"].to_numpy(float),
    ]
    tcerr = [
        df_b["tcerr"].to_numpy(float),
        df_c["tcerr"].to_numpy(float),
    ]

    return tcobs, tcerr


def linear_ephemeris(tc, p_guess):
    epoch = np.round((tc - tc[0]) / p_guess)
    pfit, t0fit = np.polyfit(epoch, tc, deg=1)
    return float(t0fit), float(pfit)


# ============================================================
# Phase helpers
# ============================================================

def wrap_2pi(x):
    return np.mod(x, 2.0 * np.pi)


def lambda_from_tic(t_epoch, period, tic):
    return wrap_2pi(2.0 * np.pi * (t_epoch - tic) / period)


def tic_from_lambda(t_epoch, period, lam):
    tic = t_epoch - period * wrap_2pi(lam) / (2.0 * np.pi)

    while tic < t_epoch - period:
        tic += period
    while tic > t_epoch:
        tic -= period

    return float(tic)


def tic_d_from_three_body_phi(t_epoch, Pb, Pc, Pd, tic_b, tic_c, phi):
    """
    1:2:3 chain:
        phi = lambda_b - 4 lambda_d + 3 lambda_c
    """
    lam_b = lambda_from_tic(t_epoch, Pb, tic_b)
    lam_c = lambda_from_tic(t_epoch, Pc, tic_c)
    lam_d = wrap_2pi((lam_b + 3.0 * lam_c - phi) / 4.0)
    return tic_from_lambda(t_epoch, Pd, lam_d)


def resonance_offsets(Pb, Pd, Pc):
    delta_bd_21 = (Pd / Pb) / 2.0 - 1.0
    delta_dc_32 = (Pc / Pd) / 1.5 - 1.0

    nb = 2.0 * np.pi / Pb
    nd = 2.0 * np.pi / Pd
    nc = 2.0 * np.pi / Pc
    phidot = nb - 4.0 * nd + 3.0 * nc

    return delta_bd_21, delta_dc_32, phidot


# ============================================================
# Grid
# ============================================================

def make_grid(cfg, t_epoch, t0_b, Pb, t0_c, Pc):
    if cfg.run_wide:
        for Pd in np.linspace(cfg.wide_P_min, cfg.wide_P_max, cfg.wide_NP):
            for ed, omega_deg, phase_deg in product(
                cfg.ed_grid,
                cfg.omega_grid_deg,
                cfg.phase_grid_deg,
            ):
                omega = omega_deg * DEG
                hd = ed * np.cos(omega)
                kd = ed * np.sin(omega)

                lam_d = phase_deg * DEG
                tic_d = tic_from_lambda(t_epoch, Pd, lam_d)

                yield {
                    "search_type": "wide",
                    "Pd": float(Pd),
                    "ed": float(ed),
                    "omega_deg": float(omega_deg),
                    "phase_or_phi_deg": float(phase_deg),
                    "hd": float(hd),
                    "kd": float(kd),
                    "tic_d": float(tic_d),
                }

    if cfg.run_chain:
        for Pd in np.linspace(cfg.chain_P_min, cfg.chain_P_max, cfg.chain_NP):
            for ed, omega_deg, phi_deg in product(
                cfg.ed_grid,
                cfg.omega_grid_deg,
                cfg.phase_grid_deg,
            ):
                omega = omega_deg * DEG
                hd = ed * np.cos(omega)
                kd = ed * np.sin(omega)

                phi = phi_deg * DEG
                tic_d = tic_d_from_three_body_phi(
                    t_epoch=t_epoch,
                    Pb=Pb,
                    Pc=Pc,
                    Pd=Pd,
                    tic_b=t0_b,
                    tic_c=t0_c,
                    phi=phi,
                )

                yield {
                    "search_type": "chain_1_2_3",
                    "Pd": float(Pd),
                    "ed": float(ed),
                    "omega_deg": float(omega_deg),
                    "phase_or_phi_deg": float(phi_deg),
                    "hd": float(hd),
                    "kd": float(kd),
                    "tic_d": float(tic_d),
                }


# ============================================================
# Bounds and chi2
# ============================================================

def build_bounds_for_grid(cfg, jttv, t0_b, Pb, t0_c, Pc, g):
    npl = 3

    t0_guess = np.array([t0_b, g["tic_d"], t0_c], dtype=float)
    p_guess = np.array([Pb, g["Pd"], Pc], dtype=float)

    param_bounds = ttv_default_parameter_bounds(
        jttv,
        npl=npl,
        t0_guess=t0_guess,
        p_guess=p_guess,
        dtic=cfg.dtic,
        dp_frac=cfg.dp_frac,
        emax=cfg.emax,
        mmin=cfg.mmin,
        mmax=cfg.mmax,
    )

    # Include coplanar angle bounds explicitly.
    param_bounds["cosi"] = [
        np.zeros(npl),
        np.zeros(npl) + 1.0e-8,
    ]
    param_bounds["lnode"] = [
        np.zeros(npl),
        np.zeros(npl) + 1.0e-8,
    ]

    # Fix d's period near grid Pd.
    param_bounds["period"][0][1] = g["Pd"] - cfg.fixed_P_width
    param_bounds["period"][1][1] = g["Pd"] + cfg.fixed_P_width

    # Fix d's eccentricity vector near grid hd/kd.
    param_bounds["ecosw"][0][1] = g["hd"] - cfg.fixed_hk_width
    param_bounds["ecosw"][1][1] = g["hd"] + cfg.fixed_hk_width

    param_bounds["esinw"][0][1] = g["kd"] - cfg.fixed_hk_width
    param_bounds["esinw"][1][1] = g["kd"] + cfg.fixed_hk_width

    # Fix d's tic near grid tic_d.
    param_bounds["tic"][0][1] = g["tic_d"] - cfg.fixed_tic_width
    param_bounds["tic"][1][1] = g["tic_d"] + cfg.fixed_tic_width

    return param_bounds


def chi2_from_pdic(jttv, pdic, transit_orbit_idx):
    tc_model, ediff = jttv.get_transit_times_obs(
        pdic,
        transit_orbit_idx=jnp.array(transit_orbit_idx),
    )

    resid = (
        np.asarray(tc_model)
        - np.asarray(jttv.tcobs_flatten)
    ) / np.asarray(jttv.errorobs_flatten)

    return float(np.sum(resid**2))


def fit_one_grid(cfg, jttv, param_bounds, transit_orbit_idx):
    popt = ttv_optim_curve_fit(
        jttv,
        param_bounds,
        jac=cfg.use_jax_jacobian,
        plot=False,
        transit_orbit_idx=np.array(transit_orbit_idx),
    )

    chi2 = chi2_from_pdic(jttv, popt, transit_orbit_idx)

    return popt, chi2

def plot_2p_baseline(cfg, jttv, t0_b, Pb_lin, t0_c, Pc_lin):
    """
    Fit ordinary 2-planet baseline [b,c] and save jnkepler TTV plots.
    No chi2 print, no CSV output.
    """

    bounds_2p = ttv_default_parameter_bounds(
        jttv,
        npl=2,
        t0_guess=np.array([t0_b, t0_c], dtype=float),
        p_guess=np.array([Pb_lin, Pc_lin], dtype=float),
        dtic=cfg.dtic,
        dp_frac=cfg.dp_frac,
        emax=cfg.emax,
        mmin=cfg.mmin,
        mmax=cfg.mmax,
    )

    popt_2p = ttv_optim_curve_fit(
        jttv,
        bounds_2p,
        jac=cfg.use_jax_jacobian,
        plot=False,
    )

    tcmodellist_2p = jttv.get_transit_times_all_list(
        popt_2p,
        truncate=True,
    )

    plot_dir = Path("baseline_plots")
    plot_dir.mkdir(parents=True, exist_ok=True)

    jttv.plot_model(
        tcmodellist_2p,
        save=str(plot_dir / "baseline_2p"),
        unit=1440.0,
        xlabel="Transit time [day]",
        ylabel="TTV from linear ephemeris [min]",
    )

    print("Saved 2p baseline plots:")
    print("  baseline_plots/baseline_2p_planet1.png")
    print("  baseline_plots/baseline_2p_planet2.png")

def plot_3p_baseline_from_popt(jttv, popt_base):
    """
    Plot baseline TTV curve using optimized 3-planet baseline parameters.

    Dynamical model:
        orbit 0 = b
        orbit 1 = nearly massless d
        orbit 2 = c

    Observed transiting planets:
        b and c only -> transit_orbit_idx=[0, 2]
    """

    tcmodellist_base = jttv.get_transit_times_all_list(
        popt_base,
        truncate=True,
        transit_orbit_idx=jnp.array([0, 2]),
    )

    plot_dir = Path("baseline_plots")
    plot_dir.mkdir(parents=True, exist_ok=True)

    jttv.plot_model(
        tcmodellist_base,
        save=str(plot_dir / "baseline_3p_massless_d"),
        unit=1440.0,
        xlabel="Transit time [day]",
        ylabel="TTV from linear ephemeris [min]",
    )

    print("Saved 3p baseline plots:")
    print("  baseline_plots/baseline_3p_massless_d_planet1.png")
    print("  baseline_plots/baseline_3p_massless_d_planet2.png")


# ============================================================
# Main
# ============================================================

def main(cfg):
    out = Path(cfg.output_csv)
    if out.exists():
        out.unlink()

    tcobs, tcerr = load_transit_times(cfg.transit_file)

    t0_b, Pb_lin = linear_ephemeris(tcobs[0], cfg.Pb0)
    t0_c, Pc_lin = linear_ephemeris(tcobs[1], cfg.Pc0)

    print("Linear ephemerides:")
    print(f"  b: T0={t0_b:.8f}, P={Pb_lin:.8f}")
    print(f"  c: T0={t0_c:.8f}, P={Pc_lin:.8f}")

    t_start = min(np.min(tcobs[0]), np.min(tcobs[1])) - cfg.t_margin
    t_end = max(np.max(tcobs[0]), np.max(tcobs[1])) + cfg.t_margin

    jttv = JaxTTV(
        t_start=t_start,
        t_end=t_end,
        dt=cfg.dt,
        tcobs=tcobs,
        p_init=[Pb_lin, Pc_lin],
        errorobs=tcerr,
        print_info=True,
        transit_time_method=cfg.transit_time_method,
    )

    plot_2p_baseline(
        cfg,
        jttv,
        t0_b,
        Pb_lin,
        t0_c,
        Pc_lin,
    )

    # Baseline: b/c only in practice, represented by a nearly massless middle d.
    baseline_grid = {
        "Pd": 12.7,
        "hd": 0.0,
        "kd": 0.0,
        "tic_d": t0_b,
    }

    # baseline_boundはdef関数を使わず自分で作る
    baseline_bounds = ttv_default_parameter_bounds(
        jttv,
        npl=3,
        t0_guess=np.array([t0_b, baseline_grid["tic_d"], t0_c]),
        p_guess=np.array([Pb_lin, baseline_grid["Pd"], Pc_lin]),
        dtic=cfg.dtic,
        dp_frac=cfg.dp_frac,
        emax=cfg.emax,
        mmin=cfg.mmin,
        mmax=cfg.mmax,
    )

    baseline_bounds["cosi"] = [
        np.zeros(3),
        np.zeros(3) + 1.0e-8,
    ]
    baseline_bounds["lnode"] = [
        np.zeros(3),
        np.zeros(3) + 1.0e-8,
    ]

    # Force d to be almost massless for baseline.
    baseline_bounds["lnpmass"][0][1] = np.log(1.0e-10)
    baseline_bounds["lnpmass"][1][1] = np.log(1.0e-7)

    # Keep d's other parameters harmless and fixed.
    baseline_bounds["period"][0][1] = baseline_grid["Pd"] - cfg.fixed_P_width
    baseline_bounds["period"][1][1] = baseline_grid["Pd"] + cfg.fixed_P_width
    baseline_bounds["ecosw"][0][1] = -cfg.fixed_hk_width
    baseline_bounds["ecosw"][1][1] = +cfg.fixed_hk_width
    baseline_bounds["esinw"][0][1] = -cfg.fixed_hk_width
    baseline_bounds["esinw"][1][1] = +cfg.fixed_hk_width
    baseline_bounds["tic"][0][1] = baseline_grid["tic_d"] - cfg.fixed_tic_width
    baseline_bounds["tic"][1][1] = baseline_grid["tic_d"] + cfg.fixed_tic_width

    print("Fitting baseline model...")
    try:
        popt_base, chi2_base = fit_one_grid(
            cfg,
            jttv,
            baseline_bounds,
            transit_orbit_idx=[0, 2],
        )
    except Exception as e:
        raise RuntimeError(f"Baseline optimization failed: {repr(e)}")

    print(f"Baseline chi2 = {chi2_base:.6f}")

    print("\n=== Optimized parameters ===")
    for key, val in popt_base.items():
        try:
            print(key, np.asarray(val))
        except Exception:
            print(key, val)

    plot_3p_baseline_from_popt(jttv, popt_base)

    grid = list(make_grid(cfg, t_start, t0_b, Pb_lin, t0_c, Pc_lin))
    print(f"Number of grid points = {len(grid)}")

    rows = []

    # ここからgirdでparams_boundを作る
    for i, g in enumerate(grid, start=1):
        param_bounds = build_bounds_for_grid(
            cfg,
            jttv,
            t0_b,
            Pb_lin,
            t0_c,
            Pc_lin,
            g,
        )

        try:
            popt, chi2 = fit_one_grid(
                cfg,
                jttv,
                param_bounds,
                transit_orbit_idx=[0, 2],
            )
            success = True
            message = "ok"
        except Exception as e:
            popt = None
            chi2 = np.inf
            success = False
            message = repr(e)

        # ここからまとめ

        delta_bd, delta_dc, phidot = resonance_offsets(Pb_lin, g["Pd"], Pc_lin)

        if popt is not None:
            md_mearth = float(np.asarray(popt["pmass"])[1] / M_EARTH_IN_MSUN)
            mb_mearth = float(np.asarray(popt["pmass"])[0] / M_EARTH_IN_MSUN)
            mc_mearth = float(np.asarray(popt["pmass"])[2] / M_EARTH_IN_MSUN)
            Pb_best = float(np.asarray(popt["period"])[0])
            Pd_best = float(np.asarray(popt["period"])[1])
            Pc_best = float(np.asarray(popt["period"])[2])
            h_best = np.asarray(popt["ecosw"], dtype=float)
            k_best = np.asarray(popt["esinw"], dtype=float)
            tic_best = np.asarray(popt["tic"], dtype=float)

            hb_best, hd_best, hc_best = h_best
            kb_best, kd_best, kc_best = k_best

            eb_best = np.sqrt(hb_best**2 + kb_best**2)
            ed_best = np.sqrt(hd_best**2 + kd_best**2)
            ec_best = np.sqrt(hc_best**2 + kc_best**2)

            tic_b_best, tic_d_best, tic_c_best = tic_best

        else:
            mb_mearth = md_mearth = mc_mearth = np.nan
            Pb_best = Pd_best = Pc_best = np.nan
            hb_best = hd_best = hc_best = np.nan
            kb_best = kd_best = kc_best = np.nan
            eb_best = ed_best = ec_best = np.nan
            tic_b_best = tic_d_best = tic_c_best = np.nan

        row = {
            **g,
            "chi2_base": chi2_base,
            "chi2_3p": chi2,
            "delta_chi2": chi2_base - chi2,
            "success": success,
            "message": message,

            "delta_bd_2_1": delta_bd,
            "delta_dc_3_2": delta_dc,
            "phidot_rad_per_day": phidot,

            "mb_mearth": mb_mearth,
            "md_mearth": md_mearth,
            "mc_mearth": mc_mearth,

            "Pb_best": Pb_best,
            "Pd_best": Pd_best,
            "Pc_best": Pc_best,

            "hb_best": hb_best,
            "hd_best": hd_best,
            "hc_best": hc_best,
            "kb_best": kb_best,
            "kd_best": kd_best,
            "kc_best": kc_best,

            "eb_best": eb_best,
            "ed_best": ed_best,
            "ec_best": ec_best,

            "tic_b_best": tic_b_best,
            "tic_d_best": tic_d_best,
            "tic_c_best": tic_c_best,

            "hd_shift": hd_best - g["hd"],
            "kd_shift": kd_best - g["kd"],
            "tic_d_shift": tic_d_best - g["tic_d"],
        }

        rows.append(row)

        if i % cfg.save_every == 0 or i == len(grid):
            pd.DataFrame(rows).to_csv(out, index=False)
        print(
            f"{i:5d}/{len(grid)} "
            f"{g['search_type']:12s} "
            f"Pd={g['Pd']:.6f} "
            f"e={g['ed']:.3f} "
            f"phase/phi={g['phase_or_phi_deg']:6.1f} "
            f"chi2={chi2:.3f} "
            f"dchi2={chi2_base - chi2:.3f} "
            f"md={md_mearth:.3f} Mearth "
            f"success={success}"
        )

    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)

    print("\nSaved:", out)
    print("\nTop 20 candidates:")
    print(
        df.sort_values("delta_chi2", ascending=False)
        .head(20)[
            [
                "search_type",
                "Pd",
                "ed",
                "omega_deg",
                "phase_or_phi_deg",
                "chi2_3p",
                "delta_chi2",
                "md_mearth",
                "delta_bd_2_1",
                "delta_dc_3_2",
                "phidot_rad_per_day",
                "success",
            ]
        ]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main(CFG)