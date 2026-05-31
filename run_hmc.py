from dataclasses import dataclass
from pathlib import Path

import jax, os
jax_version = jax.__version__
major, minor, patch = (int(x) for x in jax_version.split(".")[:3])
if (major, minor, patch) >= (0, 4, 32):
    print(f"JAX version: {jax_version}")
    os.environ["XLA_FLAGS"] = "--xla_cpu_use_thunk_runtime=false"
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
from jax import config, random

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_value

import arviz as az
import corner

from jnkepler.jaxttv import JaxTTV

config.update("jax_enable_x64", True)
numpyro.set_platform("cpu")
numpyro.set_host_device_count(8)

M_EARTH_IN_MSUN = 3.0034893e-6


@dataclass
class Config:
    transit_file: str = "toi560_transit_times.txt"
    target_csv: str = "toi560_wide_hmc_targets.csv"
    output_dir: str = "toi560_wide_oc_hmc_simple"

    Pb0: float = 6.3980661
    Pc0: float = 18.8805

    dt: float = 0.4
    t_margin: float = 6.17
    transit_time_method: str = "fast"

    # HMC prior widths around best-fit params
    # このwidthはもっと広げていいのでは！！！
    P_width: float = 3.0
    tic_width: float = 0.5
    hk_width: float = 1.0
    ln_mass_width: float = 3.0

    # HMC settings
    num_warmup: int = 20000
    num_samples: int = 20000
    num_chains: int = 4
    target_accept_prob: float = 0.85
    max_tree_depth: int = 10
    rng_seed: int = 42


CFG = Config()


# ============================================================
# Data
# ============================================================

def load_transit_times(path):
    df = pd.read_csv(
        path,
        header=None,
        names=["tnum", "tc", "tcerr", "dnum", "planum"],
        comment="#",
        skip_blank_lines=True,
    )

    for col in ["tnum", "tc", "tcerr", "dnum", "planum"]:
        df[col] = pd.to_numeric(df[col], errors="raise")

    df["tnum"] = df["tnum"].astype(int)
    df["dnum"] = df["dnum"].astype(int)
    df["planum"] = df["planum"].astype(int)

    df_b = df[df["planum"] == 1].sort_values("tc").copy()
    df_c = df[df["planum"] == 2].sort_values("tc").copy()

    tcobs = [
        df_b["tc"].to_numpy(float),
        df_c["tc"].to_numpy(float),
    ]
    tcerr = [
        df_b["tcerr"].to_numpy(float),
        df_c["tcerr"].to_numpy(float),
    ]

    return df_b, df_c, tcobs, tcerr


def linear_ephemeris(tc, p_guess):
    epoch = np.round((tc - tc[0]) / p_guess)
    pfit, t0fit = np.polyfit(epoch, tc, deg=1)
    return float(t0fit), float(pfit)


# ============================================================
# Params
# ============================================================

def row_to_pdic(row):
    period = jnp.array([
        row["Pb_best"],
        row["Pd_best"],
        row["Pc_best"],
    ])

    ecosw = jnp.array([
        row["hb_best"],
        row["hd_best"],
        row["hc_best"],
    ])

    esinw = jnp.array([
        row["kb_best"],
        row["kd_best"],
        row["kc_best"],
    ])

    tic = jnp.array([
        row["tic_b_best"],
        row["tic_d_best"],
        row["tic_c_best"],
    ])

    pmass = jnp.array([
        row["mb_mearth"] * M_EARTH_IN_MSUN,
        row["md_mearth"] * M_EARTH_IN_MSUN,
        row["mc_mearth"] * M_EARTH_IN_MSUN,
    ])

    return {
        "period": period,
        "ecosw": ecosw,
        "esinw": esinw,
        "tic": tic,
        "pmass": pmass,
        "smass": 1.0,
    }


def make_bounds_around_pdic(pdic, cfg):
    period = np.asarray(pdic["period"])
    tic = np.asarray(pdic["tic"])
    ecosw = np.asarray(pdic["ecosw"])
    esinw = np.asarray(pdic["esinw"])
    lnpmass = np.log(np.asarray(pdic["pmass"]))

    bounds = {
        "period": [
            period - cfg.P_width,
            period + cfg.P_width,
        ],
        "tic": [
            tic - cfg.tic_width,
            tic + cfg.tic_width,
        ],
        "ecosw": [
            ecosw - cfg.hk_width,
            ecosw + cfg.hk_width,
        ],
        "esinw": [
            esinw - cfg.hk_width,
            esinw + cfg.hk_width,
        ],
        "lnpmass": [
            lnpmass - cfg.ln_mass_width,
            lnpmass + cfg.ln_mass_width,
        ],
    }

    return bounds



# ============================================================
# TTV relative to linear ephemeris
# ============================================================

def fit_linear_ephemeris_from_tnum(df_planet):
    """
    Fit linear ephemeris:
        tc = T0 + P * tnum

    Uses the tnum column in the transit file.
    """
    tnum = df_planet["tnum"].to_numpy(float)
    tc = df_planet["tc"].to_numpy(float)
    tcerr = df_planet["tcerr"].to_numpy(float)

    # Weighted linear fit: tc = P * tnum + T0
    coeff = np.polyfit(
        tnum,
        tc,
        deg=1,
        w=1.0 / tcerr,
    )

    P = float(coeff[0])
    T0 = float(coeff[1])

    return T0, P

def get_oc_table(jttv, pdic, df_b, df_c):
    """
    Compute O-C table from a best-fit parameter dictionary.

    Here:
        O-C = observed transit time - model transit time

    This is not TTV from a linear ephemeris.
    This is the residual of the dynamical model fit.
    """

    tc_model, ediff = jttv.get_transit_times_obs(
        pdic,
        transit_orbit_idx=jnp.array([0, 2]),
    )

    tc_model = np.asarray(tc_model, dtype=float)
    tc_obs = np.asarray(jttv.tcobs_flatten, dtype=float)
    tc_err = np.asarray(jttv.errorobs_flatten, dtype=float)

    df_obs = pd.concat(
        [
            df_b.assign(planet="b"),
            df_c.assign(planet="c"),
        ],
        ignore_index=True,
    )

    if len(df_obs) != len(tc_model):
        raise ValueError(
            f"Length mismatch: len(df_obs)={len(df_obs)}, "
            f"len(tc_model)={len(tc_model)}"
        )

    out = df_obs.copy()

    out["tc_model"] = tc_model

    # Observed - calculated/model
    out["O_minus_C_day"] = out["tc"] - out["tc_model"]
    out["O_minus_C_min"] = out["O_minus_C_day"] * 24.0 * 60.0

    # Standardized residual
    out["resid_sigma"] = out["O_minus_C_day"] / out["tcerr"]
    out["resid2"] = out["resid_sigma"] ** 2

    out["ediff"] = float(np.asarray(ediff))

    chi2 = float(np.sum(out["resid2"]))

    return out, chi2


def get_ttv_table_from_linear(jttv, pdic, df_b, df_c):
    """
    Compute observed and model TTVs relative to a linear ephemeris.

    For each planet:
        T_linear = T0 + P * tnum
        TTV_obs = tc_obs - T_linear
        TTV_model = tc_model - T_linear
    """
    tc_model, ediff = jttv.get_transit_times_obs(
        pdic,
        transit_orbit_idx=jnp.array([0, 2]),
    )

    tc_model = np.asarray(tc_model, dtype=float)

    df_obs = pd.concat(
        [
            df_b.assign(planet="b"),
            df_c.assign(planet="c"),
        ],
        ignore_index=True,
    ).copy()

    if len(df_obs) != len(tc_model):
        raise ValueError(
            f"Length mismatch: len(df_obs)={len(df_obs)}, len(tc_model)={len(tc_model)}"
        )

    df_obs["tc_model"] = tc_model

    # Fit separate linear ephemerides for b and c using observed tc.
    T0_b, P_b = fit_linear_ephemeris_from_tnum(df_b)
    T0_c, P_c = fit_linear_ephemeris_from_tnum(df_c)

    T_linear = np.empty(len(df_obs), dtype=float)

    mask_b = df_obs["planet"].to_numpy() == "b"
    mask_c = df_obs["planet"].to_numpy() == "c"

    T_linear[mask_b] = (
        T0_b + P_b * df_obs.loc[mask_b, "tnum"].to_numpy(float)
    )
    T_linear[mask_c] = (
        T0_c + P_c * df_obs.loc[mask_c, "tnum"].to_numpy(float)
    )

    df_obs["tc_linear"] = T_linear

    # TTV relative to linear ephemeris
    df_obs["ttv_obs_day"] = df_obs["tc"] - df_obs["tc_linear"]
    df_obs["ttv_model_day"] = df_obs["tc_model"] - df_obs["tc_linear"]

    df_obs["ttv_obs_min"] = df_obs["ttv_obs_day"] * 24.0 * 60.0
    df_obs["ttv_model_min"] = df_obs["ttv_model_day"] * 24.0 * 60.0

    # Residual after subtracting the TTV model
    df_obs["obs_minus_model_day"] = df_obs["tc"] - df_obs["tc_model"]
    df_obs["obs_minus_model_min"] = df_obs["obs_minus_model_day"] * 24.0 * 60.0
    df_obs["resid_sigma"] = df_obs["obs_minus_model_day"] / df_obs["tcerr"]
    df_obs["resid2"] = df_obs["resid_sigma"] ** 2

    df_obs["ediff"] = float(np.asarray(ediff))

    chi2 = float(np.sum(df_obs["resid2"]))

    return df_obs, chi2


def plot_ttv_with_jnkepler_plot_model(jttv, pdic, out_prefix):
    """
    Plot TTVs relative to linear ephemeris using jttv.plot_model().

    For TOI-560 [b,d,c], observed planets are [b,c],
    so transit_orbit_idx=[0,2] is required.
    """

    tcmodellist = jttv.get_transit_times_all_list(
        pdic,
        truncate=True,
        transit_orbit_idx=jnp.array([0, 2]),
    )

    jttv.plot_model(
        tcmodellist,
        save=out_prefix,
        unit=1440.0,
        ylabel="TTV from linear ephemeris [min]",
        xlabel="Transit time [day]",
    )


def plot_ttv_from_linear(df_ttv, title, out_png):
    """
    Plot observed and model TTVs relative to the linear ephemeris.
    """
    fig, ax = plt.subplots(figsize=(8, 4.8))

    for planet, sub in df_ttv.groupby("planet"):
        sub = sub.sort_values("tc")

        # Observed TTV points
        ax.errorbar(
            sub["tc"],
            sub["ttv_obs_min"],
            yerr=sub["tcerr"] * 24.0 * 60.0,
            fmt="o",
            ms=4,
            capsize=2,
            label=f"{planet} obs",
        )

        # Model TTV curve/points
        ax.plot(
            sub["tc"],
            sub["ttv_model_min"],
            "-",
            lw=1.5,
            label=f"{planet} model",
        )

    ax.axhline(0.0, lw=1)
    ax.set_xlabel("Time")
    ax.set_ylabel("Transit timing deviation from linear ephemeris [min]")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
# ============================================================
# HMC model
# ============================================================

def make_numpyro_model(jttv, bounds):
    tc_obs = jnp.asarray(jttv.tcobs_flatten)
    tc_err = jnp.asarray(jttv.errorobs_flatten)

    lo_period = jnp.asarray(bounds["period"][0])
    hi_period = jnp.asarray(bounds["period"][1])

    lo_tic = jnp.asarray(bounds["tic"][0])
    hi_tic = jnp.asarray(bounds["tic"][1])

    lo_ecosw = jnp.asarray(bounds["ecosw"][0])
    hi_ecosw = jnp.asarray(bounds["ecosw"][1])

    lo_esinw = jnp.asarray(bounds["esinw"][0])
    hi_esinw = jnp.asarray(bounds["esinw"][1])

    lo_lnpmass = jnp.asarray(bounds["lnpmass"][0])
    hi_lnpmass = jnp.asarray(bounds["lnpmass"][1])

    def model():
        period = numpyro.sample(
            "period",
            dist.Uniform(lo_period, hi_period).to_event(1),
        )
        tic = numpyro.sample(
            "tic",
            dist.Uniform(lo_tic, hi_tic).to_event(1),
        )
        ecosw = numpyro.sample(
            "ecosw",
            dist.Uniform(lo_ecosw, hi_ecosw).to_event(1),
        )
        esinw = numpyro.sample(
            "esinw",
            dist.Uniform(lo_esinw, hi_esinw).to_event(1),
        )
        lnpmass = numpyro.sample(
            "lnpmass",
            dist.Uniform(lo_lnpmass, hi_lnpmass).to_event(1),
        )

        pdic = {
            "period": period,
            "tic": tic,
            "ecosw": ecosw,
            "esinw": esinw,
            "pmass": jnp.exp(lnpmass),
            "smass": 1.0,
        }

        tc_model, ediff = jttv.get_transit_times_obs(
            pdic,
            transit_orbit_idx=jnp.array([0, 2]),
        )

        numpyro.deterministic("ediff", ediff)
        numpyro.deterministic(
            "eccentricity",
            jnp.sqrt(ecosw**2 + esinw**2),
        )

        numpyro.sample(
            "tcobs",
            dist.Normal(tc_model, tc_err).to_event(1),
            obs=tc_obs,
        )

    return model


def pdic_to_init_values(pdic):
    return {
        "period": jnp.asarray(pdic["period"]),
        "tic": jnp.asarray(pdic["tic"]),
        "ecosw": jnp.asarray(pdic["ecosw"]),
        "esinw": jnp.asarray(pdic["esinw"]),
        "lnpmass": jnp.log(jnp.asarray(pdic["pmass"])),
    }


def run_hmc(jttv, pdic, bounds, cfg, rng_key):
    model = make_numpyro_model(jttv, bounds)

    init_values = pdic_to_init_values(pdic)

    kernel = NUTS(
        model,
        init_strategy=init_to_value(values=init_values),
        target_accept_prob=cfg.target_accept_prob,
        max_tree_depth=cfg.max_tree_depth,
    )

    mcmc = MCMC(
        kernel,
        num_warmup=cfg.num_warmup,
        num_samples=cfg.num_samples,
        num_chains=cfg.num_chains,
        progress_bar=True,
    )

    mcmc.run(rng_key)
    mcmc.print_summary()

    return mcmc


def summarize_samples(samples):
    rows = {}

    period = np.asarray(samples["period"])
    tic = np.asarray(samples["tic"])
    ecosw = np.asarray(samples["ecosw"])
    esinw = np.asarray(samples["esinw"])
    lnpmass = np.asarray(samples["lnpmass"])
    pmass = np.exp(lnpmass)

    names = ["b", "d", "c"]

    for i, name in enumerate(names):
        e = np.sqrt(ecosw[:, i] ** 2 + esinw[:, i] ** 2)
        omega = np.degrees(np.arctan2(esinw[:, i], ecosw[:, i]))
        mass_me = pmass[:, i] / M_EARTH_IN_MSUN

        values = {
            f"P_{name}": period[:, i],
            f"tic_{name}": tic[:, i],
            f"ecosw_{name}": ecosw[:, i],
            f"esinw_{name}": esinw[:, i],
            f"e_{name}": e,
            f"omega_deg_{name}": omega,
            f"pmass_mearth_{name}": mass_me,
        }

        for key, arr in values.items():
            rows[f"{key}_median"] = np.nanmedian(arr)
            rows[f"{key}_p16"] = np.nanpercentile(arr, 16)
            rows[f"{key}_p84"] = np.nanpercentile(arr, 84)

    return rows

def sample_means_and_stds_for_plot_model(
    jttv,
    samples,
    n_draws=100,
    transit_orbit_idx=jnp.array([0, 2]),
):
    """
    Make posterior mean/std transit-time lists for jttv.plot_model().

    Model order is [b, d, c], but observed planets are [b, c],
    so transit_orbit_idx=[0,2] is required.
    """

    n_total = len(np.asarray(samples["period"]))
    n_use = min(n_draws, n_total)

    rng = np.random.default_rng(123)
    idxs = rng.choice(n_total, size=n_use, replace=False)

    tcmodel_samples = []

    for idx in idxs:
        pdic_i = {
            "period": jnp.asarray(samples["period"][idx]),
            "tic": jnp.asarray(samples["tic"][idx]),
            "ecosw": jnp.asarray(samples["ecosw"][idx]),
            "esinw": jnp.asarray(samples["esinw"][idx]),
            "pmass": jnp.exp(jnp.asarray(samples["lnpmass"][idx])),
            "smass": 1.0,
        }

        tc_list_i = jttv.get_transit_times_all_list(
            pdic_i,
            truncate=True,
            transit_orbit_idx=transit_orbit_idx,
        )

        tcmodel_samples.append([np.asarray(x, dtype=float) for x in tc_list_i])

    means = []
    stds = []

    for planet_idx in range(len(tcmodel_samples[0])):
        arr = np.array([
            tcmodel_samples[s][planet_idx]
            for s in range(n_use)
        ])

        means.append(np.mean(arr, axis=0))
        stds.append(np.std(arr, axis=0))

    return means, stds


def save_hmc_plots(jttv, mcmc, samples, cand_dir):
    """
    Save HMC diagnostic plots:
      1. posterior TTV plot with jttv.plot_model
      2. ArviZ trace plot
      3. corner plot
    """

    cand_dir = Path(cand_dir)

    # --------------------------------------------------------
    # 1. Posterior TTV plot: jttv.plot_model(means, stds)
    # --------------------------------------------------------

    means, stds = sample_means_and_stds_for_plot_model(
        jttv,
        samples,
        n_draws=100,
        transit_orbit_idx=jnp.array([0, 2]),
    )

    jttv.plot_model(
        means,
        tcmodelunclist=stds,
        save=str(cand_dir / "posterior_ttv"),
        unit=1440.0,
        xlabel="Transit time [day]",
        ylabel="TTV from linear ephemeris [min]",
    )

    # This saves:
    #   posterior_ttv_planet1.png
    #   posterior_ttv_planet2.png

    # --------------------------------------------------------
    # 2. Trace plot
    # --------------------------------------------------------

    idata = az.from_numpyro(mcmc)

    # Convert lnpmass to physical mass.
    idata.posterior["pmass"] = np.exp(idata.posterior["lnpmass"])
    idata.posterior["mu"] = idata.posterior["pmass"] / M_EARTH_IN_MSUN

    sample_keys = ["period", "tic", "ecosw", "esinw", "lnpmass"]

    axes = az.plot_trace(
        idata,
        var_names=sample_keys,
        compact=False,
    )

    fig = np.ravel(axes)[0].figure
    fig.tight_layout(pad=0.2)
    fig.savefig(cand_dir / "trace_plot.png", dpi=200)
    plt.close(fig)

    # --------------------------------------------------------
    # 3. Corner plot
    # --------------------------------------------------------

    names = ["period", "tic", "ecosw", "esinw", "mu"]

    fig = corner.corner(
        idata,
        var_names=names,
        show_titles=True,
    )

    fig.savefig(cand_dir / "corner_plot.png", dpi=200)
    plt.close(fig)

    # Save idata as well
    idata.to_netcdf(cand_dir / "posterior_inferencedata.nc")

    return idata

# ============================================================
# Main
# ============================================================

def main(cfg):
    outdir = Path(cfg.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    df_b, df_c, tcobs, tcerr = load_transit_times(cfg.transit_file)

    t0_b, Pb_lin = linear_ephemeris(tcobs[0], cfg.Pb0)
    t0_c, Pc_lin = linear_ephemeris(tcobs[1], cfg.Pc0)

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

    targets = pd.read_csv(cfg.target_csv)

    required = [
        "Pb_best", "Pd_best", "Pc_best",
        "hb_best", "hd_best", "hc_best",
        "kb_best", "kd_best", "kc_best",
        "tic_b_best", "tic_d_best", "tic_c_best",
        "mb_mearth", "md_mearth", "mc_mearth",
    ]

    missing = [c for c in required if c not in targets.columns]
    if missing:
        raise ValueError(f"target_csv is missing columns: {missing}")

    targets = targets.reset_index(drop=True)

    summary_rows = []
    rng = random.PRNGKey(cfg.rng_seed)

    # ========================================================
    # Step 1: O-C plots for all candidates
    # ========================================================

    print("\n=== Step 1: making O-C plots ===")

    for i, row in targets.iterrows():
        cand_dir = outdir / f"candidate_{i:03d}"
        cand_dir.mkdir(parents=True, exist_ok=True)

        pdic = row_to_pdic(row)
        bounds = make_bounds_around_pdic(pdic, cfg)

        df_oc, chi2 = get_oc_table(jttv, pdic, df_b, df_c)
        df_oc.to_csv(cand_dir / "oc_table.csv", index=False)

        plot_ttv_with_jnkepler_plot_model(
            jttv,
            pdic,
            out_prefix=str(cand_dir / "ttv_linear")
        )

        print(f"candidate {i}: TTV plots saved, chi2={chi2:.3f}")

        summary_rows.append({
            "candidate_index": i,
            "chi2_before_hmc": chi2,
            "oc_plot": str(cand_dir / "oc_plot.png"),
            "oc_table": str(cand_dir / "oc_table.csv"),
        })

    pd.DataFrame(summary_rows).to_csv(outdir / "oc_summary.csv", index=False)

    # ========================================================
    # Step 2: HMC for all candidates
    # ========================================================

    print("\n=== Step 2: running HMC/NUTS ===")

    hmc_rows = []

    for i, row in targets.iterrows():
        print("\n" + "=" * 80)
        print(f"Running HMC for candidate {i}")

        cand_dir = outdir / f"candidate_{i:03d}"
        cand_dir.mkdir(parents=True, exist_ok=True)

        pdic = row_to_pdic(row)
        bounds = make_bounds_around_pdic(pdic, cfg)

        rng, subkey = random.split(rng)

        try:
            mcmc = run_hmc(jttv, pdic, bounds, cfg, subkey)

            samples = mcmc.get_samples(group_by_chain=False)
            extra = mcmc.get_extra_fields(group_by_chain=False)

            np.savez(
                cand_dir / "posterior_samples.npz",
                **{k: np.asarray(v) for k, v in samples.items()},
            )

            post_summary = summarize_samples(samples)
            pd.DataFrame([post_summary]).to_csv(
                cand_dir / "posterior_summary.csv",
                index=False,
            )

            # Save posterior TTV, trace, and corner plots
            idata = save_hmc_plots(
                jttv,
                mcmc,
                samples,
                cand_dir,
            )

            if "diverging" in extra:
                div = np.asarray(extra["diverging"])
                n_div = int(np.sum(div))
                frac_div = float(np.mean(div))
            else:
                n_div = np.nan
                frac_div = np.nan

            hmc_success = True
            message = "ok"

        except Exception as e:
            post_summary = {}
            n_div = np.nan
            frac_div = np.nan
            hmc_success = False
            message = repr(e)

        hmc_row = {
            "candidate_index": i,
            "hmc_success": hmc_success,
            "message": message,
            "n_divergent": n_div,
            "frac_divergent": frac_div,
        }
        hmc_row.update(post_summary)

        hmc_rows.append(hmc_row)

        pd.DataFrame(hmc_rows).to_csv(outdir / "hmc_summary.csv", index=False)

        print(f"candidate {i}: HMC success={hmc_success}, message={message}")

    print("\nDone.")
    print("Saved:", outdir)


if __name__ == "__main__":
    main(CFG)