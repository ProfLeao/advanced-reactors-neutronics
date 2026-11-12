"""Plot neutron-flux, spectrum, pin-power, and temperature-feedback results."""
import os
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    HAS_MPL = True
except Exception:
    HAS_MPL = False

from .peaking import compute_pin_powers

# Fronteiras espectrais convencionais (eV)
E_THERMAL_TOP = 0.625      # eV
E_FAST_BOTTOM = 1.0e5      # eV (0.1 MeV)


def _setup(outdir):
    os.makedirs(outdir, exist_ok=True)


def plot_flux_slices(flux: np.ndarray, cfg: dict, outdir: str,
                     flux_rel: np.ndarray = None) -> str:
    """Plot XY/XZ flux slices and the axial profile, with optional uncertainty."""
    if not HAS_MPL:
        return ""
    _setup(outdir)
    nx, ny, nz = flux.shape
    mid_z, mid_y = nz // 2, ny // 2
    half = cfg["core"]["hex_side_cm"] * 2.0
    hz = cfg["core"]["active_height_cm"] / 2.0

    if flux_rel is not None:
        fig, axes = plt.subplots(2, 2, figsize=(13, 11))
        axes = axes.ravel()
    else:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    im0 = axes[0].imshow(flux[:, :, mid_z].T, origin="lower", cmap="viridis",
                         aspect="equal",
                         extent=[-half, half, -half, half])
    axes[0].set_title("Neutron Flux — XY Slice")
    axes[0].set_xlabel("x (cm)"); axes[0].set_ylabel("y (cm)")
    plt.colorbar(im0, ax=axes[0], label="flux (a.u.)")

    im1 = axes[1].imshow(flux[:, mid_y, :].T, origin="lower", cmap="viridis",
                         aspect="auto",
                         extent=[-half, half, -hz, hz])
    axes[1].set_title("Neutron Flux — XZ Slice")
    axes[1].set_xlabel("x (cm)"); axes[1].set_ylabel("z (cm)")
    plt.colorbar(im1, ax=axes[1], label="flux (a.u.)")

    axz = flux.mean(axis=(0, 1))
    zz = np.linspace(-hz, hz, nz)
    axes[2].plot(zz, axz / axz.max(), "r-", linewidth=2)
    if flux_rel is not None:
        sig_axz = np.sqrt((((flux_rel * flux) ** 2).sum(axis=(0, 1)))) / nx / ny
        band = sig_axz / axz.max()
        axes[2].fill_between(zz, (axz / axz.max()) - band,
                             (axz / axz.max()) + band, alpha=0.25,
                             color="tab:red", label=r"±1$\sigma$")
        axes[2].legend(fontsize=8)
    axes[2].set_title("Axial Flux Profile")
    axes[2].set_xlabel("z (cm)"); axes[2].set_ylabel("Normalized flux")
    axes[2].grid(True, alpha=0.3)

    if flux_rel is not None:
        rel = np.ma.masked_less_equal(flux_rel[:, :, mid_z].T * 100.0, 0.0)
        cmr = plt.get_cmap("inferno").copy()
        cmr.set_bad("#e8e8e8")
        im3 = axes[3].imshow(rel, origin="lower", cmap=cmr,
                             aspect="equal", extent=[-half, half, -half, half],
                             norm=LogNorm(vmin=0.1, vmax=10.0))
        axes[3].set_title("Relative uncertainty of flux (log)")
        axes[3].set_xlabel("x (cm)"); axes[3].set_ylabel("y (cm)")
        plt.colorbar(im3, ax=axes[3], label="rel. error (%), 1\u03c3")

    fig.tight_layout()
    path = os.path.join(outdir, "flux_distribution.png")
    fig.savefig(path, dpi=cfg["output"]["dpi"]); plt.close(fig)
    print(f"  [OK] {path}")
    return path


def plot_flux_group_maps(flux_th: np.ndarray, flux_fast: np.ndarray,
                         cfg: dict, outdir: str,
                         rel_fast: np.ndarray = None) -> str:
    """Plot thermal and fast flux maps and the fast-flux fraction.

    Thermal flux can be statistically unresolved in a fast-spectrum UN core.
    """
    if not HAS_MPL:
        return ""
    _setup(outdir)
    nx, ny, nz = flux_th.shape
    mid_z = nz // 2
    half = cfg["core"]["hex_side_cm"] * 2.0

    tot = flux_th + flux_fast
    # Fast-flux fraction, masked where total flux is negligible.
    frac_fast = np.where(tot > 1e-4 * tot.max(),
                         flux_fast / np.maximum(tot, 1e-30), np.nan)

    n_panels = 4 if rel_fast is not None else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    ims = []
    for ax, data, title, cmap, label in [
            (axes[0], flux_th[:, :, mid_z].T,
             "Thermal flux (< 0.625 eV)", "Blues", "flux (a.u.)"),
            (axes[1], flux_fast[:, :, mid_z].T,
             "Fast flux (> 0.625 eV)", "Reds", "flux (a.u.)"),
            (axes[2], frac_fast[:, :, mid_z].T,
             "Fast fraction of flux", "viridis", "fraction")]:
        im = ax.imshow(data, origin="lower", cmap=cmap, aspect="equal",
                       extent=[-half, half, -half, half])
        ax.set_title(title)
        ax.set_xlabel("x (cm)"); ax.set_ylabel("y (cm)")
        ims.append(im)
        plt.colorbar(im, ax=ax, label=label)

    if rel_fast is not None:
        rel = np.ma.masked_less_equal(rel_fast[:, :, mid_z].T * 100.0, 0.0)
        cmr = plt.get_cmap("inferno").copy()
        cmr.set_bad("#e8e8e8")
        im = axes[3].imshow(rel, origin="lower", cmap=cmr,
                            aspect="equal", extent=[-half, half, -half, half],
                            norm=LogNorm(vmin=0.1, vmax=10.0))
        axes[3].set_title("Relative uncertainty of fast flux (log)")
        axes[3].set_xlabel("x (cm)"); axes[3].set_ylabel("y (cm)")
        plt.colorbar(im, ax=axes[3], label="rel. error (%), 1\u03c3")

    fig.tight_layout()
    path = os.path.join(outdir, "flux_group_maps.png")
    fig.savefig(path, dpi=cfg["output"]["dpi"]); plt.close(fig)
    print(f"  [OK] {path}")
    return path


def plot_spectrum(e_edges: np.ndarray, phi: np.ndarray,
                  cfg: dict, outdir: str,
                  phi_std: np.ndarray = None) -> str:
    """Plot neutron flux per unit lethargy on log-log axes.

    ``phi`` is the group flux. ``phi_std`` optionally supplies one-sigma
    uncertainty for error bands and unresolved-group markers.
    """
    if not HAS_MPL:
        return ""
    _setup(outdir)
    e_mid = np.sqrt(e_edges[:-1] * e_edges[1:])
    du = np.log(e_edges[1:] / e_edges[:-1])
    phi = np.asarray(phi, dtype=float)
    norm = (phi / du).max()
    phi_u = (phi / du) / norm

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogx(e_mid, phi_u, "b-", linewidth=2)
    if phi_std is not None:
        std = np.asarray(phi_std, dtype=float)
        sig_u = (std / du) / norm
        ax.fill_between(e_mid, np.clip(phi_u - sig_u, 0.0, None),
                        phi_u + sig_u, color="tab:blue", alpha=0.25,
                        label=r"±1$\sigma$ (statistical)")
        rel = np.divide(std, phi, out=np.full_like(std, np.inf),
                        where=phi > 0)
        noisy = rel > 0.10
        if np.any(noisy):
            ax.plot(e_mid[noisy], phi_u[noisy], "o", mfc="none", mec="red",
                    ms=4, label="rel. error > 10% (not resolved)")
        ax.legend(loc="upper left", fontsize=8)
    # Mark conventional energy ranges.
    for lo, hi, name in [(1e-5, E_THERMAL_TOP, "thermal"),
                         (E_THERMAL_TOP, E_FAST_BOTTOM, "epithermal"),
                         (E_FAST_BOTTOM, 2e7, "fast")]:
        ax.axvspan(lo, hi, alpha=0.08)
        ax.text(np.sqrt(lo * hi), 0.95, name, ha="center", va="top",
                fontsize=9, alpha=0.7)
    ax.axvline(0.625, color="gray", ls="--", lw=1)
    ax.set_title("Neutron Energy Spectrum (per unit lethargy)")
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Normalized flux per unit lethargy")
    ax.set_xlim(1e-5, 2e7)
    ax.set_ylim(0, 1.05)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    path = os.path.join(outdir, "neutron_spectrum.png")
    fig.savefig(path, dpi=cfg["output"]["dpi"]); plt.close(fig)
    print(f"  [OK] {path}")
    return path


def plot_pin_power_map(power: np.ndarray, cfg: dict, outdir: str,
                       pin_sigma: np.ndarray = None,
                       peaking_sigma: float = None) -> str:
    """Plot relative pin power, with optional uncertainty.

    ``pin_sigma`` contains one-sigma relative pin-power uncertainties;
    ``peaking_sigma`` is the one-sigma uncertainty of the peaking factor.
    """
    if not HAS_MPL:
        return ""
    _setup(outdir)
    pm = compute_pin_powers(power, cfg)
    x, y, ring, pnorm = pm["x"], pm["y"], pm["ring"], pm["pin_power"]

    fig, ax = plt.subplots(figsize=(9, 9))
    sc = ax.scatter(x, y, s=380, c=pnorm, cmap="plasma",
                    edgecolors="k", linewidths=0.4, vmin=0.0,
                    vmax=max(1.0, pnorm.max()))
    # Show heat-pipe positions as gray markers.
    try:
        from .geometry import generate_hex_positions_aziz
        hps = [(xx, yy) for xx, yy, t, _ in generate_hex_positions_aziz(
            cfg["core"]["pitch_cm"]) if t == "heatpipe"]
        ax.scatter([p[0] for p in hps], [p[1] for p in hps],
                   s=150, marker="s", color="0.75", edgecolors="0.4",
                   linewidths=0.4, label="heat pipe")
    except Exception:
        pass
    ax.set_aspect("equal")
    title = (f"Pin Power Map — relative to core mean "
             f"(pin peaking = {pm['peaking']:.3f}")
    title += (f" ± {peaking_sigma:.4f})" if peaking_sigma
              else ")")
    ax.set_title(title)
    if pin_sigma is not None:
        ip = int(np.argmax(pnorm))
        ax.annotate(f"hottest: {pnorm[ip]:.3f} ± {pin_sigma[ip]:.4f}",
                    (x[ip], y[ip]), textcoords="offset points",
                    xytext=(10, -16), fontsize=8, color="k")
    ax.set_xlabel("x (cm)"); ax.set_ylabel("y (cm)")
    plt.colorbar(sc, ax=ax, label="Relative pin power")
    ax.legend(loc="upper right")
    fig.tight_layout()
    path = os.path.join(outdir, "pin_power_map.png")
    fig.savefig(path, dpi=cfg["output"]["dpi"]); plt.close(fig)
    print(f"  [OK] {path}")
    return path


def plot_reactivity_temperature(temps_k, keffs, kerrs, cfg, outdir) -> str:
    """Plot k-effective and reactivity versus fuel temperature.

    temps_k: temperaturas (K); keffs: k_eff; kerrs: incertezas (1σ).
    """
    if not HAS_MPL:
        return ""
    _setup(outdir)
    temps_k = np.asarray(temps_k, float)
    keffs = np.asarray(keffs, float)
    kerrs = np.asarray(kerrs, float) if kerrs is not None else None

    rho = (keffs - 1.0) / keffs * 1e5  # pcm

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.errorbar(temps_k, keffs, yerr=kerrs, fmt="o-", color="tab:blue",
                 capsize=4)
    ax1.axhline(1.0, color="k", ls="--", lw=1)
    ax1.set_title("$k_{eff}$ vs Fuel Temperature")
    ax1.set_xlabel("Temperature (K)"); ax1.set_ylabel("$k_{eff}$")
    ax1.grid(True, alpha=0.3)

    ax2.plot(temps_k, rho, "o-", color="tab:red")
    if len(temps_k) >= 2:
        coef = np.polyfit(temps_k, rho, 1)[0]  # pcm/K
        ax2.plot(temps_k, np.polyval(np.polyfit(temps_k, rho, 1), temps_k),
                 "--", color="gray", alpha=0.7)
        ax2.set_title(f"Isothermal Temp. Coeff. = {coef:.3f} pcm/K")
    ax2.set_xlabel("Temperature (K)"); ax2.set_ylabel("Reactivity (pcm)")
    ax2.axhline(0, color="k", ls="--", lw=1)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(outdir, "reactivity_temperature.png")
    fig.savefig(path, dpi=cfg["output"]["dpi"]); plt.close(fig)
    print(f"  [OK] {path}")
    return path
