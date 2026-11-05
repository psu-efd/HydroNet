"""
Case 4 (Savannah River): longitudinal section, FVM-PINN runs vs SRH-2D at
t = T: (a) plan view with the line; (b) water-surface elevation and
(c) its difference from SRH-2D, (d) velocity magnitude and (e) its
difference, along the thalweg (default) or centreline; (f) discharge through
bank-to-bank cross-sections relative to SRH-2D's discharge through the same
section, a check of mass conservation along the reach.

The centreline is the locus of cells equidistant (geodesic distance through
the mesh) from the two banks (BC 3 and BC 4), ordered by the normalised
distance from the inlet (BC 1) to the outlet (BC 2), smoothed with a spline
and resampled every ``--ds`` m. The thalweg is the lowest bed point of
cross-sections normal to the centreline every 10 m, smoothed (Savitzky-Golay
over 21 sections, then the same spline); it depends on the bathymetry only.
Below the bend it lies 20-40 m toward the outer bank of the centreline,
with the velocity core. Cell-centred fields (SRH-2D and the
networks, evaluated at cell centres as in build_manuscript_figures.py) are
averaged to the nodes and interpolated linearly on the mesh triangulation,
so all curves use the same sampling operator.

Q(s) integrates the normal unit discharge over sections normal to the
centreline every 10 m (the end 20 m are skipped: those sections cross the
oblique open boundaries). Each section is placed at the s where it crosses
the plotted line. SRH-2D's own Q(s)/Q_in deviates from 1 by about +/-2%;
this is the accuracy of the section sampling.

Checkpoints default to the A100 timing tree, whose runs give the reported
Table 3 numbers.

Usage
-----
    python examples/FVM_PINN/fig_savannah_longitudinal.py                      # recommended runs, thalweg
    python examples/FVM_PINN/fig_savannah_longitudinal.py --line centreline --out fig_savannah_longitudinal_cl
    python examples/FVM_PINN/fig_savannah_longitudinal.py --runs SR_A SR_B SR_C SR_D SR_E SR_F SR_G \\
        --out fig_savannah_longitudinal_all
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from scipy.interpolate import splev, splprep
from scipy.signal import savgol_filter
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

import build_manuscript_figures as bmf

logger = logging.getLogger(__name__)

A100_SR_DIR = Path("/storage/work/axt5780/Projects/Hydro/HydroNet_a100_timing/examples/FVM_PINN/savannah_river")
BC_INLET, BC_OUTLET, BC_BANK_L, BC_BANK_R = 1, 2, 3, 4

# Same colours as fig_savannah_l2_by_time for SR-A..E
STYLE = {
    "SR_A": ("#7f7f7f", "--", "SR-A (physics only)"),
    "SR_B": ("#d62728", "-",  "SR-B (standard)"),
    "SR_C": ("#1f77b4", "-",  "SR-C (window(5))"),
    "SR_D": ("#2ca02c", "-",  "SR-D (window(10))"),
    "SR_E": ("#ff7f0e", "-",  "SR-E (FVM teacher)"),
    "SR_F": ("#9467bd", "-.", "SR-F (200 sparse velocities)"),
    "SR_G": ("#8c564b", ":",  r"SR-G ($\lambda_{\mathrm{fvm\mbox{-}pinn}}=0$)"),
    # Strong-form PINN baselines: colour of the FVM-PINN counterpart, dotted
    "SR_A_SF": ("#7f7f7f", ":", "SR-A-SF (strong form, physics only)"),
    "SR_F_SF": ("#9467bd", ":", "SR-F-SF (strong form, 200 sparse)"),
    "SR_B_SF": ("#d62728", ":", "SR-B-SF (strong form, standard)"),
}
# Strong-form runs live in the main repo (not the A100 timing tree)
SF_RUNS = [(f"SR_{k}_SF", f"fvm_pinn_config_SR_{k}_SF.yaml", "ckpt_final.pt") for k in "AFB"]
# SR-B/C/E/G overlap SR-D at plot scale (thalweg RMS <= 0.033 m, 0.022 m/s)
DEFAULT_RUNS = ["SR_A", "SR_F", "SR_D"]


def centreline(md, ds_step: float):
    """Smoothed centreline polyline [n, 2] and its arc length s [n]."""
    fl, fr = md["face_left"].numpy(), md["face_right"].numpy()
    bid, xc = md["face_bc_id"].numpy(), md["cell_center"].numpy()
    n = int(md["n_cells"])
    inner = fr >= 0
    a, b = fl[inner], fr[inner]
    w = np.linalg.norm(xc[a] - xc[b], axis=1)
    graph = coo_matrix((np.r_[w, w], (np.r_[a, b], np.r_[b, a])), shape=(n, n)).tocsr()

    def dist_from(bc):
        src = np.unique(fl[(~inner) & (bid == bc)])
        return dijkstra(graph, indices=src, min_only=True)

    d_in, d_out = dist_from(BC_INLET), dist_from(BC_OUTLET)
    d_l, d_r = dist_from(BC_BANK_L), dist_from(BC_BANK_R)
    s_norm = d_in / (d_in + d_out)

    # Most bank-equidistant cell in each streamwise bin
    n_bins = 60
    pts = []
    for k in range(n_bins):
        m = (s_norm >= k / n_bins) & (s_norm < (k + 1) / n_bins)
        if m.any():
            idx = np.where(m)[0]
            pts.append(xc[idx[np.argmin(np.abs(d_l[idx] - d_r[idx]))]])
    fc = md["face_center"].numpy()
    start = fc[(~inner) & (bid == BC_INLET)].mean(0)
    end = fc[(~inner) & (bid == BC_OUTLET)].mean(0)
    return _resample(np.vstack([start, np.array(pts), end]), ds_step)


def _resample(pts, ds_step: float):
    """Spline through ordered points [n, 2], resampled every ds_step m along its arc."""
    tck, _ = splprep(pts.T, s=len(pts) * 25.0)          # light smoothing (m^2 per point)
    fine = np.array(splev(np.linspace(0, 1, 4000), tck)).T
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(fine, axis=0), axis=1))]
    s = np.arange(0.0, arc[-1], ds_step)
    line = np.column_stack([np.interp(s, arc, fine[:, 0]), np.interp(s, arc, fine[:, 1])])
    return line, s


def thalweg(md, ref, triang, ds_step: float, spacing: float = 10.0, half_width: float = 150.0):
    """Smoothed thalweg polyline [n, 2] and its arc length s [n]."""
    cl, s_cl = centreline(md, 1.0)
    bed = mtri.LinearTriInterpolator(
        triang, bmf._cell_to_node(ref["node_xy"], ref["cell_nodes"], md["bed_elev"].numpy()))
    finder = triang.get_trifinder()
    tang = np.gradient(cl, axis=0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True)
    nrm = np.column_stack([-tang[:, 1], tang[:, 0]])
    eta = np.arange(-half_width, half_width + 0.01, 1.0)
    c = int(np.argmin(np.abs(eta)))
    pts = []
    for i in range(0, len(s_cl), int(round(spacing))):
        sec = cl[i] + eta[:, None] * nrm[i]
        inside = finder(sec[:, 0], sec[:, 1]) >= 0
        if not inside[c]:
            continue
        lo, hi = c, c                       # contiguous wet section through the centreline
        while lo > 0 and inside[lo - 1]:
            lo -= 1
        while hi < len(eta) - 1 and inside[hi + 1]:
            hi += 1
        zb = np.ma.filled(bed(sec[lo:hi + 1, 0], sec[lo:hi + 1, 1]), np.inf)
        pts.append(sec[lo + int(np.argmin(zb))])
    pts = np.array(pts)
    k = min(21, (len(pts) // 2) * 2 - 1)
    pts = np.column_stack([savgol_filter(pts[:, 0], k, 2), savgol_filter(pts[:, 1], k, 2)])
    return _resample(pts, ds_step)


def sample(triang, ref, field_cells, line):
    nv = bmf._cell_to_node(ref["node_xy"], ref["cell_nodes"], field_cells)
    return np.ma.filled(mtri.LinearTriInterpolator(triang, nv)(line[:, 0], line[:, 1]), np.nan)


def section_discharge(md, ref, preds, runs, triang, line, s_line,
                      spacing: float = 10.0, trim: float = 20.0, half_width: float = 150.0):
    """Discharge through bank-to-bank sections normal to the centreline.

    Returns s [n] (position along ``line`` where each section crosses it,
    sorted) and {run: Q [n]} in m^3/s.
    """
    cl, s_cl = centreline(md, 1.0)
    finder = triang.get_trifinder()
    tang = np.gradient(cl, axis=0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True)
    nrm = np.column_stack([-tang[:, 1], tang[:, 0]])
    d_eta = 0.5
    eta = np.arange(-half_width, half_width + 0.01, d_eta)
    c = int(np.argmin(np.abs(eta)))
    node = lambda f: bmf._cell_to_node(ref["node_xy"], ref["cell_nodes"], f)
    interp = {r: (mtri.LinearTriInterpolator(triang, node(preds[r]["h"] * preds[r]["u"])),
                  mtri.LinearTriInterpolator(triang, node(preds[r]["h"] * preds[r]["v"])))
              for r in runs}
    S, Q = [], {r: [] for r in runs}
    for i in range(0, len(s_cl), int(round(spacing))):
        if s_cl[i] < trim or s_cl[i] > s_cl[-1] - trim:
            continue
        sec = cl[i] + eta[:, None] * nrm[i]
        inside = finder(sec[:, 0], sec[:, 1]) >= 0
        if not inside[c]:
            continue
        lo, hi = c, c
        while lo > 0 and inside[lo - 1]:
            lo -= 1
        while hi < len(eta) - 1 and inside[hi + 1]:
            hi += 1
        pts = sec[lo:hi + 1]
        dist = np.linalg.norm(line[:, None, :] - pts[None, ::4, :], axis=2)
        S.append(s_line[np.unravel_index(np.argmin(dist), dist.shape)[0]])
        for r in runs:
            qx = np.ma.filled(interp[r][0](pts[:, 0], pts[:, 1]), 0.0)
            qy = np.ma.filled(interp[r][1](pts[:, 0], pts[:, 1]), 0.0)
            Q[r].append(np.sum(qx * tang[i, 0] + qy * tang[i, 1]) * d_eta)
    order = np.argsort(S)
    return np.array(S)[order], {r: np.array(q)[order] for r, q in Q.items()}


def inlet_discharge(case_dir: Path) -> float:
    cfg = bmf.Config(str(case_dir / bmf.SR_RUNS[0][1]))
    for bc in (cfg.get("boundary_conditions", {}) or {}).values():
        if bc.get("type") == "inlet-q":
            return float(bc["value"])
    raise ValueError("no inlet-q boundary condition in the config")


def mark_off_scale(ax, rids, x_frac=0.03):
    """One note at the bottom of ``ax`` for the runs whose curves lie below the axis."""
    if not rids:
        return
    rids = sorted(rids)
    label = ", ".join(STYLE[r][2].split(" (")[0] for r in rids)
    color = STYLE[rids[0]][0]
    ax.annotate(rf"{label}: off scale (below)", xy=(x_frac, 0.0), xycoords="axes fraction",
                xytext=(x_frac, 0.13), textcoords="axes fraction", fontsize=12, color=color,
                arrowprops=dict(arrowstyle="->", color=color, lw=1.4))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--runs", nargs="+", default=DEFAULT_RUNS, choices=list(STYLE))
    p.add_argument("--case-dir", type=Path, default=A100_SR_DIR)
    p.add_argument("--line", choices=["thalweg", "centreline"], default="thalweg")
    p.add_argument("--ds", type=float, default=2.0, help="Sampling interval along the line [m]")
    p.add_argument("--out", default="fig_savannah_longitudinal")
    p.add_argument("--regenerate", action="store_true", help="Re-predict instead of using the cache")
    args = p.parse_args()

    cache = bmf.CACHE_DIR / f"sr_predictions_{args.case_dir.parents[2].name}.npz"
    preds = bmf.collect_predictions(args.case_dir, bmf.SR_RUNS, bmf.SR_T_END, cache,
                                    regenerate=args.regenerate)
    sf_wanted = [r for r in SF_RUNS if r[0] in args.runs]
    if sf_wanted:
        sf = bmf.collect_predictions(bmf.SR_DIR, SF_RUNS, bmf.SR_T_END,
                                     bmf.CACHE_DIR / "sr_predictions_strong_form.npz",
                                     regenerate=args.regenerate)
        preds.update({k: v for k, v in sf.items() if k != "_ref"})
    ref = preds["_ref"]
    _, ds = bmf._load_run(args.case_dir, bmf.SR_RUNS[0][1], bmf.SR_RUNS[0][2])
    md = ds.get_mesh_data()
    zb = md["bed_elev"].numpy()
    triang = bmf._triangulation_from_meta(ref["node_xy"], ref["cell_nodes"])
    line, s = (thalweg(md, ref, triang, args.ds) if args.line == "thalweg"
               else centreline(md, args.ds))
    speed = lambda d: np.sqrt(d["u"] ** 2 + d["v"] ** 2)
    runs = [r for r in args.runs if preds.get(r) is not None]
    for r in set(args.runs) - set(runs):
        logger.warning("%s has no predictions; skipped", r)

    q_in = inlet_discharge(args.case_dir)
    s_q, Q = section_discharge(md, ref, preds, ["_ref"] + runs, triang, line, s)
    wse = {r: sample(triang, ref, preds[r]["h"] + zb, line) for r in ["_ref"] + runs}
    vel = {r: sample(triang, ref, speed(preds[r]), line) for r in ["_ref"] + runs}
    for r in runs:
        wd = wse[r][np.isfinite(wse[r])]
        logger.info("%s along %s: RMS WSE %.3f m (drop %+.3f m), |u| %.3f m/s; Q/Q_in mean %.3f, "
                    "max |Q - Q_SRH|/Q_SRH %.1f%%", r, args.line,
                    np.sqrt(np.nanmean((wse[r] - wse["_ref"]) ** 2)), wd[0] - wd[-1],
                    np.sqrt(np.nanmean((vel[r] - vel["_ref"]) ** 2)), Q[r].mean() / q_in,
                    100 * np.max(np.abs(Q[r] - Q["_ref"]) / Q["_ref"]))

    # Layout: (a) plan view; (b) z_s with (c) its difference from SRH-2D;
    # (d) |u| with (e) its difference; (f) section discharge vs SRH-2D.
    # Panel (a) is sized so the plan view (aspect ~2:1) fills the full width.
    fig = plt.figure(figsize=(11, 19), constrained_layout=True)
    gs = fig.add_gridspec(6, 1, height_ratios=[1.75, 1, 0.55, 1, 0.55, 0.85])
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1]); ax1d = fig.add_subplot(gs[2], sharex=ax1)
    ax2 = fig.add_subplot(gs[3]); ax2d = fig.add_subplot(gs[4], sharex=ax2)
    ax3 = fig.add_subplot(gs[5], sharex=ax1)

    # (a) plan view: SRH-2D |u| with the line and distance ticks; the
    # colourbar sits in the empty land area at the lower left of the map
    tcf = ax0.tricontourf(triang, bmf._cell_to_node(ref["node_xy"], ref["cell_nodes"], speed(ref)),
                          levels=25, cmap="RdBu_r")
    cax = ax0.inset_axes([0.05, 0.16, 0.32, 0.045])
    cb = fig.colorbar(tcf, cax=cax, orientation="horizontal", ticks=np.arange(0.0, 1.21, 0.2))
    cb.set_label(r"$|\mathbf{u}|$ (m\,s$^{-1}$)", fontsize=16); cb.ax.tick_params(labelsize=12)
    ax0.plot(line[:, 0], line[:, 1], "k-", lw=1.6)
    for sk in np.arange(0.0, s[-1], 200.0):
        i = int(np.argmin(np.abs(s - sk)))
        ax0.plot(*line[i], "ko", ms=5)
        # Labels near the top of the frame go below their point
        top = line[i, 1] > 0.9 * line[:, 1].max()
        ax0.annotate(rf"${sk:.0f}$", line[i], xytext=(7, -17 if top else 7), textcoords="offset points", fontsize=13,
                     bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
    ax0.set_aspect("equal")
    ax0.set_xlabel(r"$x$ (m)", fontsize=18); ax0.set_ylabel(r"$y$ (m)", fontsize=18)
    ax0.set_title(rf"SRH-2D $|\mathbf{{u}}|$ at $t = {bmf.SR_T_END:.0f}$~s; {args.line} with $s$ (m)",
                  fontsize=16)
    ax0.tick_params(labelsize=14)

    def style(ax, ylab, title=None, xlabel=True):
        ax.set_xlim(0, s[-1]); ax.set_ylabel(ylab, fontsize=16)
        if title:
            ax.set_title(title, fontsize=16)
        if xlabel:
            ax.set_xlabel(r"$s$ (m)", fontsize=18)
        else:
            ax.tick_params(labelbottom=False)
        ax.tick_params(labelsize=13); ax.grid(True, alpha=0.3)

    # Profiles: axis limits follow SRH-2D and the runs that stay near it; a
    # run far below (SR-A) is marked off scale instead of setting the axis.
    def plot_profile(ax, x, data, lim=None):
        ax.plot(x, data["_ref"], "k-", lw=2.2, label="SRH-2D")
        for r in runs:
            color, ls, lab = STYLE[r]
            ax.plot(x, data[r], color=color, ls=ls, lw=1.6, label=lab)
        off = set()
        if lim is not None:
            ax.set_ylim(*lim)
            off = {r for r in runs if np.nanmedian(data[r]) < lim[0]}
            mark_off_scale(ax, off)
        return off

    # Differences from SRH-2D (zero line = SRH-2D); runs in ``off`` are noted
    # instead of plotted so they do not set the scale.
    def plot_diff(ax, x, diff, off=()):
        ax.axhline(0.0, color="k", lw=1.6)
        shown = [r for r in runs if r not in off]
        for r in shown:
            color, ls, _ = STYLE[r]
            ax.plot(x, diff[r], color=color, ls=ls, lw=1.4)
        if shown:
            vals = np.concatenate([diff[r] for r in shown])
            lo, hi = np.nanmin(vals), np.nanmax(vals)
            pad = 0.12 * max(hi - lo, 1e-9)
            ax.set_ylim(min(lo, 0.0) - pad, max(hi, 0.0) + pad)
        mark_off_scale(ax, off)

    # (b)/(c) water-surface elevation and its difference (cm); headroom in
    # (b) for the legend (two rows when many runs)
    w_all = np.concatenate([wse[r] for r in ["_ref"] + runs])
    w_lo, w_hi = np.nanmin(w_all), np.nanmax(w_all)
    plot_profile(ax1, s, wse, lim=(w_lo - 0.02, w_hi + (0.45 if len(runs) > 3 else 0.3) * (w_hi - w_lo)))
    style(ax1, r"$z_s$ (m)", f"Water-surface elevation along the {args.line}", xlabel=False)
    plot_diff(ax1d, s, {r: 100.0 * (wse[r] - wse["_ref"]) for r in runs})
    style(ax1d, r"$\Delta z_s$ (cm)")

    # (d)/(e) velocity magnitude and its difference (m/s)
    v_ok = np.concatenate([vel[r] for r in ["_ref"] + runs if np.nanmedian(vel[r]) > 0.2])
    off_v = plot_profile(ax2, s, vel, lim=(np.floor(np.nanmin(v_ok) * 10) / 10 - 0.05, np.nanmax(v_ok) + 0.05))
    style(ax2, r"$|\mathbf{u}|$ (m/s)", f"Velocity magnitude along the {args.line}", xlabel=False)
    plot_diff(ax2d, s, {r: vel[r] - vel["_ref"] for r in runs}, off=off_v)
    style(ax2d, r"$\Delta|\mathbf{u}|$ (m/s)")

    # (f) discharge through each cross-section relative to SRH-2D's discharge
    # through the same section (cancels the section-sampling error)
    dq = {r: 100.0 * (Q[r] - Q["_ref"]) / Q["_ref"] for r in runs}
    plot_diff(ax3, s_q, dq, off={r for r in runs if abs(np.median(dq[r])) > 50.0})
    style(ax3, r"$\Delta Q/Q_{\mathrm{SRH}}$ (\%)",
          rf"Discharge through cross-sections vs SRH-2D ($Q_{{\mathrm{{in}}}} = {q_in:.1f}$~m$^3$\,s$^{{-1}}$)")

    many = len(runs) > 3
    ax1.legend(fontsize=10.5 if many else 12, loc="upper right", ncol=4 if many else 2,
               framealpha=0.9, columnspacing=1.0 if many else 2.0)
    for ax, lab in zip((ax0, ax1, ax2, ax3), ("(a)", "(b)", "(d)", "(f)")):
        ax.text(-0.08, 1.04, lab, transform=ax.transAxes, fontsize=18)
    # Difference strips have no title row: label inside the top-left corner
    for ax, lab in zip((ax1d, ax2d), ("(c)", "(e)")):
        ax.text(0.008, 0.94, lab, transform=ax.transAxes, fontsize=16, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))

    out = bmf.SR_OUT_DIR / args.out
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s.{pdf,png}  (%s length %.0f m, %d sections)", out, args.line, s[-1], len(s_q))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
