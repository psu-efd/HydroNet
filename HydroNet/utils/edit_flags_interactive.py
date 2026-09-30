#!/usr/bin/env python3
"""Interactive PINN / FVM-PINN data flag editor.

Usage:
    python edit_flags_interactive.py [dir_path]
    python edit_flags_interactive.py --fvm-config fvm_pinn_config.yaml

    dir_path: directory containing data_flags.npy and data_points.npy.
              Defaults to the current working directory.

    --fvm-config: FVM-PINN YAML with data.measurements.flags_file set.
              Run from the case directory (YAML paths are relative). One
              point per mesh cell (its centre), flags [xi, u|hu, v|hv],
              drawn over the SRH-2D wet-cell speed at the last measurement
              time. If flags_file does not exist yet, all cells start
              unflagged. Save writes (n_cells, 3) flags to flags_file
              (.npy, or .csv / .txt as integer text).

Controls:
    - Drag (left mouse) on the plot to select points inside a rectangle.
    - Use radio buttons to choose which variable to visualize (h, u, v).
    - Use checkboxes to choose which flags to modify (h, u, v).
    - Random % slider: only a random subset of selected points will be affected.
    - Toggle: flip the flag (0->1, 1->0) for the target subset.
    - Enable / Disable: force flags to 1 or 0 for the target subset.
    - Select All / Unselect All: modify the selection.
    - Undo: revert the last flag modification (up to 50 steps).
    - Save: overwrite data_flags.npy in dir_path.
"""

import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.widgets as mwidgets
import numpy as np
from matplotlib.lines import Line2D

FLAG_NAMES = ["h", "u", "v"]
COLOR_ACTIVE = "steelblue"
COLOR_INACTIVE = "tomato"
COLOR_SELECTED = "gold"


class FlagEditor:
    def __init__(
        self,
        dir_path: str = ".",
        *,
        points: np.ndarray = None,
        flags: np.ndarray = None,
        save_path: str = None,
        flag_names=None,
        title: str = "PINN Data Flag Editor",
        background=None,
        view_var: int = 0,
        modify_vars=None,
    ):
        """Edit ``data_flags.npy`` in ``dir_path``, or in-memory ``points`` /
        ``flags`` saved to ``save_path``. ``background(ax)`` draws under the
        points (e.g. the mesh)."""
        self.flag_names = list(flag_names or FLAG_NAMES)
        self._title = title
        self._background = background
        if points is not None:
            self.data_points = np.asarray(points)
            self.data_flags = np.asarray(flags).astype(np.int32)
            self.N = self.data_points.shape[0]
            self.save_path = os.path.abspath(save_path)
            self.dir_path = os.path.dirname(self.save_path)
            print(f"Loaded {self.N} points  |  flags shape: {self.data_flags.shape}")
        else:
            self.dir_path = os.path.abspath(dir_path)
            self.save_path = os.path.join(self.dir_path, "data_flags.npy")
            self._load_data()

        self._view_var = view_var   # index into flag_names for the displayed variable
        self._modify_vars = list(modify_vars or [True, True, True])  # which flags to edit
        self._rand_pct = 100.0      # percentage of selected points to affect
        self._selected = np.zeros(self.N, dtype=bool)
        self._history: list[np.ndarray] = []

        self._build_ui()
        self._init_scatter()
        self._update_scatter()
        self._update_status()

    # ------------------------------------------------------------------
    # Data I/O
    # ------------------------------------------------------------------

    def _load_data(self) -> None:
        flags_path = os.path.join(self.dir_path, "data_flags.npy")
        points_path = os.path.join(self.dir_path, "data_points.npy")

        if not os.path.exists(flags_path):
            raise FileNotFoundError(f"data_flags.npy not found in {self.dir_path}")
        if not os.path.exists(points_path):
            raise FileNotFoundError(f"data_points.npy not found in {self.dir_path}")

        self.data_flags = np.load(flags_path).astype(np.int32)
        self.data_points = np.load(points_path)
        self.N = self.data_points.shape[0]

        print(f"Loaded {self.N} points  |  flags shape: {self.data_flags.shape}")

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self.fig = plt.figure(figsize=(15, 8))
        try:
            self.fig.canvas.manager.set_window_title(self._title)
        except Exception:
            pass

        # Main scatter axes
        self.ax_main = self.fig.add_axes([0.05, 0.10, 0.57, 0.84])
        self.ax_main.set_aspect("equal", adjustable="box")

        # Status bar below main axes
        self.ax_status = self.fig.add_axes([0.05, 0.01, 0.57, 0.07])
        self.ax_status.axis("off")
        self.status_text = self.ax_status.text(
            0.01, 0.5, "",
            transform=self.ax_status.transAxes,
            va="center", ha="left", fontsize=9,
        )

        # Right panel ---------------------------------------------------
        x0 = 0.66   # left edge of the right panel

        def _label_ax(rect, text):
            ax = self.fig.add_axes(rect)
            ax.axis("off")
            ax.text(0.0, 0.5, text, transform=ax.transAxes,
                    va="center", fontsize=10, fontweight="bold")

        # View variable
        _label_ax([x0, 0.88, 0.30, 0.05], "View variable:")
        ax_view = self.fig.add_axes([x0, 0.73, 0.13, 0.14])
        self.radio_view = mwidgets.RadioButtons(ax_view, self.flag_names, active=self._view_var)
        self.radio_view.on_clicked(self._on_view_change)

        # Modify variables
        _label_ax([x0, 0.68, 0.30, 0.05], "Modify flags:")
        ax_check = self.fig.add_axes([x0, 0.53, 0.13, 0.14])
        self.check_modify = mwidgets.CheckButtons(
            ax_check, self.flag_names, list(self._modify_vars)
        )
        self.check_modify.on_clicked(self._on_modify_change)

        # Random % slider
        _label_ax([x0, 0.48, 0.30, 0.04], "Random % of selection:")
        ax_slider = self.fig.add_axes([x0, 0.42, 0.30, 0.04])
        self.slider_rand = mwidgets.Slider(
            ax_slider, "", 0, 100, valinit=100, valstep=1
        )
        self.slider_rand.on_changed(self._on_rand_change)

        # Toggle / Enable / Disable
        ax_toggle = self.fig.add_axes([x0, 0.34, 0.30, 0.06])
        btn = mwidgets.Button(ax_toggle, "Toggle Selected Flags", color="lightyellow")
        btn.on_clicked(self._on_toggle)
        self.btn_toggle = btn

        ax_enable = self.fig.add_axes([x0, 0.26, 0.14, 0.06])
        self.btn_enable = mwidgets.Button(ax_enable, "Enable\nSelected", color="lightcyan")
        self.btn_enable.on_clicked(self._on_enable)

        ax_disable = self.fig.add_axes([x0 + 0.16, 0.26, 0.14, 0.06])
        self.btn_disable = mwidgets.Button(ax_disable, "Disable\nSelected", color="mistyrose")
        self.btn_disable.on_clicked(self._on_disable)

        # Select all / unselect all
        ax_sel = self.fig.add_axes([x0, 0.18, 0.14, 0.06])
        self.btn_sel_all = mwidgets.Button(ax_sel, "Select All")
        self.btn_sel_all.on_clicked(self._on_select_all)

        ax_unsel = self.fig.add_axes([x0 + 0.16, 0.18, 0.14, 0.06])
        self.btn_unsel_all = mwidgets.Button(ax_unsel, "Unselect All")
        self.btn_unsel_all.on_clicked(self._on_unselect_all)

        # Undo / Save
        ax_undo = self.fig.add_axes([x0, 0.09, 0.14, 0.06])
        self.btn_undo = mwidgets.Button(ax_undo, "Undo")
        self.btn_undo.on_clicked(self._on_undo)

        ax_save = self.fig.add_axes([x0 + 0.16, 0.09, 0.14, 0.06])
        self.btn_save = mwidgets.Button(ax_save, "Save", color="lightgreen")
        self.btn_save.on_clicked(self._on_save)

        # Rectangle selector — created last so it sits on top of the axes.
        # Never call ax_main.cla() after this point: it invalidates the
        # selector's blit cache and makes it stop responding.
        self.selector = mwidgets.RectangleSelector(
            self.ax_main,
            self._on_rect_select,
            useblit=True,
            button=[1],
            minspanx=1e-12,
            minspany=1e-12,
            spancoords="data",
            interactive=False,
            props=dict(facecolor="yellow", edgecolor="orange", alpha=0.25, fill=True),
        )

    def _init_scatter(self) -> None:
        """Create the scatter and legend once; set static axes properties."""
        self.ax_main.set_xlabel("X")
        self.ax_main.set_ylabel("Y")
        self.ax_main.set_aspect("equal", adjustable="datalim")
        if self._background is not None:
            self._background(self.ax_main)

        colors = self._point_colors()
        sizes = self._point_sizes()

        self._scatter = self.ax_main.scatter(
            self.data_points[:, 0],
            self.data_points[:, 1],
            c=colors,
            s=sizes,
            alpha=0.85,
            linewidths=0.3,
            edgecolors="k",
            zorder=3,
        )

        # Placeholder legend handles — text updated in _update_scatter
        self._legend_handles = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_ACTIVE,
                   markersize=8, label="Active"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_INACTIVE,
                   markersize=8, label="Inactive"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_SELECTED,
                   markersize=8, label="Selected"),
        ]
        self._legend = self.ax_main.legend(
            handles=self._legend_handles, loc="upper right", fontsize=8
        )

    # ------------------------------------------------------------------
    # Drawing  (never calls ax_main.cla() — updates data in-place)
    # ------------------------------------------------------------------

    def _point_colors(self) -> np.ndarray:
        flags = self.data_flags[:, self._view_var].astype(bool)
        colors = np.where(flags, COLOR_ACTIVE, COLOR_INACTIVE)
        colors[self._selected] = COLOR_SELECTED
        return colors

    def _point_sizes(self) -> np.ndarray:
        sizes = np.full(self.N, 18.0)
        sizes[self._selected] = 55.0
        return sizes

    def _update_scatter(self) -> None:
        colors = self._point_colors()
        sizes = self._point_sizes()

        self._scatter.set_facecolors(colors)
        self._scatter.set_sizes(sizes)

        var_name = self.flag_names[self._view_var]
        n_active = int(self.data_flags[:, self._view_var].sum())
        n_sel = int(self._selected.sum())

        self.ax_main.set_title(
            f"Viewing: {var_name}  |  active={n_active}  inactive={self.N - n_active}  "
            f"selected={n_sel}\n"
            "Drag left mouse to select  |  gold=selected  blue=active  red=inactive",
            fontsize=9,
        )

        # Update legend labels in-place
        self._legend_handles[0].set_label(f"Active ({n_active})")
        self._legend_handles[1].set_label(f"Inactive ({self.N - n_active})")
        self._legend_handles[2].set_label(f"Selected ({n_sel})")
        self._legend = self.ax_main.legend(
            handles=self._legend_handles, loc="upper right", fontsize=8
        )

        self.fig.canvas.draw_idle()

    def _update_status(self) -> None:
        n_sel = int(self._selected.sum())
        modify_str = (
            "+".join(self.flag_names[i] for i, v in enumerate(self._modify_vars) if v) or "none"
        )
        self.status_text.set_text(
            f"dir: {self.dir_path}  |  selected: {n_sel}/{self.N}  |  "
            f"modify: {modify_str}  |  random%: {self._rand_pct:.0f}%  |  "
            f"undo steps: {len(self._history)}"
        )
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Widget callbacks
    # ------------------------------------------------------------------

    def _on_view_change(self, label: str) -> None:
        self._view_var = self.flag_names.index(label)
        self._update_scatter()
        self._update_status()

    def _on_modify_change(self, label: str) -> None:
        idx = self.flag_names.index(label)
        self._modify_vars[idx] = not self._modify_vars[idx]
        self._update_status()

    def _on_rand_change(self, val: float) -> None:
        self._rand_pct = float(val)
        self._update_status()

    def _on_rect_select(self, eclick, erelease) -> None:
        x0 = min(eclick.xdata, erelease.xdata)
        x1 = max(eclick.xdata, erelease.xdata)
        y0 = min(eclick.ydata, erelease.ydata)
        y1 = max(eclick.ydata, erelease.ydata)
        in_rect = (
            (self.data_points[:, 0] >= x0) & (self.data_points[:, 0] <= x1) &
            (self.data_points[:, 1] >= y0) & (self.data_points[:, 1] <= y1)
        )
        self._selected = in_rect
        self._update_scatter()
        self._update_status()

    def _on_select_all(self, _event) -> None:
        self._selected[:] = True
        self._update_scatter()
        self._update_status()

    def _on_unselect_all(self, _event) -> None:
        self._selected[:] = False
        self._update_scatter()
        self._update_status()

    # ------------------------------------------------------------------
    # Flag modification helpers
    # ------------------------------------------------------------------

    def _target_indices(self) -> np.ndarray:
        """Selected point indices filtered by random percentage."""
        sel_idx = np.where(self._selected)[0]
        if len(sel_idx) == 0:
            return sel_idx
        n_target = max(1, round(len(sel_idx) * self._rand_pct / 100.0))
        if n_target < len(sel_idx):
            sel_idx = np.random.choice(sel_idx, size=n_target, replace=False)
        return sel_idx

    def _push_history(self) -> None:
        self._history.append(self.data_flags.copy())
        if len(self._history) > 50:
            self._history.pop(0)

    def _on_toggle(self, _event) -> None:
        idx = self._target_indices()
        if len(idx) == 0:
            return
        self._push_history()
        for v in range(3):
            if self._modify_vars[v]:
                self.data_flags[idx, v] ^= 1
        self._update_scatter()
        self._update_status()

    def _on_enable(self, _event) -> None:
        idx = self._target_indices()
        if len(idx) == 0:
            return
        self._push_history()
        for v in range(3):
            if self._modify_vars[v]:
                self.data_flags[idx, v] = 1
        self._update_scatter()
        self._update_status()

    def _on_disable(self, _event) -> None:
        idx = self._target_indices()
        if len(idx) == 0:
            return
        self._push_history()
        for v in range(3):
            if self._modify_vars[v]:
                self.data_flags[idx, v] = 0
        self._update_scatter()
        self._update_status()

    def _on_undo(self, _event) -> None:
        if not self._history:
            return
        self.data_flags = self._history.pop()
        self._update_scatter()
        self._update_status()

    def _on_save(self, _event) -> None:
        out_path = self.save_path
        if out_path.lower().endswith(".npy"):
            np.save(out_path, self.data_flags)
        else:
            delim = "," if out_path.lower().endswith(".csv") else " "
            np.savetxt(out_path, self.data_flags, fmt="%d", delimiter=delim)
        print(f"Saved -> {out_path}")
        self.ax_main.set_title(
            f"  SAVED to {out_path}  ", color="green", fontsize=11, fontweight="bold"
        )
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.8)
        self._update_scatter()
        self._update_status()


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

def fvm_editor_kwargs(config_path: str) -> dict:
    """FlagEditor arguments for an FVM-PINN config's ``flags_file``."""
    from HydroNet.utils.config import Config
    from HydroNet.models.FVM_PINN.data import FVM_PINNDataset
    from HydroNet.models.FVM_PINN._internal.mesh.srh2d_reader import SRH2DMeshReader
    from HydroNet.models.FVM_PINN._internal.mesh.mesh_topology import build_mesh

    config = Config(config_path)
    flags_file = config.get("data.measurements.flags_file", None)
    if not flags_file:
        raise SystemExit(
            f"{config_path}: set data.measurements.flags_file to the file to edit."
        )
    mesh = build_mesh(SRH2DMeshReader(str(config.get_required_config("data.srhhydro"))).read())
    n_cells = mesh.n_cells

    if os.path.exists(flags_file):
        flags = FVM_PINNDataset._load_table(flags_file)
        if flags.ndim == 2 and flags.shape[1] == 1:
            flags = flags[:, 0]
        if flags.ndim == 1:
            # 1-column file: the cell flag applies to every variable.
            flags = np.repeat(flags[:, None], 3, axis=1)
        if flags.shape != (n_cells, 3):
            raise SystemExit(
                f"{flags_file} has shape {flags.shape}; expected ({n_cells},) or ({n_cells}, 3)."
            )
        flags = (flags != 0).astype(np.int32)
    else:
        print(f"{flags_file} not found: starting with all {n_cells} cells unflagged.")
        flags = np.zeros((n_cells, 3), dtype=np.int32)

    variables = list(config.get("data.measurements.variables", ["xi", "hu", "hv"]))
    names = ["xi", "u", "v"] if {"u", "v"} & set(variables) else ["xi", "hu", "hv"]

    # Background: SRH-2D wet-cell speed at the last measurement time (if available).
    speed, t_bg = None, None
    h5_file = config.get("data.srh2d_h5_file", None)
    if h5_file and os.path.exists(str(h5_file)):
        import h5py
        meas = config.get("data.measurements", {}) or {}
        req = list(meas.get("times", []) or meas.get("sparse_times", []) or [])
        t_bg = float(req[-1]) if req else float(config.get("training.t_end", 0.0))
        h_dry = float(config.get("physics.h_dry", 1e-2))
        with h5py.File(str(h5_file), "r") as f:
            times = f["Water_Depth_m/Times"][:]
            ti = int(np.argmin(np.abs(times - t_bg)))
            h = f["Water_Depth_m/Values"][ti, :]
            vel = f["Velocity_m_p_s/Values"][ti, :, :]
        t_bg = float(times[ti])
        speed = np.where(h > h_dry, np.hypot(vel[:, 0], vel[:, 1]), np.nan)

    def background(ax):
        tris, tri_cell = [], []
        for ci, cn in enumerate(mesh.cell_nodes):
            for k in range(1, len(cn) - 1):
                tris.append([cn[0], cn[k], cn[k + 1]])
                tri_cell.append(ci)
        x, y = mesh.node_xy[:, 0], mesh.node_xy[:, 1]
        if speed is not None:
            pc = ax.tripcolor(x, y, tris, facecolors=speed[tri_cell], cmap="Greys",
                              alpha=0.5, zorder=1)
            ax.figure.colorbar(pc, ax=ax, shrink=0.6, pad=0.01,
                               label=f"SRH-2D |V| at t = {t_bg:g} s [m/s]")
        ax.triplot(x, y, tris, color="k", lw=0.15, alpha=0.3, zorder=2)

    return dict(
        points=mesh.cell_center,
        flags=flags,
        save_path=flags_file,
        flag_names=names,
        title=f"FVM-PINN Flag Editor: {flags_file}",
        background=background,
        view_var=1,
        modify_vars=["xi" in variables, True, True],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive PINN / FVM-PINN data flag editor")
    parser.add_argument(
        "dir_path",
        nargs="?",
        default=".",
        help="Directory containing data_flags.npy and data_points.npy",
    )
    parser.add_argument(
        "--fvm-config",
        default=None,
        help="FVM-PINN YAML; edits its data.measurements.flags_file (one flag row per cell)",
    )
    args = parser.parse_args()
    if args.fvm_config:
        FlagEditor(**fvm_editor_kwargs(args.fvm_config))
    else:
        FlagEditor(args.dir_path)
    plt.show()


if __name__ == "__main__":
    main()
