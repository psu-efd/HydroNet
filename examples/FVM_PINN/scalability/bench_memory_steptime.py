"""Peak GPU memory and Adam step time against mesh size (reviewer item R2-6).

For one mesh, times n_steps Adam steps of the Case 4 physics-only configuration (SR-A: FVM residual
+ IC + BC, 10 sampled times per step, MLP 6x128, float64) under each training setting:
  full        full-batch residual (the setting used in the paper)
  full_ckpt   full batch + gradient checkpointing of the network forward
  mb<f>       cell mini-batch, fraction f of the cells per step (1-ring stencil, inlet cells kept)
  mb<f>_ckpt  mini-batch + checkpointing
No convergence is sought; the network starts from the same seed in every setting.
Writes one JSON per mesh to scalability/results/.

Usage (from savannah_river/):
  python ../scalability/bench_memory_steptime.py --srhhydro data_refined/n20000/savana_n20000.srhhydro --tag n20000
"""
import argparse
import copy
import json
import platform
import time
from pathlib import Path

import torch
import torch.nn as nn

from HydroNet import Config, FVM_PINNDataset, FVM_PINNTrainer, FVM_SWE_PINN

HERE = Path(__file__).resolve().parent
SETTINGS = {
    "full":       ("standard", None, False),
    "full_ckpt":  ("standard", None, True),
    "mb0.25":     ("minibatch", 0.25, False),
    "mb0.10":     ("minibatch", 0.10, False),
    "mb0.10_ckpt": ("minibatch", 0.10, True),
    "mb0.02":     ("minibatch", 0.02, False),
}


def run_setting(base_cfg, dataset, name, n_warm, n_steps):
    strategy, frac, ckpt = SETTINGS[name]
    cfg = Config()
    cfg.config = copy.deepcopy(base_cfg.config)
    cfg.set("training.strategy", strategy)
    cfg.set("training.use_grad_checkpoint", ckpt)
    if frac is not None:
        cfg.set("training.minibatch.cell_sample_frac", frac)
        cfg.set("training.minibatch.sampling", "uniform")
    cfg.set("training.logging.checkpoint_dir", str(HERE / "results" / "ckpt_unused"))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cuda = dev.type == "cuda"
    rec = {"setting": name, "strategy": strategy, "cell_frac": frac, "grad_checkpoint": ckpt}
    tr = model = opt = None
    try:
        torch.manual_seed(0)
        model = FVM_SWE_PINN(cfg)
        tr = FVM_PINNTrainer(model, dataset, cfg)._trainer
        opt = torch.optim.Adam(tr.network.parameters(), lr=1e-3)
        if cuda:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            base_alloc = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
        times = []
        for k in range(n_warm + n_steps):
            t0 = time.perf_counter()
            opt.zero_grad()
            losses = tr._step()
            losses["total"].backward()
            nn.utils.clip_grad_norm_(tr.network.parameters(), 1.0)
            opt.step()
            if cuda:
                torch.cuda.synchronize()
            if k >= n_warm:
                times.append(time.perf_counter() - t0)
        t = torch.tensor(times, dtype=torch.float64)
        rec.update(status="ok", s_per_step_median=float(t.median()), s_per_step_mean=float(t.mean()),
                   s_per_step_p90=float(t.quantile(0.9)), final_total_loss=losses["total"].item(),
                   final_fvm_loss=losses["fvm"].item())
        if cuda:
            rec.update(peak_mib=torch.cuda.max_memory_allocated() / 2**20,
                       peak_above_start_mib=(torch.cuda.max_memory_allocated() - base_alloc) / 2**20,
                       reserved_peak_mib=torch.cuda.max_memory_reserved() / 2**20)
    except torch.OutOfMemoryError as e:
        rec.update(status="oom", error=str(e).splitlines()[0])
    finally:
        del tr, model, opt
        if cuda:
            torch.cuda.empty_cache()
    print(json.dumps(rec), flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="fvm_pinn_config_SR_A.yaml")
    ap.add_argument("--srhhydro", default=None, help="mesh to use (default: the config's)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--settings", nargs="+", default=list(SETTINGS))
    ap.add_argument("--n-warm", type=int, default=10)
    ap.add_argument("--n-steps", type=int, default=50)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    cfg = Config(a.config)
    if a.srhhydro:
        cfg.set("data.srhhydro", a.srhhydro)
    cfg.set("data.srh2d_h5_file", None)  # physics-only: no reference needed
    if a.device:
        cfg.set("device.type", a.device)
    t0 = time.perf_counter()
    dataset = FVM_PINNDataset(cfg)
    md = dataset.get_mesh_data()
    info = {"tag": a.tag, "srhhydro": a.srhhydro or cfg.get("data.srhhydro"), "n_cells": int(md["n_cells"]),
            "n_faces": int(md["n_faces"]), "dataset_build_s": time.perf_counter() - t0,
            "n_time_samples": int(cfg.get("training.n_time_samples")), "n_warm": a.n_warm, "n_steps": a.n_steps,
            "torch": torch.__version__, "host": platform.node(),
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "gpu_total_mib": (torch.cuda.get_device_properties(0).total_memory / 2**20
                              if torch.cuda.is_available() else None)}
    print(json.dumps(info), flush=True)
    recs = [run_setting(cfg, dataset, s, a.n_warm, a.n_steps) for s in a.settings]
    out = HERE / "results" / f"bench_{a.tag}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"info": info, "results": recs}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
