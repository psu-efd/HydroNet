#!/bin/bash
# Build flat_view/: the Case 4 runs reported in the manuscript under their SR_X names, with
# the flat-start SRH-2D reference as data/Savana_XMDFC.h5, for plot_SR_ablation_grid.py
# and plot_SR_memory_history.py. SR_A uses no SRH-2D data, so it stays the A100 checkpoint.
set -e
cd "$(dirname "$0")"
A100=/storage/work/axt5780/Projects/Hydro/HydroNet_a100_timing/examples/FVM_PINN/savannah_river
rm -rf flat_view; mkdir -p flat_view/data flat_view/checkpoints
for f in data/*; do ln -s "$PWD/$f" flat_view/data/; done
rm flat_view/data/Savana_XMDFC.h5
ln -s "$PWD/data_savannah_flat/Savana_flat_5snap_XMDFC.h5" flat_view/data/Savana_XMDFC.h5
ln -s "$PWD/data_savannah_flat" flat_view/data_savannah_flat
ln -s "$PWD/fvm_pinn_config_SR_B_FL.yaml" flat_view/fvm_pinn_config_SR_B.yaml
ln -s "$A100/checkpoints/SR_A" flat_view/checkpoints/SR_A
for x in B C D E F G; do ln -s "$PWD/checkpoints/SR_${x}_FL" flat_view/checkpoints/SR_$x; done
