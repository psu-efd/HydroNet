# Block-in-channel with an inflow pulse (unsteady case, R1-M3 / R2-5)

Same mesh, roughness, and exit stage as the steady BIC case. The inflow is
held at the steady value for a 360 s spin-up, then a pulse is applied:

| t (s) | t (h)        | Q (m³/s) |
|------:|-------------:|---------:|
| 0     | 0.0          | 0.38     |
| 360   | 0.1          | 0.38     |
| 390   | 0.1083333333 | 0.76     |
| 480   | 0.1333333333 | 0.38     |
| 600   | 0.1666666667 | 0.38     |

FVM-PINN trains on [360, 600] s from the SRH-2D state at t = 360 s
(`initial_condition.mode: srh2d-snapshot`). The same table, in seconds, is in
the `hydrograph:` entry of `fvm_pinn_config_BICU_*.yaml`. Keep the two in sync.

Time scales: channel 15 m x 5 m, h ~ 0.4 m, so a gravity wave crosses it in
~7.5 s. The 30 s rise is ~4 crossings. The pulse is fast compared with the
flow but not a bore.

## Running SRH-2D (Windows / SMS)

The files `block_in_channel_hydrograph.srhhydro` and `inflow_hydrograph.xys`
were written by hand. The `IQParams ... .xys` time-series syntax has NOT
been checked against an SMS-written case. The most reliable route is:

1. Open the steady BIC case in SMS. Set BC 1 (Inlet-Q) to a time-series
   (hydrograph) discharge with the table above (SMS time unit: hours), and
   keep the CONVEYANCE distribution.
2. Set the simulation end time to 0.1666666667 h (600 s), time step 0.5 s,
   output interval 0.0027777778 h (10 s), output format XMDFC, SI.
3. Run SRH-2D. Compare the generated `.srhhydro` with the one here.
4. Copy the XMDFC result to this folder as
   `block_in_channel_hydrograph_XMDFC.h5` (the name the configs expect).
   Also note the SRH-2D wall time and the CPU (needed for R1-M2).

Check: the t = 360 s snapshot should match the steady case
(`../block_in_channel_XMDFC.h5`, t = 360 s) to within the spin-up tolerance.

## FVM cross-check

    cd examples/FVM_PINN/block_in_channel
    python block_in_channel_fvm_only.py --config fvm_pinn_config_BICU_A.yaml \
        --n-vtk 24 --save-h5 data/hydrograph/fvm_reference_XMDFC.h5

This integrates the in-house Roe solver over the same window and compares it
with SRH-2D at t = 600 s.
