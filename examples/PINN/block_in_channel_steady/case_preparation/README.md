# Case Preparation: Block in Channel (Steady)

This folder produces the training data for the `block_in_channel_steady` PINN example. The workflow has three parts. First, build the mesh and run SRH-2D. Second, convert the SRH-2D results into PINN points. Third, copy the results into the example folder and train.

```
case_preparation/
├── SRH_2D/                  # mesh generation + SRH-2D simulation
│   ├── block_in_mesh.geo
│   ├── gen_mesh.py
│   └── run_SRH_2D_Model.py
└── generate_PINN_points/    # converts SRH-2D results into PINN training points
    ├── simulations_config.json
    └── generate_points_for_PINN.py
```

## Prerequisites

- The HydroNet environment is active (`.venv`), with `HydroNet` and `pyHMT2D` installed (see the top-level `CLAUDE.md` / README).
- SMS (Aquaveo) is installed, because it ships the SRH-2D executables. This is Windows only.

## Step 1: Generate the mesh

```bash
cd case_preparation/SRH_2D
python gen_mesh.py
```

This reads the Gmsh mesh `block_in_channel.msh` and writes `block_in_channel_3d.msh` (the mesh with bed elevation assigned). It then converts that mesh to the SRH-2D case files `block_in_channel.srhgeom` and `block_in_channel.srhmat`. You can open `check_mesh_before.vtk` and `check_mesh_after.vtk` in ParaView to check the mesh.

## Step 2: Run SRH-2D

```bash
python run_SRH_2D_Model.py
```

**Before running, check that the SRH-2D paths in `run_SRH_2D_Model.py` (lines 18 to 21) match your SMS installation:**

```python
version = "3.6.5"
srh_pre_path   = r"C:\Program Files\SMS 13.3 64-bit\python\Lib\site-packages\srh2d_exe\SRH_Pre_Console.exe"
srh_path       = r"C:\Program Files\SMS 13.3 64-bit\python\Lib\site-packages\srh2d_exe\SRH-2D_Console_v365.exe"
extra_dll_path = r"C:\Program Files\SMS 13.3 64-bit\python\Lib\site-packages\srh2d_exe"
```

Other SMS / SRH-2D versions also work, but you must update these lines to match:

- `version` is the SRH-2D version string.
- The `SMS 13.3 64-bit` part of each path must name your installed SMS folder.
- The console executable name (`SRH-2D_Console_v365.exe`) changes with the SRH-2D version. Look in your `srh2d_exe` folder for the correct name.

When the run finishes, the folder contains the SRH-2D results, including `block_in_channel_XMDFC.h5`, which is the input to the next step.

## Step 3: Generate the PINN points

Copy the SRH-2D result into `generate_PINN_points`, then run the script:

```bash
cp block_in_channel_XMDFC.h5 ../generate_PINN_points/
cd ../generate_PINN_points
python generate_points_for_PINN.py
```

> If you changed the mesh in Step 1, also copy the new `block_in_channel.srhhydro`, `.srhgeom`, `.srhmat`, and `.msh` files into `generate_PINN_points/`. The copies already there match the default mesh.

The script reads `simulations_config.json` and writes the following into `data/PINN/`:

| File | Contents |
|------|----------|
| `pde_points.npy`, `pde_data.npy` | PDE (interior collocation) points |
| `boundary_points.npy`, `boundary_info.npy` | Boundary points and their info (boundary ID, nx, ny, represented length) |
| `data_points.npy`, `data_values.npy`, `data_flags.npy` | Data points, SRH-2D values `(h, u, v)`, and per-variable flags that set which values are used in the loss |
| `all_PINN_stats.json`, `all_PINN_stats.npy` | Normalization statistics |

It also writes `equation_points.vtk`, `boundary_points.vtk`, and `data_points.vtk` so you can inspect the points in ParaView.

## Step 4: Copy the data into the example folder

Copy the `data` folder you just created into `block_in_channel_steady/`. If an older `data/` folder is already there, replace it.

```bash
cp -r data ../../
```

The training config (`pinn_config.yaml`) reads its points from `data/PINN`.

## Step 5: Train the PINN

```bash
cd ../..          # back to block_in_channel_steady/
python block_in_channel_PINN.py
```
