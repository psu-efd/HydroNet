"""Refined Savannah River meshes for the memory and step-time benchmark (reviewer item R2-6).

Same boundary strings, BCs and solver settings as Case 4; triangles from gmsh at a uniform size
lc = 15.5 m * sqrt(1306 / target). Bed linear in the original nodes, Manning zone of the nearest
original cell (as in make_second_mesh.py). No SRH-2D run is needed: the benchmark only times
training steps. Writes savannah_river/data_refined/n<target>/savana_n<target>.srh*.

Usage: python make_refined_meshes.py [targets...]   (default 5000 10000 20000 50000 100000 200000)
"""
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from make_second_mesh import mesh, old_cells, read_geom, read_mat, resample, write_case  # noqa: E402

SR = HERE.parent / "savannah_river"


def build(target):
    src = SR / "data_savannah_flat"
    nodes, elems, strings = read_geom(src / "savana_SI.srhgeom")
    keys, cen, area = old_cells(nodes, elems)
    oxy = np.array([nodes[i][:2] for i in sorted(nodes)])
    oz = np.array([nodes[i][2] for i in sorted(nodes)])
    mats = read_mat(src / "savana_SI.srhmat")
    lc = 15.5 * np.sqrt(len(keys) / target)
    loops = [[(b, resample(np.array([nodes[i][:2] for i in strings[b]]), lc)) for b in (3, 2, 4, 1)]]
    xy, tri, st, a = mesh(loops, lc)
    z = LinearNDInterpolator(oxy, oz)(xy)
    miss = np.isnan(z)
    z[miss] = NearestNDInterpolator(oxy, oz)(xy[miss])
    m = [mats[keys[j]] for j in cKDTree(cen).query(xy[tri - 1].mean(1))[1]]
    stem = f"savana_n{target}"
    out = SR / "data_refined" / f"n{target}"
    out.parent.mkdir(exist_ok=True)
    names = [(1, "zone_1"), (2, "zone_2"), (3, "main_channel"), (4, "zone_3"), (5, "zone_4")]
    write_case(out, stem, xy, z, tri, st, m, src / "savana_SI.srhhydro", names, 6,
               [(r'^Case .*$', f'Case "{stem}"'), (r'^Grid .*$', f'Grid "{stem}.srhgeom"'),
                (r'^HydroMat .*$', f'HydroMat "{stem}.srhmat"')])
    print(f"target {target}: lc {lc:.2f} m, {len(tri)} cells, {len(xy)} nodes, total area {a.sum():.0f} "
          f"(old {area.sum():.0f}), inlet nodes {len(st[1])}, bed {z.min():.2f}-{z.max():.2f}", flush=True)


if __name__ == "__main__":
    for t in (list(map(int, sys.argv[1:])) or [5000, 10000, 20000, 50000, 100000, 200000]):
        build(t)
