"""Second, independent meshes for the off-mesh evaluation of Cases 3 and 4 (reviewer item R2-3).

Triangular meshes from gmsh with the same mean cell area as the training mesh but independent
node placement. Case 4: bed linear in the training-mesh nodes, Manning zone of the nearest
training cell. Boundary IDs, boundary conditions and solver settings are copied unchanged.
Writes block_in_channel/data_mesh2/ and savannah_river/data_mesh2/ (.srhgeom/.srhmat/.srhhydro).
"""
import re
from pathlib import Path

import gmsh
import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent


def read_geom(path):
    nodes, elems, strings, cur = {}, {}, {}, None
    for line in open(path):
        t = line.split()
        if not t:
            continue
        if t[0] == "Node":
            nodes[int(t[1])] = tuple(map(float, t[2:5]))
            cur = None
        elif t[0] == "Elem":
            elems[int(t[1])] = list(map(int, t[2:]))
            cur = None
        elif t[0] == "NodeString":
            cur = int(t[1])
            strings[cur] = list(map(int, t[2:]))
        elif cur is not None and t[0].isdigit():
            strings[cur] += list(map(int, t))
    return nodes, elems, strings


def read_mat(path):
    mats, cur = {}, None
    for line in open(path):
        t = line.split()
        if not t or t[0] in ("SRHMAT", "NMaterials", "MatName"):
            continue
        if t[0] == "Material":
            cur = int(t[1])
            mats[cur] = list(map(int, t[2:]))
        elif cur is not None:
            mats[cur] += list(map(int, t))
    return {e: m for m, es in mats.items() for e in es}


def resample(P, step):
    """Open polyline with new interior points every ~step; end points are kept."""
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(P, axis=0).T))])
    n = max(int(round(s[-1] / step)), 1)
    q = np.linspace(0.0, s[-1], n + 1)
    return np.column_stack([np.interp(q, s, P[:, j]) for j in (0, 1)])


def direction_hint(candidates, xy, target):
    return min(candidates, key=lambda n: (xy[n - 1, 0] - target[0]) ** 2 + (xy[n - 1, 1] - target[1]) ** 2)


def mesh(loops, lc):
    """loops = [outer, hole...]; each a list of (bc_id, polyline) forming a closed curve."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeMax", lc)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.model.add("m")
    geo = gmsh.model.geo
    groups, wires = {}, []
    start_xy = {bc: P[0] for loop in loops for bc, P in loop}
    next_xy = {bc: P[1] for loop in loops for bc, P in loop}
    for loop in loops:
        first, prev, lines = None, None, []
        for bc, P in loop:
            ids = []
            start = 0 if prev is None else 1
            for p in P[start:]:
                ids.append(geo.addPoint(p[0], p[1], 0, lc))
            seq = ([prev] if prev is not None else []) + ids
            if first is None:
                first = seq[0]
            groups.setdefault(bc, [])
            pts = seq
            for a, b in zip(pts[:-1], pts[1:]):
                groups[bc].append(geo.addLine(a, b))
                lines.append(groups[bc][-1])
            prev = seq[-1]
        # the last point of the loop duplicates the first: rebuild its closing line
        px = gmsh.model.geo
        px.synchronize()
        a = gmsh.model.getBoundary([(1, lines[-1])], oriented=True)[0][1]
        geo.remove([(1, lines[-1])])
        bc_last = loop[-1][0]
        groups[bc_last].remove(lines[-1])
        closing = geo.addLine(a, first)
        groups[bc_last].append(closing)
        lines[-1] = closing
        wires.append(geo.addCurveLoop(lines))
    surf = geo.addPlaneSurface(wires)
    geo.synchronize()
    for ls in groups.values():
        for l in ls:  # the boundary points are the mesh nodes; only the interior density varies
            gmsh.model.mesh.setTransfiniteCurve(l, 2)
    for bc, ls in groups.items():
        gmsh.model.addPhysicalGroup(1, ls, bc)
    gmsh.model.addPhysicalGroup(2, [surf], 100)
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.optimize("Laplace2D")
    tags, xyz, _ = gmsh.model.mesh.getNodes()
    idx = {int(t): i + 1 for i, t in enumerate(tags)}
    xy = xyz.reshape(-1, 3)[:, :2]
    _, _, conn = gmsh.model.mesh.getElements(2, surf)
    tri = np.array([idx[int(t)] for t in conn[0]]).reshape(-1, 3)
    strings = {}
    for bc, ls in groups.items():
        adj = {}
        for l in ls:
            _, _, ct = gmsh.model.mesh.getElements(1, l)
            for n1, n2 in np.array(ct[0], dtype=np.int64).reshape(-1, 2):
                adj.setdefault(idx[int(n1)], []).append(idx[int(n2)])
                adj.setdefault(idx[int(n2)], []).append(idx[int(n1)])
        ends = [n for n, v in adj.items() if len(v) == 1]
        if len(ends) not in (0, 2):
            raise ValueError(f"string {bc} has {len(ends)} ends")
        # Keep the loop direction: start at the point the polyline starts from
        sx, sy = start_xy[bc]
        d = {n: (xy[n - 1, 0] - sx) ** 2 + (xy[n - 1, 1] - sy) ** 2 for n in (ends or adj)}
        start = min(d, key=d.get)
        second = direction_hint(adj[start], xy, next_xy[bc])
        chain, prev, cur = [start], start, second
        while True:
            chain.append(cur)
            nxt = [n for n in adj[cur] if n != prev]
            if not nxt or cur == start:
                break
            prev, cur = cur, nxt[0]
        if len(chain) != len(adj) + (0 if ends else 1):
            raise ValueError(f"string {bc}: walked {len(chain)} of {len(adj)} nodes")
        strings[bc] = chain
    gmsh.finalize()
    # counter-clockwise elements
    P = xy[tri - 1]
    area = 0.5 * ((P[:, 1, 0] - P[:, 0, 0]) * (P[:, 2, 1] - P[:, 0, 1]) - (P[:, 2, 0] - P[:, 0, 0]) * (P[:, 1, 1] - P[:, 0, 1]))
    tri[area < 0] = tri[area < 0][:, [0, 2, 1]]
    return xy, tri, strings, np.abs(area)


def write_case(out, stem, xy, z, tri, strings, mat, hydro_src, mat_names, n_materials, edits):
    out.mkdir(exist_ok=True)
    with open(out / f"{stem}.srhgeom", "w") as f:
        f.write(f'SRHGEOM 30\nName "Second mesh for the off-mesh evaluation"\n\nGridUnit "METER"\n')
        for i, e in enumerate(tri, 1):
            f.write(f"Elem {i} {e[0]} {e[1]} {e[2]}\n")
        for i, (p, zz) in enumerate(zip(xy, z), 1):
            f.write(f"Node {i} {p[0]:.10f} {p[1]:.10f} {zz:.10f}\n")
        for bc, s in sorted(strings.items()):
            f.write(f"NodeString {bc} " + "\n ".join(" ".join(map(str, s[i:i + 10])) for i in range(0, len(s), 10)) + "\n")
    with open(out / f"{stem}.srhmat", "w") as f:
        f.write(f"SRHMAT 30\nNMaterials {n_materials}\n")
        for k, name in mat_names:
            f.write(f'MatName {k} "{name}"\n')
        for k, _ in mat_names:
            ids = [i + 1 for i, m in enumerate(mat) if m == k]
            f.write(f"Material {k} " + "\n ".join(" ".join(map(str, ids[i:i + 10])) for i in range(0, len(ids), 10)) + "\n")
    text = Path(hydro_src).read_text()
    for pat, rep in edits:
        text, n = re.subn(pat, rep, text, flags=re.M)
        if n != 1:
            raise ValueError(pat)
    (out / f"{stem}.srhhydro").write_text(text)


def tune(loops, target, step):
    """Boundary spacing fixed at step; scan the interior size for the target cell count."""
    best = None
    for lc in step * np.linspace(0.7, 1.5, 321):
        res = mesh(loops, lc)
        if best is None or abs(len(res[1]) - target) < abs(len(best[1]) - target):
            best = (*res, lc)
    return best


def report(name, xy, tri, area, old_xy, old_tri_centres, old_area):
    c = xy[tri - 1].mean(1)
    d = cKDTree(old_tri_centres).query(c)[0]
    dn = cKDTree(old_xy).query(xy)[0]
    print(f"{name}: {len(tri)} cells, {len(xy)} nodes; mean area {area.mean():.4f} (old {old_area.mean():.4f}); total {area.sum():.1f} (old {old_area.sum():.1f}); "
          f"area range {area.min():.3f}-{area.max():.3f}; distance centre to nearest old centre median {np.median(d):.3f}, nodes coinciding with old nodes {int((dn < 1e-6).sum())}")


def old_cells(nodes, elems):
    keys = sorted(elems)
    cen, area = [], []
    for k in keys:
        P = np.array([nodes[i][:2] for i in elems[k]])
        x, y = P[:, 0], P[:, 1]
        a = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
        cx = np.sum((x + np.roll(x, -1)) * (x * np.roll(y, -1) - np.roll(x, -1) * y)) / (6 * a)
        cy = np.sum((y + np.roll(y, -1)) * (x * np.roll(y, -1) - np.roll(x, -1) * y)) / (6 * a)
        cen.append((cx, cy))
        area.append(abs(a))
    return keys, np.array(cen), np.array(area)


def bic():
    src = HERE / "block_in_channel" / "data"
    nodes, elems, strings = read_geom(src / "block_in_channel.srhgeom")
    keys, cen, area = old_cells(nodes, elems)
    oxy = np.array([nodes[i][:2] for i in sorted(nodes)])
    corner = {1: [(0, 5), (0, 0)], 4: [(0, 0), (15, 0)], 2: [(15, 0), (15, 5)], 3: [(15, 5), (0, 5)]}

    def loops(lc):
        outer = [(b, resample(np.array(corner[b], float), lc)) for b in (4, 2, 3, 1)]
        corners = np.array([(4.5, 2), (4.5, 3), (6.5, 3), (6.5, 2), (4.5, 2)], float)
        sides = [resample(corners[i:i + 2], lc) for i in range(4)]  # keeps the block corners
        ring = np.vstack([sides[0]] + [q[1:] for q in sides[1:]])
        return [outer, [(5, ring)]]
    xy, tri, st, a, lc = tune(loops(0.41), len(keys), 0.41)
    report("BIC", xy, tri, a, oxy, cen, area)
    write_case(HERE / "block_in_channel" / "data_mesh2", "block_in_channel_mesh2", xy, np.zeros(len(xy)), tri, st,
               [1] * len(tri), src / "block_in_channel.srhhydro", [(1, "channel")], 2,
               [(r'^Case .*$', 'Case "block_in_channel_mesh2"'), (r'^Grid .*$', 'Grid "block_in_channel_mesh2.srhgeom"'),
                (r'^HydroMat .*$', 'HydroMat "block_in_channel_mesh2.srhmat"'), (r'^OutputInterval .*$', 'OutputInterval 0.01')])
    return lc


def sr():
    src = HERE / "savannah_river" / "data_savannah_flat"
    nodes, elems, strings = read_geom(src / "savana_SI.srhgeom")
    keys, cen, area = old_cells(nodes, elems)
    oid = sorted(nodes)
    oxy = np.array([nodes[i][:2] for i in oid])
    oz = np.array([nodes[i][2] for i in oid])
    mats = read_mat(src / "savana_SI.srhmat")

    def loops(lc):
        return [[(b, resample(np.array([nodes[i][:2] for i in strings[b]]), lc)) for b in (3, 2, 4, 1)]]
    xy, tri, st, a, lc = tune(loops(15.5), len(keys), 15.5)
    report("SR", xy, tri, a, oxy, cen, area)
    lin, near = LinearNDInterpolator(oxy, oz), NearestNDInterpolator(oxy, oz)
    z = lin(xy)
    miss = np.isnan(z)
    z[miss] = near(xy[miss])
    c = xy[tri - 1].mean(1)
    m = [mats[keys[j]] for j in cKDTree(cen).query(c)[1]]
    print(f"SR bed {z.min():.2f}-{z.max():.2f} (old {oz.min():.2f}-{oz.max():.2f}); nodes outside the old hull {int(miss.sum())}; zone fractions",
          {k: round(float(np.mean(np.array(m) == k)), 3) for k in sorted(set(m))},
          "old", {k: round(float(np.mean(np.array([mats[e] for e in keys]) == k)), 3) for k in sorted(set(mats.values()))})
    names = [(1, "zone_1"), (2, "zone_2"), (3, "main_channel"), (4, "zone_3"), (5, "zone_4")]
    write_case(HERE / "savannah_river" / "data_mesh2", "savana_mesh2", xy, z, tri, st, m,
               src / "savana_SI.srhhydro", names, 6,
               [(r'^Case .*$', 'Case "Savana_mesh2"'), (r'^Grid .*$', 'Grid "savana_mesh2.srhgeom"'),
                (r'^HydroMat .*$', 'HydroMat "savana_mesh2.srhmat"')])
    return lc


if __name__ == "__main__":
    print("lc", bic(), sr())
