# =========================================================
# Ladder Polymer Classification — Latest Complete Version (supports system-centric rules)
# Dependencies: rdkit-pypi, networkx, pandas, tqdm (optional)
# =========================================================
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
import networkx as nx
import pandas as pd
import numpy as np
import math

# ----------------- Basic Utilities -----------------
def _get_star_indices(mol):
    """Find atom indices of all terminals ([*] or *)."""
    return [a.GetIdx() for a in mol.GetAtoms()
            if a.GetAtomicNum() == 0 or a.GetSymbol() == '*']

def _mol_to_graph(mol):
    """Build an undirected molecular graph (hydrogens removed); edge attributes include bond_idx."""
    G = nx.Graph()
    for a in mol.GetAtoms():
        if a.GetAtomicNum() == 1:
            continue
        G.add_node(a.GetIdx())
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if mol.GetAtomWithIdx(i).GetAtomicNum() == 1 or mol.GetAtomWithIdx(j).GetAtomicNum() == 1:
            continue
        G.add_edge(i, j, bond_idx=b.GetIdx())
    return G

def _pairwise_sp_len(G, nodes):
    d = {}
    for i in nodes:
        sp = nx.single_source_shortest_path_length(G, i)
        for j in nodes:
            d[(i, j)] = sp.get(j, float('inf'))
    return d

def _cluster_stars_two_sets(G, star_idxs):
    """
    Split multiple [*] terminals into left/right sets:
    pick the farthest two stars as seeds; assign other stars to the nearer seed.
    """
    if len(star_idxs) < 2:
        return star_idxs, []
    d = _pairwise_sp_len(G, star_idxs)
    far_pair, far_dist = None, -1
    for i in range(len(star_idxs)):
        for j in range(i + 1, len(star_idxs)):
            a, b = star_idxs[i], star_idxs[j]
            if d[(a, b)] > far_dist:
                far_dist = d[(a, b)]
                far_pair = (a, b)
    s1, s2 = far_pair
    L, R = [], []
    for s in star_idxs:
        (L if d[(s, s1)] <= d[(s, s2)] else R).append(s)
    if not L:
        L.append(R.pop())
    if not R:
        R.append(L.pop())
    return L, R

def _ring_bond_indices(mol):
    rings = set()
    for r in Chem.GetSymmSSSR(mol):
        for bi in r:
            rings.add(int(bi))
    return rings

# ----------------- Fused / Bridged Bond Detection (bond-centric) -----------------
def detect_fused_and_bridged_bonds(mol):
    """
    Return (fused_bonds, bridged_bonds) as sets of bond_idx.
    - Fused: bond belongs to >= 2 rings.
    - Bridged (engineering approximation): bond is not in a ring, but both end atoms are ring atoms.
    """
    ringinfo = mol.GetRingInfo()
    num_bonds = mol.GetNumBonds()
    num_atoms = mol.GetNumAtoms()

    bond_in_ring_count = [ringinfo.NumBondRings(i) for i in range(num_bonds)]
    atom_in_ring = [ringinfo.NumAtomRings(i) > 0 for i in range(num_atoms)]

    fused = {i for i, c in enumerate(bond_in_ring_count) if c >= 2}

    bridged = set()
    for b in mol.GetBonds():
        bi = b.GetIdx()
        a = b.GetBeginAtomIdx()
        c = b.GetEndAtomIdx()
        if bond_in_ring_count[bi] == 0 and atom_in_ring[a] and atom_in_ring[c]:
            bridged.add(bi)

    return fused, bridged

# ----------------- System-Centric Criterion -----------------
def path_hits_fused_or_bridged_system(mol, G, path):
    """
    Decide whether the path goes through a fused/bridged ring system:
    - Fused system: the path contains an atom with NumAtomRings >= 2 (ring–ring junction, includes fused/spiro nodes).
    - Bridged system: along the path there exists a bond that is non-ring while both end atoms are ring atoms (external bridge).
    Satisfying either condition counts as hitting the system.
    """
    ri = mol.GetRingInfo()
    atom_ring_cnt = [ri.NumAtomRings(i) for i in range(mol.GetNumAtoms())]
    bond_ring_cnt = [ri.NumBondRings(i) for i in range(mol.GetNumBonds())]

    fused_sys = any(atom_ring_cnt[a] >= 2 for a in path)
    bridged_sys = any(
        bond_ring_cnt[G.get_edge_data(u, v)['bond_idx']] == 0 and
        atom_ring_cnt[u] > 0 and atom_ring_cnt[v] > 0
        for u, v in zip(path[:-1], path[1:])
    )
    return fused_sys or bridged_sys

# ----------------- Vertex-Disjoint Paths via Max-Flow (node splitting) -----------------
def _build_node_capacity_flow_network(G, L, R, distinct_endpoints=False):
    """
    Node splitting: each atom v -> (v,in) -> (v,out)
    - Internal node capacity = 1 (prevent two paths from sharing internal atoms)
    - Terminal node capacity = +∞ (default) or 1 (when distinct_endpoints=True)
    Edge capacity = +∞
    """
    DG = nx.DiGraph()
    S, T = "S", "T"
    for v in G.nodes():
        DG.add_node((v, "in")); DG.add_node((v, "out"))
        cap = (1.0 if distinct_endpoints else (float('inf') if (v in L or v in R) else 1.0))
        DG.add_edge((v, "in"), (v, "out"), capacity=cap)
    for u, v, data in G.edges(data=True):
        DG.add_edge((u, "out"), (v, "in"), capacity=float('inf'), bond_idx=data.get('bond_idx'))
        DG.add_edge((v, "out"), (u, "in"), capacity=float('inf'), bond_idx=data.get('bond_idx'))
    src_cap  = 1.0 if distinct_endpoints else float('inf')
    sink_cap = 1.0 if distinct_endpoints else float('inf')
    for l in L: DG.add_edge(S, (l, "in"),  capacity=src_cap)
    for r in R: DG.add_edge((r, "out"), T, capacity=sink_cap)
    return DG, S, T

def _max_vertex_disjoint_paths(G, L, R, distinct_endpoints=False):
    DG, S, T = _build_node_capacity_flow_network(G, L, R, distinct_endpoints=distinct_endpoints)
    flow_val, flow_dict = nx.maximum_flow(DG, S, T, flow_func=nx.algorithms.flow.edmonds_karp)
    return int(round(flow_val)), flow_dict, DG

def _extract_paths_from_flow(DG, flow_dict, S, T, max_paths=2):
    import copy
    fd = copy.deepcopy(flow_dict)
    paths = []
    for _ in range(max_paths):
        pos_edges = [(u, v) for u, nbrs in fd.items() for v, f in nbrs.items() if f > 0]
        H = nx.DiGraph(); H.add_edges_from(pos_edges)
        try:
            p = nx.shortest_path(H, S, T)
        except Exception:
            break
        for a, b in zip(p[:-1], p[1:]):
            if fd.get(a, {}).get(b, 0) > 0:
                fd[a][b] -= 1
        atoms = []
        for node in p:
            if isinstance(node, tuple) and len(node) == 2 and isinstance(node[0], int):
                atoms.append(node[0])
        comp = []
        for a in atoms:
            if not comp or a != comp[-1]:
                comp.append(a)
        paths.append(comp)
    return paths

def _path_uses_ring(atom_path, G, ring_bonds):
    for a, b in zip(atom_path[:-1], atom_path[1:]):
        bidx = G.get_edge_data(a, b)['bond_idx']
        if bidx in ring_bonds:
            return True
    return False

# Check if path uses any bond from given sets (bond-centric)
def _path_uses_any_bond_from_sets(atom_path, G, bond_sets):
    target = set().union(*bond_sets) if bond_sets else set()
    for u, v in zip(atom_path[:-1], atom_path[1:]):
        bidx = G.get_edge_data(u, v)['bond_idx']
        if bidx in target:
            return True
    return False

def _path_uses_all_bond_sets(atom_path, G, bond_sets):
    if not bond_sets:
        return True
    seen = set()
    for u, v in zip(atom_path[:-1], atom_path[1:]):
        bidx = G.get_edge_data(u, v)['bond_idx']
        for k, s in enumerate(bond_sets):
            if bidx in s:
                seen.add(k)
    return len(seen) == len(bond_sets)

# ----------------- Robust Path Enumeration -----------------
def _k_simple_paths_with_ring(G, mol, s, t, ring_bonds, max_paths=300):
    """Return up to max_paths simple paths s->t (no strict ring preference here); safely return [] if disconnected."""
    if (s not in G) or (t not in G):
        return []
    try:
        if not nx.has_path(G, s, t):
            return []
    except Exception:
        return []
    try:
        gen = nx.shortest_simple_paths(G, s, t, weight=None)
    except Exception:
        return []
    res = []
    try:
        for _, path in zip(range(max_paths), gen):
            res.append(path)
    except Exception:
        pass
    return res

# ----------------- Four Terminals: Clockwise Ordering + Two-Pair Check -----------------
def _order_stars_clockwise(mol, stars):
    """Order four [*] terminals clockwise using 2D coordinates; return [s1, s2, s3, s4]."""
    rdDepictor.Compute2DCoords(mol)
    conf = mol.GetConformer()
    pts = [(i, conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y) for i in stars]
    cx = sum(x for _, x, _ in pts) / len(pts)
    cy = sum(y for _, _, y in pts) / len(pts)
    angs = [(i, math.atan2(y - cy, x - cx)) for i, x, y in pts]
    ordered = [i for i, _ in sorted(angs, key=lambda t: t[1], reverse=True)]
    return ordered

def _two_pair_vertex_disjoint(
    G, mol, pair1, pair2, ring_bonds,
    require_ring=True,
    max_first_paths=300,
    max_second_paths=300,
    # — disjointness mode —
    disjoint_mode="vertex",          # "vertex" (default) or "edge"
    # — special structure constraints (kept as-is) —
    special_mode="off",              # "bond" | "system" | "off"
    special_bond_sets=None,
    per_path_mode="any",
    require_both_paths_special=True
):
    def passes_special(path):
        if special_mode == "off":
            return True
        if special_mode == "system":
            return path_hits_fused_or_bridged_system(mol, G, path)
        if not special_bond_sets:
            return True
        if per_path_mode == "all":
            return _path_uses_all_bond_sets(path, G, special_bond_sets)
        else:
            return _path_uses_any_bond_from_sets(path, G, special_bond_sets)

    def try_order(p_first, p_second):
        a, b = p_first
        c, d = p_second
        cand1 = _k_simple_paths_with_ring(G, mol, a, b, ring_bonds, max_paths=max_first_paths)
        for p1 in cand1:
            H = G.copy()
            if disjoint_mode == "vertex":
                # Vertex-disjoint: ban internal vertices of p1
                banned_nodes = set(p1[1:-1])
                H.remove_nodes_from(banned_nodes)
            else:
                # Edge-disjoint only: ban all edges of p1, but allow node reuse
                banned_edges = [(u, v) if H.has_edge(u, v) else (v, u)
                                for u, v in zip(p1[:-1], p1[1:])]
                H.remove_edges_from(banned_edges)

            try:
                if not (c in H and d in H and nx.has_path(H, c, d)):
                    continue
            except Exception:
                continue

            cand2 = _k_simple_paths_with_ring(H, mol, c, d, ring_bonds, max_paths=max_second_paths)
            for p2 in cand2:
                if require_ring:
                    def uses_ring(path):
                        return any(G.get_edge_data(x, y)['bond_idx'] in ring_bonds
                                   for x, y in zip(path[:-1], path[1:]))
                    if not (uses_ring(p1) or uses_ring(p2)):
                        continue
                if require_both_paths_special:
                    if not (passes_special(p1) and passes_special(p2)):
                        continue
                else:
                    if not (passes_special(p1) or passes_special(p2)):
                        continue
                return True, [p1, p2]
        return False, []

    ok, paths = try_order(pair1, pair2)
    if ok:
        return True, paths
    ok, paths = try_order(pair2, pair1)
    return ok, paths


# ----------------- Main Decision (with 1↔2 swap retry & system/bond modes) -----------------
def is_ladder_polymer(
    mol,
    require_ring_path=True,
    min_vertex_disjoint_paths=2,
    require_both_pairings_for_4stars=True,
    require_special_in_one_pair_for_4stars=True,  # At least one pairing must have both paths through the "special structure"
    enforce_distinct_endpoints=True,              # For multi-terminals, force two paths to occupy different terminals
    special_mode="off",                           # "system" | "bond" | "off"
    special_per_path_mode="any",                  # Effective only in "bond" mode
    try_swap_12_on_fail=True,
    disjoint_mode="vertex",                       # Four-terminal disjointness: "vertex" or "edge"
):
    stars = _get_star_indices(mol)
    if len(stars) < 2:
        return False, "Insufficient [*] terminals (<2); cannot define repeat-unit ends", {}

    G = _mol_to_graph(mol)
    ring_bonds = _ring_bond_indices(mol)

    # ---- Four-terminal special case: support swapping 1↔2 if the first attempt fails ----
    if require_both_pairings_for_4stars and len(stars) == 4:
        order0 = _order_stars_clockwise(mol, stars)

        def _eval_with_order(order, note=""):
            s1, s2, s3, s4 = order
            pairs_A = ((s1, s2), (s3, s4))
            pairs_B = ((s1, s4), (s2, s3))

            # First check “pairing feasibility” without special-structure constraints
            okA_base, _ = _two_pair_vertex_disjoint(
                G, mol, pairs_A[0], pairs_A[1], ring_bonds,
                require_ring=require_ring_path,
                special_mode="off",
                disjoint_mode=disjoint_mode
            )
            okB_base, _ = _two_pair_vertex_disjoint(
                G, mol, pairs_B[0], pairs_B[1], ring_bonds,
                require_ring=require_ring_path,
                special_mode="off",
                disjoint_mode=disjoint_mode
            )
            if not (okA_base and okB_base):
                reason = f"{note}pairings not {disjoint_mode}-disjoint feasible: A={okA_base}, B={okB_base}"
                return False, reason, {"order": order, "pairs_A": pairs_A, "pairs_B": pairs_B}

            # Then check special-structure constraints
            fused_bonds, bridged_bonds = detect_fused_and_bridged_bonds(mol)
            special_sets = [fused_bonds, bridged_bonds] if special_mode == "bond" else None

            okA_spec, pathsA = _two_pair_vertex_disjoint(
                G, mol, pairs_A[0], pairs_A[1], ring_bonds,
                require_ring=require_ring_path,
                special_mode=special_mode,
                special_bond_sets=special_sets,
                per_path_mode=special_per_path_mode,
                require_both_paths_special=True,
                disjoint_mode=disjoint_mode
            )
            okB_spec, pathsB = _two_pair_vertex_disjoint(
                G, mol, pairs_B[0], pairs_B[1], ring_bonds,
                require_ring=require_ring_path,
                special_mode=special_mode,
                special_bond_sets=special_sets,
                per_path_mode=special_per_path_mode,
                require_both_paths_special=True,
                disjoint_mode=disjoint_mode
            )

            if (not require_special_in_one_pair_for_4stars) or (okA_spec or okB_spec):
                return True, f"{note}4-star: both pairings feasible; special-structure OK ({special_mode}, {disjoint_mode}-disjoint)", {
                    "order": order,
                    "pairs_A": pairs_A, "pairs_B": pairs_B,
                    "paths_A": pathsA, "paths_B": pathsB,
                    "special_mode": special_mode,
                    "disjoint_mode": disjoint_mode
                }
            else:
                reason = f"{note}both pairings feasible, but neither pairing has both paths through fused/bridged {('system' if special_mode=='system' else 'bonds')}"
                return False, reason, {"order": order, "pairs_A": pairs_A, "pairs_B": pairs_B, "special_mode": special_mode}

        # Try the clockwise order first
        ok0, reason0, dbg0 = _eval_with_order(order0)

        # If failed, swap 1↔2 and retry
        if (not ok0) and try_swap_12_on_fail:
            order_swapped = [order0[1], order0[0], order0[2], order0[3]]
            ok1, reason1, dbg1 = _eval_with_order(order_swapped, note="[swap 1↔2] ")
            if ok1:
                dbg1["original_order"] = order0
                dbg1["swapped_order"] = order_swapped
                return True, "4-star: passed after swapping 1↔2", dbg1
            else:
                return False, f"{reason0}；{reason1}", {
                    "original_order": order0,
                    "swapped_order": order_swapped,
                    "dbg_original": dbg0,
                    "dbg_swapped": dbg1
                }
        else:
            return ok0, reason0, dbg0

    # ---- Regular multi-terminal / two-terminal path decision ----
    if len(stars) >= 3:
        L, R = _cluster_stars_two_sets(G, stars)
    else:
        L, R = [stars[0]], [stars[1]]

    k, flow_dict, DG = _max_vertex_disjoint_paths(G, L, R, distinct_endpoints=enforce_distinct_endpoints)
    if k < min_vertex_disjoint_paths:
        return False, f"max vertex-disjoint paths = {k} (< {min_vertex_disjoint_paths})", {"L": L, "R": R, "k": k}

    if require_ring_path:
        atom_paths = _extract_paths_from_flow(DG, flow_dict, "S", "T", max_paths=min_vertex_disjoint_paths)
        uses_ring = any(_path_uses_ring(p, G, ring_bonds) for p in atom_paths)
        if not uses_ring:
            return False, "no path crosses ring bonds (likely parallel sidechains)", {"L": L, "R": R, "k": k, "paths": atom_paths}

    return True, f"max vertex-disjoint paths = {k} (ladder topology satisfied)", {"L": L, "R": R, "k": k}

# ----------------- Convenience Wrapper: decide from SMILES -----------------
def is_ladder_smiles(smi, disjoint_mode="edge"):
    """
    Input SMILES, return (True/False, reason).
    - Four terminals: both pairings must be feasible; at least one pairing must have both paths pass through a fused/bridged system (system mode, default).
    - Other cases: need two vertex-disjoint paths (distinct terminals) and at least one path goes through ring bonds.
    """
    s = str(smi).strip()
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return False, "invalid SMILES"
    try:
        ok, reason, _dbg = is_ladder_polymer(
            mol,
            require_ring_path=True,
            min_vertex_disjoint_paths=2,
            require_both_pairings_for_4stars=True,
            require_special_in_one_pair_for_4stars=True,
            enforce_distinct_endpoints=True,
            special_mode="system",          # Set to "bond" or "off" to switch behavior
            special_per_path_mode="any",    # Effective only in "bond" mode
            try_swap_12_on_fail=True, disjoint_mode=disjoint_mode
        )
        return ok, reason
    except Exception as e:
        return False, f"error:{type(e).__name__}: {e}"

# ----------------- CSV Batch Processing (with progress bar & UTF-8-SIG) -----------------
def process_csv(
    input_csv="Nate.csv",
    smiles_col="Smiles",
    output_csv="Nate_with_ladder.csv",
    output_true_csv="Nate_ladder_only.csv",
    show_progress=True,
    use_manual_loop=False,   # Set True to use manual loop + progress bar
):
    """
    Batch process CSV (with progress bar):
    - Read the column `smiles_col`;
    - Add columns `is_ladder` and `ladder_reason`;
    - Save as UTF-8-SIG (openable directly in Excel without mojibake).
    """
    df = pd.read_csv(input_csv)

    if smiles_col not in df.columns:
        for alt in ["SMILES", "smiles"]:
            if alt in df.columns:
                smiles_col = alt; break
    if smiles_col not in df.columns:
        raise ValueError("Cannot find SMILES column (Smiles/SMILES/smiles).")

    _cache = {}
    def _judge(s):
        s = str(s).strip()
        if s in _cache:
            return _cache[s]
        try:
            ok, reason = is_ladder_smiles(s, disjoint_mode="vertex")
        except Exception as e:
            ok, reason = False, f"error:{type(e).__name__}: {e}"
        _cache[s] = (ok, reason)
        return ok, reason

    used_progress = False
    if show_progress:
        try:
            from tqdm.auto import tqdm  # noqa
            used_progress = True
        except Exception:
            used_progress = False

    if not use_manual_loop:
        if used_progress:
            from tqdm.auto import tqdm
            tqdm.pandas(desc="Ladder check")
            res = df[smiles_col].progress_apply(_judge)
        else:
            res = df[smiles_col].apply(_judge)
        df["is_ladder"]     = res.apply(lambda x: x[0])
        df["ladder_reason"] = res.apply(lambda x: x[1])
    else:
        ok_list, reason_list = [], []
        iterable = df[smiles_col].tolist()
        if used_progress:
            from tqdm.auto import tqdm
            iterable = tqdm(iterable, total=len(iterable), desc="Ladder check", unit="mol")
        for s in iterable:
            ok, reason = _judge(s)
            ok_list.append(ok); reason_list.append(reason)
        df["is_ladder"] = ok_list
        df["ladder_reason"] = reason_list

    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    df_true = df[df["is_ladder"]].copy().reset_index(drop=True)
    df_true.to_csv(output_true_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] wrote: {output_csv} ({len(df)} rows); {output_true_csv} (True={len(df_true)} rows)")
    return df, df_true

if __name__ == "__main__":
    import sys
    import os

    if len(sys.argv) < 2:
        print("Usage: python xx.py input.csv [smiles_column]")
        sys.exit(1)

    input_csv = sys.argv[1]
    smiles_col = sys.argv[2] if len(sys.argv) > 2 else "Smiles"

    base = os.path.splitext(input_csv)[0]

    output_csv = f"{base}_with_ladder.csv"
    output_true_csv = f"{base}_ladder_only.csv"

    print(f"[INFO] Input: {input_csv}")
    print(f"[INFO] SMILES column: {smiles_col}")

    process_csv(
        input_csv=input_csv,
        smiles_col=smiles_col,
        output_csv=output_csv,
        output_true_csv=output_true_csv,
        show_progress=True
    )

    print(f"[DONE] Ladder polymer extraction finished.")
