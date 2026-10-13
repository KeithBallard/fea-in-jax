import sys
import os
import time
import argparse

# 1. Parse CLI arguments FIRST before importing JAX / Warp / CuPy
parser = argparse.ArgumentParser(description="Run a single Angle Interlock contact search benchmark in an isolated process.")
parser.add_argument(
    "--method",
    type=str,
    required=True,
    choices=["JAX_HASH", "JZTREE", "WARP_NODE_CLOUD", "WARP_ROD", "SCIPY_KDTREE", "CUPY_KDTREE"],
    help="Contact search method to benchmark",
)
parser.add_argument("--n_dx", type=int, default=0, help="Refinement level exponent n (dX = diameter / 2^n)")
parser.add_argument("--tow", type=str, default="1", choices=["1", "1_2", "2_3_2"], help="Tow refinement pattern")
parser.add_argument("--self_adj", type=int, default=50000, help="Self-adjacency block size")
parser.add_argument(
    "--json_out",
    type=str,
    default=f"ContactScaling/{time.strftime('%Y%b%d')}_angle_interlock_scaling_results.json",
    help="Relative path to output JSON file (resolved via get_output)",
)
args = parser.parse_args()

# 2. Configure backend-specific JAX VRAM preallocation BEFORE importing JAX
if args.method in ["WARP_NODE_CLOUD", "WARP_ROD", "CUPY_KDTREE", "SCIPY_KDTREE"]:
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.10"  # Gives 90% of GPU VRAM to Warp or CuPy!
else:
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"  # Gives JAX 90% of GPU VRAM!

# 3. Import heavy libraries
import json
import gc
import numpy as np
import jax
import jax.numpy as jnp
from fe_jax.helper import *
from fe_jax.contact import *
from fe_jax.paths import get_output

try:
    import warp as wp
    import newton
    HAS_WARP = True
except ImportError:
    HAS_WARP = False

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


class SingleContactSearch:
    def __init__(
        self,
        fabric,
        self_adjacency_block=50000,
        contact_search_alpha=2.0,
        rigid_contact_factor=40,
        shape_pairs_factor=150,
    ):
        self.fabric = fabric
        self.dummy_pair = find_nonzero_length_dummy_contact_pair(points=fabric.points)
        self.rigid_contact_factor = rigid_contact_factor
        self.shape_pairs_factor = shape_pairs_factor
        self.rigid_contact_max = self.rigid_contact_factor * len(self.fabric.points)
        self.shape_pairs_max = self.shape_pairs_factor * len(self.fabric.points)

        self.point_fiber_ids_np = np.concatenate(
            [
                np.full((self.fabric.fiber_offsets[i + 1] - self.fabric.fiber_offsets[i],), i)
                for i in range(self.fabric.fiber_offsets.shape[0] - 1)
            ]
        )
        self.point_diameters_np = np.concatenate(
            [
                np.full(
                    (
                        self.fabric.fiber_offsets[self.fabric.bundle_offsets[b_i + 1]]
                        - self.fabric.fiber_offsets[self.fabric.bundle_offsets[b_i]],
                    ),
                    self.fabric.get_diameter(b_i),
                )
                for b_i in range(self.fabric.get_n_bundles())
            ]
        )
        self.points_np = np.asarray(self.fabric.points)

        self.point_fiber_ids = jnp.asarray(self.point_fiber_ids_np)
        self.point_diameters = jnp.asarray(self.point_diameters_np)
        self.points = jnp.asarray(self.fabric.points)
        self.self_adjacency_block = self_adjacency_block
        self.contact_search_alpha = contact_search_alpha

    def prep_jax_hash(self, pad_size=3):
        point_radii = jnp.asarray(0.5 * self.point_diameters, dtype=jnp.float32)
        query_radius = self.contact_search_alpha * 2.0 * np.max(point_radii)
        self.domain_min, self.Nx, self.Ny, self.Nz, self.total_cells, self.C_max = pad_hash_cells(
            points=self.points,
            query_radius=query_radius,
            pad_size=pad_size,
        )

    def jaxhash_contact(self):
        return jaxhash_contact_batch(
            points=self.points,
            point_fiber_ids=self.point_fiber_ids,
            adjacency_block=self.self_adjacency_block,
            point_diameters=self.point_diameters,
            search2radius_ratio=self.contact_search_alpha,
            rigid_contact_max=self.rigid_contact_max,
            domain_min=self.domain_min,
            Nx=self.Nx,
            Ny=self.Ny,
            Nz=self.Nz,
            total_cells=self.total_cells,
            C_max=self.C_max,
            dummy_pair=self.dummy_pair,
        )

    def scipy_contact(self):
        return contact_batch(
            points=self.points,
            point_fiber_ids=self.point_fiber_ids,
            adjacency_block=self.self_adjacency_block,
            point_diameters=self.point_diameters,
            search2radius_ratio=self.contact_search_alpha,
        )

    def build_warp(self):
        self.ctx = build_newton_node_cloud_contact(
            points=self.points_np,
            point_diameters=self.point_diameters_np,
            contact_search_alpha=self.contact_search_alpha,
            self_adjacency_block=self.self_adjacency_block,
            point_fiber_ids=self.point_fiber_ids_np,
            rigid_contact_max=self.rigid_contact_max,
        )

    def warp_contact(self):
        return warp_contact_batch(self.ctx, self.points, self.dummy_pair)

    def build_warp_rod(self):
        self.ctx_rod = build_newton_rods_contact(
            fabric=self.fabric,
            rigid_contact_max=self.rigid_contact_max,
            shape_pairs_max=self.shape_pairs_max,
        )

    def warp_contact_rod(self):
        return NewtonRodContactSearch(self.ctx_rod, self.points)

    def jztree_contact(self, k=160):
        return jztree_contact_batch(
            points=self.points,
            point_fiber_ids=self.point_fiber_ids,
            adjacency_block=self.self_adjacency_block,
            point_diameters=self.point_diameters,
            search2radius_ratio=self.contact_search_alpha,
            rigid_contact_max=self.rigid_contact_max,
            dummy_pair=self.dummy_pair,
            k=k,
        )

    def cupyx_contact(self, k=160):
        if not HAS_CUPY:
            return None
        res = cupyx_contact_batch(
            points=self.points,
            point_fiber_ids=self.point_fiber_ids,
            adjacency_block=self.self_adjacency_block,
            point_diameters=self.point_diameters,
            search2radius_ratio=self.contact_search_alpha,
            rigid_contact_max=self.rigid_contact_max,
            dummy_pair=self.dummy_pair,
        )
        if HAS_CUPY:
            cp.cuda.Stream.null.synchronize()
        return res


def run_and_time(f, n=30, timeout_sec=15.0):
    try:
        t_start = time.perf_counter()
        F = f()
        if F is not None:
            if hasattr(F, "block_until_ready"):
                F.block_until_ready()
            elif HAS_CUPY:
                cp.cuda.Stream.null.synchronize()
        t_first = time.perf_counter() - t_start

        # If a single run takes longer than cutoff, return immediately
        if t_first > timeout_sec:
            return float(t_first * 1000.0), F

        t = [t_first * 1000.0]
        for _ in range(n - 1):
            t_s = time.perf_counter()
            F = f()
            if F is not None:
                if hasattr(F, "block_until_ready"):
                    F.block_until_ready()
                elif HAS_CUPY:
                    cp.cuda.Stream.null.synchronize()
            t.append(1000 * (time.perf_counter() - t_s))
        return float(np.median(t)), F
    except Exception as e:
        print(f"  Warning: search failed: {e}")
        return None, None


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, jnp.integer)):
            return int(obj)
        elif isinstance(obj, (np.floating, jnp.floating)):
            return float(obj)
        elif isinstance(obj, (np.ndarray, jnp.ndarray)):
            return obj.tolist()
        return super().default(obj)


def update_results_file(json_path, entry):
    results = []
    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                results = json.load(f)
        except Exception:
            results = []

    # Find matching entry by (FIBERS_PER_BUNDLE, NODES)
    updated = False
    for r in results:
        if r.get("FIBERS_PER_BUNDLE") == entry["FIBERS_PER_BUNDLE"] and r.get("NODES") == entry["NODES"]:
            r.update(entry)
            updated = True
            break

    if not updated:
        results.append(entry)

    os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)


def main():
    base_fab = read_fabric("experiments/initial_single_fiber/initial_single_fiber.fab")

    # Step 1: Tow Refinement
    if args.tow == "1":
        f_tow = base_fab
        fibers_in_bundle = f_tow.get_n_fibers_in_bundle(0)
    elif args.tow == "1_2":
        f_tow = refine_tow(base_fab, [1, 2])
        fibers_in_bundle = f_tow.get_n_fibers_in_bundle(0)
    elif args.tow == "2_3_2":
        f_tow = refine_tow(base_fab, [2, 3, 2])
        fibers_in_bundle = f_tow.get_n_fibers_in_bundle(0)
    else:
        raise ValueError(f"Unknown tow pattern: {args.tow}")

    # Step 2: Mesh Refinement based on diameter AFTER refining tow
    dx = f_tow.get_diameter(0) / (2**args.n_dx)
    fabric = refine_fabric(fabric_in=f_tow, dX=dx)
    n_nodes = len(fabric.points)

    print(f"--> Geometry: Tow={args.tow} ({fibers_in_bundle} fibers/tow), n_dx={args.n_dx} (dX={dx:.4f}), Nodes={n_nodes}")

    entry = {"NODES": n_nodes, "FIBERS_PER_BUNDLE": fibers_in_bundle}
    cs = SingleContactSearch(fabric, self_adjacency_block=args.self_adj)

    # Step 3: Run requested benchmark method
    if args.method == "JAX_HASH":
        t0 = time.perf_counter()
        cs.prep_jax_hash()
        entry["JAX_HASH_PREP"] = 1000 * (time.perf_counter() - t0)

        t_search, JH = run_and_time(cs.jaxhash_contact, n=50)
        entry["JAX_HASH"] = t_search
        entry["JAX_HASH_CONTACTS"] = int(JH[2]) if JH is not None else None

    elif args.method == "JZTREE":
        t_search, JZ = run_and_time(cs.jztree_contact, n=50)
        entry["JZTREE"] = t_search
        entry["JZTREE_CONTACTS"] = int(JZ[2]) if JZ is not None else None

    elif args.method == "WARP_NODE_CLOUD":
        t0 = time.perf_counter()
        cs.build_warp()
        entry["WARP_NODE_CLOUD_BUILD"] = 1000 * (time.perf_counter() - t0)

        t_search, WN = run_and_time(cs.warp_contact, n=50)
        entry["WARP_NODE_CLOUD"] = t_search
        entry["WARP_NODE_CLOUD_CONTACTS"] = int(WN[2]) if WN is not None else None

    elif args.method == "WARP_ROD":
        t0 = time.perf_counter()
        cs.build_warp_rod()
        entry["WARP_ROD_BUILD"] = 1000 * (time.perf_counter() - t0)

        t_search, WR = run_and_time(cs.warp_contact_rod, n=50)
        entry["WARP_ROD"] = t_search
        entry["WARP_ROD_CONTACTS"] = int(np.asarray(WR.count).squeeze()) if WR is not None else None

    elif args.method == "SCIPY_KDTREE":
        t_search, SC = run_and_time(cs.scipy_contact, n=5, timeout_sec=10.0)
        entry["SCIPY_KDTREE"] = t_search
        entry["SCIPY_KDTREE_CONTACTS"] = int(len(SC)) if SC is not None else None

    elif args.method == "CUPY_KDTREE":
        if n_nodes < 25000:
            t_search, _ = run_and_time(cs.cupyx_contact, n=5)
            entry["CUPY_KDTREE"] = t_search
        else:
            entry["CUPY_KDTREE"] = None

    out_path = get_output(args.json_out)
    update_results_file(out_path, entry)
    print(f"--> Saved results to {out_path}")


if __name__ == "__main__":
    main()
