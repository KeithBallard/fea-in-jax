import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"

from fe_jax.helper import *
import matplotlib.pyplot as plt
import numpy as np
import json
import time
import gc
import warp as wp
import newton
import jax
import jax.numpy as jnp
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

fabric = read_fabric("experiments/initial_single_fiber/initial_single_fiber.fab")
fabric = refine_fabric(fabric, dX=0.25)

class AllContactSearch:
    def __init__(
        self,
        fabric,
        self_adjacency_block=1000,
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
        
        # Keep CPU NumPy copies to avoid GPU-to-CPU stalls during setup loops
        self.point_fiber_ids_np = np.concatenate(
            [
                np.full((self.fabric.fiber_offsets[i + 1] - self.fabric.fiber_offsets[i],), i)
                for i in range(self.fabric.fiber_offsets.shape[0] - 1)
            ]
        )
        self.point_diameters_np = np.concatenate(
            [
                np.full(
                    (self.fabric.fiber_offsets[self.fabric.bundle_offsets[b_i+1]]-self.fabric.fiber_offsets[self.fabric.bundle_offsets[b_i]],),
                    self.fabric.get_diameter(b_i)
                ) for b_i in range(self.fabric.get_n_bundles())
            ]
        )
        self.points_np = np.asarray(self.fabric.points)
        
        # JAX device arrays for search execution
        self.point_fiber_ids = jnp.asarray(self.point_fiber_ids_np)
        self.point_diameters = jnp.asarray(self.point_diameters_np)
        self.points = jnp.asarray(self.fabric.points)
        self.self_adjacency_block = self_adjacency_block
        self.contact_search_alpha = contact_search_alpha
    
    # KD-tree approach 
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
        # Pre-compile JIT search for rods
        self._rod_search_jit = jax.jit(lambda pts: NewtonRodContactSearch(self.ctx_rod, pts))
        
    def warp_contact_rod(self):
        return self._rod_search_jit(self.points)

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
    
    def cupyx_contact(self):
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
    
def sort_contact(mat):
    mat = np.array(mat)
    mat = np.sort(mat, axis = 1)
    idx = np.lexsort((mat[:,1], mat[:,0]))
    return mat[idx]
    
def run_and_time(f, n=30):
    try:
        F = f()
        if F is not None:
            jax.block_until_ready(F)
        t = []
        for i in range(n):
            t_start = time.perf_counter()
            F = f()
            if F is not None:
                jax.block_until_ready(F)
            t_end = time.perf_counter()
            t.append(1000 * (t_end - t_start))
        return float(np.median(t)), F
    except Exception as e:
        print(f"  Warning: search failed: {e}")
        return None, None

def all_timings(fabric):
    timing_results = {}
    acs = AllContactSearch(fabric)
    n_pts = len(fabric.points)
    
    # 1. JAX Hash Prep
    try:
        t0 = time.perf_counter()
        acs.prep_jax_hash()
        timing_results['JAX_HASH_PREP'] = 1000 * (time.perf_counter() - t0)
    except Exception as e:
        print(f"  Warning: JAX Hash Prep failed at {n_pts} nodes: {e}")
        timing_results['JAX_HASH_PREP'] = None

    # 2. Warp Node Cloud Build
    try:
        t0 = time.perf_counter()
        acs.build_warp()
        timing_results['WARP_NODE_CLOUD_BUILD'] = 1000 * (time.perf_counter() - t0)
    except Exception as e:
        print(f"  Warning: Warp Node Cloud Build failed at {n_pts} nodes: {e}")
        timing_results['WARP_NODE_CLOUD_BUILD'] = None

    # 3. Warp Rod Build
    try:
        t0 = time.perf_counter()
        acs.build_warp_rod()
        timing_results['WARP_ROD_BUILD'] = 1000 * (time.perf_counter() - t0)
    except Exception as e:
        print(f"  Warning: Warp Rod Build failed at {n_pts} nodes: {e}")
        timing_results['WARP_ROD_BUILD'] = None

    # 4. Search timings and contact counts
    # SciPy
    timing_results['SCIPY_KDTREE'], SC = run_and_time(acs.scipy_contact, n=5)
    timing_results['SCIPY_KDTREE_CONTACTS'] = int(len(SC)) if SC is not None else None

    # JAX Hash
    if timing_results['JAX_HASH_PREP'] is not None:
        timing_results['JAX_HASH'], JH = run_and_time(acs.jaxhash_contact, n=50)
        timing_results['JAX_HASH_CONTACTS'] = int(JH[2]) if JH is not None else None
    else:
        timing_results['JAX_HASH'] = None
        timing_results['JAX_HASH_CONTACTS'] = None

    # JZTree
    timing_results['JZTREE'], JZ = run_and_time(acs.jztree_contact, n=50)
    timing_results['JZTREE_CONTACTS'] = int(JZ[2]) if JZ is not None else None

    # Warp Node Cloud
    if timing_results['WARP_NODE_CLOUD_BUILD'] is not None:
        timing_results['WARP_NODE_CLOUD'], WN = run_and_time(acs.warp_contact, n=50)
        timing_results['WARP_NODE_CLOUD_CONTACTS'] = int(WN[2]) if WN is not None else None
    else:
        timing_results['WARP_NODE_CLOUD'] = None
        timing_results['WARP_NODE_CLOUD_CONTACTS'] = None

    # Warp Rod
    if timing_results['WARP_ROD_BUILD'] is not None:
        timing_results['WARP_ROD'], WR = run_and_time(acs.warp_contact_rod, n=50)
        timing_results['WARP_ROD_CONTACTS'] = int(np.asarray(WR.count).squeeze()) if WR is not None else None
    else:
        timing_results['WARP_ROD'] = None
        timing_results['WARP_ROD_CONTACTS'] = None

    # CuPy
    if n_pts < 25000:
        timing_results['CUPY_KDTREE'], _ = run_and_time(acs.cupyx_contact, n=5)
    else:
        timing_results['CUPY_KDTREE'] = None

    timing_results['NODES'] = n_pts
    timing_results['FIBERS_PER_BUNDLE'] = fabric.get_n_fibers_in_bundle(0)
    
    # Cleanup memory
    del acs
    gc.collect()
    
    return timing_results

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, jnp.integer)):
            return int(obj)
        elif isinstance(obj, (np.floating, jnp.floating)):
            return float(obj)
        elif isinstance(obj, (np.ndarray, jnp.ndarray)):
            return obj.tolist()
        return super().default(obj)

def save_results(results, filename="contact_scaling_results.json"):
    with open(filename, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

def load_results(filename="contact_scaling_results.json"):
    with open(filename, "r") as f:
        return json.load(f)

def plot_scaling_results(results_or_filename, output_prefix=None):
    if isinstance(results_or_filename, str):
        results = load_results(results_or_filename)
    else:
        results = results_or_filename

    results = sorted(results, key=lambda x: (x["FIBERS_PER_BUNDLE"], x["NODES"]))
    fibers_in_tow = sorted(list(set(r["FIBERS_PER_BUNDLE"] for r in results)))
    tow_styles = {fibers_in_tow[i]: ["-", "--", ":", "-."][i % 4] for i in range(len(fibers_in_tow))}

    # --- Plot 1: Contact Search Scaling ---
    plt.figure(figsize=(9, 6))
    search_methods = [
        ("JAX_HASH", "JAX Hash", "#1f77b4", "o"),
        ("JZTREE", "JZTree", "#ff7f0e", "s"),
        ("WARP_ROD", "Warp Rod", "#2ca02c", "^"),
        ("WARP_NODE_CLOUD", "Warp Node Cloud", "#9467bd", "v"),
        ("SCIPY_KDTREE", "SciPy KDTree", "#d62728", "D"),
        ("CUPYX_KDTREE", "CuPy KDTree", "#8c564b", "x"),
    ]

    for method_key, label, color, marker in search_methods:
        for tow in fibers_in_tow:
            pts = [r for r in results if r["FIBERS_PER_BUNDLE"] == tow and r.get(method_key) is not None]
            if not pts:
                continue
            x = [p["NODES"] for p in pts]
            y = [p[method_key] for p in pts]
            style = tow_styles[tow]
            leg_label = f"{label} ({tow} f/tow)" if len(fibers_in_tow) > 1 else label
            plt.plot(x, y, style, marker=marker, color=color, label=leg_label, lw=2, ms=6)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of Nodes ($N$)", fontsize=12)
    plt.ylabel("Execution Time (ms)", fontsize=12)
    plt.title("Contact Search Scaling Comparison (Log-Log)", fontsize=14, fontweight="bold")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=10)
    plt.tight_layout()
    if output_prefix:
        plt.savefig(f"{output_prefix}_search_scaling.png", dpi=900)
        plt.savefig(f"{output_prefix}_search_scaling.pdf")
    plt.show()

    # --- Plot 2: Prep & Build Time Scaling ---
    plt.figure(figsize=(9, 6))
    prep_methods = [
        ("JAX_HASH_PREP", "JAX Hash Prep", "#1f77b4", "o"),
        ("WARP_NODE_CLOUD_BUILD", "Warp Node Cloud Build", "#9467bd", "v"),
        ("WARP_ROD_BUILD", "Warp Rod Build", "#2ca02c", "^"),
    ]

    for method_key, label, color, marker in prep_methods:
        for tow in fibers_in_tow:
            pts = [r for r in results if r["FIBERS_PER_BUNDLE"] == tow and r.get(method_key) is not None]
            if not pts:
                continue
            x = [p["NODES"] for p in pts]
            y = [p[method_key] for p in pts]
            style = tow_styles[tow]
            leg_label = f"{label} ({tow} f/tow)" if len(fibers_in_tow) > 1 else label
            plt.plot(x, y, style, marker=marker, color=color, label=leg_label, lw=2, ms=6)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of Nodes ($N$)", fontsize=12)
    plt.ylabel("Build / Prep Time (ms)", fontsize=12)
    plt.title("Model Build & Prep Scaling Comparison (Log-Log)", fontsize=14, fontweight="bold")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=10)
    plt.tight_layout()
    if output_prefix:
        plt.savefig(f"{output_prefix}_prep_scaling.png", dpi=900)
        plt.savefig(f"{output_prefix}_prep_scaling.pdf")
    plt.show()

def scaling_times(fabric):
    TR = []
    for n in range(4):
        dx = fabric.get_diameter(0) / (2**n)
        print(f"\n--- Running scaling study for dX={dx:.4f} (n={n}) ---")
        
        f1 = refine_fabric(fabric_in=fabric, dX=dx)
        print(f"Refinement 1 (1 fiber/tow): {len(f1.points)} nodes")
        TR.append(all_timings(f1))
        
        f2 = refine_tow(fabric, [1, 2])
        f2 = refine_fabric(fabric_in=f2, dX=f2.get_diameter(0)/(2**n))
        print(f"Refinement 2 (3 fibers/tow): {len(f2.points)} nodes")
        TR.append(all_timings(f2))
        
        f3 = refine_tow(fabric, [2, 3, 2])
        f3 = refine_fabric(fabric_in=f3, dX=f3.get_diameter(0)/(2**n))
        print(f"Refinement 3 (7 fibers/tow): {len(f3.points)} nodes")
        TR.append(all_timings(f3))
        
        save_results(TR, get_output("Oct7_ContactScaling/contact_scaling_results.json"))
    return TR
        