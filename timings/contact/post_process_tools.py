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
        ("CUPY_KDTREE", "CuPy KDTree", "#8c564b", "x"),
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
        ("JZTREE_PREP", "JZTree", "#ff7f0e", "s"),
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

    # --- Plot 3: number of contacts found ---
    plt.figure(figsize=(9, 6))
    contacts_found_methods = [
        ("JAX_HASH_CONTACTS", "JAX Hash", "#1f77b4", "o"),
        ("JZTREE_CONTACTS", "JZTree", "#ff7f0e", "s"),
        ("WARP_ROD_CONTACTS", "Warp Rod", "#2ca02c", "^"),
        ("WARP_NODE_CLOUD_CONTACTS", "Warp Node Cloud", "#9467bd", "v"),
        ("SCIPY_KDTREE_CONTACTS", "SciPy KDTree", "#d62728", "D"),
    ]

    for method_key, label, color, marker in contacts_found_methods:
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
    plt.ylabel("Number of contacts", fontsize=12)
    plt.title("Model Contacts Identified Comparison (Log-Log)", fontsize=14, fontweight="bold")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=10)
    plt.tight_layout()
    if output_prefix:
        plt.savefig(f"{output_prefix}_contacts_found.png", dpi=900)
        plt.savefig(f"{output_prefix}_contacts_found.pdf")
    plt.show()

    # --- Plot 4: Contact Search Scaling versus Contacts Found---
    plt.figure(figsize=(9, 6))
    search_methods_v_contacts = [
        ("JAX_HASH", "JAX Hash", "#1f77b4", "o"),
        ("JZTREE", "JZTree", "#ff7f0e", "s"),
        ("WARP_ROD", "Warp Rod", "#2ca02c", "^"),
        ("WARP_NODE_CLOUD", "Warp Node Cloud", "#9467bd", "v"),
        ("SCIPY_KDTREE", "SciPy KDTree", "#d62728", "D"),
    ]

    for method_key, label, color, marker in search_methods_v_contacts:
        for tow in fibers_in_tow:
            pts = [r for r in results if r["FIBERS_PER_BUNDLE"] == tow and r.get(method_key) is not None]
            if not pts:
                continue
            x = [p[method_key + "_CONTACTS"] for p in pts]
            y = [p[method_key] for p in pts]
            style = tow_styles[tow]
            leg_label = f"{label} ({tow} f/tow)" if len(fibers_in_tow) > 1 else label
            plt.plot(x, y, style, marker=marker, color=color, label=leg_label, lw=2, ms=6)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of Contacts Found", fontsize=12)
    plt.ylabel("Execution Time (ms)", fontsize=12)
    plt.title("Contact Search Scaling Comparison (Log-Log)", fontsize=14, fontweight="bold")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=10)
    plt.tight_layout()
    if output_prefix:
        plt.savefig(f"{output_prefix}_search_scaling_v_contacts.png", dpi=900)
        plt.savefig(f"{output_prefix}_search_scaling_v_contacts.pdf")
    plt.show()


def export_scaling_tables(results_or_filename, output_prefix=None):
    if isinstance(results_or_filename, str):
        results = load_results(results_or_filename)
    else:
        results = results_or_filename

    # Sort by fibers per bundle and nodes
    results = sorted(results, key=lambda x: (x["FIBERS_PER_BUNDLE"], x["NODES"]))

    # Mapping fibers_per_bundle -> tow pattern label
    tow_map = {1: "(1)", 3: "(1, 2)", 7: "(2, 3, 2)"}
    dx_labels = ["d_0", "d_0 / 2", "d_0 / 4", "d_0 / 8"]
    dx_latex_labels = ["$d_0$", "$d_0 / 2$", "$d_0 / 4$", "$d_0 / 8$"]

    # Assign dX indices by sorting nodes within each bundle
    grouped = {}
    for r in results:
        b = r["FIBERS_PER_BUNDLE"]
        grouped.setdefault(b, []).append(r)

    rows = []
    for b in sorted(grouped.keys()):
        b_results = sorted(grouped[b], key=lambda x: x["NODES"])
        for idx, r in enumerate(b_results):
            r["TOW_LABEL"] = tow_map.get(b, f"({b})")
            r["DX_MD"] = dx_labels[idx] if idx < len(dx_labels) else f"d_0 / {2**idx}"
            r["DX_TEX"] = dx_latex_labels[idx] if idx < len(dx_latex_labels) else f"$d_0 / {2**idx}$"
            rows.append(r)

    def fmt_num(val):
        if val is None:
            return "-"
        return f"{int(val):,}"

    def fmt_delta(val, base):
        if val is None or base is None:
            return "-"
        diff = int(val) - int(base)
        if diff == 0:
            return "0 (0.00%)"
        pct = (diff / float(base)) * 100.0
        sign_diff = f"+{diff:,}" if diff > 0 else f"-{abs(diff):,}"
        if abs(pct) < 0.01:
            sign_pct = f"{pct:+.3f}%"
        else:
            sign_pct = f"{pct:+.2f}%"
        return f"{sign_diff} ({sign_pct})"

    # =========================================================================
    # TABLE 1: RAW CONTACTS FOUND TABLE
    # =========================================================================
    # Markdown
    md_t1 = []
    md_t1.append("# Contact Search Candidates Identified\n")
    md_t1.append("| Tow Pattern | dX | Nodes (N) | SciPy KDTree | JAX Hash | JZTREE | Warp Node Cloud |")
    md_t1.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for r in rows:
        scipy_str = fmt_num(r.get("SCIPY_KDTREE_CONTACTS"))
        jax_str = fmt_num(r.get("JAX_HASH_CONTACTS"))
        jz_str = fmt_num(r.get("JZTREE_CONTACTS"))
        warp_nc_str = fmt_num(r.get("WARP_NODE_CLOUD_CONTACTS"))
        md_t1.append(
            f"| {r['TOW_LABEL']} | {r['DX_MD']} | {r['NODES']:,} | {scipy_str} | {jax_str} | {jz_str} | {warp_nc_str} |"
        )
    md_t1_content = "\n".join(md_t1) + "\n"

    # LaTeX
    tex_t1 = []
    tex_t1.append("% Table 1: Raw Contacts Found")
    tex_t1.append("\\begin{table}[htbp]")
    tex_t1.append("\\centering")
    tex_t1.append("\\caption{Total contact search candidate pairs identified across grid refinements and tow patterns.}")
    tex_t1.append("\\label{tab:contact_scaling_counts}")
    tex_t1.append("\\begin{tabular}{llrrrrr}")
    tex_t1.append("\\toprule")
    tex_t1.append("Tow Pattern & $dX$ & Nodes ($N$) & SciPy KDTree & JAX Hash & JZTREE & Warp Cloud \\\\")
    tex_t1.append("\\midrule")
    for r in rows:
        scipy_str = fmt_num(r.get("SCIPY_KDTREE_CONTACTS"))
        jax_str = fmt_num(r.get("JAX_HASH_CONTACTS"))
        jz_str = fmt_num(r.get("JZTREE_CONTACTS"))
        warp_nc_str = fmt_num(r.get("WARP_NODE_CLOUD_CONTACTS"))
        tex_t1.append(
            f"{r['TOW_LABEL']} & {r['DX_TEX']} & {r['NODES']:,} & {scipy_str} & {jax_str} & {jz_str} & {warp_nc_str} \\\\"
        )
    tex_t1.append("\\bottomrule")
    tex_t1.append("\\end{tabular}")
    tex_t1.append("\\end{table}")
    tex_t1_content = "\n".join(tex_t1) + "\n"

    # =========================================================================
    # TABLE 2: GROUND-TRUTH COMPARISON TABLE (DELTA RELATIVE TO SCIPY)
    # =========================================================================
    # Markdown
    md_t2 = []
    md_t2.append("# Contacts Difference Relative to SciPy Baseline (Delta = Method - SciPy)\n")
    md_t2.append("| Tow Pattern | dX | Nodes (N) | SciPy Baseline | JAX Hash (Delta, %) | JZTREE (Delta, %) | Warp Node Cloud (Delta, %) |")
    md_t2.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for r in rows:
        scipy_base = r.get("SCIPY_KDTREE_CONTACTS")
        scipy_str = fmt_num(scipy_base)
        jax_d = fmt_delta(r.get("JAX_HASH_CONTACTS"), scipy_base)
        jz_d = fmt_delta(r.get("JZTREE_CONTACTS"), scipy_base)
        warp_nc_d = fmt_delta(r.get("WARP_NODE_CLOUD_CONTACTS"), scipy_base)
        md_t2.append(
            f"| {r['TOW_LABEL']} | {r['DX_MD']} | {r['NODES']:,} | {scipy_str} | {jax_d} | {jz_d} | {warp_nc_d} |"
        )
    md_t2_content = "\n".join(md_t2) + "\n"

    # LaTeX
    tex_t2 = []
    tex_t2.append("% Table 2: Delta relative to SciPy Baseline")
    tex_t2.append("\\begin{table}[htbp]")
    tex_t2.append("\\centering")
    tex_t2.append("\\caption{Comparison of identified contacts relative to exact SciPy KDTree baseline ($\\Delta = N_{\\text{found}} - N_{\\text{SciPy}}$ and \\% difference).}")
    tex_t2.append("\\label{tab:contact_scaling_delta}")
    tex_t2.append("\\begin{tabular}{llrrrr}")
    tex_t2.append("\\toprule")
    tex_t2.append("Tow Pattern & $dX$ & Nodes ($N$) & SciPy Baseline & JAX Hash ($\\Delta$, \\%) & JZTREE ($\\Delta$, \\%) & Warp Cloud ($\\Delta$, \\%) \\\\")
    tex_t2.append("\\midrule")
    for r in rows:
        scipy_base = r.get("SCIPY_KDTREE_CONTACTS")
        scipy_str = fmt_num(scipy_base)
        jax_d = fmt_delta(r.get("JAX_HASH_CONTACTS"), scipy_base)
        jz_d = fmt_delta(r.get("JZTREE_CONTACTS"), scipy_base)
        warp_nc_d = fmt_delta(r.get("WARP_NODE_CLOUD_CONTACTS"), scipy_base)
        tex_t2.append(
            f"{r['TOW_LABEL']} & {r['DX_TEX']} & {r['NODES']:,} & {scipy_str} & {jax_d} & {jz_d} & {warp_nc_d} \\\\"
        )
    tex_t2.append("\\bottomrule")
    tex_t2.append("\\end{tabular}")
    tex_t2.append("\\end{table}")
    tex_t2_content = "\n".join(tex_t2) + "\n"

    # Save to files if output_prefix provided
    if output_prefix:
        t1_md_path = f"{output_prefix}_contacts_table.md"
        t1_tex_path = f"{output_prefix}_contacts_table.tex"
        t2_md_path = f"{output_prefix}_contacts_delta_table.md"
        t2_tex_path = f"{output_prefix}_contacts_delta_table.tex"

        with open(t1_md_path, "w") as f:
            f.write(md_t1_content)
        with open(t1_tex_path, "w") as f:
            f.write(tex_t1_content)
        with open(t2_md_path, "w") as f:
            f.write(md_t2_content)
        with open(t2_tex_path, "w") as f:
            f.write(tex_t2_content)

        print(f"--> Saved Table 1 Markdown: {t1_md_path}")
        print(f"--> Saved Table 1 LaTeX:    {t1_tex_path}")
        print(f"--> Saved Table 2 Markdown: {t2_md_path}")
        print(f"--> Saved Table 2 LaTeX:    {t2_tex_path}")

    return {
        "table1_md": md_t1_content,
        "table1_tex": tex_t1_content,
        "table2_md": md_t2_content,
        "table2_tex": tex_t2_content,
    }