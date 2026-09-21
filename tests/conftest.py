"""Shared pytest configuration for test-side plotting behavior."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
import ctypes
import glob

import site
import importlib.util


def _preload_nvidia_shared_libraries():
    """Discover and preload pip-installed NVIDIA shared libraries (.so) programmatically.

    Finds the 'nvidia' package directory dynamically via importlib and site-packages
    without making any assumptions about virtualenv location, Python version, or paths.
    Loaded with RTLD_GLOBAL so PETSc and petsc4py find CUDA symbols in process memory.
    """
    candidate_dirs = set()

    # 1. Discover via importlib module spec
    try:
        spec = importlib.util.find_spec("nvidia")
        if spec and spec.submodule_search_locations:
            for loc in spec.submodule_search_locations:
                if os.path.isdir(loc):
                    candidate_dirs.add(os.path.abspath(loc))
    except Exception:
        pass

    # 2. Discover via Python's active site-packages / user-site paths
    try:
        site_dirs = []
        if hasattr(site, "getsitepackages"):
            site_dirs.extend(site.getsitepackages())
        if hasattr(site, "getusersitepackages"):
            user_site = site.getusersitepackages()
            if isinstance(user_site, str):
                site_dirs.append(user_site)
        for s_dir in site_dirs:
            n_dir = os.path.join(s_dir, "nvidia")
            if os.path.isdir(n_dir):
                candidate_dirs.add(os.path.abspath(n_dir))
    except Exception:
        pass

    for nvidia_dir in candidate_dirs:
        for root, _, files in os.walk(nvidia_dir):
            for f in files:
                if ".so" in f:
                    try:
                        ctypes.CDLL(os.path.join(root, f), mode=ctypes.RTLD_GLOBAL)
                    except OSError:
                        pass


_preload_nvidia_shared_libraries()

# Configure Matplotlib before any test imports pyplot. This keeps pytest runs
# non-interactive and avoids writing cache files into the user's home directory.
_mpl_config_dir = Path(
    os.environ.get("MPLCONFIGDIR", Path(tempfile.gettempdir()) / "matplotlib")
)
_mpl_config_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_config_dir))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pytest


def _noop(*args, **kwargs):
    return None


# Apply immediately so import-time plotting in test modules cannot write files
# or open windows before pytest fixtures start.
plt.show = _noop
plt.savefig = _noop
Figure.savefig = _noop


@pytest.fixture(autouse=True)
def suppress_plot_output(monkeypatch):
    """Disable plot display and plot file writes during pytest."""
    monkeypatch.setattr(plt, "show", _noop)
    monkeypatch.setattr(plt, "savefig", _noop)
    monkeypatch.setattr(Figure, "savefig", _noop)
