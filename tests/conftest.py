"""Shared pytest configuration for test-side plotting behavior."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

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
