# run_truss_tests.sh
export MPLBACKEND=Agg
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${TMPDIR:-/tmp}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/matplotlib}"

pytest \
  test_truss_1D.py \
  test_truss_2D_bar.py \
  test_Neumann_truss_2D_bridge.py \
  test_Neumann_truss_2element_linear.py \
  test_Neumann_truss_2D_6element.py \
  test_truss_3D_bar.py \
  test_Neumann_truss_3D_stadium.py \
  -q
