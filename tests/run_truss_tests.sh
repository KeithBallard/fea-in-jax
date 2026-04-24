# run_truss_tests.sh
pytest \
  test_truss_1D.py \
  test_truss_2D_bar.py \
  test_Neumann_truss_2D_bridge.py \
  test_Neumann_truss_2element.py \
  test_truss_3D_bar.py \
  -q
