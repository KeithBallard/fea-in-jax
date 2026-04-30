from helper import *


def test_uniform_tri_grid_writes_triangle_mesh():
    import igl

    V, F = uniform_tri_grid(7, 10)
    print(f"V {V.shape} {V}")
    print(f"F {F.shape} {F}")

    assert V.shape == (70, 3)
    assert F.shape == (108, 3)

    output_path = get_output("test_uniform_grid.stl")
    igl.write_triangle_mesh(output_path, V, F)
    assert Path(output_path).exists()
