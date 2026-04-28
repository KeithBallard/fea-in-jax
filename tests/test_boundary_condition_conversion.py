import numpy as np
import pytest
import jax.numpy as jnp

from fe_jax.boundary_conditions import BCType, DirichletBC, NeumannBC, PeriodicBC
from fe_jax.constraints import convert_boundary_conditions_to_constraints
from fe_jax.dof_enumeration import DofEnumeration
from fe_jax.fea import (
    convert_boundary_conditions_to_external_load,
    convert_external_load_to_system,
)


def _centralized_dof_enumeration(n_total_dofs: int, n_field_dofs: int) -> DofEnumeration:
    return DofEnumeration(
        n_owned_elements=1,
        n_owned_dofs=n_total_dofs,
        n_local_ghost_dofs=0,
        n_exclusive_ghost_dofs=0,
        n_free_global_dofs=n_total_dofs - n_field_dofs,
        free_global_dof_rank_begin=n_field_dofs,
        owned_global_dof_begin=0,
        owned_global_dof_end=n_total_dofs,
        rank_to_global_map=jnp.arange(n_total_dofs),
    )


def test_boundary_conditions_convert_periodic_and_global_constraints():
    vertices = np.array([[0.0, 0.0], [2.0, 3.0]], dtype=np.float64)
    dof_enumeration = _centralized_dof_enumeration(
        n_total_dofs=7,
        n_field_dofs=4,
    )

    fixed_point_constraints, multipoint_constraints = (
        convert_boundary_conditions_to_constraints(
            boundary_conditions=[
                DirichletBC(index=0, component=1, value=0.5),
                DirichletBC(
                    index=0,
                    component=2,
                    value=0.25,
                    bc_type=BCType.GLOBAL_VALUE,
                ),
                PeriodicBC(
                    primary_index=0,
                    secondary_index=1,
                    global_gradient_index=0,
                ),
            ],
            vertices_vd=vertices,
            dof_enumeration=dof_enumeration,
            n_solution_components=2,
            global_values=[3],
        )
    )

    assert [(c.dep_dof, c.value) for c in fixed_point_constraints] == [
        (1, 0.5),
        (6, 0.25),
    ]

    mpcs_by_dep_dof = {mpc.dep_dof: mpc for mpc in multipoint_constraints}
    assert sorted(mpcs_by_dep_dof) == [2, 3]
    assert mpcs_by_dep_dof[2].indep_dof_terms == {0: 1.0, 4: 2.0, 6: 3.0}
    assert mpcs_by_dep_dof[3].indep_dof_terms == {1: 1.0, 6: 2.0, 5: 3.0}


def test_boundary_condition_conversion_validates_global_value_blocks():
    vertices = np.array([[0.0], [1.0]], dtype=np.float64)
    dof_enumeration = _centralized_dof_enumeration(
        n_total_dofs=2,
        n_field_dofs=2,
    )

    with pytest.raises(ValueError, match="global value 0"):
        convert_boundary_conditions_to_constraints(
            boundary_conditions=[
                DirichletBC(
                    index=0,
                    component=0,
                    value=1.0,
                    bc_type=BCType.GLOBAL_VALUE,
                )
            ],
            vertices_vd=vertices,
            dof_enumeration=dof_enumeration,
            n_solution_components=1,
        )


def test_neumann_boundary_conditions_convert_to_load_system():
    vertices = np.zeros((3, 2), dtype=np.float64)
    dof_enumeration = _centralized_dof_enumeration(
        n_total_dofs=6,
        n_field_dofs=6,
    )

    external_load = convert_boundary_conditions_to_external_load(
        boundary_conditions=[
            NeumannBC(index=2, component=1, value=4.0),
        ],
        vertices_vd=vertices,
        dof_enumeration=dof_enumeration,
        n_solution_components=2,
    )
    load_system = convert_external_load_to_system(external_load)

    assert load_system.dep_dofs.tolist() == [5]
    assert load_system.loads.tolist() == [4.0]
