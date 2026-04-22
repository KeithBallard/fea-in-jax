"""
# Boundary conditions vs. constraints
There is a need to specify "boundary conditions" (i.e. fixing the solution at a node) and "constraints"
(i.e. fixing the value for a DoF in the system of linear equations). Boundary conditions are the
high-level interface for a user, while constraints are the low-level interface for the solver. Boundary
conditions do not have strict rules, while constraints do (e.g. any dependant DoF cannot show up as
an independent DoF in another constraint).

# Boundary conditions and indices
The intention is that boundary conditions are specified for either nodes, elements, faces, edges,
or global values (e.g. global gradients for periodic boundary conditions).  However, only nodes and
global values are supported currently. Due to this flexibility, boundary conditions hold a type and
corresponding index. For example, a Dirichlet boundary condition on a node would have type BCType.NODE
and index corresponding to the node index. Typically, if there is a global value, there will only be
one, so the index should be `0`. However, in theory, there can be multiple global values to accommodate
periodic boundary conditions discretely on different subdomains of a problem.
"""

from enum import Enum, auto
from dataclasses import dataclass


class BCType(Enum):
    NODE = auto()
    GLOBAL_VALUE = auto()


@dataclass
class DirichletBC:
    """
    Represents a Dirichlet boundary condition (i.e. fixed value for a DoF).
    """

    index: int  # Index of node, element, etc. to constrain (if bc_type is GLOBAL_DOF, then this is ignored)
    component: int  # Component of the solution to constrain (e.g. 0=u, 1=v, 2=w for linear elasticity)
    value: float  # Specified value for the boundary condition
    bc_type: BCType = (
        BCType.NODE
    )  # Type of boundary condition (e.g. fixing the solution at a node)

@dataclass
class NeumannBC:
    """
    Represents a Neumann boundary condition (i.e. external load at a DoF).
    """

    index: int  # Index of node, element, etc. to constrain (if bc_type is GLOBAL_DOF, then this is ignored)
    component: int  # Component of the solution to constrain (e.g. 0=u, 1=v, 2=w for linear elasticity)
    value: float  # Specified value for the external load
    bc_type: BCType = (
        BCType.NODE
    )  # Type of boundary condition (e.g. fixing the solution at a node)

@dataclass
class PeriodicBC:
    """
    Represents a periodic boundary condition (i.e. the solution at one node is a function of
    the solution at another node plus a term corresponding to the volume averaged gradient).
    """

    primary_index: int  # Index of the node corresponding to the independent DoFs
    secondary_index: int  # Index of the node corresponding to the dependent DoFs (i.e. a function of the primary)
    global_gradient_index: (
        int  # Index of global value corresponding to volume average gradients
    )
