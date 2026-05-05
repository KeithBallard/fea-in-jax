"""
User facing constraint definitions.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Tuple

from enum import Enum, auto
import numpy as np

from .boundary_conditions import BCType, DirichletBC, PeriodicBC
from .dof_enumeration import DofEnumeration
from .utils import tensor_to_voigt_indices

# Using a simplified tolerance for float comparisons
TOLERANCE = 1e-16


@dataclass
class FixedPointConstraint:
    """
    Represents a fixed value for a DoF.
    """

    dep_dof: int
    value: float


# Used during consolidation of constratints
class CheckResult(Enum):
    BAD = auto()
    GOOD = auto()
    TRIVIAL = auto()


@dataclass(eq=False)
class MultiPointConstraint:
    """
    Represents a multi-point constraint equation:
    [dep_dof] = sum(factor_i * indep_dof_i) + constant
    """

    # The dependent degree of freedom
    dep_dof: int
    # Terms that are independent degrees of freedom
    # Stored as a dictionary mapping indep_dof -> factor for easier access
    indep_dof_terms: Dict[int, float] = field(default_factory=dict)
    # Terms that get moved to the constant due to fixed point constraints
    # Stored as list of (FixedPointConstraint, factor)
    fixed_point_terms: List[Tuple[FixedPointConstraint, float]] = field(
        default_factory=list
    )
    # The constant part of the RHS (including resolved fixed_point terms)
    rhs_constant: float = 0.0

    def __init__(
        self,
        dep_dof: int,
        indep_dofs: List[int],
        factors: List[float],
        rhs_constant: float = 0.0,
    ):
        self.dep_dof = dep_dof
        self.rhs_constant = rhs_constant

        if len(indep_dofs) != len(factors):
            raise ValueError("Number of independent DoFs must match number of factors")

        # Store terms as a dictionary mapping indep_dof -> factor for easier access
        self.indep_dof_terms: Dict[int, float] = {}
        for dof, factor in zip(indep_dofs, factors):
            self.add_new_term(dof, factor)

        # Terms that get moved to the constant due to fixed_point constraints
        # Stored as list of (FixedPointConstraint, factor)
        self.fixed_point_terms: List[Tuple[FixedPointConstraint, float]] = []

    def evaluate(self, independent_dof_values: Dict[int, float]) -> float:
        """
        Evaluates the constraint given values for independent DoFs.
        """
        val = self.get_total_constant()
        for dof, factor in self.indep_dof_terms.items():
            if dof not in independent_dof_values:
                raise ValueError(f"Value for independent DoF {dof} not provided")
            val += factor * independent_dof_values[dof]
        return val

    def get_total_constant(self) -> float:
        """
        Returns the total constant part of the RHS (including resolved fixed_point terms).
        """
        total = self.rhs_constant
        for dc, factor in self.fixed_point_terms:
            total += factor * dc.value
        return total

    def add_new_term(self, indep_dof: int, factor: float):
        """
        Adds a new linear term to the RHS.
        """
        if abs(factor) < TOLERANCE:
            return

        if indep_dof in self.indep_dof_terms:
            self.indep_dof_terms[indep_dof] += factor
            if abs(self.indep_dof_terms[indep_dof]) < TOLERANCE:
                del self.indep_dof_terms[indep_dof]
        else:
            self.indep_dof_terms[indep_dof] = factor

    def remove_term(self, indep_dof: int):
        """Removes a term by independent DoF index."""
        if indep_dof in self.indep_dof_terms:
            del self.indep_dof_terms[indep_dof]

    def substitute_mpc(
        self, indep_dof_to_replace: int, eqn_to_insert: "MultiPointConstraint"
    ):
        """
        Substitutes another MPC into this one for a specific independent DoF.
        """
        if indep_dof_to_replace not in self.indep_dof_terms:
            return  # Term not found, nothing to do

        old_factor = self.indep_dof_terms[indep_dof_to_replace]

        # Remove the term we are replacing
        del self.indep_dof_terms[indep_dof_to_replace]

        # Add terms from the substituted equation
        for dof, factor in eqn_to_insert.indep_dof_terms.items():
            self.add_new_term(dof, factor * old_factor)

        # Add constant part
        self.rhs_constant += eqn_to_insert.get_total_constant() * old_factor

    def substitute_term(
        self, indep_dof_to_replace: int, new_indep_dof: int, factor: float
    ):
        """
        Replaces [indep_dof_to_replace] with [factor * new_indep_dof].
        """
        if indep_dof_to_replace not in self.indep_dof_terms:
            return

        old_term_factor = self.indep_dof_terms[indep_dof_to_replace]
        del self.indep_dof_terms[indep_dof_to_replace]

        self.add_new_term(new_indep_dof, factor * old_term_factor)

    def swap_dep_dof_with_indep(self, indep_dof_to_swap: int):
        """
        Swaps the dependent DoF with an independent DoF.
        """
        if indep_dof_to_swap not in self.indep_dof_terms:
            raise ValueError(f"Independent DoF {indep_dof_to_swap} not found in terms")

        factor = self.indep_dof_terms[indep_dof_to_swap]
        if abs(factor) < TOLERANCE:
            raise ValueError("Cannot swap with zero factor term")

        # New equation:
        # [new_dep] = ( [old_dep] - sum(other_terms) - constant ) / factor
        #           = (1/factor)*[old_dep] + sum((-other_factor/factor)*other_terms) - constant/factor

        old_dep_dof = self.dep_dof
        new_dep_dof = indep_dof_to_swap

        # Remove the term that is becoming the new dependent variable from RHS
        del self.indep_dof_terms[new_dep_dof]

        # Divide everything by -factor (since moving to LHS effectively flips sign relative to RHS summation,
        # then dividing by factor to isolate).
        # Actually logic trace:
        # y = a*x + b*z + C
        # Want x as subject
        # a*x = y - b*z - C
        # x = (1/a)*y - (b/a)*z - (C/a)

        scale = 1.0 / factor

        # Modify existing terms
        for dof in list(self.indep_dof_terms.keys()):
            self.indep_dof_terms[dof] *= -1.0  # move to other side
            self.indep_dof_terms[dof] *= scale

        # Add old dependent as new independent
        self.add_new_term(old_dep_dof, scale)

        self.rhs_constant *= -scale  # move to other side and scale
        for i in range(len(self.fixed_point_terms)):
            dc, f = self.fixed_point_terms[i]
            self.fixed_point_terms[i] = (dc, f * -scale)

        self.dep_dof = new_dep_dof

    def simplify(self) -> CheckResult:
        """
        Simplifies the equation: removes zero terms, checks for issues.
        Returns CheckResult enum.
        """
        # Remove zero factors
        vals_to_remove = [
            k for k, v in self.indep_dof_terms.items() if abs(v) < TOLERANCE
        ]
        for k in vals_to_remove:
            del self.indep_dof_terms[k]

        # Check if dependent variable is on RHS
        if self.dep_dof in self.indep_dof_terms:
            factor = self.indep_dof_terms[self.dep_dof]
            del self.indep_dof_terms[self.dep_dof]

            # y = a*y + ... -> (1-a)y = ...
            lhs_factor = 1.0 - factor

            if abs(lhs_factor) < TOLERANCE:
                # 0 = ...
                if len(self.indep_dof_terms) == 0:
                    if abs(self.get_total_constant()) > TOLERANCE:
                        return CheckResult.BAD  # 0 = 5
                    else:
                        return CheckResult.TRIVIAL  # 0 = 0
                else:
                    # Pick new dependent variable
                    # 0 = a*z + b -> a*z = -b -> z = -b/a
                    new_dep = next(iter(self.indep_dof_terms))
                    new_factor = self.indep_dof_terms[new_dep]
                    del self.indep_dof_terms[new_dep]

                    # 0 = new_factor*new_dep + rest
                    # new_factor*new_dep = -rest
                    # new_dep = (-1/new_factor) * rest
                    scale = -1.0 / new_factor

                    for k in self.indep_dof_terms:
                        self.indep_dof_terms[k] *= scale
                    self.rhs_constant *= scale
                    # Update fixed_point terms too? Yes if they contribute to constant
                    for i in range(len(self.fixed_point_terms)):
                        dc, f = self.fixed_point_terms[i]
                        self.fixed_point_terms[i] = (dc, f * scale)

                    self.dep_dof = new_dep
                    return CheckResult.GOOD

            # Divide by lhs_factor
            scale = 1.0 / lhs_factor
            for k in self.indep_dof_terms:
                self.indep_dof_terms[k] *= scale
            self.rhs_constant *= scale
            for i in range(len(self.fixed_point_terms)):
                dc, f = self.fixed_point_terms[i]
                self.fixed_point_terms[i] = (dc, f * scale)

        return CheckResult.GOOD

    def __str__(self):
        parts = [f"[{self.dep_dof}] ="]
        for dof, factor in self.indep_dof_terms.items():
            parts.append(f"{factor} * [{dof}] +")
        parts.append(f"{self.get_total_constant()}")
        return " ".join(parts)

    def __repr__(self):
        return self.__str__()


def consolidate_multipoint_constraints(
    fixed_point_constraints: List[FixedPointConstraint],
    multipoint_constraints: List[MultiPointConstraint],
) -> List[MultiPointConstraint]:
    """
    Consolidates constraints so that each dependent DoF appears only once on LHS,
    and never on RHS.
    """
    dep_dof_map: Dict[int, MultiPointConstraint] = {}
    indep_dof_map: Dict[int, Set[MultiPointConstraint]] = (
        {}
    )  # Maps indep dof -> set of MPCs using it

    def add_constraint(new_mpc: MultiPointConstraint) -> bool:
        # Simplify first
        status = new_mpc.simplify()
        if status == CheckResult.BAD:
            raise RuntimeError("Impossible constraint encountered")
        if status == CheckResult.TRIVIAL:
            return False

        # Step 1: Apply previous constraints to new constraint (substitute RHS vars)
        # Verify keys carefully to avoid runtime err due to modification during iteration
        current_indep_dofs = list(new_mpc.indep_dof_terms.keys())
        for dof in current_indep_dofs:
            if dof in dep_dof_map:
                prev_mpc = dep_dof_map[dof]
                new_mpc.substitute_mpc(dof, prev_mpc)

        status = new_mpc.simplify()
        if status == CheckResult.TRIVIAL:
            return False
        if status == CheckResult.BAD:
            raise RuntimeError("Impossible constraint")

        # Step 2: Handle conflict if new dep_dof is already a dependent DoF
        if new_mpc.dep_dof in dep_dof_map:
            prev_mpc = dep_dof_map[new_mpc.dep_dof]
            # Conflict!
            # If new mpc has no indep terms (it is a value constraint), swap with prev
            # Or simplified logic from C++:

            # Simple approach: If new constraint has terms, use it to substitute into old?
            # Or swap dep dof of new constraint with one of its indep dofs

            if len(new_mpc.indep_dof_terms) == 0:
                if len(prev_mpc.indep_dof_terms) > 0:
                    # Swap definitions essentially?
                    # Actually C++ logic says: swap RHS of previous and new if new is effectively const
                    # Easier: Just swap dep dof of NEW constraint with an indep dof
                    pass  # Fall through to swap logic below
                else:
                    # Both constant
                    if abs(new_mpc.rhs_constant - prev_mpc.rhs_constant) > 1e-8:
                        raise RuntimeError("Conflicting constant constraints")
                    return False

            # Swap dep dof of new constraint with one of its proper independent dofs
            # Find a suitable independent dof
            if len(new_mpc.indep_dof_terms) == 0:
                # Should have been handled or is impossible conflict
                raise RuntimeError(
                    "Cannot resolve constraint conflict (no indep vars to swap)"
                )

            swap_dof = next(iter(new_mpc.indep_dof_terms.keys()))
            new_mpc.swap_dep_dof_with_indep(swap_dof)

            # Now new_mpc.dep_dof is different. But we must substitute the PREVIOUS constraint
            # (which owned the old dep_dof) into this new definition if it appears on RHS?
            # Actually, we effectively inverted the relation.
            # Old: X = ...
            # New: X = Y... -> Y = X... substitute Old into New for X.

            # Substitute prev_mpc into new_mpc (the term that was the old dep_dof)
            # The swap logic put old_dep_dof into RHS.
            # self.add_new_term(old_dep_dof, scale) is in swap code

            # So now new_mpc has 'old_dep_dof' on RHS.
            # dep_dof_map[old_dep_dof] is prev_mpc.
            new_mpc.substitute_mpc(prev_mpc.dep_dof, prev_mpc)

            status = new_mpc.simplify()
            if status == CheckResult.TRIVIAL:
                return False
            if status == CheckResult.BAD:
                raise RuntimeError("Impossible")

        # Step 3: Apply new constraint to all previous constraints (if they use new dep_dof on RHS)
        # Check if new_mpc.dep_dof is used as indep in existing
        if new_mpc.dep_dof in indep_dof_map:
            mpcs_to_update = list(indep_dof_map[new_mpc.dep_dof])
            for mpc in mpcs_to_update:
                mpc.substitute_mpc(new_mpc.dep_dof, new_mpc)
                mpc.simplify()

                # Re-register modified MPCs in maps if their independent variables changed?
                # The indep_dof_map needs to be kept consistent.
                # It's cleaner to rebuild map or carefully update.
                # For this implementation, let's update indep_dof_map hooks for `mpc`
                # But substitute_mpc changes keys. This is tricky.
                # A full refresh of maps might be safer or careful book-keeping.
                # Let's do careful update:
                # remove mpc from all indep_dof_map entries corresponding to its OLD terms?
                # That is hard to track.
                # EASIER: We only need to fix the specific indep_dof entry we are iterating?
                # No, mpc has changed entirely.
                pass

            # Clear the entry since now nobody depends on this dof (it was substituted)
            # Wait, if we substituted, they might now depend on NEW indep vars of new_mpc.
            # So those maps need updating.
            del indep_dof_map[new_mpc.dep_dof]

            # We need to re-index the updated MPCs
            for mpc in mpcs_to_update:
                for dof in mpc.indep_dof_terms:
                    if dof not in indep_dof_map:
                        indep_dof_map[dof] = set()
                    indep_dof_map[dof].add(mpc)

        # Register new constraint
        dep_dof_map[new_mpc.dep_dof] = new_mpc
        for dof in new_mpc.indep_dof_terms:
            if dof not in indep_dof_map:
                indep_dof_map[dof] = set()
            indep_dof_map[dof].add(new_mpc)

        return True

    # Process all multipoint constraints
    for mpc_orig in multipoint_constraints:
        # Clone to avoid modifying input list objects directly if that's a concern?
        # User said "given the list of MPC's, the list is modified".
        # We will work on the objects provided or copies. Let's assume we can modify them.
        add_constraint(mpc_orig)

    # Process fixed_point constraints
    for dc in fixed_point_constraints:
        # Check if it conflicts with existing MPC dependent variable
        if dc.dep_dof in dep_dof_map:
            mpc = dep_dof_map[dc.dep_dof]

            if len(mpc.indep_dof_terms) == 0:
                # Conflict of constants
                if abs(mpc.get_total_constant() - dc.value) > 1e-8:
                    raise RuntimeError(
                        "Conflict between MPC and Fixed Point constraint"
                    )
            else:
                # Resolve conflict by swapping
                # We need to force mpc.dep_dof to be something else
                # because dc.dep_dof MUST be fixed.

                # Remove from maps
                del dep_dof_map[mpc.dep_dof]
                for dof in mpc.indep_dof_terms:
                    if dof in indep_dof_map and mpc in indep_dof_map[dof]:
                        indep_dof_map[dof].remove(mpc)

                swap_dof = next(iter(mpc.indep_dof_terms.keys()))
                mpc.swap_dep_dof_with_indep(swap_dof)

                # Now mpc.dep_dof is different.
                # The old dep_dof (which is now dc.dep_dof) is on the RHS.
                # We must substitute the Fixed Point value into the RHS term.
                # Actually swap_dof logic puts old dep_dof into RHS.
                # So substitute Fixed Point value for it.

                # mpc has term for dc.dep_dof.
                factor = mpc.indep_dof_terms[dc.dep_dof]
                del mpc.indep_dof_terms[dc.dep_dof]
                mpc.fixed_point_terms.append((dc, factor))

                # Now re-integrate this MPC
                add_constraint(mpc)

        # Check if fixed_point dof is used as independent in any MPC
        if dc.dep_dof in indep_dof_map:
            # We need to copy the set because we might modify it
            mpcs = list(indep_dof_map[dc.dep_dof])
            for mpc in mpcs:
                if dc.dep_dof in mpc.indep_dof_terms:
                    factor = mpc.indep_dof_terms[dc.dep_dof]
                    del mpc.indep_dof_terms[dc.dep_dof]
                    mpc.fixed_point_terms.append((dc, factor))

                    # Also remove from indep map
                    # indep_dof_map[dc.dep_dof].remove(mpc) # Done implicitly by clearing entry later or loop logic

            del indep_dof_map[dc.dep_dof]

    # Return only active constraints
    return list(dep_dof_map.values())


def convert_boundary_conditions_to_constraints(
    boundary_conditions: List[DirichletBC | PeriodicBC],
    vertices_vd: np.ndarray[Any, np.dtype[np.floating[Any]]],
    dof_enumeration: DofEnumeration,
    n_solution_components: int,
    global_values: List[int] | None = None,
) -> Tuple[List[FixedPointConstraint], List[MultiPointConstraint]]:
    """
    Converts a list of DirichletBC and PeriodicBC to a list of constraints.
    """
    if global_values is None:
        global_values = []

    fixed_point_constraints = []
    multi_point_constraints = []

    global_dof_offsets = np.array(
        [
            dof_enumeration.free_global_dof_rank_begin + sum(global_values[:i])
            for i in range(len(global_values))
        ],
        dtype=np.int64,
    )
    voigt_indices = tensor_to_voigt_indices(vertices_vd.shape)

    for bc in boundary_conditions:
        if isinstance(bc, DirichletBC):
            if bc.bc_type == BCType.NODE:
                fixed_point_constraints.append(
                    FixedPointConstraint(
                        dep_dof=n_solution_components * bc.index + bc.component,
                        value=bc.value,
                    )
                )
            elif bc.bc_type == BCType.GLOBAL_VALUE:
                if bc.index >= len(global_values):
                    raise ValueError(
                        f"DirichletBC references global value {bc.index}, but only "
                        f"{len(global_values)} global value blocks were declared."
                    )
                if bc.component >= global_values[bc.index]:
                    raise ValueError(
                        f"DirichletBC component {bc.component} is outside global "
                        f"value block {bc.index} with size {global_values[bc.index]}."
                    )
                fixed_point_constraints.append(
                    FixedPointConstraint(
                        dep_dof=int(global_dof_offsets[bc.index] + bc.component),
                        value=bc.value,
                    )
                )
        elif isinstance(bc, PeriodicBC):
            if bc.global_gradient_index >= len(global_values):
                raise ValueError(
                    "PeriodicBC references global gradient block "
                    f"{bc.global_gradient_index}, but only {len(global_values)} "
                    "global value blocks were declared."
                )
            d = vertices_vd[bc.secondary_index] - vertices_vd[bc.primary_index]
            for i in range(n_solution_components):
                indep_terms = {
                    n_solution_components * bc.primary_index + i: 1.0,
                    **{
                        int(
                            global_dof_offsets[bc.global_gradient_index]
                            + voigt_indices[i][j]
                        ): float(d[j])
                        for j in range(n_solution_components)
                    },
                }
                multi_point_constraints.append(
                    MultiPointConstraint(
                        dep_dof=n_solution_components * bc.secondary_index + i,
                        indep_dofs=list(indep_terms.keys()),
                        factors=list(indep_terms.values()),
                    )
                )

    return fixed_point_constraints, multi_point_constraints
