#pragma once

#include "sparse_linear_algebra/constraints/DirichletConstraint.h"
#include "sparse_linear_algebra/constraints/MultiPointConstraint.h"

#include <set>

namespace sparse_linear_algebra
{
/// In practice, a DoF can be involved in multiple multipoint constraints, especially with periodic
/// boundary conditions.  This class is responsible for reorganizing the multipoint constraints
/// such that each constraint is an equation in the form:
/// <dependent DoF> = (linear function of independent DoF's) + a constant
/// The reorganization is such that each dependent DoF only appears once within all
/// constraints.  Additionally, any independent DoF only appears on the RHS of any constraint.
///
/// Note: it is assumed that no two Dirichlet constraints constrain the same DoF.
///
/// Note: this algorithm not work for the distributed case unless each rank has all of the
/// constraints (in the same order), which would not scale well at all.
/// @param dirichlet_constraints (in) - the list of Dirichlet constraints to incoporate in to the
///        system of constraints. This will not be modified.
/// @param multipoint_constraints (in/out) - given the list of MPC's, the list is modified
///        throughout the algorithm to reduce the list to the set of constraints as described above.
void consolidate_multipoint_constraints(
  const std::vector<DirichletConstraint>& dirichlet_constraints,
  std::vector<MultiPointConstraint>& multipoint_constraints);



} // end namespace sparse_linear_algebra
