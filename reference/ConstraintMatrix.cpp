#include "sparse_linear_algebra/ConstraintMatrix.h"
#include "sparse_linear_algebra/constraints/ConstraintConsolidation.h"
#include "utility/MPIUtils.h"

#include <iostream>

namespace sparse_linear_algebra
{
ConstraintMatrix::ConstraintMatrix(const DofMapRefs& dof_refs,
                                   const ConstraintMethod constraint_method)
  : dof_refs_(dof_refs)
  , constraint_method_(constraint_method)
{
}

void ConstraintMatrix::add_dirichlet_constraint(const GlobalIndex global_dof, const double value)
{
    if (dof_refs_.is_global_index_owned(global_dof))
    {
        dirichlet_constraints_.push_back(DirichletConstraint({ global_dof, value }));
    }
}

void ConstraintMatrix::add_multi_point_constraint(
  const GlobalIndex dependent_doF,
  const std::vector<GlobalIndex>& independent_doFs,
  const std::vector<double>& independent_dof_factors,
  const double rhs_constant)
{
    if (dof_refs_.is_global_index_owned(dependent_doF))
    {
        multi_point_constraints_.push_back(
          { dependent_doF, independent_doFs, independent_dof_factors, rhs_constant });
    }
}

void ConstraintMatrix::add_neumann_term(const GlobalIndex global_dof, const double value)
{
    if (dof_refs_.is_global_index_owned(global_dof))
    {
        neumann_terms_.push_back({ global_dof, value });
    }
}

const ConstraintMethod& ConstraintMatrix::get_constraint_method() const
{
    return constraint_method_;
}

void ConstraintMatrix::ensure_valid_constraints()
{
    // Check no two dirichlet constraints have the same dependent DoF
    // Reverse the list of Dirichlet constraints so that during the next loop, the most recent
    // constraint is kept.
    std::reverse(dirichlet_constraints_.begin(), dirichlet_constraints_.end());
    std::vector<bool> owned_dof_has_dirichlet(dof_refs_.n_owned_dofs, false);
    std::vector<std::size_t> indices_to_remove;
    for (auto i : utility::indices(dirichlet_constraints_))
    {
        auto rank_dof_index = dirichlet_constraints_[i].dep_dof - dof_refs_.owned_global_dof_begin;
        if (!owned_dof_has_dirichlet[rank_dof_index])
        {
            owned_dof_has_dirichlet[rank_dof_index] = true;
        }
        else
        {
            _UTIL_WARN_C("Conflicting Dirichlet constraints for DoF: " +
                         std::to_string(dirichlet_constraints_[i].dep_dof) +
                         " keeping the last constraint added.");
            indices_to_remove.push_back(i);
        }
    }
    // Reverse the indices so the indices are not invalidated in the loop
    std::reverse(indices_to_remove.begin(), indices_to_remove.end());
    for (auto& index_to_remove : indices_to_remove)
    {
        dirichlet_constraints_.erase(dirichlet_constraints_.begin() + index_to_remove);
    }

    if (!utility::MPIUtils::is_distributed() &&
        constraint_method_ == ConstraintMethod::IN_PLACE_ELIMINATION)
    {
        // For the centralized case, this process has all constraints, so it is possible to
        // reorganize the constraints, see consolidate_multipoint_constraints for a description of
        // what happens
        consolidate_multipoint_constraints(dirichlet_constraints_, multi_point_constraints_);
    }
    else if (utility::MPIUtils::is_distributed() &&
             constraint_method_ == ConstraintMethod::IN_PLACE_ELIMINATION)
    {
        // For the distributed case, we cannot consolidate MPC's and in fact, we cannot incoporate
        // the Dirichlet constraints into the MPC's.  This is because this rank is not guaranteed to
        // have all Dirichlet constraints that would modify an MPC on this rank, and even if each
        // rank had all Dirichlet constraints for any DoF (ghost and owned) on the rank, if there is
        // a conflict with an MPC such that the dependent DoF needs to be switched for the MPC with
        // one of the independent DoF's, then the rank is also not guaranteed to have all MPCs that
        // connect in the graph to the one being switch, precluding the ability to update it.

        // So, to make sure the Dirichlet constraints do not break any of the MPCs, no Dirichlet
        // constraint can constrain any DoF involved in a MPC.
        for (auto& mpc : multi_point_constraints_)
        {
            if (owned_dof_has_dirichlet[mpc.dep_dof - dof_refs_.owned_global_dof_begin])
            {
                throw _UTIL_MSG_EXCEPT_C("A Dirchlet constraint already exists for a DoF that is "
                                         "constrained by an MPC, which is not allowed for the "
                                         "distributed case.");
            }
            for (auto& indep_term : mpc.indep_dof_terms)
            {
                if (owned_dof_has_dirichlet[indep_term.dof - dof_refs_.owned_global_dof_begin])
                {
                    throw _UTIL_MSG_EXCEPT_C(
                      "A Dirchlet constraint already exists for a DoF that is "
                      "constrained by an MPC, which is not allowed for the "
                      "distributed case.");
                }
            }
        }

        // TODO consider checking what MPCs you can on this rank to ensure no conflicts or
        // substitutions are needed.
    }
}

void ConstraintMatrix::clear()
{
    dirichlet_constraints_.clear();
    multi_point_constraints_.clear();
}

void ConstraintMatrix::print()
{
    std::cout << "ConstraintMatrix:\n";
    std::cout << "    Dirchlet Constraints (# of: " << dirichlet_constraints_.size() << ")\n";
    for (auto& c : dirichlet_constraints_)
    {
        std::cout << "        " << c.str() << "\n";
    }

    std::cout << "    Multi Point Constraints (# of: " << multi_point_constraints_.size() << ")\n";
    for (auto& c : multi_point_constraints_)
    {
        std::cout << "        " << c.str() << "\n";
    }
}

std::vector<double> ConstraintMatrix::get_owned_nodal_neumann_terms() const
{
    std::vector<double> neumann_terms(dof_refs_.n_owned_dofs.get(), 0.0);
    for (auto& term : neumann_terms_)
    {
        neumann_terms[term.first - dof_refs_.owned_global_dof_begin.get()] += term.second;
    }
    return neumann_terms;
}

} // end namespace sparse_linear_algebra
