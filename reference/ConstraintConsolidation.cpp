#include "sparse_linear_algebra/constraints/ConstraintConsolidation.h"

#include "utility/Exceptions.h"
#include "utility/MPIUtils.h"

#include <cmath>
#include <iostream>
#include <stdlib.h>
#include <unordered_set>

#ifdef HAVE_MPI
#include <mpi.h>
#endif

using namespace std;

namespace sparse_linear_algebra
{

//////////////////////////////////////////////////////////////////////////////////////////////////
// Helper functions

/// Replaces all the DoFs in the new constraint that are already an independent DoF
/// in an constraint that already exists.
static bool consolidate_previous_constraints_into_new(
  MultiPointConstraint& new_constraint,
  const std::map<GlobalIndex, MultiPointConstraint*> dep_dof_to_mpc_map,
  std::map<GlobalIndex, std::set<MultiPointConstraint*>>& indep_dof_to_mpcs_map)
{
    //// PART 1a - if any independent DOF in the new constraint is already a dependent in a previous
    /// constraint, substitute /           the previous constraint into this one.
    for (int i = 0; i < new_constraint.num_terms(); ++i)
    {
        auto find_iter = dep_dof_to_mpc_map.find(new_constraint.indep_dof_terms[i].dof);
        if (find_iter != dep_dof_to_mpc_map.end())
        {
            new_constraint.substitute_term(i, *find_iter->second);
        }
    }

    auto result = new_constraint.simplify();
    switch (result)
    {
        case (CheckResult::GOOD): break;
        case (CheckResult::BAD): throw _UTIL_MSG_EXCEPT("Bad constraint.");
        case (CheckResult::TRIVIAL): return false;
        default: throw _UTIL_MSG_EXCEPT("Don't understand the result of simplify.");
    }
    // Now there are no independent DOFs in the new constraint that are dependent in previous
    // constraints

    // PART 1b - check that dependent DOF of new constraint is not a dependent DOF in a previous
    // constraint, and fix if it is.
    auto find_iter = dep_dof_to_mpc_map.find(new_constraint.dep_dof);
    // If a previous constraint has the new constraint's dependent DOF
    if (find_iter != dep_dof_to_mpc_map.end())
    {
        MultiPointConstraint& previous_constraint = *find_iter->second;

        // A previous multi-point constraint has the new constraint's dependent DOF.
        // The basic approach here is that if the new constraint has the same dependent DOF as a
        // previous constraint, then the dependent DOF of the new constraint will be swapped
        // with its first independent DOF. If the new constraint doesn't have independent DOFs,
        // then we will first swap the RHS of the previous constraint with that of the new
        // constraint.
        if (new_constraint.num_terms() == 0)
        {
            // This constraint has no RHS terms (other than the constant)
            if (previous_constraint.num_terms() > 0)
            {
                // The previous constraint has independent DOFs, swap the RHSs of the two
                // constraints
                double newEqnConstant = new_constraint.rhs_constant;

                // Move RHS from previous constraint to new constraint
                new_constraint.rhs_constant    = previous_constraint.rhs_constant;
                new_constraint.indep_dof_terms = previous_constraint.indep_dof_terms;

                // Update indep_dof_to_mpcs_map
                for (auto i = 0; i < previous_constraint.num_terms(); ++i)
                {
                    indep_dof_to_mpcs_map[previous_constraint.indep_dof_terms[i].dof].erase(
                      &previous_constraint);
                }

                // Clear RHS of previous constraint
                previous_constraint.indep_dof_terms.clear();
                previous_constraint.rhs_constant = newEqnConstant;
            }
            else
            {
                // Both constraints set the dependent DoF to a constant, it must be the same,
                // which is trivial
                if (fabs(previous_constraint.rhs_constant - new_constraint.rhs_constant) > 1.0e-8)
                {
                    throw _UTIL_MSG_EXCEPT(
                      "A new constraint has the same dependent DoF as a previous constraint "
                      "but a different RHS constant.");
                }
                else
                {
                    return false;
                }
            }

            // Done swapping constraint RHSs if necessary...
            // Now we are in a position to modify the new constraint by swapping the dependent DOF
            // with the first suitable independent DOF
            auto swap_term = 0;
            new_constraint.swap_dep_dof_with_indep(swap_term);
            // The previous constraint needs to be substituted into the swapped DoF
            new_constraint.substitute_term(swap_term, previous_constraint);
            result = new_constraint.simplify();
            switch (result)
            {
                case (CheckResult::GOOD): break;
                case (CheckResult::BAD): throw _UTIL_MSG_EXCEPT("Bad constraint.");
                case (CheckResult::TRIVIAL): return false;
                default: throw _UTIL_MSG_EXCEPT("Don't understand the result of simplify.");
            }
        }
    }
    return true;
}

bool consolidate_new_constraint_into_previous(
  const MultiPointConstraint& new_constraint,
  const std::map<GlobalIndex, MultiPointConstraint*> dep_dof_to_mpc_map,
  std::map<GlobalIndex, std::set<MultiPointConstraint*>>& indep_dof_to_mpcs_map)
{
    const GlobalIndex dependent_dof = new_constraint.dep_dof;
    // PART 2 - Examine the independent DOFs of previous constraints.  If any are the same as the
    // dependent DOF of the
    //          new constraint, substitue the new constraint into the previous.
    auto find_iter = indep_dof_to_mpcs_map.find(dependent_dof);
    if (find_iter != indep_dof_to_mpcs_map.end())
    {
        if (find_iter->second.size() > 0)
        {
            auto& prev_constraints = find_iter->second;
            // Loop over all previous constraints that have the new constraint's dependent DOF as a
            // independent DOF
            for (auto& prev_constraint : prev_constraints)
            {
                MultiPointConstraint* prev_mpc =
                  dynamic_cast<MultiPointConstraint*>(prev_constraint);
                if (prev_mpc != nullptr)
                {
                    // Loop through independent DOFs in prev. constraint to find one to substitue
                    for (auto i = 0; i < prev_mpc->num_terms(); ++i)
                    {
                        if (prev_mpc->indep_dof_terms[i].dof == dependent_dof)
                        {
                            prev_mpc->substitute_term(i, new_constraint);
                            auto result = prev_mpc->simplify();
                            switch (result)
                            {
                                case (CheckResult::GOOD): break;
                                case (CheckResult::BAD): throw _UTIL_MSG_EXCEPT("Bad constraint.");
                                case (CheckResult::TRIVIAL):
                                    throw _UTIL_MSG_EXCEPT(
                                      "Unexpected - shouldn't discover a trivial constraint while "
                                      "subbing a new constraint into previous");
                                default:
                                    throw _UTIL_MSG_EXCEPT(
                                      "Don't understand the result of simplify.");
                            }

                            break; // No need to continue looping once you find the independent DOF
                                   // to replace.
                        }
                    }

                    // Recalculate constraint_map_by_indep_dof_ for this constraint now that
                    // substitution is done
                    for (auto i = 0; i < prev_mpc->num_terms(); ++i)
                    {
                        // Does nothing if prev_mpc is already in constraint_map_by_indep_dof_[i]
                        // (property of STL set)
                        indep_dof_to_mpcs_map[prev_mpc->indep_dof_terms[i].dof].insert(prev_mpc);
                    }
                }
                else
                    throw _UTIL_MSG_EXCEPT(
                      "The dependent variable of this constraint has already been assigned to a "
                      "constant variable.  This shouldn't happen unless a DoF is overconstrained.");
            } // End loop over previous constraints that have independent DOFs that are the new
              // constraint's dependent DOF
        }     // End check to see if there are any entries in the set for
              // constraint_map_by_indep_dof_[dependent_dof]

        // Now, no constraint should have the current constraint's dependent DOF as an independent
        // DOF.
        indep_dof_to_mpcs_map.erase(dependent_dof);
    }
    else
    {
        // If no previous constraints have the new constraint's dependent DOF as an independent, all
        // is well.
        return true;
    }
    // End check to see if there is an entry for the new constraint's dependent DoF
    return true;
}

static bool add_multipoint_constraint(
  MultiPointConstraint& new_constraint,
  std::map<GlobalIndex, MultiPointConstraint*>& dep_dof_to_mpc_map,
  std::map<GlobalIndex, std::set<MultiPointConstraint*>>& indep_dof_to_mpcs_map)
{
    // The logic for this function is explained very well in Clint Chapman's dissertation page 26
    // and 27

    // Procedure for adding a constraint:
    // 1. Replace all dof on rhs which are already a dependent dof.
    // 2. If the new constraint's dependent dof is already a dependent, switch with one of the
    // independent dof's on rhs 2a. Replace the dependent dof which is now on rhs
    // 3. This constraint should be unique.
    // 4. Replace the new dependent dof in all existing constraints.

    // First simplify the constraint to ensure it is in the form:
    // [DependentDoF] = [IndependentDoF_0] * [IndependentFactor_0] + ... + RHSConstant
    auto result = new_constraint.simplify();
    switch (result)
    {
        case (CheckResult::GOOD): break;
        case (CheckResult::BAD):
            throw _UTIL_MSG_EXCEPT("A constraint was given that is impossible to satify.");
        case (CheckResult::TRIVIAL): return false;
        default: throw _UTIL_MSG_EXCEPT("Unrecognized result from simplify.");
    }

    // std::cout << "Trying to add " << *new_constraint << std::endl;

    // Step 1: Apply all previous constraints to the new constraint
    auto nontrivial = consolidate_previous_constraints_into_new(new_constraint,
                                                                dep_dof_to_mpc_map,
                                                                indep_dof_to_mpcs_map);
    // If the constraint turns out to be trivial, then we are done here.
    if (nontrivial == false)
        return false;

    // std::cout << "  After consoliding previous " << *new_constraint << std::endl;

    // Step 2: Apply new constraint to all previous constraints
    nontrivial = consolidate_new_constraint_into_previous(new_constraint,
                                                          dep_dof_to_mpc_map,
                                                          indep_dof_to_mpcs_map);
    // If the constraint turns out to be trivial, then we are done here.
    if (nontrivial == false)
        return false;

    // std::cout << "  After consoliding into previous " << *new_constraint << std::endl;

    // Incorporate the new constraint into the maps
    dep_dof_to_mpc_map[new_constraint.dep_dof] = &new_constraint;
    for (auto i = 0; i < new_constraint.num_terms(); ++i)
    {
        indep_dof_to_mpcs_map[new_constraint.indep_dof_terms[i].dof].insert(&new_constraint);
    }

    // The constraint is unique
    return true;
}

static void resolve_conflict(
  MultiPointConstraint& multi_point_constraint,
  const DirichletConstraint& dirichlet_constraint,
  const bool multi_point_constraint_is_new,
  std::map<GlobalIndex, MultiPointConstraint*>& dep_dof_to_mpc_map,
  std::map<GlobalIndex, std::set<MultiPointConstraint*>>& indep_dof_to_mpcs_map)
{
    // if the multi-point constraint has the same dependent DoF as the Dirichlet constraint.
    if (multi_point_constraint.dep_dof == dirichlet_constraint.dep_dof)
    {
        if (multi_point_constraint.num_terms() == 0)
        {
            // The constraint has no RHS terms (other than the constant)
            throw _UTIL_MSG_EXCEPT(
              "The conflict between the multi-point constraint and Dirichlet constraint cannot be "
              "resolved because the multi-point constraint has no independent DoFs to swap terms "
              "with.");
        }
        else
        {
            // We need to swap the dependent DoF in the multi-point constraint to remove the
            // conflict
            if (!multi_point_constraint_is_new)
            {
                // First, take this multi-point constraint out of the system of constraints by
                // removing it from the maps
                dep_dof_to_mpc_map.erase(multi_point_constraint.dep_dof);
                for (auto i = 0; i < multi_point_constraint.num_terms(); ++i)
                    indep_dof_to_mpcs_map[multi_point_constraint.indep_dof_terms[i].dof].erase(
                      &multi_point_constraint);
            }

            // Swap the dependent DoF with an independent DoF so that this previous constraint no
            // longer shares the same dependent DoF as the new constraint.
            auto swap_term = 0;
            multi_point_constraint.swap_dep_dof_with_indep(swap_term);
            // Move the term into the value BC terms and remove the independent DoF term since it is
            // not an independent DoF
            multi_point_constraint.dirichlet_terms.push_back(
              DirichletTerm(dirichlet_constraint,
                            multi_point_constraint.indep_dof_terms[swap_term].factor));
            multi_point_constraint.remove_term(swap_term);

            // Apply the modified multi-point constraint to all multi-point constraints
            auto nontrivial = consolidate_new_constraint_into_previous(multi_point_constraint,
                                                                       dep_dof_to_mpc_map,
                                                                       indep_dof_to_mpcs_map);
            // If the constraint turns out to be trivial, then we can get rid of it
            if (nontrivial == false)
            {
                return;
            }
            else if (!multi_point_constraint_is_new)
            {
                // Reincoporate the multi-point constraint since it is nontrivial
                dep_dof_to_mpc_map[multi_point_constraint.dep_dof] = &multi_point_constraint;
                for (auto i = 0; i < multi_point_constraint.num_terms(); ++i)
                {
                    indep_dof_to_mpcs_map[multi_point_constraint.indep_dof_terms[i].dof].insert(
                      &multi_point_constraint);
                }
            }
        }
        // Now the multi-point constraint has a different dependent DoF as the Dirichlet constraint
    }

    // Check to see if the multi-point constraint has the dependent DoF of the dirichlet constraint
    // as an independent DoF
    for (auto i = 0; i < multi_point_constraint.num_terms(); i++)
    {
        if (multi_point_constraint.indep_dof_terms[i].dof == dirichlet_constraint.dep_dof)
        {
            multi_point_constraint.dirichlet_terms.push_back(
              DirichletTerm(dirichlet_constraint,
                            multi_point_constraint.indep_dof_terms[i].factor));
            multi_point_constraint.remove_term(i);
            break;
        }
    }
}

static void incorporate_dirichlet_constraint(
  const DirichletConstraint& new_constraint,
  std::map<GlobalIndex, MultiPointConstraint*>& dep_dof_to_mpc_map,
  std::map<GlobalIndex, std::set<MultiPointConstraint*>>& indep_dof_to_mpcs_map)
{
    auto find_iter = dep_dof_to_mpc_map.find(new_constraint.dep_dof);
    // if a previous constraint has the new constraint's dependent DOF.
    if (find_iter != dep_dof_to_mpc_map.end())
    {
        MultiPointConstraint* previous_constraint =
          dynamic_cast<MultiPointConstraint*>(find_iter->second);
        if (previous_constraint != nullptr)
        {
            // A previous multi-point constraint has the new constraint's dependent DOF.
            resolve_conflict(*previous_constraint,
                             new_constraint,
                             false,
                             dep_dof_to_mpc_map,
                             indep_dof_to_mpcs_map);
        }
        else
        {
            // Something other than a multi-point constraint has the same dependent DoF, something
            // is over-constrained
            throw _UTIL_MSG_EXCEPT(
              "Another Dirichlet constraint already shares the same dependent DoF for DoF: " +
              std::to_string(new_constraint.dep_dof));
        }
    }

    // Check to see if previous constraints have this DoF as an independnet DoF
    auto find_iter2 = indep_dof_to_mpcs_map.find(new_constraint.dep_dof);
    if (find_iter2 != indep_dof_to_mpcs_map.end())
    {
        auto& previous_constraintsWithDoF = find_iter2->second;
        for (auto previous_constraint : previous_constraintsWithDoF)
        {
            MultiPointConstraint* previous_mpc =
              dynamic_cast<MultiPointConstraint*>(previous_constraint);
            if (previous_mpc != nullptr)
            {
                resolve_conflict(*previous_mpc,
                                 new_constraint,
                                 false,
                                 dep_dof_to_mpc_map,
                                 indep_dof_to_mpcs_map);
            }
        }
        indep_dof_to_mpcs_map.erase(find_iter2);
    }
    // Now all the independent DoFs in existing multi-point constraints that match this dependent
    // DoF have been moved to the dirichlet_terms

    // If we get here, no other constraint has the same dependent DoF (whether it before when this
    // method started or after a previous multi-point constraint was modified.
}


//////////////////////////////////////////////////////////////////////////////////////////////////
// Header-functions definitions

void consolidate_multipoint_constraints(
  const std::vector<DirichletConstraint>& dirichlet_constraints,
  std::vector<MultiPointConstraint>& multipoint_constraints)
{
    /// A map with the value corresponding to dependent dof and value is a pointer to
    /// the associated MultiPointConstraint
    /// NOTE: There is only one constraint per a dependent dof
    std::map<GlobalIndex, MultiPointConstraint*> dep_dof_to_mpc_map;

    /// A map with the value corresponding to independent dof and value is a pointer to
    /// a set of MultiPointConstraint's where the dof is an independent.
    /// NOTE: A dof can be a independent DoF in many multipoint constraints
    std::map<GlobalIndex, std::set<MultiPointConstraint*>> indep_dof_to_mpcs_map;

    for (auto& mpc : multipoint_constraints)
    {
        add_multipoint_constraint(mpc, dep_dof_to_mpc_map, indep_dof_to_mpcs_map);
    }

    for (auto& dc : dirichlet_constraints)
    {
        incorporate_dirichlet_constraint(dc, dep_dof_to_mpc_map, indep_dof_to_mpcs_map);
    }

    std::vector<MultiPointConstraint> nontrivial_mpcs;
    for (auto& key_value : dep_dof_to_mpc_map)
    {
        nontrivial_mpcs.push_back(*key_value.second);
    }
    multipoint_constraints.swap(nontrivial_mpcs);
}
}
