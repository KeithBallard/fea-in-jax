#include "sparse_linear_algebra/constraints/MultiPointConstraint.h"

#include "utility/Exceptions.h"

#include <cmath>
#include <iostream>

using namespace std;

namespace sparse_linear_algebra
{
MultiPointConstraint::MultiPointConstraint(const GlobalIndex dependent_doF,
                                           const std::vector<GlobalIndex>& independent_doFs,
                                           const std::vector<double>& independent_dof_factors,
                                           const double rhs_constant_a,
                                           const bool immutable)
{
    dep_dof      = dependent_doF;
    rhs_constant = rhs_constant_a;

    if (independent_doFs.size() != independent_dof_factors.size())
        throw _UTIL_MSG_EXCEPT_C("The number of DoFs does not match the number of factors given.");

    for (auto i = 0; i < independent_doFs.size(); i++)
    {
        add_new_term(independent_doFs[i], independent_dof_factors[i]);
    }
}

double MultiPointConstraint::evaluate(
  const simple_math::GeneralVector& independent_doF_values) const
{
    if (independent_doF_values.size() != indep_dof_terms.size())
        throw _UTIL_MSG_EXCEPT_C("Number of independent DoF values provided does not match the "
                                 "number of independent DoF's in the equation.");

    double value = get_total_constant();
    for (auto i = 0; i < indep_dof_terms.size(); i++)
    {
        value += indep_dof_terms[i].factor * independent_doF_values(i);
    }
    return value;
}

double MultiPointConstraint::get_total_constant() const
{
    double additional_rhs = 0.0;
    for (auto& term : dirichlet_terms)
    {
        additional_rhs += term.factor * term.dirichlet_value;
    }
    return additional_rhs + rhs_constant;
}

void MultiPointConstraint::add_new_term(const GlobalIndex independent_dof, const double factor)
{
    // If the factor is 0, then we don't need it
    if (factor == 0.0)
        return;

    // Make sure this DoF doesn't already exist
    for (auto i = 0; i < indep_dof_terms.size(); i++)
    {
        if (independent_dof == indep_dof_terms[i].dof)
        {
            indep_dof_terms[i].factor += factor;
            return;
        }
    }

    // A term for this DoF did not exist, to make a new one
    indep_dof_terms.push_back(LinearTerm(independent_dof, factor));
}

void MultiPointConstraint::substitute_term(const TermIndex& term_to_replace,
                                           const MultiPointConstraint& eqn_to_insert)
{
    if (term_to_replace > indep_dof_terms.size())
        throw _UTIL_MSG_EXCEPT_C(
          "The term provided is outside the range of the terms in equation.");

    // Consolidates the process of substituting another MultiPointConstraint object into a term of
    // this equation
    double old_factor = indep_dof_terms[term_to_replace].factor;
    if (eqn_to_insert.num_terms() > 0)
    {
        substitute_term(term_to_replace,
                        eqn_to_insert.indep_dof_terms[0].dof,
                        eqn_to_insert.indep_dof_terms[0].factor * old_factor);
        for (TermIndex j = 1; j < eqn_to_insert.num_terms(); j++)
        {
            add_new_term(eqn_to_insert.indep_dof_terms[j].dof,
                         eqn_to_insert.indep_dof_terms[j].factor * old_factor);
        }
    }
    else
    {
        remove_term(term_to_replace);
    }
    rhs_constant += eqn_to_insert.rhs_constant * old_factor;
}

void MultiPointConstraint::substitute_term(const TermIndex& term_to_replace,
                                           const GlobalIndex independent_dof,
                                           const double& factor)
{
    // If master already exists just add in factor
    for (auto i = 0; i < num_terms(); i++)
    {
        if (i != term_to_replace && indep_dof_terms[i].dof == independent_dof)
        {
            indep_dof_terms[i].factor += factor;
            remove_term(term_to_replace);
            return;
        }
    }

    indep_dof_terms[term_to_replace].dof    = independent_dof;
    indep_dof_terms[term_to_replace].factor = factor;
}

void MultiPointConstraint::remove_term(const TermIndex& term)
{
    if (term >= num_terms())
        throw _UTIL_MSG_EXCEPT_C("Term to remove is out of range.");

    indep_dof_terms.erase(indep_dof_terms.begin() + term);
}

void MultiPointConstraint::swap_dep_dof_with_indep(const TermIndex& term_to_swap)
{
    if (term_to_swap >= num_terms())
        throw _UTIL_MSG_EXCEPT_C("Term to swap is out of range.");

    double depend_factor                 = -indep_dof_terms[term_to_swap].factor;
    GlobalIndex new_depend_doF           = indep_dof_terms[term_to_swap].dof;
    indep_dof_terms[term_to_swap].dof    = dep_dof;
    indep_dof_terms[term_to_swap].factor = -1.0;
    dep_dof                              = new_depend_doF;
    divide_rhs(depend_factor);
}

void MultiPointConstraint::divide_rhs(const double& factor)
{
    // This is allowed to be called for an immutable equation because it is needed for
    // simplify_equation
    if (factor == 0)
        throw _UTIL_MSG_EXCEPT_C("Divide by zero");
    for (auto& term : indep_dof_terms)
        term.factor /= factor;
    rhs_constant /= factor;
}

CheckResult MultiPointConstraint::simplify()
{
    // Remove terms with factor of 0
    auto limit = num_terms();
    for (auto i = 0; i < num_terms(); i++)
    {
        if (fabs(indep_dof_terms[i].factor) < compare_tolerance)
        {
            remove_term(i);
            limit = num_terms();
            i--;
        }
    }

    limit             = num_terms();
    double dep_factor = 1.0;
    for (auto i = 0; i < limit; i++)
    {
        // If the dependent dof is on the RHS, the we need to fix it
        if (indep_dof_terms[i].dof == dep_dof)
        {
            dep_factor -= indep_dof_terms[i].factor;
            remove_term(i);
            limit = num_terms();
            // If the term with the dependant variable on the RHS cancels
            // out the LHS, then we need to pick a new dependent variable.
            if (dep_factor == 0.0)
            {
                // There are no other terms to make sure we don't end up with
                // the situation that 1 = 2 or something.
                if (num_terms() == 0)
                {
                    if (fabs(rhs_constant) > compare_tolerance)
                        return CheckResult::BAD;
                    else
                        return CheckResult::TRIVIAL;
                }
                dep_dof    = indep_dof_terms[0].dof;
                dep_factor = -indep_dof_terms[0].factor;
                remove_term(0);
                limit = num_terms();
            }
            divide_rhs(dep_factor);
            return CheckResult::GOOD;
        }
    }
    return CheckResult::GOOD;
}

const std::string MultiPointConstraint::str() const
{
    std::string s = "[" + std::to_string(dep_dof) + "] = ";
    for (auto i = 0; i < num_terms(); i++)
    {
        s += std::to_string(indep_dof_terms[i].factor) + " * [" +
             std::to_string(indep_dof_terms[i].dof) + "] + ";
    }
    s += std::to_string(get_total_constant());
    return s;
}
}
