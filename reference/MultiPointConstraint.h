#pragma once

#include "DirichletConstraint.h"

namespace sparse_linear_algebra
{
enum class CheckResult : short
{
    BAD,
    GOOD,
    TRIVIAL
};

/// A simple struct for storing a Dirchlet term in a multi-point constraint.
/// These terms are resolved to be part of the RHS constant when the equation is evaluated.
struct DirichletTerm
{
    const double dirichlet_value;
    double factor;

    DirichletTerm(const DirichletConstraint& dirch_constraint, const double& factor_a)
      : dirichlet_value(dirch_constraint.rhs_constant)
      , factor(factor_a)
    {
    }
    ~DirichletTerm() {}

  private:
    DirichletTerm() = delete;
};

/// A simple struct for storing a term relating the dependent DoF to
/// another independnet DoF.
struct LinearTerm
{
    GlobalIndex dof;
    double factor;

    LinearTerm(const GlobalIndex& dof_a, const double& factor_a)
      : dof(dof_a)
      , factor(factor_a)
    {
    }
    ~LinearTerm() {}

  private:
    LinearTerm();
};

/// This type of constraint provides what is commonly called multi-point constraints, which is a
/// linear additional equation.  In other words, the dependent DoF is a linear function of other
/// DoFs and a constant.
/// Stucture of equation:
/// [DependentDoF] = [IndependentDoF_0] * [IndependentFactor_0] + ... + RHSConstant
struct MultiPointConstraint
{
  public:
    using TermIndex = std::size_t;

    /// TODO explain parameters
    MultiPointConstraint(const GlobalIndex dependent_doF,
                         const std::vector<GlobalIndex>& independent_doFs,
                         const std::vector<double>& independent_dof_factors,
                         const double rhs_constant,
                         const bool immutable = false);
    virtual ~MultiPointConstraint() {}

    /// For this class, it just returns the constant.  \see AdditionalEquation::evaluate
    double evaluate(const simple_math::GeneralVector& independent_doF_values) const;
    /// For this class, the RHS constant is the normal constant plus any other terms that
    /// arise from dirichlet terms (independent DoFs that are specified to be a specific value)
    double get_total_constant() const;

    /// Returns the number of linear terms (does not include the value BC terms since they
    /// are evaluated as part of the RHS constant).
    inline TermIndex num_terms() const
    {
        return indep_dof_terms.size();
    }

    /// A term to the equation in the form of ... + [independentDoF] * [factor]
    /// If a term already exists for the independent DoF given in this equation, the
    /// given factor will be added to the existing one.
    void add_new_term(const GlobalIndex independent_doF, const double factor);
    /// Substitues another additional equation into a term in this equation.  This is useful
    /// if an independent variable in this equation becomes a dependent variable in another
    /// equation which would require that equation to be substituted in to maintain only
    /// independent vars on the RHS of the equation.
    void substitute_term(const TermIndex& term_to_replace,
                         const MultiPointConstraint& eqn_to_insert);

    /// Very similar to the previous verion of substitute_term, but this one will replace a term
    /// with another single term. If there is another term (not the one to replace) already
    /// exists with the same independent DoF as the term that will be substutited into the
    /// equation, the term to replace will just be deleted and the other term will be modified.
    void substitute_term(const TermIndex& term_to_replace,
                         const GlobalIndex independent_doF,
                         const double& factor);

    /// Removes a term from the equation
    void remove_term(const TermIndex& term);

    /// This method will make the current dependent variable an independent variable by swapping
    /// it with the specified independent variable, which become the dependent variable.
    void swap_dep_dof_with_indep(const TermIndex& termToSwap);

    /// Divides all factors on the RHS by the given factor
    void divide_rhs(const double& factor);

    /// Removes terms with 0.0 factors and ensures that the dependent DoF is not on both
    /// sides of the equation
    CheckResult simplify();

    const std::string str() const;

    /// The global index of the dependent DoF on the LHS of this equation
    GlobalIndex dep_dof;
    /// The global indices of the independent DoFs on the RHS of this equation
    std::vector<LinearTerm> indep_dof_terms;
    /// As independent DoFs are specifed as value bounday conditions, the terms are moved
    /// into the constant of the additional equation.
    std::vector<DirichletTerm> dirichlet_terms;
    /// The RHS constant in the equation
    double rhs_constant;

    static constexpr double compare_tolerance = 1e-16;
};

}
