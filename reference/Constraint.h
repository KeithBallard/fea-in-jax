#pragma once

#include "sparse_linear_algebra/Defines.h"

#include <list>
#include <map>
#include <memory>
#include <vector>

// Math includes
#include "simple_math/EigenDefines.h"

// Utility includes
#include "utility/Exceptions.h"
#include "utility/IOUtils.h"

namespace sparse_linear_algebra
{
/// This class is meant to encapsulate the data needed to describe a constraint to the
/// normal system of equations that results from a FEA assumbly.
/// This class will handle both multi-point constraints and Dirchlet boundary conditions.
/// Add an Constraint object to a ConstraintManager instance.
class Constraint
{
  public:
    enum class CheckResult : short
    {
        BAD,
        GOOD,
        TRIVIAL
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


    virtual ~Constraint() {}

    /// Evaluates the additional equation to get the value of the dependent DoF given the
    /// independent DoF values. If there are no independent DoFs, then provide an empty vector.
    virtual double evaluate(const simple_math::GeneralVector& independent_doF_values) const = 0;

    /// Returns the dependent DoF index
    const GlobalIndex& get_dep_dof() const
    {
        return dep_dof_;
    }

    /// Returns the RHS constant in the equation.  This may be overloaded to incorporate things that
    /// change the RHS cosntant in an additional equation.
    virtual double get_constant() const
    {
        return rhs_constant_;
    }

    virtual void print(std::ostream& o) const
    {
        o << '[' << dep_dof_ << "] = ";
        o << rhs_constant_;
    };

    friend std::ostream& operator<<(std::ostream& o, const Constraint& e)
    {
        e.print(o);
        return o;
    };

  protected:
    /// The global index of the dependent DoF on the LHS of this equation
    GlobalIndex dep_dof_;

    /// The RHS constant in the equation
    double rhs_constant_;

    /// Tolerance for zero terms and comparison if two factors are equal
    static double get_tolerance()
    {
        return 1e-16;
    }
};

} // end namespace sparse_linear_algebra
