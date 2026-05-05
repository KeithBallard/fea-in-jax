#pragma once

#include "Constraint.h"

namespace sparse_linear_algebra
{
/// This class defines a Dirchlet constraint where a DoF is specified to have a particular
/// value.  There can only be one Dirchlet constraint for a given DoF.
struct DirichletConstraint
{
    /// The global index of the dependent DoF on the LHS of this equation
    GlobalIndex dep_dof;
    /// The RHS constant in the equation
    double rhs_constant;

    inline const std::string str() const
    {
        std::string s = "[" + std::to_string(dep_dof) + "] = " + std::to_string(rhs_constant);
        return s;
    }
};

}
