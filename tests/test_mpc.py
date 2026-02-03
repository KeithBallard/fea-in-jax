import sys
import os
import importlib.util

# Load module directly to bypass package __init__ which requires jax (missing in env)
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/fe_jax/multipoint_constraints.py"))
spec = importlib.util.spec_from_file_location("multipoint_constraints", src_path)
mpc_module = importlib.util.module_from_spec(spec)
sys.modules["multipoint_constraints"] = mpc_module
spec.loader.exec_module(mpc_module)

MultiPointConstraint = mpc_module.MultiPointConstraint
DirichletConstraint = mpc_module.DirichletConstraint
consolidate_multipoint_constraints = mpc_module.consolidate_multipoint_constraints

def test_mpc_consolidation():
    # 1. DOF 0 = DOF 1 + 0.1 * DOF 2
    mpc1 = MultiPointConstraint(dep_dof=0, indep_dofs=[1, 2], factors=[1.0, 0.1])
    
    # 2. DOF 1 = 0.1 * DOF 3
    mpc2 = MultiPointConstraint(dep_dof=1, indep_dofs=[3], factors=[0.1])
    
    # 3. DOF 3 = 1 (Dirichlet)
    dc = DirichletConstraint(dep_dof=3, value=1.0)
    
    mpcs = [mpc1, mpc2]
    dirichlet_constraints = [dc]
    
    consolidated_mpcs = consolidate_multipoint_constraints(dirichlet_constraints, mpcs)
    
    # Sort for deterministic checks
    consolidated_mpcs.sort(key=lambda x: x.dep_dof)
    
    # Expected:
    # 1. DOF 0 = 0.1 + 0.1 * DOF 2
    #    Explanation: 
    #    DOF 0 = DOF 1 + 0.1 * DOF 2
    #    DOF 1 = 0.1 * DOF 3 = 0.1 * 1.0 = 0.1
    #    DOF 0 = 0.1 + 0.1 * DOF 2
    
    # 2. DOF 1 = 0.1
    #    (as above)
    
    assert len(consolidated_mpcs) == 2
    
    # Check DOF 0
    mpc_0 = next(m for m in consolidated_mpcs if m.dep_dof == 0)
    assert abs(mpc_0.get_total_constant() - 0.1) < 1e-15
    assert len(mpc_0.indep_dof_terms) == 1
    assert 2 in mpc_0.indep_dof_terms
    assert abs(mpc_0.indep_dof_terms[2] - 0.1) < 1e-15
    
    # Check DOF 1
    mpc_1 = next(m for m in consolidated_mpcs if m.dep_dof == 1)
    assert abs(mpc_1.get_total_constant() - 0.1) < 1e-15
    assert len(mpc_1.indep_dof_terms) == 0

if __name__ == "__main__":
    test_mpc_consolidation()
    print("Test passed!")
