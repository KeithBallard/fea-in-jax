import meshio

import fea_traditional as test
from helper import *

import numpy as np

# General notes:
# 1) It might be helpful to inherit from jax.array and add labels for axes to
#    help with debugging and enable a higher level description of operations.


# mesh = meshio.read(f"test_meshes/polygon_mesh_{0.05}.vtk")
# points = np.array(mesh.points, dtype=np.float32)
# cells = np.array(mesh.cells[1].data, dtype=np.uint64)

# Make the mesh:
#    <-----3----->
#  2 o-----------o 1 ^
#    |   2    /  |   |
#    |     /     |   2
#    |  /     1  |   |
#  3 o-----------o 0 V
#    ^ origin (0, 0)
mesh = meshio.Mesh(
    points=[
        [3.0, 0.0, 0.0],
        [3.0, 2.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 0.0],
    ],
    cells=[
        ("triangle", [[0, 1, 3], [2, 3, 1]]),
    ],
)

mesh.write(get_output("two_tri.vtk"))

points = np.array(mesh.points, dtype=np.float32)[:, 0:2]
cells = np.array(mesh.cells[0].data, dtype=np.uint64)

# Sizes of arrays
U = 2  # number of solution components
V = points.shape[0]  # number of vertices
E = cells.shape[0]  # number of elements
M = 2  # number of material parameters
F = V * U  # number of DoFs
fe_type = FiniteElementType(
    cell_type=CellType.triangle,
    family=ElementFamily.P,
    basis_degree=1,
    lagrange_variant=LagrangeVariant.equispaced,
    quadrature_type=QuadratureType.default,
    quadrature_degree=0,
)
Q = get_quadrature(fe_type=fe_type)[0].shape[0] # number of quadrature points

n_cell_per_vert = get_n_cells_per_vert(points, cells)
print("n_cell_per_vert", n_cell_per_vert)

# Boundary Conditions
#  2 >o-----------o --> 1
#     |        /  |
#     |     /     |
#     |  /        |
#  3 >o-----------o --> 0
#     ^           ^
# An array that is (# of constrainted DoFs, 2) with structure [point index][component of solution]
# Fixes left two points in x, fixes bottom two points in y, and moves right edge to the right
dirichlet_bcs = np.array([[0, 0], [0, 1], [1, 0], [2, 0], [3, 0], [3, 1]], dtype=np.uint64)
# Values of the Dirichlet boundary conditions matching 'dirichlet_bcs'
# Fixes bottom-left and moves top-right to the right by 1
dirichlet_values = np.array([1.0, 0.0, 1.0, 0.0, 0.0, 0.0])
print("dirichlet_bcs = \n", dirichlet_bcs)
print("dirichlet_values = ", dirichlet_values)

# Set material properties at the quadrature point level randomly seeded such that
# E = [90e9, 100e9] and nu = 0.25
tmp_mat_params = np.zeros(shape=(E, Q, M))
tmp_mat_params[..., 0] = 30e6
tmp_mat_params[..., 1] = 0.25
mat_params_eqm = jnp.array(tmp_mat_params)

element_batches = [
    ElementBatch(
        fe_type=fe_type,
        n_dofs_per_basis=U,
        connectivity_en=cells,
        constitutive_model=elastic_isotropic,
        material_params=mat_params_eqm,
    )
]

#for solver_type in SolverType:

solver_type = LinearSolverType.CG_JAX_SCIPY_W_INFO
print(f'Solving with {solver_type}')

# Solve the boundary value problem
"""u, residual = solve_bvp(
    element_residual_func=linear_elasticity_residual,
    vertices_vd=points,
    element_batches=element_batches,
    dirichlet_bcs=dirichlet_bcs,
    dirichlet_values=dirichlet_values,
    linear_solver_type=solver_type,
)"""

B = len(element_batches)
V = points.shape[0]
D = points.shape[1]

sharding = False

material_params_beqm = [b.material_params for b in element_batches]
x_bend = [
            #tri_mesh_to_jax(vertices=points, cells=b.connectivity_en)
            mesh_to_jax(vertices=points, cells=b.connectivity_en)
            for b in element_batches
        ]

assembly_map_b = [
            mesh_to_sparse_assembly_map(n_vertices=V, cells=b.connectivity_en)
            for b in element_batches
        ]

        # Initial solution global unraveled vector
u_0_g = jnp.zeros((V * D,))

dirichlet_dofs = jnp.array(D * dirichlet_bcs[:, 0] + dirichlet_bcs[:, 1])
        # print('dirichlet_dofs: ', dirichlet_dofs)
        # Global unraveled
dirichlet_values_g = (jnp.zeros_like(u_0_g).at[dirichlet_dofs].set(dirichlet_values))
dirichlet_mask_g = jnp.zeros_like(u_0_g).at[dirichlet_dofs].set(1.0)

element_residual_func = jax.tree_util.Partial(linear_elasticity_residual)
constitutive_model_b = [jax.tree_util.Partial(b.constitutive_model) for b in element_batches]
xi_bqp, W_bq = zip(*[get_quadrature(fe_type=b.fe_type) for b in element_batches])
phi_bqn, dphi_dxi_bqnp = zip(
    *[
        eval_basis_and_derivatives(fe_type=b.fe_type, xi_qp=xi_bqp[i])
        for i, b in enumerate(element_batches)
    ]
)

#print("dphi_dxi_bqnp = ", dphi_dxi_bqnp)


callBuild = False

if callBuild:
    jac = buildJacobian(
        element_residual_func=element_residual_func,
        constitutive_model_b=constitutive_model_b,
        material_params_beqm=material_params_beqm,
        x_bend=x_bend,
        dphi_dxi_bqnp=dphi_dxi_bqnp,
        W_bq=W_bq,
        assembly_map_b=assembly_map_b,
        u_g=u_0_g, #this is the only difference in call signature
        #dirichlet_values_g=dirichlet_values_g, #make sure this is the correct thing we're passing. 
        #dirichlet_mask_g=dirichlet_mask_g, 
        )
else:
    import jacobian_test_data
    k1,B1,D1 = jacobian_test_data.buildTestValues()

    xlen = len(x_bend)
    outputByBatch = list()

    E = x_bend[0].shape[0]
    N = x_bend[0].shape[1]
    D = x_bend[0].shape[2]
    Q = W_bq[0].shape[0]
    print(f"Q = {Q}")


    for i in range(xlen):

        x_end = x_bend[i]
        constitutive_model=constitutive_model_b[i],
        material_params_eqm=material_params_beqm[i],
        x_end=x_bend[i],
        dphi_dxi_qnp=dphi_dxi_bqnp[i],
        W_q=W_bq[i],
        assembly_map=assembly_map_b[i]

        



        u_end = transform_global_unraveled_to_element_node(
            assembly_map, u_0_g, N_ge=E, N_n=N, N_u=D
        )
        u_et = u_end.reshape(E, N*D)

        print(dphi_dxi_bqnp[0].shape)
        print(x_end[0][0].shape)
        print(u_et[0].shape)

        def residual_kernel(u_t, x_nd, material_params_qm):
            u_nd = u_t.reshape(N, D)
            R_nd = element_residual_func(
                constitutive_model=constitutive_model[i],
                u_nd=u_nd,
                x_nd=x_nd[i],
                dphi_dxi_qnp=dphi_dxi_qnp[i],
                W_q=W_q[i],
                material_params_qm=material_params_qm[i],
                internal_state_qi=None #follow this variable through to make sure it's not breaking anything
            )
            return R_nd[0].reshape(N*D)
        
        #residual_kernel(u_et[0], x_end[0], material_params_eqm[0])


        J_qpd = jnp.einsum("nd,qnp->qpd", x_end[0][0], dphi_dxi_qnp[0])
        G_qpd = jnp.linalg.inv(J_qpd).transpose(0, 2, 1)
        det_J_q = jnp.linalg.det(J_qpd)
        #jax.debug.print("det_J_q = {x}", x=det_J_q)

        dphi_dx_qnd = jnp.einsum("qpd,qnp->qnd", G_qpd, dphi_dxi_qnp[0])
        assert dphi_dx_qnd.shape[1] == 3
        assert dphi_dx_qnd.shape[2] == 2
        assert jnp.linalg.norm(dphi_dx_qnd-B1) < 1e-10

        #print(dphi_dx_qnd.shape)
        #print(u_end.shape)

        du_dx_qdd = jnp.einsum("qnd,ni->qid", dphi_dx_qnd, u_end[0])
        eps_qdd = 0.5 * (du_dx_qdd + du_dx_qdd.transpose((0, 2, 1)))
        constModelJac = jax.jacfwd(constitutive_model[0])(eps_qdd, material_params_eqm[0][0])
        print("Something funny is going on, and I don't like it")
        
        print(D1.shape)
        print(len(constModelJac))
        print(constModelJac[0].shape)
        print(constModelJac[1].shape)

        assert len(constModelJac[0].shape) == len(D1.shape)+1 #extra 1 because nothing can ever be easy
        assert jnp.linalg.norm(constModelJac[0]-D1) < 1e-10

        def residual_kernel_tmp(u_t):
            u_nd = u_t.reshape(N, D)
            R_nd = element_residual_func(
                constitutive_model=constitutive_model[0],
                u_nd=u_nd,
                x_nd=x_end[0][0],
                dphi_dxi_qnp=dphi_dxi_qnp[0],
                W_q=W_q[0],
                material_params_qm=material_params_eqm[0][0],
                internal_state_qi=None
            )
            return R_nd.reshape(N*D)

        J_tmp = jax.jacfwd(residual_kernel, argnums=0)(u_et[0], x_end[0], material_params_eqm[0])
        #print(f'J_tmp = {J_tmp}')
        v = jax.random.normal(jax.random.key(0), shape=(N * D,))
        #print(f'J_tmp * v = {jnp.dot(J_tmp, v)}')
        #print(f'jvp(v) = {jax.jvp(residual_kernel_tmp, (u_et[0],), (v,))}')

        def buildfwd(u_t, x_nd, material_params_qm):
            return jax.jacfwd(residual_kernel, argnums=0)(u_t, x_nd, material_params_qm)
    
        buildfwdv = jax.vmap(buildfwd)

        jac = buildfwdv(u_et, x_end, material_params_eqm)
        #print(f'J_vmap = {jac[0]}')
        #print("expected values",k1*2)


        assert len(jac[0].shape) == len(k1.shape)
        print("difference",jnp.linalg.norm(jac[1] - k1*2))
        print("relative error",jnp.linalg.norm(jac[1] - k1*2)/jnp.linalg.norm(k1*2))

        #now to compare with function definition
        createJVPFunction(
            element_residual_func=element_residual_func,
            constitutive_model_b=constitutive_model_b,
            material_params_beqm=material_params_beqm,
            x_bend=x_bend,
            dphi_dxi_bqnp=dphi_dxi_bqnp,
            W_bq=W_bq,
            assembly_map_b=assembly_map_b,
            u_0_g=u_0_g,
            dirichlet_values_g=dirichlet_values_g,
            dirichlet_mask_g=dirichlet_mask_g,
            dirichlet_dofs=dirichlet_dofs,
            dirichlet_values=jnp.array(dirichlet_values),
        )

        key = jax.random.key(25061998)
        test_vec = jax.random.normal(key,(8,))

        trueJac = jacobian_test_data.buildFullJac()
        #print("true dot test 1",jnp.dot(test_vec,trueJac*2))

        print("""jac[0]""")

        """print(assembly_map_b[0])
        print(dir(assembly_map_b[0]))
        print(assembly_map_b[0].todense())
        print(assembly_map_b[0].indices)
        print(assembly_map_b[0].indptr)
        print(assembly_map_b[0].data)"""


        b = assembly_map_b[0].to_bcoo()

        c = b.indices[0,b.indices[:,:,1].argsort()][0,:,0]
        print(c)
        print(c.shape)

        d = jnp.vstack((2*c,2*c+1)).transpose().reshape((12,))

        print("d",d)

        djac = d.reshape(2,6)


        def getIndecies(x,y):
            return jnp.array([x,y])
        getIndecies = jax.vmap(jax.vmap(getIndecies,(0,None)),(None,0))

        indexflat0 = getIndecies(djac[0],djac[0]).reshape((36,2))
        indexflat1 = getIndecies(djac[1],djac[1]).reshape((36,2))

        valuesflat0 = jac[0].reshape((36))
        valuesflat1 = jac[1].reshape((36))


        flatvalues = jnp.append(valuesflat0,valuesflat1)
        flatindex = jnp.append(indexflat0,indexflat1,axis=0)

        import jax.experimental.sparse as sparse

        exampleSparseJac = sparse.BCOO((flatvalues.squeeze(),flatindex),shape=(8,8))



        print("dot test",jnp.dot(exampleSparseJac.todense(),test_vec))


        checkGMRES = jax.scipy.sparse.linalg.gmres(exampleSparseJac,test_vec)
        exampleCSR = sparse.BCSR.from_bcoo(exampleSparseJac)

        skipSP = 1
        if(skipSP == 0):
            checkSPsolve = sparse.linalg.spsolve(jnp.array(exampleCSR.data),jnp.array(exampleCSR.indices),jnp.array(exampleCSR.indptr),test_vec,reorder=0)
            #weird writeback error. Will invetigate

        exampleSolveCG = jax.scipy.sparse.linalg.cg(A=exampleSparseJac, b=test_vec)
        exampleSolveGMRES = jax.scipy.sparse.linalg.gmres(A=exampleSparseJac, b=test_vec,tol=0.034,maxiter=50)
        exampleSolveSOLVE = jax.scipy.linalg.solve(a=exampleSparseJac.todense(), b=test_vec)
        exampleSolveLU = jax.scipy.linalg.lu_solve(jax.scipy.linalg.lu_factor(exampleSparseJac.todense()), b=test_vec)

        print("CG matrix result \n",exampleSolveCG[0])
        print("GMRES matrix result \n",exampleSolveGMRES[0])
        print("SOLVE matrix result \n",exampleSolveSOLVE)
        print("LU matrix result \n",exampleSolveLU)


        #print(b.indices[b.indices[:,:,1].argsort()])
        #print(dir(b))

        #TODO-----------------------------------------------------
        #check residuals of all the solver outputs Ax - b
        #check code path of different calls
    
        print("residual SOLVE matrix:", jnp.linalg.norm(exampleSparseJac.todense() @ exampleSolveSOLVE - test_vec))
        print("residual LU matrix:", jnp.linalg.norm(exampleSparseJac.todense() @ exampleSolveLU - test_vec))
        print("residual CG matrix:", jnp.linalg.norm(exampleSparseJac.todense() @ exampleSolveCG[0] - test_vec))

        print(jnp.linalg.cond(exampleSparseJac.todense()))



print("finished simplish")
