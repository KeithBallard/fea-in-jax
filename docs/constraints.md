# Multi-Point Constraints and Constrained Jacobian Implementation

This document describes the design, math, and implementation of multi-point constraints (MPCs) and boundary conditions (BCs) in the JAX-based Finite Element Analysis (FEA) solver. Specifically, it details the transition from on-the-fly host-callback constraint enforcement to the highly efficient **Static Index-Mapping and Gather-Scatter** architecture.

---

## 1. How Linear Constraints are Used

In structural mechanics and multiphysics simulations, degrees of freedom (DoFs) are often subjected to linear algebraic relationships. These constraints fall into three main categories:

1. **Dirichlet Boundary Conditions (Prescribed Values)**:
   A subset of DoFs is set to a constant or prescribed value:
   $$u_d = \bar{u}_d$$

2. **Multi-Point Constraints (MPCs)**:
   A dependent degree of freedom ($u_i$) is expressed as a linear combination of independent degrees of freedom ($u_j$):
   $$u_{\text{dep}} = P \cdot u_{\text{indep}}$$
   where $P$ is a matrix of linear constraint coefficients. This is widely used for modeling:
   - **Periodic Boundary Conditions**: For unit-cell or representative volume element (RVE) analyses.
   - **Rigid Links / MPC Beams**: Modeling perfectly rigid connections between different parts of a structure.
   - **Joints and Hinges**: Restricting specific rotational/translational motion.

3. **Global Relations / Reduced Systems**:
   Global relationships where groups of DoFs are reduced out of the system of equations.

---

## 2. Implementation: Static Index-Mapping & Gather-Scatter (Architecture B)

Because the mesh topology and the constraint equations remain constant (static) throughout a simulation step or transient load path, the sparsity patterns of both the unconstrained and constrained Jacobians are completely static. 

This enables us to decouple the expensive index searching and algebraic block transformation from the iterative solver loop.

### Mathematical Formulation of Block Transformations
Applying the constraint matrix $P$ to the global stiffness matrix $K$ involves reducing out the dependent degrees of freedom ($d$) in favor of the independent degrees of freedom ($i$):
$$K_{\text{constrained}} = \begin{bmatrix} I & P^T \end{bmatrix} \begin{bmatrix} K_{ii} & K_{id} \\ K_{di} & K_{dd} \end{bmatrix} \begin{bmatrix} I \\ P \end{bmatrix}$$

This expands algebraically into four block contributions:
1. **Block 1 (Independent-Independent)**: $K_{ii}$
2. **Block 2 (Independent-Dependent)**: $K_{id} P$
3. **Block 3 (Dependent-Independent)**: $P^T K_{di}$
4. **Block 4 (Dependent-Dependent)**: $P^T K_{dd} P$

At the end of the global system, identity diagonal entries ($1.0$) are appended for the dependent rows to maintain a square, solvable system.

### Preprocessing Phase (Outside JAX JIT)
Before entering the JIT-compiled Newton-Raphson or solver loop, `precompute_constrained_jacobian_mapping` evaluates these algebraic blocks:
1. Generates unconstrained global Jacobian coordinates `(rows, cols)`.
2. Identifies all dependent DoFs (`dep_dofs`) and applies the constraint matrix $P$ block-by-block.
3. Tracks the original index in the raw element stiffness array (`J_ett`) and computes the associated algebraic weight multiplier (e.g., $1.0$, $P_{c, c'}$, $P_{r, r'}$, or $P_{r, r'} P_{c, c'}$).
4. Sorts the transformed coordinates, identifies unique coordinates, and compiles five static arrays:
   - `row_final`, `col_final`: Unique, static coordinates of the final constrained global Jacobian.
   - `source_indices`: Indices mapping entries in the raw unrolled element stiffness vector `J_ett` to their destinations.
   - `target_indices`: Destinations in the unique data vector of the constrained Jacobian.
   - `target_weights`: Algebraic multipliers combining constraint coefficients.

### Iterative Solver Phase (Inside JAX JIT)
In every Newton step, the constrained Jacobian is assembled in a single $O(N)$ GPU-optimized gather-scatter operation:
```python
# 1. Allocate unique data array for the constrained Jacobian
data_constrained = jnp.zeros((precomputed_constrained_jacobian_nnz,))

# 2. Scatter-add the raw stiffness contributions multiplied by constraint weights
data_constrained = data_constrained.at[target_indices].add(J_ett[source_indices] * target_weights)

# 3. Impose the 1.0 identity diagonals for dependent constraints equations
data_constrained = data_constrained.at[row_unique_shape:].set(1.0)

# 4. Construct JAX-native jsparse.COO matrix
return jsparse.COO(
    (data_constrained, row_final, col_final),
    shape=(u_f.shape[0], u_f.shape[0]),
)
```

---

## 3. Implementation Options and Design Decisions

During the development process, two primary architectures were evaluated:

### Option A: On-the-Fly Host-Callback (SpGEMM)
*   **Mechanism**: The unconstrained Jacobian was assembled inside JIT and passed via `pure_callback` to the host. The host called SciPy/CuPy to perform sparse-sparse matrix multiplication (SpGEMM) for $P^T K P$, saved the result in a registry, and passed a reference handle to the solver.
*   **Pros**: Simple to write; minimal preprocessing setup.
*   **Cons**:
    - **Massive Memory Overhead**: Required keeping both the unconstrained and constrained matrices in memory, plus temporary arrays during SpGEMM.
    - **Tracing Limitations**: Prevented solvers from accepting the Jacobian as a clean JAX-native `jsparse.COO` matrix, which hindered future compilation and solver breakout plans (e.g., JetSCI integration).
    - **Performance Bottleneck**: Executed index sorting, duplicate summing, and sparse matrix multiplication *on every single Newton iteration*.

### Option B: Precomputed Static Mapping & Gather-Scatter (Selected)
*   **Mechanism**: Precomputed the static sparsity maps once outside JIT, then assembled the constrained Jacobian inside JIT using fast index-mapping and array scatter-adds.
*   **Pros**:
    - **Extremely Fast**: Assembly is reduced to a single GPU kernel. No SpGEMM, sorting, or duplicates-summing during iterations.
    - **Highly Memory Efficient**: No unconstrained global Jacobian is ever allocated during the solve loop, reducing iterative peak memory footprint to the absolute minimum.
    - **JAX-Native & Decoupled**: Produces a clean `jsparse.COO` matrix directly within JAX trace, allowing straightforward integration with native linear solvers and downstream libraries like JetSCI.
    - **Thread/GPU Safety**: Avoids mutable host-side registries (`_OBJECT_STORE`) during solver execution.

---

## 4. Limitations of the Current Implementation

While the current implementation offers optimal performance and memory efficiency, developers should be aware of the following limitations:

1. **Static Constraint Topology**:
   The preprocessing step assumes that the set of constrained DoFs, constraint coefficients ($P$), and mesh connectivity do not change. If boundary conditions or MPCs are added or removed dynamically during a step (e.g., in contact mechanics), the mapping arrays must be re-precomputed.
2. **Linear Constraints Only**:
   The gather-scatter mapping is designed around constant, linear coefficients. Nonlinear constraint equations (e.g., $u_i = f(u_j)$) cannot be precomputed this way and would require on-the-fly linearization and assembly adjustments.
