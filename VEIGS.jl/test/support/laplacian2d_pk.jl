# test/support/laplacian2d_pk.jl
#
# Higher-order P1 / P2 / P3 stiffness and mass matrices for the 2D
# Dirichlet Laplacian eigenproblem `K x = λ M x` on the unit square,
# assembled via Ferrite.jl. Companion to `laplacian2d_p1` (which is
# hand-rolled on the Friedrichs-Keller triangulation, 5/7-point stencils).
#
# Ferrite is a *test-only* dependency (declared in [extras] / [targets]).
# Loading this file requires `using Ferrite` to succeed; tests that need
# higher-order matrices include this file directly.

using Ferrite, SparseArrays, LinearAlgebra

"""
    laplacian2d_pk(n_cells::Int, k::Int) -> (K, M, n_dofs)

Triangular Pk Lagrange FEM matrices for the 2D Dirichlet Laplacian
eigenproblem on the unit square `(0,1)²`, with `n_cells × n_cells`
quadrilateral cells each split into 2 right triangles. Polynomial order
`k ∈ {1, 2, 3}`. Dirichlet u=0 on all four boundaries.

`K` is the stiffness (`∫ ∇φ · ∇ψ dx`) and `M` the consistent mass
(`∫ φψ dx`). Both are returned as `SparseMatrixCSC{Float64, Int}`.
`n_dofs` is the number of *interior* (free) degrees of freedom.

Discrete eigenvalues approximate `λ_{j,k} = π²(j² + k²)`, with
convergence rate `O(h^{2k})` for the eigenvalues (Strang-Fix theorem).

Notes:
- For k=1, this gives a different mesh than `laplacian2d_p1` from the
  hand-written matrix_factory.jl (Ferrite's default triangulation is
  not Friedrichs-Keller — it stitches triangles a different way).
  Eigenvalues converge to the same continuum limit at the same rate.
- For k=2 (P2): each triangle has 6 DOFs (3 vertices + 3 edge midpoints).
- For k=3 (P3): each triangle has 10 DOFs (3 vertices + 6 edge thirds + 1 face).
"""
function laplacian2d_pk(n_cells::Int, k::Int)
    n_cells ≥ 1 || throw(ArgumentError("laplacian2d_pk: n_cells must be ≥ 1"))
    k ∈ (1, 2, 3) || throw(ArgumentError("laplacian2d_pk: k must be 1, 2, or 3"))

    # Unit-square mesh (Ferrite's default is [-1,1]² — we override).
    grid = generate_grid(Triangle, (n_cells, n_cells),
                         Vec(0.0, 0.0), Vec(1.0, 1.0))

    # Pk Lagrange interpolation on the reference triangle.
    ip = Lagrange{RefTriangle, k}()

    dh = DofHandler(grid)
    add!(dh, :u, ip)
    close!(dh)

    # Quadrature: degree 2k integrates Pk² exactly.
    qr = QuadratureRule{RefTriangle}(2 * k)
    cv = CellValues(qr, ip)
    n_basefuncs = getnbasefunctions(cv)

    K = allocate_matrix(dh)
    M = allocate_matrix(dh)

    Ke = zeros(n_basefuncs, n_basefuncs)
    Me = zeros(n_basefuncs, n_basefuncs)

    K_assembler = start_assemble(K)
    M_assembler = start_assemble(M)

    for cell in CellIterator(dh)
        reinit!(cv, cell)
        fill!(Ke, 0)
        fill!(Me, 0)
        for q in 1:getnquadpoints(cv)
            dΩ = getdetJdV(cv, q)
            for i in 1:n_basefuncs
                v_i  = shape_value(cv, q, i)
                ∇v_i = shape_gradient(cv, q, i)
                for j in 1:n_basefuncs
                    v_j  = shape_value(cv, q, j)
                    ∇v_j = shape_gradient(cv, q, j)
                    Ke[i, j] += (∇v_i ⋅ ∇v_j) * dΩ
                    Me[i, j] += (v_i  *  v_j) * dΩ
                end
            end
        end
        assemble!(K_assembler, celldofs(cell), Ke)
        assemble!(M_assembler, celldofs(cell), Me)
    end

    # Dirichlet BC: u = 0 on the 4 boundary facets.
    ch = ConstraintHandler(dh)
    for set in ("left", "right", "top", "bottom")
        add!(ch, Dirichlet(:u, getfacetset(grid, set), x -> 0.0))
    end
    close!(ch)
    update!(ch, 0.0)

    # Extract interior submatrix by dropping constrained rows/cols.
    free_dofs = setdiff(1:ndofs(dh), ch.prescribed_dofs)
    return K[free_dofs, free_dofs], M[free_dofs, free_dofs], length(free_dofs)
end
