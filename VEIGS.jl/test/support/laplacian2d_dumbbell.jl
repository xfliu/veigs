# test/support/laplacian2d_dumbbell.jl
#
# Pk Lagrange FEM matrices for the 2D Dirichlet Laplacian eigenproblem
# on a *dumbbell* domain — two unit squares connected by a thin bar:
#
#       ┌────────┐                                       ┌────────┐
#       │        │                                       │        │
#       │  [-2,  │                                       │ [1, 2] │
#       │  -1] × │                                       │ × [0,1]│
#       │ [0, 1] │                                       │        │
#       │        ┝═══════════════════════════════════════┥        │
#       │        │       [-1, 1] × [0, w]   bar          │        │
#       └────────┴───────────────────────────────────────┴────────┘
#
# This is a textbook example for *clustered* (near-degenerate)
# Dirichlet eigenvalues. As w → 0 the two lobes decouple and the
# spectrum becomes that of a single unit square with multiplicity 2.
# At finite w, that degeneracy splits into close pairs (symmetric /
# antisymmetric modes); the splitting is exponentially small in 1/w
# (a quantum-mechanical tunneling effect).

using Ferrite, SparseArrays, LinearAlgebra

"""
    laplacian2d_dumbbell(h, k; w=0.2) -> (K, M, ndof, grid)

P_k Lagrange FEM stiffness `K` and consistent mass `M` for the
Dirichlet Laplacian on the dumbbell domain
    [-2, -1] × [0, 1] ∪ [-1, 1] × [0, w] ∪ [1, 2] × [0, 1].

`h` is the mesh size (must divide 4, 1, and `w` evenly).
`k ∈ {1, 2, 3}` is the polynomial order.

The first two eigenvalues form a tight cluster — exactly the case
`veigs` was designed to handle.
"""
function laplacian2d_dumbbell(h::Float64, k::Int; w::Float64 = 0.2)
    k ∈ (1, 2, 3) || throw(ArgumentError("laplacian2d_dumbbell: k must be 1, 2, or 3"))

    grid = _build_dumbbell_grid(h, w)

    ip = Lagrange{RefTriangle, k}()
    dh = DofHandler(grid)
    add!(dh, :u, ip)
    close!(dh)

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
        fill!(Ke, 0); fill!(Me, 0)
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

    # Dirichlet u = 0 on all boundary facets (those with no neighbour cell
    # across them). Ferrite's structured-grid helpers don't apply here
    # because we built the mesh manually — detect boundary by edge count.
    boundary_facets = _detect_boundary_facets(grid)

    ch = ConstraintHandler(dh)
    add!(ch, Dirichlet(:u, boundary_facets, x -> 0.0))
    close!(ch)
    update!(ch, 0.0)

    free_dofs = setdiff(1:ndofs(dh), ch.prescribed_dofs)
    return K[free_dofs, free_dofs], M[free_dofs, free_dofs], length(free_dofs), grid
end

# Build the dumbbell grid manually as a structured triangulation of the
# bounding box [-2, 2] × [0, 1], retaining only quads whose centroid
# lies inside the dumbbell. Each retained quad is split SW-NE into two
# triangles (Friedrichs-Keller convention).
function _build_dumbbell_grid(h::Float64, w::Float64)
    isapprox(round(4/h)*h, 4, atol = 1e-10) ||
        throw(ArgumentError("h must divide 4 evenly (got h=$h)"))
    isapprox(round(1/h)*h, 1, atol = 1e-10) ||
        throw(ArgumentError("h must divide 1 evenly"))
    isapprox(round(w/h)*h, w, atol = 1e-10) ||
        throw(ArgumentError("h must divide w=$w evenly"))

    nx = Int(round(4/h))
    ny = Int(round(1/h))

    @inline function quad_inside(i::Int, j::Int)
        xc = -2.0 + (i - 0.5) * h
        yc = (j - 0.5) * h
        (-2 ≤ xc ≤ -1 && 0 ≤ yc ≤ 1) ||
        (-1 ≤ xc ≤  1 && 0 ≤ yc ≤ w) ||
        ( 1 ≤ xc ≤  2 && 0 ≤ yc ≤ 1)
    end

    node_id_at = Dict{Tuple{Int,Int}, Int}()
    nodes = Ferrite.Node{2, Float64}[]
    cells = Triangle[]

    function get_node(i::Int, j::Int)
        get!(node_id_at, (i, j)) do
            x = -2.0 + (i - 1) * h
            y = (j - 1) * h
            push!(nodes, Ferrite.Node((x, y)))
            length(nodes)
        end
    end

    for i in 1:nx, j in 1:ny
        quad_inside(i, j) || continue
        a = get_node(i, j)
        b = get_node(i + 1, j)
        c = get_node(i + 1, j + 1)
        d = get_node(i, j + 1)
        push!(cells, Triangle((a, b, c)))
        push!(cells, Triangle((a, c, d)))
    end

    return Grid(cells, nodes)
end

# Boundary facets = edges shared by exactly one triangle.
function _detect_boundary_facets(grid::Ferrite.Grid)
    cells = getcells(grid)
    edge_facets = Dict{NTuple{2,Int}, Vector{FacetIndex}}()
    for (ci, c) in enumerate(cells)
        v = c.nodes
        for fl in 1:3
            a = v[fl]
            b = v[fl == 3 ? 1 : fl + 1]
            e = a < b ? (a, b) : (b, a)
            push!(get!(() -> FacetIndex[], edge_facets, e), FacetIndex(ci, fl))
        end
    end
    boundary = Set{FacetIndex}()
    for (_e, fis) in edge_facets
        length(fis) == 1 && push!(boundary, fis[1])
    end
    return boundary
end
