# test/support/matrix_factory.jl
#
# Reusable matrix generators for the per-routine test files. Imported via
# `include("support/matrix_factory.jl")` from each `test_*.jl`.

using LinearAlgebra, SparseArrays, Random
using IntervalArithmetic: interval

"""
    spd_random(n; rng=GLOBAL_RNG) -> Symmetric{Float64}

A random `n×n` symmetric positive definite matrix with eigenvalues bounded
away from zero (smallest ≈ 0.5, largest ≈ n + 0.5).
"""
function spd_random(n::Int; rng::AbstractRNG = Random.GLOBAL_RNG)
    Q, _ = qr(randn(rng, n, n))
    Q = Matrix(Q)
    Λ = Diagonal(0.5 .+ rand(rng, n) .* n)
    return Symmetric(Q * Λ * Q')
end

"""
    indef_random(n; rng=GLOBAL_RNG) -> Symmetric{Float64}

A random `n×n` symmetric indefinite matrix with eigenvalues in [-n, n].
"""
function indef_random(n::Int; rng::AbstractRNG = Random.GLOBAL_RNG)
    Q, _ = qr(randn(rng, n, n))
    Q = Matrix(Q)
    Λ = Diagonal(2 .* (rand(rng, n) .- 0.5) .* n)
    return Symmetric(Q * Λ * Q')
end

"""
    near_singular_spd(n; tail_eig=1e-12, rng=GLOBAL_RNG)

SPD with a single tiny eigenvalue (`tail_eig`); the rest spread in [1, n].
Useful for stressing `verified_isspd` near the borderline.
"""
function near_singular_spd(n::Int; tail_eig::Float64 = 1e-12,
                           rng::AbstractRNG = Random.GLOBAL_RNG)
    Q, _ = qr(randn(rng, n, n)); Q = Matrix(Q)
    eigs = collect(range(1.0, n; length = n))
    eigs[1] = tail_eig
    return Symmetric(Q * Diagonal(eigs) * Q')
end

"""
    barely_indef(n; eps=1e-10, rng=GLOBAL_RNG)

Symmetric matrix with `λ_min = -eps`; SOUND `verified_isspd` MUST return false.
"""
function barely_indef(n::Int; eps::Float64 = 1e-10,
                      rng::AbstractRNG = Random.GLOBAL_RNG)
    Q, _ = qr(randn(rng, n, n)); Q = Matrix(Q)
    eigs = collect(range(1.0, n; length = n))
    eigs[1] = -eps
    return Symmetric(Q * Diagonal(eigs) * Q')
end

"""
    hilbert(n) -> Symmetric{Float64}

The classic Hilbert matrix `H[i,j] = 1 / (i + j - 1)`.
"""
hilbert(n::Int) = Symmetric([1.0 / (i + j - 1) for i in 1:n, j in 1:n])

"""
    banded_test(n) -> Matrix{Float64}

The 5-banded test matrix from veigs README example #2.
"""
function banded_test(n::Int)
    A = Matrix{Float64}(I, n, n) .* 6
    A[1, 1] = 5
    A[n, n] = 5
    for i in 1:n-1
        A[i, i+1] = -4; A[i+1, i] = -4
    end
    for i in 1:n-2
        A[i, i+2] = 1;  A[i+2, i] = 1
    end
    return A
end

"""
    interval_perturb(M; r=1e-13)

Return `interval.(M .- r, M .+ r)`, an interval matrix of radius `r`.
"""
interval_perturb(M::AbstractMatrix; r::Real = 1e-13) =
    interval.(M .- r, M .+ r)

"""
    laplacian2d_p1(n_side) -> (K, M, h)

P1 (linear) FEM stiffness `K` and consistent mass `M` for the
2D Dirichlet Laplacian eigenproblem `K x = λ M x` on the unit square
`(0,1)²`, using the Friedrichs–Keller triangulation (each unit cell
split along its SW–NE diagonal). Interior nodes only — `N = n_side²`
DOFs total. `h = 1 / (n_side + 1)` is the grid spacing.

`K` is the 5-point sparse stencil: diagonal `4`, axis neighbours `-1`.
`M` is the 7-point sparse stencil: diagonal `h²/2`, axis neighbours
`h²/12`, and the SW–NE diagonal neighbours `h²/12` (NW–SE neighbours
are zero — those edges don't exist in this triangulation).

Discrete eigenvalues approximate `λ_{j,k} = π²(j² + k²)` with
`O(h²)` accuracy. Useful as a non-tridiagonal sparse test problem:
verified_ldl's tridiag fast-path is skipped, so the dense fallback
is exercised.
"""
function laplacian2d_p1(n_side::Int)
    n_side ≥ 1 || throw(ArgumentError("laplacian2d_p1: n_side must be ≥ 1"))
    N = n_side^2
    h = 1.0 / (n_side + 1)
    h2 = h * h

    @inline idx(i, j) = (j - 1) * n_side + i

    rowsK = Int[]; colsK = Int[]; valsK = Float64[]
    rowsM = Int[]; colsM = Int[]; valsM = Float64[]
    sizehint!(rowsK, 5N); sizehint!(colsK, 5N); sizehint!(valsK, 5N)
    sizehint!(rowsM, 7N); sizehint!(colsM, 7N); sizehint!(valsM, 7N)

    push_K!(r, c, v) = (push!(rowsK, r); push!(colsK, c); push!(valsK, v))
    push_M!(r, c, v) = (push!(rowsM, r); push!(colsM, c); push!(valsM, v))

    for j in 1:n_side, i in 1:n_side
        c = idx(i, j)
        push_K!(c, c, 4.0)
        push_M!(c, c, h2 / 2)

        if i < n_side
            push_K!(c, idx(i + 1, j), -1.0)
            push_M!(c, idx(i + 1, j), h2 / 12)
        end
        if i > 1
            push_K!(c, idx(i - 1, j), -1.0)
            push_M!(c, idx(i - 1, j), h2 / 12)
        end
        if j < n_side
            push_K!(c, idx(i, j + 1), -1.0)
            push_M!(c, idx(i, j + 1), h2 / 12)
        end
        if j > 1
            push_K!(c, idx(i, j - 1), -1.0)
            push_M!(c, idx(i, j - 1), h2 / 12)
        end
        # SW–NE diagonal edges (mass only).
        if i < n_side && j < n_side
            push_M!(c, idx(i + 1, j + 1), h2 / 12)
        end
        if i > 1 && j > 1
            push_M!(c, idx(i - 1, j - 1), h2 / 12)
        end
    end

    K = sparse(rowsK, colsK, valsK, N, N)
    M = sparse(rowsM, colsM, valsM, N, N)
    return K, M, h
end
