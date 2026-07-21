# test/test_lehmann_behnke.jl
#
# CORNER-CASE TAXONOMY (plan §7.5, reviewed 2026-05-01):
#   1. scalar case s == r (single eigenvalue): returns one tight interval
#      that contains the true λ_r. Both do_shift=true and do_shift=false
#      must produce sound bounds.
#   2. cluster of 2: returns 2 intervals, each containing the corresponding
#      true eigenvalue.
#   3. cluster of 5: returns 5 intervals; soundness on a synthetic case.
#   4. do_shift=true vs do_shift=false: bounds agree on validity.
#   5. SB barely PD: still works (succeeds when separators are reasonable).
#   6. SB not PD: throws VeigsLehmannBehnkeError (separators too tight).
#   7. orthogonal vs non-orthogonal V: both work; orthogonalization is not
#      required by the algorithm.
#   8. interval A with non-zero radius: bounds wider but still sound.
#   9. invalid inputs: shape mismatch, λ_B_min ≤ 0, r > s, V wrong cols.
#  10. soundness battery on diagonal known-spectrum problems.

using Test, Random, LinearAlgebra
import Veigs
using Veigs: lehmann_behnke, veig, VeigsLehmannBehnkeError, VeigsSizeError
using IntervalArithmetic: interval, mid, inf, sup

include(joinpath(@__DIR__, "support", "matrix_factory.jl"))

Random.seed!(20260501)

# Helper: build approximate eigenvectors via float eigen-decomposition.
function _approx_eigvecs(A, B)
    F = eigen(Symmetric(A), Symmetric(B))
    perm = sortperm(real.(F.values))
    return real.(F.values[perm]), real.(F.vectors[:, perm])
end

# Helper: synthesise verified separators ρ, σ for cluster (r..s) by stepping
# halfway between adjacent float eigenvalues (sound when there's a real gap).
function _separators(eig_list, r, s)
    n = length(eig_list)
    ρ = r > 1 ? (eig_list[r-1] + eig_list[r]) / 2 : eig_list[1] - 1.0
    σ = s < n ? (eig_list[s] + eig_list[s+1]) / 2 : eig_list[n] + 1.0
    return ρ, σ
end

@testset "lehmann_behnke" begin

    # ----------- Build a workhorse problem with well-separated spectrum -----
    # A = diag(1, 2, 4, 8, 16), B = I.  Eigenvalues = 1, 2, 4, 8, 16.
    n = 5
    diag_vals = [1.0, 2.0, 4.0, 8.0, 16.0]
    A_diag = Matrix(Diagonal(diag_vals))
    B_diag = Matrix{Float64}(I, n, n)
    eig_list_diag, V_diag = _approx_eigvecs(A_diag, B_diag)
    λ_B_min_diag = 1.0     # B = I

    @testset "1. scalar case s == r" begin
        for r in 1:n
            ρ, σ = _separators(eig_list_diag, r, r)
            for shift in (true, false)
                bounds = lehmann_behnke(A_diag, B_diag, eig_list_diag,
                                        V_diag[:, r:r], ρ, σ, λ_B_min_diag,
                                        r, r; do_shift = shift)
                @test length(bounds) == 1
                # Soundness: true eigenvalue is inside the bound.
                @test inf(bounds[1]) ≤ diag_vals[r] ≤ sup(bounds[1])
            end
        end
    end

    @testset "2. cluster of 2" begin
        # Use an indistinguishable pair.  diag(3.0, 3.0 + 1e-3, 6, 8, 16).
        diag2 = [3.0, 3.0 + 1e-3, 6.0, 8.0, 16.0]
        A2 = Matrix(Diagonal(diag2))
        B2 = Matrix{Float64}(I, 5, 5)
        ev, V = _approx_eigvecs(A2, B2)
        ρ, σ = _separators(ev, 1, 2)
        bounds = lehmann_behnke(A2, B2, ev, V[:, 1:2], ρ, σ, 1.0, 1, 2)
        @test length(bounds) == 2
        @test inf(bounds[1]) ≤ diag2[1] ≤ sup(bounds[1])
        @test inf(bounds[2]) ≤ diag2[2] ≤ sup(bounds[2])
    end

    @testset "3. cluster of 5 (full spectrum)" begin
        # Cluster covering the entire spectrum; ρ < λ_1, σ > λ_n.
        ev, V = _approx_eigvecs(A_diag, B_diag)
        ρ = ev[1] - 1.0    # well below λ_1
        σ = ev[end] + 1.0  # well above λ_n
        bounds = lehmann_behnke(A_diag, B_diag, ev, V, ρ, σ, λ_B_min_diag, 1, 5;
                                do_shift = false)
        @test length(bounds) == 5
        for i in 1:5
            @test inf(bounds[i]) ≤ diag_vals[i] ≤ sup(bounds[i])
        end
    end

    @testset "4. do_shift true vs false agree on soundness" begin
        r, s = 2, 3
        ρ, σ = _separators(eig_list_diag, r, s)
        for shift in (true, false)
            bounds = lehmann_behnke(A_diag, B_diag, eig_list_diag,
                                    V_diag[:, r:s], ρ, σ, λ_B_min_diag,
                                    r, s; do_shift = shift)
            for (i, idx) in enumerate(r:s)
                @test inf(bounds[i]) ≤ diag_vals[idx] ≤ sup(bounds[i])
            end
        end
    end

    @testset "5. SB barely PD (separators near-tight)" begin
        # Make σ very close to λ_3 = 4.0 from above, but still beyond λ_2 = 2.
        r, s = 1, 2
        ρ = -10.0
        σ = 4.0 - 1e-3                           # tight; near boundary
        bounds = lehmann_behnke(A_diag, B_diag, eig_list_diag,
                                V_diag[:, 1:2], ρ, σ, λ_B_min_diag, 1, 2)
        for i in 1:2
            @test inf(bounds[i]) ≤ diag_vals[i] ≤ sup(bounds[i])
        end
    end

    @testset "6. SB not PD throws VeigsLehmannBehnkeError" begin
        # σ < λ_s (not a valid right separator) makes the lower SB indefinite.
        r, s = 1, 2
        ρ = 0.0
        σ = 1.5     # between λ_1 = 1 and λ_2 = 2 — invalid
        @test_throws VeigsLehmannBehnkeError lehmann_behnke(
            A_diag, B_diag, eig_list_diag, V_diag[:, 1:2], ρ, σ, λ_B_min_diag,
            r, s; do_shift = false)
    end

    @testset "7. non-orthogonal V" begin
        # Use V = I[:, 1:2] which is B-orthonormal here, then perturb columns.
        r, s = 1, 2
        ρ, σ = _separators(eig_list_diag, r, s)
        Vp = copy(V_diag[:, 1:2])
        Vp[:, 2] .+= 0.1 * Vp[:, 1]               # de-orthogonalize
        bounds = lehmann_behnke(A_diag, B_diag, eig_list_diag, Vp, ρ, σ,
                                λ_B_min_diag, r, s)
        for i in 1:2
            @test inf(bounds[i]) ≤ diag_vals[i] ≤ sup(bounds[i])
        end
    end

    @testset "8. interval A — wider but still sound" begin
        Ai = interval_perturb(A_diag; r = 1e-12)
        Bi = interval.(B_diag)
        r, s = 2, 3
        ρ, σ = _separators(eig_list_diag, r, s)
        bounds = lehmann_behnke(Ai, Bi, eig_list_diag, V_diag[:, r:s], ρ, σ,
                                λ_B_min_diag, r, s)
        for (i, idx) in enumerate(r:s)
            @test inf(bounds[i]) ≤ diag_vals[idx] ≤ sup(bounds[i])
        end
    end

    @testset "9. invalid inputs" begin
        # Shape mismatch
        @test_throws VeigsSizeError lehmann_behnke(
            zeros(3, 3), zeros(2, 2), [0.0, 0.0, 0.0], zeros(3, 1),
            -1.0, 1.0, 1.0, 1, 1)
        # λ_B_min ≤ 0
        @test_throws ArgumentError lehmann_behnke(
            A_diag, B_diag, eig_list_diag, V_diag[:, 1:1],
            -10.0, 2.0, 0.0, 1, 1)
        # r > s
        @test_throws ArgumentError lehmann_behnke(
            A_diag, B_diag, eig_list_diag, V_diag[:, 2:2],
            -10.0, 5.0, 1.0, 3, 2)
        # V wrong column count
        @test_throws ArgumentError lehmann_behnke(
            A_diag, B_diag, eig_list_diag, V_diag[:, 1:1],
            -10.0, 5.0, 1.0, 1, 2)
    end

    @testset "10. soundness battery — random diagonal SPD" begin
        for trial in 1:6
            n = rand(4:8)
            d = sort(0.5 .+ rand(n) .* 5)
            A = Matrix(Diagonal(d))
            B = Matrix{Float64}(I, n, n)
            ev, V = _approx_eigvecs(A, B)
            r = rand(1:n-1); s = rand(r:n)
            ρ, σ = _separators(ev, r, s)
            bounds = lehmann_behnke(A, B, ev, V[:, r:s], ρ, σ, 1.0, r, s)
            for (i, idx) in enumerate(r:s)
                @test inf(bounds[i]) ≤ d[idx] ≤ sup(bounds[i])
            end
        end
    end

    @testset "11. outer_matmul kwarg: :fast vs :tight" begin
        # Both must enclose the truth. :tight should be at least as tight as
        # :fast (its per-entry intervals come from directed-rounded matmul,
        # i.e. the actual rounding rather than the Wilkinson worst case).
        for r in 1:n
            ρ, σ = _separators(eig_list_diag, r, r)
            fast = lehmann_behnke(A_diag, B_diag, eig_list_diag,
                                  V_diag[:, r:r], ρ, σ, λ_B_min_diag,
                                  r, r; outer_matmul = :fast)
            tight = lehmann_behnke(A_diag, B_diag, eig_list_diag,
                                   V_diag[:, r:r], ρ, σ, λ_B_min_diag,
                                   r, r; outer_matmul = :tight)
            @test inf(fast[1])  ≤ diag_vals[r] ≤ sup(fast[1])
            @test inf(tight[1]) ≤ diag_vals[r] ≤ sup(tight[1])
            @test (sup(tight[1]) - inf(tight[1])) ≤ (sup(fast[1]) - inf(fast[1])) + 1e-14
        end
        @test_throws ArgumentError lehmann_behnke(
            A_diag, B_diag, eig_list_diag, V_diag[:, 1:1],
            -10.0, 2.0, 1.0, 1, 1; outer_matmul = :bogus)
    end

end
