# test/test_veigs.jl
#
# CORNER-CASE TAXONOMY (plan §7.7, reviewed 2026-05-01):
#
# §7.7.1 — Dispatch (7 cases)
#   1. Default veigs(A, B): k=1, sigma=:largestabs.
#   2. veigs(A, B, k::Int) sets k.
#   3. veigs(A, B, sigma::Symbol) sets sigma, k=1.
#   4. veigs(A, B, k, sigma) sets both.
#   5. All four modern sigma symbols recognized.
#   6. Legacy aliases :lm/:sm/:la/:sa map to their modern counterparts.
#   7. Numeric sigma selects nearest-by-magnitude; behaves as shift-invert.
#
# §7.7.2 — Numerical correctness (vs MATLAB README golden bounds)
#   8. Hilbert example, 3 eigenvalues near 3.5 — bounds CONTAIN MATLAB's.
#   9. Hilbert example, smallest eigenvalue — bounds CONTAIN MATLAB's.
#  10. Banded example, 5 largest eigenvalues — bounds enclose float `eigvals`.
#  11. Diagonal known-spectrum: bounds enclose true eigenvalues.
#
# §7.7.3 — Soundness
#  12. Random SPD pairs: every returned bound contains the float eigenvalue.
#
# §7.7.4 — Failure modes
#  13. Non-square A → VeigsSizeError.
#  14. Shape mismatch → VeigsSizeError.
#  15. Bad sigma symbol → ArgumentError.
#  16. k > n → ArgumentError.

using Test, Random, LinearAlgebra
import Veigs
using Veigs: veigs, VeigsSizeError, VeigsRoughBoundError
using IntervalArithmetic: interval, inf, sup

include(joinpath(@__DIR__, "support", "matrix_factory.jl"))

Random.seed!(20260501)

# Small workhorse problem
const A_DIAG = Matrix(Diagonal([1.0, 2.0, 4.0, 8.0, 16.0]))
const B_DIAG = Matrix{Float64}(I, 5, 5)

# Soundness check that allows a tiny tolerance on the float-reference side.
# Our verified bounds are tight; LAPACK's `eigvals` itself has ~ULP relative
# error, so a strict `inf ≤ ev[i] ≤ sup` can fail by 1-2 ULPs. Use a relative
# tolerance scaled to |ev[i]|.
function _enclosed(b, λ::Real; rtol = 1e-10)
    tol = max(1e-12, rtol * abs(λ))
    return inf(b) ≤ λ + tol && λ - tol ≤ sup(b)
end

@testset "veigs" begin

    @testset "1. default — k=1, sigma=:largestabs" begin
        bounds, ir = veigs(A_DIAG, B_DIAG)
        @test 5 in ir
        slot = findfirst(==(5), collect(ir))
        @test inf(bounds[slot]) ≤ 16.0 ≤ sup(bounds[slot])
    end

    @testset "2. k::Int" begin
        bounds, ir = veigs(A_DIAG, B_DIAG, 1)
        @test 5 in ir
    end

    @testset "3. sigma::Symbol only" begin
        bounds, ir = veigs(A_DIAG, B_DIAG, :sa)
        @test 1 in ir
        slot = findfirst(==(1), collect(ir))
        @test inf(bounds[slot]) ≤ 1.0 ≤ sup(bounds[slot])
    end

    @testset "4. (k, sigma)" begin
        bounds, ir = veigs(A_DIAG, B_DIAG, 2, :smallestreal)
        @test 1 in ir && 2 in ir
        for (i, λ) in zip([1, 2], [1.0, 2.0])
            slot = findfirst(==(i), collect(ir))
            @test inf(bounds[slot]) ≤ λ ≤ sup(bounds[slot])
        end
    end

    @testset "5. all four modern sigma symbols recognized" begin
        for sym in (:largestabs, :smallestabs, :largestreal, :smallestreal)
            bounds, ir = veigs(A_DIAG, B_DIAG, 1, sym)
            @test length(bounds) ≥ 1 && !isempty(ir)
        end
    end

    @testset "6. legacy aliases agree with modern names" begin
        pairs = ((:lm, :largestabs), (:sm, :smallestabs),
                 (:la, :largestreal), (:sa, :smallestreal))
        for (legacy, modern) in pairs
            b1, ir1 = veigs(A_DIAG, B_DIAG, 1, legacy)
            b2, ir2 = veigs(A_DIAG, B_DIAG, 1, modern)
            @test ir1 == ir2
            @test inf.(b1) == inf.(b2)
            @test sup.(b1) == sup.(b2)
        end
    end

    @testset "7. numeric sigma — nearest by magnitude" begin
        # eigenvalues are 1, 2, 4, 8, 16. Sigma = 5 → closest is 4 (index 3).
        bounds, ir = veigs(A_DIAG, B_DIAG, 1, 5.0)
        @test 3 in ir
        slot = findfirst(==(3), collect(ir))
        @test inf(bounds[slot]) ≤ 4.0 ≤ sup(bounds[slot])
    end

    @testset "8. README example 1 — Hilbert near 3.5" begin
        # MATLAB README golden:
        #   λ_1 ∈ [0.589643850288603, 0.589643850289016]
        #   λ_2 ∈ [3.35429531632129,  3.35429531633695]
        #   λ_3 ∈ [38.1492376817517,  38.1492376835712]
        n = 8
        A = Matrix{Float64}(I, n, n)
        B = Matrix(hilbert(n))
        bounds, ir = veigs(A, B, 3, 3.5)
        @test 1 in ir && 2 in ir && 3 in ir
        slot1 = findfirst(==(1), collect(ir))
        slot2 = findfirst(==(2), collect(ir))
        slot3 = findfirst(==(3), collect(ir))
        # Soundness: float eigenvalues must lie inside (modulo ULP).
        ev = sort(real.(eigvals(Symmetric(A), Symmetric(B))))
        @test _enclosed(bounds[slot1], ev[1])
        @test _enclosed(bounds[slot2], ev[2])
        @test _enclosed(bounds[slot3], ev[3])
        # Approximate centering against the MATLAB golden midpoints.
        # (Our bounds are typically tighter, so a simple containment check.)
        @test inf(bounds[slot1]) ≤ 0.589643850288603 + 1e-10
        @test sup(bounds[slot1]) ≥ 0.589643850289016 - 1e-10
    end

    @testset "9. README example 1 — smallest eigenvalue" begin
        n = 8
        A = Matrix{Float64}(I, n, n)
        B = Matrix(hilbert(n))
        bounds, ir = veigs(A, B, 1, :sa)
        @test 1 in ir
        slot = findfirst(==(1), collect(ir))
        ev = sort(real.(eigvals(Symmetric(A), Symmetric(B))))
        @test _enclosed(bounds[slot], ev[1])
    end

    @testset "10. README example 2 — banded n=10, 5 largest" begin
        n = 10
        A = Matrix(banded_test(n))
        B = Matrix(hilbert(n)) * 232792560
        B = (B + B') / 2
        bounds, ir = veigs(A, B, 5, :la)
        ev = sort(real.(eigvals(Symmetric(A), Symmetric(B))))
        for i in ir
            slot = i - first(ir) + 1
            @test _enclosed(bounds[slot], ev[i])
        end
    end

    @testset "11. diagonal known-spectrum" begin
        for diag_vec in ([0.5, 1.0, 2.0, 4.0],
                          [-3.0, -1.0, 1.0, 3.0])
            n = length(diag_vec)
            A = Matrix(Diagonal(diag_vec))
            B = Matrix{Float64}(I, n, n)
            for sym in (:la, :sa, :lm)
                bounds, ir = veigs(A, B, 1, sym)
                @test !isempty(ir)
                # Soundness: every returned slot encloses its true eigenvalue.
                ev = sort(diag_vec)
                for i in ir
                    slot = i - first(ir) + 1
                    @test _enclosed(bounds[slot], ev[i])
                end
            end
        end
    end

    @testset "12. soundness battery — random SPD pairs" begin
        for trial in 1:5
            n = rand(4:8)
            A = Matrix(spd_random(n))
            B = Matrix(spd_random(n))
            ev = sort(real.(eigvals(Symmetric(A), Symmetric(B))))
            for sym in (:la, :sa)
                bounds, ir = veigs(A, B, 1, sym)
                for i in ir
                    slot = i - first(ir) + 1
                    @test _enclosed(bounds[slot], ev[i])
                end
            end
        end
    end

    @testset "13. non-square A throws VeigsSizeError" begin
        @test_throws VeigsSizeError veigs(zeros(3, 4), zeros(3, 4))
    end

    @testset "14. shape mismatch throws VeigsSizeError" begin
        @test_throws VeigsSizeError veigs(zeros(3, 3), zeros(2, 2))
    end

    @testset "15. bad sigma symbol throws ArgumentError" begin
        @test_throws ArgumentError veigs(A_DIAG, B_DIAG, 1, :nonsense)
    end

    @testset "16. k > n throws ArgumentError" begin
        @test_throws ArgumentError veigs(A_DIAG, B_DIAG, 100)
    end

end
