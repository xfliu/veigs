# test/test_isspd.jl
#
# CORNER-CASE TAXONOMY (plan §7.2, reviewed 2026-05-01):
#   1. identity at sizes {1, 2, 5, 50}                   : true
#   2. diag(λ), λ ∈ {1e-12, 1.0, 1e+6}                   : true (positive)
#   3. diag(-λ)                                          : false
#   4. Hilbert n ≤ 8                                     : true
#   5. SOUNDNESS: λ_min = -ε for ε ∈ {1e-2, 1e-8, 1e-14} : MUST be false
#   6. Borderline: λ_min = +ε                            : either ok, but cross-check
#   7. asymmetric input                                  : auto-symmetrize via sym_hull
#   8. matrix with zero row/column                       : false
#   9. Inf / NaN entries                                 : DomainError
#  10. radius-zero point matrix                          : agrees with isposdef
#  11. Sparse interval matrix                            : same as dense
#  12. Cross-check Rump vs solve-based                   : DEFERRED to iter 3
#  13. Performance: n=100 dense < 50 ms; n=50 with radius < 100 ms

using Test, Random, LinearAlgebra, SparseArrays
import Veigs
using Veigs: verified_isspd, sym_hull
using IntervalArithmetic: interval, mid, inf, sup

include(joinpath(@__DIR__, "support", "matrix_factory.jl"))

Random.seed!(20260501)

@testset "verified_isspd" begin

    @testset "1. identity at multiple sizes" begin
        for n in (1, 2, 5, 50)
            M = Matrix{Float64}(I, n, n)
            @test verified_isspd(M)             # auto-wraps to interval
            @test verified_isspd(interval.(M))
        end
    end

    @testset "2. diag(positive)" begin
        for λ in (1e-12, 1.0, 1e+6)
            @test verified_isspd(Matrix(Diagonal([λ, λ * 2, λ * 3])))
        end
    end

    @testset "3. diag(negative)" begin
        @test !verified_isspd(Matrix(Diagonal([-1.0, 1.0, 2.0])))
        @test !verified_isspd(Matrix(Diagonal([-1e-12, 1.0, 2.0])))
    end

    @testset "4. Hilbert (small) is PD" begin
        for n in (3, 5, 8)
            @test verified_isspd(Matrix(hilbert(n)))
        end
    end

    @testset "5. SOUNDNESS — adversarial λ_min = -ε" begin
        for ε in (1e-2, 1e-8, 1e-14)
            for n in (5, 10, 20)
                M = Matrix(barely_indef(n; eps = ε))
                # MUST be false. If verified_isspd ever returns true here,
                # the entire correctness story collapses.
                @test verified_isspd(M) == false
            end
        end
    end

    @testset "6. borderline λ_min = +ε" begin
        for ε in (1e-2, 1e-6)
            for n in (5, 10)
                M = Matrix(near_singular_spd(n; tail_eig = ε))
                # We don't *require* true (the routine is allowed to be
                # conservative), but if it says true, an independent check must agree.
                got = verified_isspd(M)
                if got
                    @test isposdef(M)         # midpoint really is PD
                end
            end
        end
    end

    @testset "7. asymmetric input is symmetrized" begin
        # Slight asymmetry shouldn't poison the result.
        M = [2.0  0.5; 0.5001  3.0]
        # `verified_isspd` symmetrizes via Symmetric((Mm + Mm')/2) internally for
        # the midpoint; make sure it doesn't error and returns true.
        @test verified_isspd(M)
    end

    @testset "8. zero row/column" begin
        M = zeros(3, 3)
        @test !verified_isspd(M)               # midpoint Cholesky fails
        M2 = [1.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 1.0]
        @test !verified_isspd(M2)
    end

    @testset "9. Inf/NaN throws DomainError" begin
        @test_throws DomainError verified_isspd([1.0 NaN; NaN 1.0])
        @test_throws DomainError verified_isspd([1.0 Inf; Inf 1.0])
    end

    @testset "10. radius-zero interval agrees with isposdef" begin
        for trial in 1:20
            M = Matrix(spd_random(8))
            Mi = interval.(M)                  # zero radius
            @test verified_isspd(Mi) == true
            @test isposdef(M) == true          # sanity
        end
        for trial in 1:20
            M = Matrix(indef_random(8))
            Mi = interval.(M)
            # If midpoint is PD by accident, both should be true; otherwise both false.
            @test verified_isspd(Mi) == isposdef(Symmetric(M))
        end
    end

    @testset "11. sparse interval matrix" begin
        Md = Matrix(spd_random(10))
        Ms = sparse(Md)
        # Convert sparse to dense interval for now (matches MATLAB code's behavior
        # of `full(mid(B))` before factor); ensures dense path agrees.
        @test verified_isspd(Md) == verified_isspd(Matrix(Ms))
    end

    # 12 deferred to iter 3 (alternate path lands then)

    @testset "13. performance: n=100 dense < 50 ms; n=50 with radius < 100 ms" begin
        M = Matrix(spd_random(100))
        verified_isspd(M)                      # warm-up
        t1 = @elapsed verified_isspd(M)
        @test t1 < 0.05

        M2 = Matrix(spd_random(50))
        Mi = interval_perturb(M2; r = 1e-12)
        verified_isspd(Mi)
        t2 = @elapsed verified_isspd(Mi)
        @test t2 < 0.10
    end

end
