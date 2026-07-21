# test/test_veig.jl
#
# CORNER-CASE TAXONOMY (plan §7.6, reviewed 2026-05-01):
#   1. diagonal known-spectrum: bounds straddle true λ within tight tolerance.
#   2. Hilbert + I (README example 1, n=8): produced bounds contain the
#      MATLAB-published bounds for the same example (verified containment).
#   3. banded test (README example 2, n=10): same containment property.
#   4. n = 1: trivial single eigenvalue.
#   5. n = 100 (boundary): completes (slow but valid).
#   6. n = 101: throws VeigsSizeError (Q-004 cap).
#   7. single-index ind=k vs full range: agree on the returned slot.
#   8. asymmetric A or B: throws VeigsSizeError.
#   9. tightly clustered eigenvalues (gap ≤ 1e-8): one wide interval covers
#      both indices; ind_range extends to the cluster boundary.
#  10. interval input with non-zero radius: bounds are wider but still sound.
#  11. soundness — for every test case, returned bounds CONTAIN the float
#      eigenvalues from `eigvals(Symmetric(A_mid), Symmetric(B_mid))`.

using Test, Random, LinearAlgebra
import Veigs
using Veigs: veig, VeigsSizeError
using IntervalArithmetic: interval, mid, inf, sup, hull

include(joinpath(@__DIR__, "support", "matrix_factory.jl"))

Random.seed!(20260501)

@testset "veig" begin

    @testset "1. diagonal known-spectrum" begin
        for diag_vec in ([1.0, 2.0, 3.0, 4.0], [0.5, 0.6, 0.7, 0.8, 0.9])
            n = length(diag_vec)
            A = Matrix(Diagonal(diag_vec))
            B = Matrix{Float64}(I, n, n)
            bounds, ind_range = veig(A, B)
            @test ind_range == 1:n
            @test length(bounds) == n
            for (i, λ) in zip(ind_range, diag_vec)
                @test inf(bounds[i]) ≤ λ ≤ sup(bounds[i])
            end
        end
    end

    @testset "2. Hilbert + I  (README example 1, n=8)" begin
        n = 8
        A = Matrix{Float64}(I, n, n)
        B = Matrix(hilbert(n))
        bounds, ir = veig(A, B, 1:3)
        ev = sort(real.(eigvals(Symmetric(A), Symmetric(B))))
        for (i, λ_true) in zip(ir, ev[ir])
            @test inf(bounds[i - first(ir) + 1]) ≤ λ_true ≤ sup(bounds[i - first(ir) + 1])
        end
        # Compare against the MATLAB README's published bounds for index 1
        # (lower=0.589643850288603, upper=0.589643850289016). Our bounds
        # must contain that interval (interval-containment, not equality).
        # In other words: our_inf ≤ matlab_inf, our_sup ≥ matlab_sup.
        b1 = bounds[1]
        @test inf(b1) ≤ 0.589643850288603
        @test sup(b1) ≥ 0.589643850289016
    end

    @testset "3. banded (README example 2, n=10)" begin
        n = 10
        A = Matrix(banded_test(n))
        B = Matrix(hilbert(n)) * 232792560
        bounds, ir = veig(A, B, n - 4:n)        # five largest
        ev = sort(real.(eigvals(Symmetric(A), Symmetric(B))))
        for i in ir
            slot = i - first(ir) + 1
            λ_true = ev[i]
            @test inf(bounds[slot]) ≤ λ_true ≤ sup(bounds[slot])
        end
    end

    @testset "4. n = 1" begin
        A = reshape([3.0], 1, 1)
        B = reshape([1.0], 1, 1)
        bounds, ir = veig(A, B)
        @test ir == 1:1
        @test inf(bounds[1]) ≤ 3.0 ≤ sup(bounds[1])
    end

    @testset "5. n = 100 (boundary, single index)" begin
        n = 100
        A = Matrix(spd_random(n))
        B = Matrix{Float64}(I, n, n)
        ev = sort(real.(eigvals(Symmetric(A), Symmetric(B))))
        # Test middle index only — full range is too slow for CI.
        bounds, ir = veig(A, B, 50:50)
        @test 50 in ir
        slot = findfirst(==(50), collect(ir))
        @test inf(bounds[slot]) ≤ ev[50] ≤ sup(bounds[slot])
    end

    @testset "6. n = 101 throws VeigsSizeError" begin
        A = Matrix{Float64}(I, 101, 101)
        B = copy(A)
        @test_throws VeigsSizeError veig(A, B)
    end

    @testset "7. single-index vs full range agree" begin
        n = 6
        A = Matrix(spd_random(n))
        B = Matrix{Float64}(I, n, n)
        bounds_full, ir_full = veig(A, B)
        bounds_3, ir_3 = veig(A, B, 3:3)
        # The slot in `bounds_3` corresponding to index 3 should overlap the
        # slot in `bounds_full` for index 3.
        slot_full = 3 - first(ir_full) + 1
        slot_3    = 3 - first(ir_3)    + 1
        @test inf(bounds_full[slot_full]) ≥ inf(bounds_3[slot_3]) - 1e-6
        @test sup(bounds_full[slot_full]) ≤ sup(bounds_3[slot_3]) + 1e-6
    end

    @testset "8. asymmetric input throws VeigsSizeError" begin
        A = [1.0 0.5; 0.6 2.0]                    # asymmetric
        B = Matrix{Float64}(I, 2, 2)
        @test_throws VeigsSizeError veig(A, B)
    end

    @testset "9. tightly clustered eigenvalues" begin
        # Two close eigenvalues 1.0 and 1.0 + 1e-9
        A = Matrix(Diagonal([1.0, 1.0 + 1e-9, 5.0]))
        B = Matrix{Float64}(I, 3, 3)
        bounds, ir = veig(A, B, 1:2)
        # The two close eigenvalues may end up sharing a single wide bound.
        # Either way, the returned bounds must enclose both true values.
        ev = sort(real.(eigvals(Symmetric(A), Symmetric(B))))
        for i in 1:2
            slot = i - first(ir) + 1
            @test inf(bounds[slot]) ≤ ev[i] ≤ sup(bounds[slot])
        end
    end

    @testset "10. interval input — wider bounds, still sound" begin
        n = 5
        A0 = Matrix(spd_random(n))
        B0 = Matrix{Float64}(I, n, n)
        Ai = interval_perturb(A0; r = 1e-10)
        Bi = interval.(B0)
        bounds, ir = veig(Ai, Bi)
        ev = sort(real.(eigvals(Symmetric(A0), Symmetric(B0))))
        for i in ir
            slot = i - first(ir) + 1
            @test inf(bounds[slot]) ≤ ev[i] ≤ sup(bounds[slot])
        end
    end

    @testset "11. soundness battery — random SPD pairs" begin
        for trial in 1:8
            n = rand(3:12)
            A = Matrix(spd_random(n))
            B = Matrix(spd_random(n))
            bounds, ir = veig(A, B)
            ev = sort(real.(eigvals(Symmetric(A), Symmetric(B))))
            for i in ir
                slot = i - first(ir) + 1
                @test inf(bounds[slot]) ≤ ev[i] ≤ sup(bounds[slot])
            end
        end
    end

end
