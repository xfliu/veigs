# test/test_rough_bounds.jl
#
# CORNER-CASE TAXONOMY (plan §7.4, reviewed 2026-05-01):
#   1. seed already valid (A - λ_seed B is PD): returns ~λ_seed.
#   2. seed fails / midway floor succeeds: returns some λ ∈ (floor, seed).
#   3. floor fails everywhere: throws VeigsRoughBoundError.
#   4. pathological λ_seed = 0 with B = I, A SPD: lower bound is 0 - ε.
#   5. barely-PD B (smallest eig ~ 1e-6): rough_lower(B, I, λ_seed, λ_floor)
#      converges to a positive lower bound on λ_min(B).
#   6. identity-shaped problem A = αI, B = I: bounds tightly bracket α.
#   7. iter-count cap: pass an absurd seed/floor pair where the schedule
#      explodes; routine must still terminate (and either succeed or throw).
#   8. soundness — the returned λ for rough_lower must satisfy λ < eigvals(A,B)
#      for every concrete eigenvalue (verified by float eigvals on a battery).
#   9. Symmetric A wrapper / interval input: still works.
#  10. wrong direction (λ_floor ≥ λ_seed) throws ArgumentError.
#  11. mismatched A/B shapes throws VeigsSizeError.

using Test, Random, LinearAlgebra
import Veigs
using Veigs: rough_lower, rough_upper, VeigsRoughBoundError, VeigsSizeError
using IntervalArithmetic: interval

include(joinpath(@__DIR__, "support", "matrix_factory.jl"))

Random.seed!(20260501)

@testset "rough_bounds" begin

    @testset "1. seed already valid" begin
        # A SPD, B = I. λ_min(A) > 0, so any λ < λ_min(A) gives PD A - λI.
        n = 8
        A = Matrix(spd_random(n))
        B = Matrix{Float64}(I, n, n)
        λ_min_true = minimum(eigvals(Symmetric(A)))
        # Pick seed barely below λ_min; floor much further below.
        seed  = λ_min_true - 0.05
        floor = λ_min_true - 5.0
        λ = rough_lower(A, B, seed, floor)
        @test λ < λ_min_true
        @test λ ≥ floor - 1e-6
    end

    @testset "2. seed fails / midway succeeds" begin
        # Seed too high (A - seed*B is indefinite), floor low enough.
        n = 6
        A = Matrix(spd_random(n))
        B = Matrix{Float64}(I, n, n)
        λ_min_true = minimum(eigvals(Symmetric(A)))
        seed  = λ_min_true + 0.5     # too high — A - seed*I has a neg eig
        floor = λ_min_true - 10.0
        λ = rough_lower(A, B, seed, floor)
        @test λ < λ_min_true
    end

    @testset "3. floor fails everywhere" begin
        # Pick seed and floor BOTH inside the spectrum so no λ certifies PD.
        n = 5
        A = Matrix(spd_random(n))
        B = Matrix{Float64}(I, n, n)
        ev = eigvals(Symmetric(A))
        seed  = ev[2] + 1e-6        # > λ_1 (so A - seed*I has 1 neg eig)
        floor = ev[1] + 1e-6        # > λ_1 too — every λ in [floor,seed] gives at least 1 neg eig
        @test_throws VeigsRoughBoundError rough_lower(A, B, seed, floor)
    end

    @testset "4. λ_seed = 0 with SPD A" begin
        n = 4
        A = Matrix(spd_random(n)) + 2.0 * I    # min eig ≥ 2.5
        B = Matrix{Float64}(I, n, n)
        # Seed = 0 is already a valid lower bound; should return ~ -delta.
        λ = rough_lower(A, B, 0.0, -10.0)
        @test λ < 0
        @test λ > -10.0
    end

    @testset "5. barely-PD B" begin
        # Use rough_lower(B, I, λ_seed, λ_floor) to lower-bound λ_min(B).
        n = 6
        Q, _ = qr(randn(n, n)); Q = Matrix(Q)
        Λ = Diagonal([1e-6, 0.5, 1.0, 2.0, 3.0, 4.0])
        B = Q * Λ * Q'
        B = (B + B') / 2
        I_n = Matrix{Float64}(I, n, n)
        λ_min_true = 1e-6
        seed  = λ_min_true               # exact lambda_min as seed
        floor = λ_min_true / 100         # half-decade below
        λ = rough_lower(B, I_n, seed, floor)
        @test 0 < λ < λ_min_true
        @test λ ≥ floor - 1e-12
    end

    @testset "6. identity-shaped: A = αI, B = I" begin
        α = 3.0
        n = 5
        A = α * Matrix{Float64}(I, n, n)
        B = Matrix{Float64}(I, n, n)
        # A - λB = (α - λ)I is PD iff λ < α.
        λ = rough_lower(A, B, α - 0.1, α - 5.0)
        @test λ < α
        # rough_upper: λB - A = (λ - α)I is PD iff λ > α.
        λu = rough_upper(A, B, α + 0.1, α + 5.0)
        @test λu > α
    end

    @testset "7. iter-count cap — absurd schedule still terminates" begin
        n = 4
        A = Matrix(spd_random(n))
        B = Matrix{Float64}(I, n, n)
        # Floor very far below seed; delta small relative to span.
        # Should succeed quickly (k=1 already PD), but exercise the path.
        λ = rough_lower(A, B, -100.0, -1e10)
        @test λ < -100.0
    end

    @testset "8. soundness — returned λ < every concrete eigenvalue" begin
        for trial in 1:10
            n = rand(3:8)
            A = Matrix(spd_random(n))
            B = Matrix(spd_random(n))
            ev = eigvals(Symmetric(A), Symmetric(B))
            λ_min = minimum(ev); λ_max = maximum(ev)
            # rough_lower
            λL = rough_lower(A, B, λ_min - 0.01, λ_min - 10.0)
            @test λL < λ_min
            # rough_upper
            λU = rough_upper(A, B, λ_max + 0.01, λ_max + 10.0)
            @test λU > λ_max
        end
    end

    @testset "9. interval input still works" begin
        n = 4
        A0 = Matrix(spd_random(n))
        B0 = Matrix{Float64}(I, n, n)
        Ai = interval_perturb(A0; r = 1e-12)
        Bi = interval.(B0)
        λ_min_true = minimum(eigvals(Symmetric(A0)))
        λ = rough_lower(Ai, Bi, λ_min_true - 0.05, λ_min_true - 5.0)
        @test λ < λ_min_true
    end

    @testset "10. wrong direction throws ArgumentError" begin
        n = 3
        A = Matrix{Float64}(I, n, n); B = copy(A)
        @test_throws ArgumentError rough_lower(A, B, 1.0, 2.0)    # floor > seed
        @test_throws ArgumentError rough_upper(A, B, 2.0, 1.0)    # ceiling < seed
    end

    @testset "11. shape mismatch throws VeigsSizeError" begin
        @test_throws VeigsSizeError rough_lower(zeros(3, 3), zeros(2, 2), 0.0, -1.0)
        @test_throws VeigsSizeError rough_lower(zeros(3, 4), zeros(3, 4), 0.0, -1.0)
    end

end
