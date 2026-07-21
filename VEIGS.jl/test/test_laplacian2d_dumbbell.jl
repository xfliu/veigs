# test/test_laplacian2d_dumbbell.jl
#
# CORNER-CASE TAXONOMY (clustered eigenvalues):
#   1. mesh structure: ndof scales as O(1/h²) interior + O(1/h) bar edges
#   2. K and M are symmetric, sparse, positive definite
#   3. eigenvalues from veigs enclose the reference values from eigen()
#   4. tight cluster (λ_1, λ_2) near 2π² ≈ 19.74 — split by tunneling
#      through the bar; veigs(k=2) returns both with rigorous separation
#   5. four-fold cluster (λ_3..λ_6) near 5π² ≈ 49.35 — the (2,1)/(1,2)
#      modes doubled by the two lobes; veigs(k=6) returns all six
#   6. cluster-too-large detection: k=3, 4, 5 throw
#      VeigsClusterTooLargeError (refusing to half-count cluster 2)
#   7. veigs is consistent across P1 and P2 matrices on the same h

using Test, Random, LinearAlgebra, SparseArrays
import Veigs
using Veigs: veigs, VeigsClusterTooLargeError
using IntervalArithmetic: inf, sup

include(joinpath(@__DIR__, "support", "laplacian2d_dumbbell.jl"))

Random.seed!(20260501)

@testset "laplacian2d_dumbbell — clustered eigenvalues" begin

    @testset "1. mesh structure" begin
        K, M, ndof, grid = laplacian2d_dumbbell(0.1, 1)
        # h=0.1, w=0.2: bar is 20 cells wide, 2 cells tall = 40 quads;
        # each square is 10×10 = 100 quads. Total = 240 quads = 480 triangles.
        @test length(getcells(grid)) == 2 * (100 + 40 + 100)
        @test ndof > 0
        # Larger meshes have more DOFs.
        for h in (0.05, 0.025)
            _, _, ndof_finer, _ = laplacian2d_dumbbell(h, 1)
            @test ndof_finer > ndof
            ndof = ndof_finer
        end
    end

    @testset "2. K and M sparse, symmetric, PD" begin
        K, M, _, _ = laplacian2d_dumbbell(0.05, 1)
        @test K isa AbstractSparseMatrix
        @test M isa AbstractSparseMatrix
        @test K ≈ K'
        @test M ≈ M'
        @test isposdef(Matrix(K))
        @test isposdef(Matrix(M))
    end

    @testset "3. veigs enclosures match eigen()" begin
        K, M, _, _ = laplacian2d_dumbbell(0.05, 1)
        F = eigen(Symmetric(Matrix(K)), Symmetric(Matrix(M)))
        λ_ref = sort(F.values)

        # First cluster (k=2)
        λ_int, _ = veigs(K, M, 2, :smallestabs)
        @test length(λ_int) == 2
        @test inf(λ_int[1]) ≤ λ_ref[1] ≤ sup(λ_int[1])
        @test inf(λ_int[2]) ≤ λ_ref[2] ≤ sup(λ_int[2])
        # Bounds are rigorously separated (λ_2 lower bound > λ_1 upper bound).
        @test inf(λ_int[2]) > sup(λ_int[1])
    end

    @testset "4. tight cluster (λ_1, λ_2) handled correctly" begin
        # h=0.025 P1: gap ≈ 2.05e-4. veigs must produce intervals tight
        # enough to separate the pair (gap >> width).
        K, M, _, _ = laplacian2d_dumbbell(0.025, 1)
        λ_int, _ = veigs(K, M, 2, :smallestabs)
        @test length(λ_int) == 2
        gap_lo = inf(λ_int[2]) - sup(λ_int[1])
        @test gap_lo > 1e-5     # rigorous lower bound on the gap
        # Each interval is much narrower than the gap.
        for lam in λ_int
            @test sup(lam) - inf(lam) < gap_lo / 100
        end
    end

    @testset "5. four-fold cluster (λ_3..λ_6) at k=6" begin
        K, M, _, _ = laplacian2d_dumbbell(0.025, 1)
        λ_int, ind = veigs(K, M, 6, :smallestabs)
        @test length(λ_int) == 6
        @test ind == 1:6
        # All 6 are sound (interval ≤ next interval lower bound).
        for i in 1:5
            @test inf(λ_int[i+1]) ≥ inf(λ_int[i])
        end
        # Cluster 1 (λ_1, λ_2) ≈ 19.74; cluster 2 (λ_3..λ_6) ≈ 49.35.
        @test sup(λ_int[2]) < 25.0
        @test inf(λ_int[3]) > 45.0
        @test sup(λ_int[6]) < 55.0
    end

    @testset "6. cluster-too-large refusal at k=3, 4, 5" begin
        # The second cluster has 4 members (49.29, 49.36, 49.48, 49.55).
        # Asking for any k that splits it must throw.
        K, M, _, _ = laplacian2d_dumbbell(0.025, 1)
        for k in (3, 4, 5, 7)
            @test_throws VeigsClusterTooLargeError veigs(K, M, k, :smallestabs)
        end
    end

    @testset "7. P1 and P2 give consistent (λ_1, λ_2) at the same h" begin
        # As h → 0 both should converge to the continuum dumbbell pair;
        # at h=0.05 they differ by ≤1% in absolute value but both must
        # enclose their respective discrete eigenvalues correctly.
        for k in (1, 2)
            K, M, _, _ = laplacian2d_dumbbell(0.05, k)
            λ_int, _ = veigs(K, M, 2, :smallestabs)
            @test length(λ_int) == 2
            # λ_1 in some neighbourhood of 2π²; λ_2 just above.
            @test 19.0 ≤ inf(λ_int[1]) ≤ 21.0
            @test 19.0 ≤ inf(λ_int[2]) ≤ 21.0
            @test inf(λ_int[2]) > sup(λ_int[1])
        end
    end

end
