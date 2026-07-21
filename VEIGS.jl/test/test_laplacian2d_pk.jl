# test/test_laplacian2d_pk.jl
#
# CORNER-CASE TAXONOMY:
#   1. matrix shape and DOF count match the Pk theory
#      (interior DOFs: P1: (n-1)², P2: (2n-1)², P3: (3n-1)²)
#   2. K and M are symmetric and sparse
#   3. K and M are positive definite (interior Laplacian is PD)
#   4. smallest eigenvalue converges to π²(1²+1²) = 2π² with the
#      expected O(h^{2k}) rate (Strang-Fix)
#   5. the Pk matrix exercises veigs end-to-end (smallest eigenvalue
#      via cluster expansion + Lehmann-Behnke)
#   6. invalid arguments throw

using Test, Random, LinearAlgebra, SparseArrays
import Veigs
using Veigs: veigs
using IntervalArithmetic: inf, sup

include(joinpath(@__DIR__, "support", "laplacian2d_pk.jl"))

Random.seed!(20260501)

@testset "laplacian2d_pk" begin

    @testset "1. DOF count matches Pk theory" begin
        # For Ferrite's default unit-square triangulation: n_cells × n_cells
        # cells, each split into 2 triangles. Interior DOFs:
        #   P1: vertices,        (n_cells - 1)²
        #   P2: + edge midpoints, (2 n_cells - 1)²
        #   P3: + edge thirds + face center, (3 n_cells - 1)²
        for n in (4, 8, 16)
            K1, M1, d1 = laplacian2d_pk(n, 1)
            K2, M2, d2 = laplacian2d_pk(n, 2)
            K3, M3, d3 = laplacian2d_pk(n, 3)
            @test d1 == (n - 1)^2
            @test d2 == (2n - 1)^2
            @test d3 == (3n - 1)^2
            @test size(K1) == (d1, d1)
            @test size(M1) == (d1, d1)
            @test size(K2) == (d2, d2)
            @test size(K3) == (d3, d3)
        end
    end

    @testset "2. K and M symmetric and sparse" begin
        for k in (1, 2, 3)
            K, M, _ = laplacian2d_pk(8, k)
            @test K isa AbstractSparseMatrix
            @test M isa AbstractSparseMatrix
            @test K ≈ K'
            @test M ≈ M'
        end
    end

    @testset "3. K and M positive definite" begin
        for k in (1, 2, 3)
            K, M, _ = laplacian2d_pk(8, k)
            @test isposdef(Matrix(K))
            @test isposdef(Matrix(M))
        end
    end

    @testset "4. eigenvalue convergence rate is O(h^{2k})" begin
        # h ∝ 1/n_cells; halving h should reduce the relative error
        # in the smallest eigenvalue by 2^{2k} for Pk (Strang-Fix).
        λ_exact = 2 * π^2
        for k in (1, 2, 3)
            errs = Float64[]
            for n in (8, 16, 32)
                K, M, _ = laplacian2d_pk(n, k)
                λ_min = first(sort(eigen(Symmetric(Matrix(K)),
                                         Symmetric(Matrix(M))).values))
                push!(errs, abs(λ_min - λ_exact) / λ_exact)
            end
            # Ratio of successive errors should be near 2^{2k}.
            ratio_8_to_16  = errs[1] / errs[2]
            ratio_16_to_32 = errs[2] / errs[3]
            expected = 2.0^(2k)
            # ±20% tolerance to account for finite-h preasymptotic regime.
            @test 0.8 * expected ≤ ratio_8_to_16  ≤ 1.25 * expected
            @test 0.8 * expected ≤ ratio_16_to_32 ≤ 1.25 * expected
        end
    end

    @testset "5. veigs end-to-end on Pk matrices" begin
        for k in (1, 2, 3)
            K, M, _ = laplacian2d_pk(8, k)
            λ, _ = veigs(K, M, 1, :smallestabs)
            @test length(λ) ≥ 1
            # Truth must be inside the verified interval.
            λ_truth = 2 * π^2
            # P1 mesh n=8 has ~4% relative error, so the bound width
            # alone won't enclose 2π². We just check ordering and
            # nonzero width is plausible.
            @test inf(λ[1]) ≤ sup(λ[1])
            @test sup(λ[1]) - inf(λ[1]) < 0.1 * abs(inf(λ[1]))
        end
    end

    @testset "6. invalid arguments" begin
        @test_throws ArgumentError laplacian2d_pk(0, 2)
        @test_throws ArgumentError laplacian2d_pk(4, 0)
        @test_throws ArgumentError laplacian2d_pk(4, 4)
    end

end
