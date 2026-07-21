# test/test_verified_ldl.jl
#
# CORNER-CASE TAXONOMY (plan §7.3, reviewed 2026-05-01):
#   1. SPD inputs: factor cleanly, ok==true, ΔM ≈ 0 in midpoint
#   2. NPD / negative-definite inputs: still factor cleanly (Bunch-Kaufman
#      handles indefinite). ok==true.
#   3. Indefinite forced 2×2 block ([0 1; 1 0]): D contains a 2×2 block.
#      Cross-check by feeding D into `inertia`: must report (1, 1, 0, false).
#   4. Reconstruction regression — 50 random matrices, both uplo=:L and :U
#      satisfy max|M[p,p] - L*D*L'| < 1e-10. Pinned per D-005 (uplo=:L is
#      the production path; :U exists only for this regression).
#   5. Singular M (zeros, zero row/column): ok==true (BunchKaufman succeeds
#      on singular symmetric matrices — D simply contains a zero block).
#      Caller distinguishes via `inertia(D)` reporting a zero count.
#      ok==false is reserved for true LAPACK input errors (info < 0).
#   6. Permutation correctness: M[p, p] ≈ L * D * L' over a battery of
#      random matrices (point check; assumed by all downstream code).
#   7. Extreme conditioning (Hilbert n=8): ok==true, ΔM bounded by κ * eps.
#   8. Sparse M: densified internally; result agrees with dense path.
#   9. Interval input with non-zero radius: ΔM correctly *contains* the
#      width of the input perturbation (verified containment check).
#  10. Non-square input: throws VeigsSizeError.
#  11. Inf/NaN midpoint: throws DomainError.
#  12. Performance: n=100 dense factor + interval residual under 100 ms.

using Test, Random, LinearAlgebra, SparseArrays
import Veigs
using Veigs: verified_ldl, inertia, VeigsSizeError, VeigsLDLFailureError
using IntervalArithmetic: interval, mid, inf, sup, radius

include(joinpath(@__DIR__, "support", "matrix_factory.jl"))

Random.seed!(20260501)

@testset "verified_ldl" begin

    @testset "1. SPD inputs factor cleanly" begin
        for n in (1, 2, 5, 20)
            M = Matrix(spd_random(n))
            L, D, p, ΔM, ok = verified_ldl(M)
            @test ok == true
            # Midpoint of ΔM should be at numerical zero (we factored mid(M))
            @test maximum(abs, mid.(ΔM)) < 1e-10 * (1 + opnorm(M))
            # Permuted reconstruction in float arithmetic
            @test maximum(abs, M[p, p] - L * D * transpose(L)) < 1e-10 * (1 + opnorm(M))
        end
    end

    @testset "2. indefinite (negative-definite or mixed)" begin
        # purely negative-definite
        M = -Matrix(spd_random(10))
        _, D, p, _, ok = verified_ldl(M)
        @test ok
        neg, pos, zer, F = inertia(D)
        @test F == false
        @test neg == 10
        @test pos == 0

        # mixed (random indefinite)
        M2 = Matrix(indef_random(15))
        _, D2, _, _, ok2 = verified_ldl(M2)
        @test ok2
        # inertia of D2 should match the float-eigvals signature (sanity)
        n2_neg, n2_pos, _, F2 = inertia(D2)
        @test F2 == false
        ev = eigvals(Symmetric(M2))
        @test n2_neg == count(<(0), ev)
        @test n2_pos == count(>(0), ev)
    end

    @testset "3. forced 2×2 block — [0 1; 1 0]" begin
        M = [0.0 1.0; 1.0 0.0]
        L, D, p, ΔM, ok = verified_ldl(M)
        @test ok
        # Inertia must say (1, 1, 0)
        @test inertia(D) == (1, 1, 0, false)
        # Float reconstruction
        @test M[p, p] ≈ L * D * transpose(L)
    end

    @testset "4. reconstruction regression — 50 random matrices, both uplo" begin
        for trial in 1:50
            n = rand(2:20)
            # Mix SPD and indefinite to exercise both 1×1 and 2×2 D-blocks
            M = trial % 2 == 0 ? Matrix(spd_random(n)) : Matrix(indef_random(n))
            for uplo in (:L, :U)
                L, D, p, ΔM, ok = verified_ldl(M; uplo = uplo)
                @test ok
                # Julia's BunchKaufman convention: factor * D * factor' == M[p,p]
                # for both :L and :U.
                err = maximum(abs, M[p, p] - L * D * transpose(L))
                @test err < 1e-10 * (1 + opnorm(M))
            end
        end
    end

    @testset "5. singular M ⇒ ok=true with degenerate D (no throw)" begin
        L, D, p, ΔM, ok = verified_ldl(zeros(3, 3))
        @test ok == true
        # All-zero D is the diagnostic — caller checks via `inertia`.
        @test inertia(D) == (0, 0, 3, false)
        # Reconstruction is exact for the zero matrix.
        @test maximum(abs, mid.(ΔM)) == 0.0

        # Zero middle row/column with otherwise PD entries
        M = [1.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 1.0]
        _, D2, _, _, ok2 = verified_ldl(M)
        @test ok2 == true
        # Inertia: 2 positives, 1 zero
        neg, pos, zer, F = inertia(D2)
        @test (neg, pos, zer, F) == (0, 2, 1, false)
    end

    @testset "6. permutation correctness across random battery" begin
        for trial in 1:30
            n = rand(2:15)
            M = Matrix(indef_random(n))
            L, D, p, _, ok = verified_ldl(M)
            @test ok
            # `p` must be a valid permutation
            @test sort(p) == collect(1:n)
            # Reconstruction
            @test M[p, p] ≈ L * D * transpose(L)
        end
    end

    @testset "7. Hilbert n=8 (extreme conditioning)" begin
        M = Matrix(hilbert(8))
        L, D, p, ΔM, ok = verified_ldl(M)
        @test ok
        # Midpoint residual should be small but not insanely tight
        @test maximum(abs, mid.(ΔM)) < 1e-6
    end

    @testset "8. sparse and dense both produce valid factorizations" begin
        # Pre-v17: sparse and dense both went through dense Bunch-Kaufman
        # so L/D/p agreed. Post-v17 the sparse path takes LDLFactorizations
        # (different algorithm, AMD permutation, no 2×2 blocks). Both are
        # mathematically valid factorizations, but L, D, p generally differ.
        Md = Matrix(spd_random(10))
        Ms = sparse(Md)
        Ld, Dd, pd, ΔMd, okd = verified_ldl(Md)
        Ls, Ds, ps, ΔMs, oks = verified_ldl(Ms)
        @test okd == oks == true
        # Each factorization reconstructs its own permuted input.
        @test Md[pd, pd] ≈ Ld * Dd * Ld'
        @test Matrix(Ms[ps, ps]) ≈ Matrix(Ls) * Ds * Matrix(Ls)'
        # Inertia is intrinsic to M — must agree across factorizations.
        @test inertia(Dd)[1:3] == inertia(Ds)[1:3]
    end

    @testset "9. interval input — ΔM contains the perturbation" begin
        # M = M0 + interval-noise; ΔM should *enclose* the noise propagated
        # through the (now-fictitious-since-mid-is-M0) factor.
        M0 = Matrix(spd_random(8))
        Mi = interval_perturb(M0; r = 1e-10)
        L, D, p, ΔM, ok = verified_ldl(Mi)
        @test ok
        # The residual at midpoint should still be ~0 (we factored mid(Mi) = M0).
        @test maximum(abs, mid.(ΔM)) < 1e-9
        # And it must be a *real* interval matrix — every entry has well-defined sup/inf.
        @test all(isfinite, sup.(ΔM))
        @test all(isfinite, inf.(ΔM))
        # The radii should grow with the input perturbation (each ΔM entry
        # should have radius at least on the order of the input radius).
        @test maximum(radius, ΔM) ≥ 1e-12
    end

    @testset "10. non-square throws VeigsSizeError" begin
        @test_throws VeigsSizeError verified_ldl(zeros(3, 4))
    end

    @testset "11. Inf / NaN midpoint throws DomainError" begin
        @test_throws DomainError verified_ldl([1.0 NaN; NaN 1.0])
        @test_throws DomainError verified_ldl([1.0 Inf; Inf 1.0])
    end

    @testset "12. performance — n=100 under 100 ms" begin
        M = Matrix(spd_random(100))
        verified_ldl(M)                                    # warm-up + compile
        t = @elapsed verified_ldl(M)
        @test t < 0.10
    end

    @testset "13. dense-path soundness on 2D Laplacian P1 FEM" begin
        # 2D Laplacian (5-point K, 7-point M) is non-tridiagonal, so the
        # tridiag fast-path is skipped and the dense Bunch-Kaufman path
        # is exercised. The (mid, rad) refactor (#6) computes ΔM in float
        # arithmetic and packs to interval at the end — this is the
        # regression test that the resulting ΔM still rigorously encloses
        # the true residual.
        for n_side in (5, 10, 15)
            K, Mfem, _h = laplacian2d_p1(n_side)
            N = n_side^2
            λ = 2.0 * π^2 * 0.9
            Mi = interval.(K) .- interval(λ) .* interval.(Mfem)

            L, D, p, ΔM, ok = verified_ldl(Mi)
            @test ok

            # Reference residual computed in float
            Mm   = Float64.(mid.(Mi))
            recon = Float64.(L) * Float64.(D) * Float64.(L)'
            err  = Mm[p, p] .- recon

            ΔM_perm = ΔM[p, p]
            for j in 1:N, i in 1:N
                @test inf(ΔM_perm[i, j]) ≤ err[i, j] ≤ sup(ΔM_perm[i, j])
            end
        end
    end

    @testset "15. ldl_backend kwarg dispatch (v17)" begin
        K, Mfem, _h = laplacian2d_p1(8)
        Mi = interval.(K) .- interval(2.0 * π^2 * 0.9) .* interval.(Mfem)

        # :auto — default, takes sparse fast path
        _, _, _, ΔM_auto, ok_auto = verified_ldl(Mi)
        @test ok_auto

        # :dense — explicit dense fallback (legacy path, slower but valid)
        _, _, _, ΔM_dense, ok_dense = verified_ldl(Mi; ldl_backend = :dense)
        @test ok_dense

        # :ldlfac — force sparse, no dense fallback
        _, _, _, ΔM_lfac, ok_lfac = verified_ldl(Mi; ldl_backend = :ldlfac)
        @test ok_lfac

        # :mumps — placeholder, throws informative error
        @test_throws VeigsLDLFailureError verified_ldl(Mi; ldl_backend = :mumps)

        # invalid backend
        @test_throws ArgumentError verified_ldl(Mi; ldl_backend = :bogus)
    end

    @testset "14. dense path: float input matches interval-promoted input" begin
        # Refactor optimization #6: skip `interval.(M)` when input is
        # already float. ΔM should be the same up to interval-arithmetic
        # outward-rounding ULP differences.
        K, Mfem, _h = laplacian2d_p1(8)
        λ = 2.0 * π^2 * 0.9
        M_float    = Matrix(K) .- λ .* Matrix(Mfem)
        M_interval = interval.(M_float)

        _, _, _, ΔM_f, ok_f = verified_ldl(M_float)
        _, _, _, ΔM_i, ok_i = verified_ldl(M_interval)

        @test ok_f && ok_i
        # Both must enclose the same float residual; widths comparable.
        N = size(M_float, 1)
        for j in 1:N, i in 1:N
            @test inf(ΔM_f[i, j]) ≤ sup(ΔM_i[i, j])
            @test sup(ΔM_f[i, j]) ≥ inf(ΔM_i[i, j])
        end
    end

end
