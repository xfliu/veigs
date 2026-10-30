# src/verified_isspd.jl
#
# Verified positive-definiteness test for INTERVAL symmetric matrices.
# Replaces INTLAB's `isspd`, which has no off-the-shelf Julia equivalent.
#
# Strategy (LDL + inertia + shift):
#   1. Pick a shift ρ > 0 (tiny initially).
#   2. Form an interval LDLᵀ factorization of (M − ρI) via `verified_ldl`,
#      which returns the float Bunch–Kaufman factors of mid(M − ρI) plus a
#      verified interval residual ΔA bounding (M − ρI) − PLDLᵀP'.
#   3. Read off the inertia of the float D-block diagonal:
#        - any negative pivot ⇒ λ_min(M − ρI) < 0 ⇒ λ_min(M) < ρ; bail (return false).
#        - any zero pivot     ⇒ borderline; bail.
#        - all-positive       ⇒ λ_min(LDLᵀ) > 0 strictly.
#   4. Bound the operator-2-norm of the residual via
#        ‖E‖₂ ≤ √(‖|ΔA|‖₁ · ‖|ΔA|‖_∞).
#      Weyl's inequality then gives
#        λ_min(M) = ρ + λ_min(M − ρI) ≥ ρ + 0 − ‖E‖₂ = ρ − ‖E‖₂.
#      If ‖E‖₂ < ρ, M is provably PD. Otherwise double ρ and retry.
#
# Why this beats Rump's σ_min(R)² > ρ form: the inner loop reuses the
# float LDLᵀ + scalar Wilkinson bound from `verified_ldl` (one BLAS gemm),
# whereas the σ_min(R) route requires interval back-substitution against
# n unit vectors (O(n³) interval ops, no BLAS). For n=1000 the difference
# is ~30× wall clock.
#
# CONTRACT (soundness, plan §7.2):
#   verified_isspd(M) == true   ⇒  every matrix in the interval enclosure M is SPD.
#   The reverse is NOT guaranteed (test may return false on borderline-PD inputs,
#   specifically when λ_min(M) ≤ ‖E‖₂).
#
# Throws DomainError if M contains Inf/NaN (D-015).
# Returns false if midpoint Bunch-Kaufman fails or inertia flips (D-014).

using LinearAlgebra
using SparseArrays
using IntervalArithmetic: Interval, interval, mid, inf, sup, radius, mag

const _VERIFIED_ISSPD_MAX_ATTEMPTS = 30

"""
    DEFAULT_ISSPD_METHOD::Ref{Symbol}

Process-wide default method used by [`verified_isspd`](@ref) when the
`method` kwarg is omitted. Mutable so users can switch the package-wide
behaviour atomically (all internal callers — `rough_bounds`,
`lehmann_behnke`, `veig`, `veigs` — flip in lockstep).

Default value: `:rump2006` (Rump 2006 single-Cholesky algorithm; same
algorithm as INTLAB's `isspd`). Set to `:ldl_shift` to use the LDL +
inertia + shift method (kept available for cross-checking).

```julia
Veigs.DEFAULT_ISSPD_METHOD[] = :ldl_shift  # package-wide
verified_isspd(A; method = :ldl_shift)     # per-call override
```
"""
const DEFAULT_ISSPD_METHOD = Ref{Symbol}(:rump2006)

"""
    verified_isspd(M::AbstractMatrix; method=DEFAULT_ISSPD_METHOD[]) -> Bool

Verified test that the interval matrix `M` is symmetric positive definite.

Soundness: returns `true` only when every concrete matrix in the interval
enclosure `M` is SPD. The test is conservative — it may return `false` on
a matrix that is borderline PD.

Methods:
- `:rump2006` (default) — Rump 2006 (BIT 46:433–452): one floating-point
  Cholesky of (M − cI) with closed-form `c ≥ ‖Δ(M)‖₂` from
  Theorem 2.3 + §3 eq. (I). This is the algorithm INTLAB's `isspd`
  implements. Sparse-friendly: dispatches to SuiteSparse Cholesky on
  `SparseMatrixCSC` input.
- `:ldl_shift` — Bunch–Kaufman LDLᵀ of (M − ρI), inertia of D,
  iterative residual ↔ shift comparison via Weyl's inequality.
  Always available; no sparse fallback needed (densifies sparse input).
  Kept for cross-validation against `:rump2006`.
- `:rump` — legacy alias, maps to `:ldl_shift`.

The active default is read from [`DEFAULT_ISSPD_METHOD`](@ref) when
`method` is omitted, so callers (rough_bounds, lehmann_behnke, veig,
veigs) all switch atomically.

Throws `DomainError` if `M` has `Inf`/`NaN` entries.
"""
function verified_isspd(M::AbstractMatrix; method::Symbol = DEFAULT_ISSPD_METHOD[])
    if method === :ldl_shift || method === :rump
        return _verified_isspd_ldl_shift(M)
    elseif method === :rump2006
        return _verified_isspd_rump2006(M)
    else
        throw(ArgumentError("verified_isspd: method must be :ldl_shift or :rump2006, got $method"))
    end
end

function _verified_isspd_ldl_shift(M::AbstractMatrix)
    n = size(M, 1)
    n == size(M, 2) || throw(VeigsSizeError("verified_isspd: matrix must be square"))
    n == 0 && return true                     # vacuously PD

    # Promote to interval matrix once (idempotent if already interval).
    Mi = M isa AbstractMatrix{<:Interval} ? M : interval.(M)
    _check_finite_midpoint(Mi)                # D-015

    Mm     = mid.(Mi)
    norm_M = max(opnorm(Mm, Inf), 1.0)

    # Seed ρ near the expected residual scale: γ_{3n}·‖M‖_∞ is the typical
    # magnitude of ‖E‖_∞ from a well-conditioned LDLᵀ. Starting too small
    # just costs an extra doubling step; starting too large wastes a probe
    # but never harms soundness.
    u   = eps(Float64) / 2
    γ3n = 3 * n * u >= 1.0 ? Inf : 3 * n * u / (1 - 3 * n * u)
    ρ   = max(2 * γ3n * norm_M, 4 * eps(Float64) * norm_M)

    for _attempt in 1:_VERIFIED_ISSPD_MAX_ATTEMPTS
        # Form the shifted interval matrix M − ρI.
        Mshift = copy(Mi)
        ρ_int  = interval(ρ)
        @inbounds for i in 1:n
            Mshift[i, i] = Mshift[i, i] - ρ_int
        end

        L, D, p, ΔA, ok = verified_ldl(Mshift)
        if !ok
            return false                      # LAPACK rejected the input outright
        end

        neg, pos, zer, F_unc = inertia(D)
        if F_unc || neg > 0 || zer > 0
            # Either inertia uncertain at this shift, or λ_min(M − ρI) ≤ 0,
            # which means λ_min(M) ≤ ρ. Either way, no positive certificate
            # via this route.
            return false
        end

        # Bound ‖E‖₂ via √(‖|ΔA|‖₁ · ‖|ΔA|‖_∞). For symmetric concrete
        # residuals ‖E‖₂ ≤ ‖E‖_∞, but ΔA itself may not be symmetric due
        # to interval-arithmetic conservatism, so use the safe geometric mean.
        absΔ = mag.(ΔA)
        nrm1, nrmi = _nonneg_norms_up(absΔ)   # sums rounded upward
        if !(isfinite(nrm1) && isfinite(nrmi))
            return false                      # interval blow-up
        end
        Ebnd = _sqrt_prod_up(nrm1, nrmi)

        if Ebnd < ρ
            return true                       # λ_min(M) ≥ ρ − Ebnd > 0
        end

        # Residual swallowed the shift; double up and retry. Bump past
        # 2·Ebnd to guarantee strict progress.
        ρ_new = nextfloat(2 * Ebnd)
        ρ_new > ρ || return false             # monotonicity guard
        ρ = ρ_new
    end
    return false
end

# ----------------------------------------------------------------------------
# Directed-rounding helpers.  Scalar interval operations give outward
# rounding without changing the global rounding mode.
# ----------------------------------------------------------------------------

_add_up(a::Float64, b::Float64) = sup(interval(a) + interval(b))
_sub_down(a::Float64, b::Float64) = inf(interval(a) - interval(b))
_sqrt_prod_up(a::Float64, b::Float64) = sup(sqrt(interval(a) * interval(b)))

# Upper bound for the exact sum of `nterms` nonnegative numbers whose
# floating-point sum, in any order, is `s`.  The rounding error of any
# summation order is at most γ_{n-1} times the sum of the absolute values,
# so for nonnegative terms s_exact ≤ s / (1 - γ_{n-1}).
function _nonneg_sum_up(s::Float64, nterms::Int)
    isfinite(s) || return Inf
    nterms <= 1 && return s
    k = interval(Float64(nterms - 1)) * interval(eps(Float64) / 2)
    sup(k) < 0.5 || return Inf
    γ = k / (interval(1.0) - k)
    return sup(interval(s) / (interval(1.0) - γ))
end

# Upper bounds for the 1-norm and ∞-norm of a nonnegative matrix R.
function _nonneg_norms_up(R::AbstractMatrix)
    r1   = _nonneg_sum_up(Float64(maximum(sum(R; dims = 1))), size(R, 1))
    rinf = _nonneg_sum_up(Float64(maximum(sum(R; dims = 2))), size(R, 2))
    return r1, rinf
end

# ----------------------------------------------------------------------------
# Rump 2006 — single-Cholesky verified PD test.
#
# Reference: S. M. Rump, "Verification of positive definiteness," BIT
# Numerical Mathematics 46:433–452, 2006. Theorem 2.3 + Corollary 2.4 +
# Section 3 eq. (I); Corollary 2.7 lifts the test to interval matrices.
#
# Algorithm:
#   1. c := γ_{n+1}/(1 − γ_{n+1}) · tr(|A_mid|) + n·M·eta,
#      where M := 3·(2n + max|a_νν|), eta = smallest subnormal,
#      γ_k := k·u/(1 − k·u), u = eps/2.  Note γ_{n+1}/(1 − γ_{n+1}) =
#      (n+1)u/(1 − 2(n+1)u); every quantity is rounded upward.
#   2. r := ‖radius(M)‖₂  (≤ √(‖R‖₁·‖R‖_∞), Perron-bound).
#   3. shift := c + r ; form Ã := A_mid − shift·I with the diagonal
#      rounded downward, as Theorem 2.3 requires.
#   4. Run a single floating-point Cholesky of Ã.
#      If it runs to completion ⇒ every concrete matrix in the interval
#      enclosure of M is SPD.
#
# Why this can be much faster than `:ldl_shift`:
#   - One Cholesky vs ≥1 LDLᵀ: Cholesky is ~2× cheaper and has a
#     specialised LAPACK path (`potrf`) that vectorises better than
#     Bunch–Kaufman (`sytrf`).
#   - The shift `c` is closed-form (no iteration needed; the proof
#     bakes the rounding error into the Theorem 2.3 bound).
#   - Sparse-friendly: shifting the diagonal preserves sparsity, and
#     Julia's `cholesky(::SparseMatrixCSC)` dispatches to SuiteSparse's
#     supernodal factorization. Bunch-Kaufman has no sparse path here.
# ----------------------------------------------------------------------------
function _verified_isspd_rump2006(M::AbstractMatrix)
    n = size(M, 1)
    n == size(M, 2) || throw(VeigsSizeError("verified_isspd: matrix must be square"))
    n == 0 && return true

    # Split into (midpoint, radius) — both float — without ever materialising
    # an n×n interval matrix. For float input the radius is identically zero
    # and we pass `nothing` to skip the radius-bound branch entirely
    # (DESIGN.md §1: never construct an n×n interval matrix outside the
    # ≤30×30 cluster work in lehmann_behnke).
    if M isa AbstractMatrix{<:Interval}
        Mm = Float64.(mid.(M))
        Mr = Float64.(radius.(M))
        if any(!isfinite, Mm)
            throw(DomainError(M, "matrix contains Inf or NaN entries"))
        end
    else
        Mm = M isa AbstractSparseMatrix ? sparse(Float64.(M)) : Matrix{Float64}(M)
        if any(!isfinite, Mm)
            throw(DomainError(M, "matrix contains Inf or NaN entries"))
        end
        Mr = nothing
    end
    return _verified_isspd_rump2006_midrad(Mm, Mr, n)
end

# Core implementation. Consumes (mid, radius) as separate float matrices.
# `Mr` may be:
#   - `nothing`        : exact float input (radius identically zero).
#   - an AbstractMatrix : entry-wise float radius matrix.
#   - a Tuple{Float64,Float64}: precomputed (‖Mr‖_1, ‖Mr‖_∞) scalar bounds.
#     This avoids allocating an n×n radius matrix when the caller can derive
#     scalar bounds analytically (e.g., `rough_bounds` and `veigs_driver`
#     forming `M = A − λB` from float A, B).
# DESIGN.md §1: post-processing kernel form. No interval-matrix arithmetic
# below this point.
function _verified_isspd_rump2006_midrad(Mm::AbstractMatrix,
                                         Mr::Union{AbstractMatrix, Nothing,
                                                   Tuple{Float64,Float64}},
                                         n::Int)
    # Defensive symmetrization on the midpoint (mirror v3's convention).
    # Done out-of-place to keep `Mm` writable for the diagonal shift below.
    # Every symmetric matrix in the enclosure has midpoint (Mm + Mm')/2 in
    # exact arithmetic.  Its floating-point evaluation is exact where Mm is
    # symmetric; elsewhere the rounding error is at most eps(|entry|) and is
    # added to the radius bound below.
    Msym = if Mm isa AbstractSparseMatrix
        sparse(Symmetric((Mm + Mm') / 2))
    else
        (Mm + Mm') / 2
    end
    sym_err = _symmetrization_error_up(Mm, Msym)
    isfinite(sym_err) || return false
    Mm = Msym

    # γ_{n+1}/(1 − γ_{n+1}) with γ_k = k·u/(1 − k·u), rounded upward.
    kn = interval(Float64(n + 1)) * interval(eps(Float64) / 2)
    sup(kn) < 0.5 || return false
    γnp1 = kn / (interval(1.0) - kn)
    γfac = γnp1 / (interval(1.0) - γnp1)

    # Largest |diagonal| sets both `tr_abs` (for c) and `M_const`.
    a_max = 0.0
    tr_abs = 0.0
    @inbounds for i in 1:n
        ai = abs(Mm[i, i])
        tr_abs += ai
        if ai > a_max
            a_max = ai
        end
    end
    eta     = interval(nextfloat(0.0))       # smallest subnormal, IEEE 754
    M_const = interval(3.0) * (interval(2.0) * interval(Float64(n)) + interval(a_max))
    tr_up   = _nonneg_sum_up(tr_abs, n)      # the trace sum, rounded upward
    isfinite(tr_up) || return false
    c_round = sup(γfac * interval(tr_up) + interval(Float64(n)) * M_const * eta)

    # Interval-radius bound: r ≥ ρ(R) ≤ √(‖R‖₁·‖R‖_∞).
    r_int = if Mr === nothing
        0.0                         # exact float input, radius identically zero
    elseif Mr isa Tuple
        r1, rinf = Mr               # caller-supplied upper bounds of the p-norms
        (isfinite(r1) && isfinite(rinf)) || return false
        _sqrt_prod_up(Float64(r1), Float64(rinf))
    else
        r1, rinf = _nonneg_norms_up(Mr)
        (isfinite(r1) && isfinite(rinf)) || return false
        _sqrt_prod_up(r1, rinf)
    end

    shift = _add_up(_add_up(c_round, r_int), sym_err)
    isfinite(shift) || return false

    # Tridiagonal fast path. For an n×n symmetric tridiagonal SPD matrix,
    # `cholesky` is O(n) (no fill-in) — but `Cholesky` doesn't have a
    # specialised method for `SymTridiagonal`. We use the equivalent fact:
    # the unpivoted LDLᵀ of an SPD matrix has all-positive diagonal D, and
    # `ldlt(::SymTridiagonal)` is O(n). This matches the certificate Rump
    # 2006 needs ("Cholesky runs to completion ⇔ all D pivots positive").
    if !(Mm isa AbstractSparseMatrix) && _is_tridiagonal_midpoint(Mm, n)
        return _rump2006_certify_tridiag(Mm, shift, n)
    end

    # Subtract the shift from the diagonal in-place (sparse-aware).
    if Mm isa AbstractSparseMatrix
        # Densify a copy of the diagonal entries, ensuring all n diagonal
        # slots exist in the sparse pattern (`spdiagm` materialises them).
        # Replace the diagonal by fl↓(a_ii − shift).  Removing the old
        # diagonal and adding the new one is exact.
        dg = Vector{Float64}(diag(Mm))
        Atilde = Mm - spdiagm(0 => dg) + spdiagm(0 => _sub_down.(dg, shift))
        F = cholesky(Symmetric(Atilde, :L); check = false)
        # SuiteSparse Cholesky on a non-PD input throws PosDefException
        # only with `check=true`; with `check=false` it returns a factor
        # whose `issuccess` is false.
        return issuccess(F)
    else
        Atilde = copy(Mm)
        @inbounds for i in 1:n
            Atilde[i, i] = _sub_down(Atilde[i, i], shift)
        end
        F = cholesky!(Symmetric(Atilde, :L); check = false)
        return F.info == 0
    end
end

# Tridiagonal certificate: ldlt(::SymTridiagonal) is O(n). All D pivots
# > 0 ⇒ the matrix is SPD, equivalent to Rump's "Cholesky runs to
# completion" criterion.
function _rump2006_certify_tridiag(Mm::AbstractMatrix, shift::Float64, n::Int)
    dv = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        dv[i] = _sub_down(Float64(Mm[i, i]), shift)
    end
    ev = Vector{Float64}(undef, max(n - 1, 0))
    @inbounds for i in 1:(n - 1)
        ev[i] = 0.5 * (Float64(Mm[i + 1, i]) + Float64(Mm[i, i + 1]))
    end
    T = SymTridiagonal(dv, ev)
    F = try
        ldlt(T)
    catch err
        err isa LinearAlgebra.ZeroPivotException || rethrow(err)
        return false                          # zero pivot ⇒ borderline, can't certify
    end
    @inbounds for d in F.data.dv
        (isfinite(d) && d > 0.0) || return false
    end
    return true
end

# Upper bound for the 2-norm of fl((Mm + Mm')/2) − (Mm + Mm')/2.  The error
# matrix is symmetric, so its 2-norm is at most its ∞-norm.  Each rounded
# entry is in error by at most eps(|entry|); exact entries contribute 0.
# In the sparse branch an entry may be counted twice, which only enlarges
# the bound.
function _symmetrization_error_up(Mm::AbstractMatrix, Msym::AbstractMatrix)
    n = size(Mm, 1)
    rows = zeros(Float64, n)
    if Mm isa AbstractSparseMatrix
        I_, J_, V_ = findnz(sparse(Mm))
        for k in eachindex(V_)
            i, j = I_[k], J_[k]
            V_[k] == Mm[j, i] && continue
            e = eps(abs(Float64(Msym[i, j])))
            rows[i] += e
            rows[j] += e
        end
    else
        @inbounds for j in 1:n, i in 1:n
            Mm[i, j] == Mm[j, i] && continue
            rows[i] += eps(abs(Float64(Msym[i, j])))
        end
    end
    any(!iszero, rows) || return 0.0
    return _nonneg_sum_up(maximum(rows), n)
end
