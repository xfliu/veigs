# Discussion log — `verified_isspd` performance journey

This file captures the design discussion that produced the four iterations
of `verified_isspd` (v1 → v4) and the introduction of `:rump2006` as the
package default. It is kept alongside `DECISIONS.md` (which records
*decisions*) and `report.html` (which records *measurements*) so future
maintainers can reconstruct *why* the algorithm looks the way it does.

---

## Phase 0 — initial port (v1)

Setting: porting MATLAB `veigs.m` + INTLAB to Julia. INTLAB's `isspd` has
no off-the-shelf Julia equivalent, so we wrote one.

**v1 strategy.** Mirror Rump's "Verification of Positive Definiteness"
(BIT 2006, paraphrased from memory at the time of writing): take the
midpoint of the interval matrix `M`, compute an approximate Cholesky
`R'R ≈ mid(M)`, then bound `λ_min(M)` from below via

    λ_min(M)  ≥  σ_min(R)²  −  ‖M − R'R‖₂.

Rump-style: if `σ_min(R)² > ‖M − R'R‖₂`, then `M` is provably PD.

**v1 implementation.**
- Float Cholesky on `mid(M)`.
- `R'R` reconstructed via interval matmul (`transpose(interval.(R)) * interval.(R)`).
- `‖M − R'R‖₂` bounded by `√(‖E‖₁·‖E‖_∞)` of the interval matrix.
- `σ_min(R)` lower-bounded by `1 / max_j ‖R⁻¹ e_j‖_∞`, computed via
  interval back-substitution against `n` unit basis vectors.

**v1 measured cost (sparse 1D Laplacian, smallest eigenvalue, full
`veigs` wall time).**
- n = 1000: 113.2 s
- n ≥ 2000: practically infeasible (extrapolated multiple hours)

The interval matmul of `R'R` was the dominant cost. Julia's
`IntervalArithmetic.jl` falls through to a generic loop for matmul (no
BLAS path), so this was an O(n³) inner loop with no vectorisation.

---

## Phase 1 — algorithmic insight #1 (user, 2026-05-01)

> User: "The algorithm itself does not require an interval version of LDL.
> An approximate LDL decomposition plus a post-processing
> (already implemented in the algorithm) is enough."

This was the key correction. Rephrased mathematically: we don't need
`interval(R'R)` to bound `‖M − R'R‖₂`; we just need *some* upper bound on
that operator norm. Wilkinson backward error gives

    |fl(R'R) − exact(R'R)|_{ij}  ≤  γ_n · (|R'|·|R|)_{ij},
    γ_n := n·u / (1 − n·u),    u = eps/2.

So:
1. Compute `R'R` once, in **float** (BLAS gemm, fast).
2. Compute `|R'|·|R|` once, also in float (one extra BLAS gemm of
   positive matrices, ~50% the cost of a regular gemm).
3. The interval residual is `M − interval(R'R_float) + interval(±B_round)`
   with `B_round = γ_n · |R'|·|R|`.

The downstream consumer of the residual is `verified_isspd` itself,
which only reads its operator-1 / operator-∞ norm — entry-tightness was
overkill all along.

**v2 result.**
- n = 1000: 113.2 s → **31.0 s** (3.65× speedup)
- n = 2000: 199.1 s
- n = 5000: cancelled at ~50 min mark

Now the bottleneck shifted: with `R'R` no longer dominating, the next
biggest item was `_verified_lower_smin_R` — the interval back-sub
against unit vectors. Same O(n³) complexity, no BLAS path, but with a
worse constant per op.

A first attempt used a *scalar* Wilkinson bound `γ_n·‖R'‖_∞·‖R‖_∞` and
that worked at n=1000 but **broke on the banded n=10 README example**
where `‖R‖_∞ ≈ 1e8`. The scalar form inflated to ~10, which dominates
any reasonable `σ_min(R)²`. Switched to the per-entry Wilkinson bound
`γ_n · (|R'|·|R|)_{ij}` — kept entries ≈ unit-magnitude even when `R`
has large dynamic range.

---

## Phase 2 — algorithmic insight #2 (user, 2026-05-01)

> User: "Why R\\I is needed? Basically, the original algorithm has no
> place to calculate R^{-1}. To get lower bound of the first eigenvalue,
> we try some predicted bound, denoted by rho, then verify R - rho I is
> positive definite."

This is the correct decomposition. The outer driver (`veigs`) verifies
PD-ness by trying ρ and checking `M − ρI` is PD via LDL+inertia.
`verified_isspd` can use the **same** shift trick on itself, no `R⁻¹`
needed.

**Soundness argument** (Weyl + inertia):
- Pick ρ > 0.
- Compute `verified_ldl(M − ρI) → (L, D, p, ΔA, ok)`.
- If `inertia(D)` is all-positive (no zeros, no negatives), then the
  float reconstruction `LDLᵀ` is strictly PD: `λ_min(LDLᵀ) > 0`.
- The residual `E := (M − ρI)_concrete − LDLᵀ` satisfies
  `‖E‖₂ ≤ √(‖|ΔA|‖₁·‖|ΔA|‖_∞)`.
- By Weyl: `λ_min(M − ρI) ≥ λ_min(LDLᵀ) − ‖E‖₂ > 0 − ρ + (ρ − ‖E‖₂)`.
- Therefore: `λ_min(M) = λ_min(M − ρI) + ρ ≥ ρ − ‖E‖₂`.
- If `‖E‖₂ < ρ`, then `λ_min(M) > 0`. ✓

The shift ρ absorbs the residual; **no quantitative lower bound on
`λ_min(LDLᵀ)` needed**. This is the cleanest form because:
- No `σ_min(R)` ⇒ no interval back-sub.
- The whole loop reuses `verified_ldl` (already optimised in phase 1).
- It's the same algorithmic shape as the outer driver, so the package
  is internally consistent.

**v3 implementation** (`_verified_isspd_ldl_shift`):
1. Seed ρ ≈ `2·γ_{3n}·‖M‖_∞` (typical residual scale).
2. Loop: factor `M − ρI`, check inertia; bail if non-positive inertia
   (means `λ_min(M) ≤ ρ`).
3. Bound `‖E‖₂` from `mag.(ΔA)`. If `‖E‖₂ < ρ` ⇒ return true.
4. Otherwise double ρ past `2·‖E‖₂` and retry.

Companion fix: `verified_ldl`'s rounding bound also switched from
scalar to per-entry `γ_{3n}·(|L|·|D|·|Lᵀ|)_{ij}` for the same
banded-n=10 reason as v2.

**v3 result.**
- n = 1000: 31 s → **1.11 s** (28× over v2, 102× over v1)
- n = 2000: 199 s → 7.95 s
- n = 5000: previously cancelled → 90.4 s

All 723 tests still pass; bound widths unchanged at ~8e-16.

---

## Phase 3 — literature alignment (user question, 2026-05-01)

> User: "the positive verify in INTLAB is using the a priori error
> estimation, that is, if approximate LDL decomposition succeeded, then
> one can draw the conclusion. Can you find paper or resource on such
> algorithm?"

The reference is:

> **S. M. Rump, "Verification of Positive Definiteness," BIT Numerical
> Mathematics, 46:433–452, 2006.**
> Free preprint: <https://www.tuhh.de/ti3/paper/rump/Ru06c.pdf>

The core result, paraphrased from the paper:

**Theorem 2.3.** If the floating-point Cholesky of `A` runs to
completion (all square roots real, all `r_jj ≥ 0`), then

    λ_min(A) > − ‖Δ(A)‖₂,

where `Δ(A)` is a closed-form matrix depending only on the diagonals
and the sparsity pattern of `A`:
- `α_{ij} := γ_{s(i,j)+2}` if `s(i,j) ≠ 0` else 0, with
  `s(i,j) ≤ min(i,j) − 1`.
- `d_j := ((1 − α_{jj})⁻¹ a_{jj})^{1/2}`.
- `Δ(A)_{ij} := α_{ij} d_i d_j + M·eta`,
  `M := 3·(2n + max a_{νν})`, `eta` = smallest subnormal.

**Corollary 2.4** (the actual `isspd` algorithm):
1. Compute `c ≥ ‖Δ(A)‖₂`.
2. Form `Ã = A − c·I`.
3. Run a single floating-point Cholesky on `Ã`.
4. **If it runs to completion ⇒ A is positive definite.**

§3 eq. (I) gives the simplest closed-form bound:

    ‖Δ(A)‖₂  ≤  γ_{n+1}/(1 − γ_{n+1})·tr(A)  +  n·M·eta.

This is the algorithm INTLAB's `isspd` implements. The structural
advantages over v3:
- One Cholesky vs iterative LDL+inertia (each iter does `sytrf` + an
  inertia scan; Rump does one `potrf`).
- Closed-form ρ (no doubling loop).
- Sparsity-preserving: `c·I` doesn't change the sparsity pattern,
  and Cholesky has a tuned SuiteSparse path on
  `SparseMatrixCSC` whereas Bunch–Kaufman doesn't.

---

## Phase 4 — implementation and head-to-head (v4, 2026-05-01)

Added `_verified_isspd_rump2006` alongside `_verified_isspd_ldl_shift`,
both selectable via a `method` kwarg and a process-wide
`DEFAULT_ISSPD_METHOD::Ref{Symbol}`.

User instruction: keep the existing method (v3 = `:ldl_shift`) as a
selectable option; don't replace it wholesale. Kept both. The user
later said: "we use rump2006 as the default. keep ldl_shift as a
comparison." Flipped `DEFAULT_ISSPD_METHOD[]` from `:ldl_shift` to
`:rump2006`.

**Per-call timing of `verified_isspd` alone** (well-conditioned PD
interval input, median of 5):

| n    | `:ldl_shift` (v3) | `:rump2006` (v4) | speedup |
|------|------------------:|-----------------:|--------:|
| 200  |             3.6 ms |          0.88 ms |    4.1× |
| 500  |            88 ms   |          6.4 ms  |   14×   |
| 1000 |           216 ms   |         31 ms    |    7×   |
| 2000 |          1029 ms   |        157 ms    |    6.5× |
| 5000 |        15 785 ms   |       1445 ms    |   11×   |

**Full `veigs` wall-time (sparse 1D Laplacian, smallest eig)**:

| n    | `:ldl_shift` | `:rump2006` | speedup |
|------|-------------:|------------:|--------:|
| 1000 |       1.03 s |      0.68 s |   1.5×  |
| 2000 |       4.78 s |      3.38 s |   1.4×  |
| 5000 |      80.5 s  |     64.8 s  |   1.24× |

The full-`veigs` speedup is modest because `verified_isspd` is just
one of several O(n³) components in the cluster-expansion driver. The
per-call speedup (5–14×) is the more honest measure of the algorithmic
improvement.

Bound widths are bit-identical between v3 and v4. All 723 tests pass
under either method.

---

## Phase 5 — closing the n=5000 gap (2026-05-01)

After v4 the n=5000 sparse Laplacian was 64.8 s vs MATLAB+INTLAB 13.1 s
(~5× gap). User asked: "How can we further improve the efficiency?"

### Profile-first

A 5-minute `@profile` run on the n=5000 case surfaced two findings my
back-of-envelope had missed:

1. **`sym_hull(M)` was 21 % of total wall time.** It's a one-line
   broadcast `hull.(M, transpose(M))`. For a 5000×5000 interval matrix
   the broadcast machinery + `transpose` indexing dominates the per-element
   `hull` cost, and we were calling it 5–10 times per `veigs` run.
2. **Dense `eigen(Symmetric(A), Symmetric(B))` was 24 %** (~20 s
   standalone), not the dominant cost I'd assumed. Most of `veigs` time
   went to interval-arithmetic primitives in the iterated bound work
   (`rough_bounds`, `lehmann_behnke`, `verified_isspd`).

### Phase 5a — `sym_hull` rewrite

> User: "we can have more native way to implement
> `(hull.(M, transpose(M)))`. We can consider the option to mark the
> symmetry of input matrix."

Implemented:

- `sym_hull(M)` switched from a one-line broadcast to a tight loop
  over the upper triangle (`n(n+1)/2` hull calls instead of `n²`,
  no broadcast machinery).
- `sym_hull!(H, M)` for in-place mutation (saves one allocation).
- New kwarg `sym_hull(M; assume_symmetric=true)` returns `M` unchanged.
- Audited all call-sites; only `rough_bounds.jl:106` is safe to flip
  (`A − λ·B` of two symmetric matrices is symmetric in interval
  arithmetic). The `lehmann_behnke` and `_veigs_eig_local_bound` calls
  symmetrize matrices built from non-symmetric expressions like
  `V'·A·nV − nV'·(B·nV − A·V) + Err`, where interval round-off
  introduces real asymmetry; those keep the defensive pass but get the
  faster loop.

Net at n=5000: 82 s → 73 s (≈ 9 s saved, 11 % speedup). All 723 tests
still pass; bound widths unchanged.

### Phase 5b — sparse approximate eigensolve

The dense `eigen` was now the single biggest item at 28 % of total. The
fix: replace it with a sparse Krylov method that exploits A's sparsity.

**Arpack first** (it's the named MATLAB equivalent). Tried every config:
`nev=1`, `which=:LM/sigma=0.0` (shift-invert), `:SM`, `:SR`, `:LR`,
`:BE`; tuned `maxiter`, `ncv`, `tol`. Every single one threw
`XYAUPD_Exception` (max iterations reached) on the n=5000 1D Laplacian.
Even when failing fast it costs ~1 s. A first wiring of Arpack-with-
fallback regressed total wall time from 73 s to 96 s (the failed Arpack
calls + fallback overhead). Reverted.

Arpack.jl's reliability problems on Apple Silicon and on poorly-scaled
problems (smallest eigenvalue ≈ 4e-7 here) are well-known; this is
consistent.

**KrylovKit second.** Default Lanczos also fails to converge —
λ_min ≈ 4e-7 is too poorly separated from machine zero for plain
Arnoldi. But the *shift-invert* form works:

```julia
F = lu(A - σ*B)                 # SuiteSparse sparse LU, O(n) for tridiagonal
op = x -> F \ (B * x)
μs, vecs, info = KrylovKit.eigsolve(op, n, k, :LM; tol=1e-10)
λs = σ .+ 1.0 ./ μs
```

Each Lanczos step solves `(A − σB)·x = B·y` once; for tridiagonal A
this is O(n). Total n=5000 cost: ~10 ms vs ~20 s for dense LAPACK
(≈ 2000× speedup on the eigensolve alone).

`_approx_eig` now dispatches:

- n ≥ 200 and sparse input → KrylovKit shift-invert.
- σ from sigma symbol (`smallest*` → 0.0; numeric → the value;
  `largest*` → plain Arnoldi without shift).
- Convergence failure or unsupported sigma → dense fallback.

KrylovKit added as a dep (`Project.toml` v0.10).

**Caveat — first-call compile cost.** KrylovKit's shift-invert path
has non-trivial compile-time overhead. A single warm-up wasn't enough
on n=1000 (apparent 1.8 s regression vanished after a second warm-up).
Bench scripts now warm twice.

### Phase 5 measurement

| n | v4 (pre-phase-5) | v5 (sym_hull + KrylovKit) | savings | MATLAB+INTLAB |
|---|--:|--:|--:|--:|
| 1000 | 0.68 s | **0.72 s** | ≈ 0 | 0.32 s |
| 2000 | 3.38 s | **2.95 s** | 0.4 s (13 %) | (n/a) |
| 5000 | 64.8 s | **40.9 s** | **23.9 s (37 %)** | 13.1 s |

The win concentrates at large n because that's where dense `eigen` was
expensive enough to matter. At n=1000 dense `eigen` is already ~125 ms
out of 680 ms total, so swapping it for KrylovKit's 1 ms doesn't
visibly move the needle.

### Where the remaining gap to MATLAB lives

At n=5000 we're now ~3× behind MATLAB+INTLAB (40.9 s vs 13.1 s) instead
of ~5× before. The remaining ~28 s is in the iterated bound work:

- `rough_bounds` and `lehmann_behnke` repeatedly build interval
  matrices `A − λ·B` and operate on them densely. Even with
  `assume_symmetric=true` for one of those call-sites, the dominant cost
  is the n² interval arithmetic.
- `verified_ldl` and `verified_isspd` densify sparse input. When the
  input is tridiagonal, MATLAB's INTLAB sparse LDL is O(n); ours is
  O(n³).

Both are iter-8 work: keeping `SparseMatrixCSC` end-to-end and using
SuiteSparse Cholesky directly on the interval residual.

## Phase 6 — bandwidth-preserving LDL (2026-05-01)

After Phase 5, the n=5000 sparse Laplacian was 40.3 s vs MATLAB+INTLAB
0.31 s. Re-checking the per-unit profile (with `profile on/off` in
MATLAB and standalone Julia call timings) showed the entire MATLAB
advantage was in `verified_ldl`: ~5 ms sparse LDL on the tridiagonal
in MATLAB vs 12 100 ms dense Bunch-Kaufman + 3 dense BLAS gemms in
Julia.

### User question

> "Since given matrix is sparse and banded, especially the one from FEM,
> the L in LDL is also sparse, banded matrix, right? Why dense matrix is
> needed for L?"

Correct. L preserves the bandwidth of A. For tridiagonal A, L is
bidiagonal; for FEM bandwidth b, L has bandwidth b. The dense L was
purely an artifact of `bunchkaufman(Symmetric(M))` being dense-only in
Julia stdlib — no mathematical requirement.

### v6 implementation

Added a tridiagonal fast path (smallest contained change) to two
routines:

- **`verified_ldl`**: detects bandwidth ≤ 1 in the dense interval
  midpoint, builds a `SymTridiagonal{Float64}`, factors with stdlib
  `ldlt(::SymTridiagonal)` (O(n)), and computes the pentadiagonal
  `L·D·Lᵀ` + Wilkinson bound in closed form. Falls back to dense
  Bunch-Kaufman on `LinearAlgebra.ZeroPivotException`.
- **`_verified_isspd_rump2006`**: same detection, but uses the fact
  that "Cholesky succeeds ⇔ all D pivots from `ldlt(SymTridiagonal)`
  are positive." Equivalent to Rump's certificate, O(n).

Both fast paths still walk the dense interval matrix once (the
`_is_tridiagonal_midpoint` scan is O(n²)) — that cost stays until
upstream plumbing keeps `SparseMatrixCSC` end-to-end (iter 8).

### v6 measurement

| n | v5 | **v6** | savings |
|---|--:|--:|--:|
| 1000 | 0.72 s | **0.44 s** | 0.28 s (39 %) |
| 2000 | 2.95 s | **1.55 s** | 1.40 s (47 %) |
| 5000 | 40.9 s | **19.4 s** | **21.5 s (53 %)** |

Per-call standalone timings at n=5000:

| Routine | v5 (dense) | v6 (tridiag fast path) | speedup |
|---|--:|--:|--:|
| `verified_ldl(n=5000)`        | 12 100 ms | **261 ms** | 46× |
| `verified_isspd(:rump2006)`   |  1 334 ms | **548 ms** | 2.4× |

Bounds still ULP-tight (7.93e-16 width), all 723 tests pass.

### What's left to close (the remaining 62×)

The `verified_ldl` factorization is now O(n), matching MATLAB. The
remaining 19 s at n=5000 is in **dense interval matrix overhead
elsewhere**:

- `Ai = interval.(Matrix(A))` densifies sparse input at `veigs_driver.jl:95`.
- `Mi = Ai .- λi .* Bi` is a dense O(n²) interval broadcast.
- `Mi[p, p]` and `ΔM[invp, invp]` in the dense fallback path of
  `verified_ldl` are O(n²) interval permutation copies.
- `sym_hull(M)` for non-rough_bounds callers is still O(n²).
- `radius.(Mi)` and the row-/column-sum norm bound in
  `_verified_isspd_rump2006` are O(n²) on dense Mi.

The unifying fix is iter 8: **keep `SparseMatrixCSC{Interval{Float64}}`
end-to-end through the driver**, so `Mi` itself is sparse. With that:
- The tridiagonal detection becomes free (input type is `SparseMatrixCSC`).
- All O(n²) dense passes become O(nnz).
- Should land Julia within ~2-3× of MATLAB at n=5000.

## Phase 7 — sparse end-to-end (2026-05-02)

After Phase 6, the remaining 62× gap to MATLAB at n=5000 was attributed
to "dense interval matrix overhead upstream of the factorisation."

### User question

> "Strange to me — `IntervalArithmetic.jl` doesn't have an analogous
> sparse-pattern-aware interval matrix. In Julia, we can design anything
> we need, right?"

The user was right to push back. A direct probe showed Julia +
`SparseArrays` + `IntervalArithmetic.jl` already supports
`SparseMatrixCSC{Interval{Float64}}` fully out of the box — every
operation we needed worked through generic dispatch:

| Operation | Works on `SparseMatrixCSC{Interval}`? | Cost at n=1000 |
|---|:-:|---:|
| `sparse(interval.(A))`               | ✓ | trivial |
| `Ai .- λi .* I` (broadcast)          | ✓ | preserves sparsity (2998 nnz) |
| `mid.(Ai)` / `mag.(Ai)` / `radius.(Ai)` | ✓ | sparse Float64 returns |
| `Ai[p, p]` (permutation)             | ✓ | 0.1 ms (vs O(n²) dense) |
| `SymTridiagonal(mid.(Ai))` + `ldlt`  | ✓ | 0.08 ms |
| `verified_isspd(Ai; method=:rump2006)` | ✓ | 0.3 ms (vs 31 ms dense) |

So Phase 6's diagnosis ("the type system doesn't have a sparse interval
matrix") was wrong. The blocker was a single line of plumbing.

### v7 implementation

One change in `veigs_driver.jl`:

```julia
# Before:
Ai = A isa AbstractMatrix{<:Interval} ? A : interval.(Matrix(A))   # densifies!

# After:
Ai = if A isa AbstractMatrix{<:Interval}
    A
elseif A isa AbstractSparseMatrix
    interval.(A)              # preserves sparsity
else
    interval.(Matrix(A))
end
```

Plus a small helper for the sparse case of "smallest eigenvalue of B"
that uses the existing KrylovKit shift-invert pipeline rather than
densifying B for `eigvals`.

That's it. No changes to `verified_ldl`, `verified_isspd`,
`rough_bounds`, `lehmann_behnke`, or `_veigs_eig_local_bound` were
needed — Julia's multiple dispatch routes the sparse interval matrix
through the sparse methods these routines already had (the sparse
Cholesky branch in `_verified_isspd_rump2006`, the sparse permutation
in `verified_ldl`, etc.).

### v7 measurement

| n | v6 | **v7** | savings | MATLAB+INTLAB | Julia/MATLAB |
|---|--:|--:|--:|--:|--:|
| 1000 | 0.44 s | **0.103 s** | 4.3× | 0.093 s | 1.1× |
| 2000 | 1.55 s | **0.191 s** | 8.1× | 0.147 s | 1.3× |
| 5000 | 19.4 s | **0.758 s** | **25×** | 0.311 s | **2.4×** |

Bounds still ULP-tight (7.93e-16 width at n=5000). All 723 tests pass.

### What's left

Julia is now within a small constant factor of MATLAB on this problem.
The remaining ~2.4× at n=5000 is generic constant-factor overhead in
`IntervalArithmetic.jl`'s arithmetic on `Interval{Float64}` (the per-op
construction-and-validation cost) vs INTLAB's specialised `intval` class
that's been optimised for ~25 years. That's a much smaller, more
incremental gap than the 130× we started at.

### Lesson

The biggest insight of the session was at the very end: the limiting
factor wasn't algorithmic, library, or type-system — it was a single
`Matrix(A)` call that I had inherited from earlier iterations and never
questioned. The user's "in Julia we can design anything we need, right?"
forced a re-examination that revealed everything was already in place.

## Phase 8 — Diagonal inertia + redundant-broadcast cleanup (2026-05-02)

After Phase 7 the sparse-end-to-end plumbing landed and Julia was
within 2.4× of MATLAB at n=5000. The user said "do it now" to the
follow-up investigation of the residual constant-factor gap.

### Hypothesis from the v7 profile

The flat profile of `veigs(A, B, 1, :sa)` at n=5000 (851 ms total) showed:
- Broadcast machinery (`materialize`, `copy`, `_broadcast_getindex`): ~42 % of leaf time
- `inertia` (line 49 of `inertia.jl`): 142 samples (~14 %)
- `IntervalArithmetic._fastmul!`: ~10 %
- `lehmann_behnke`: ~10 %
- Generic memory allocation: ~17 %

The first finding was a microbenchmark surprise: `.- Ai` (negation of
a 5000×5000 sparse interval matrix) took **113 μs vs 12 μs** for the
parallel `(SparseMatrixCSC{Float64}, SparseMatrixCSC{Float64})`
inf/sup representation. ~10× ratio confirms per-element interval
construction (with `Decoration` validation) is a dominant constant
factor.

### Fix 1 — Diagonal D from the tridiagonal LDL fast path

Looking at `inertia(D::AbstractMatrix)`, line 49 does
`Di = interval.(D)` to promote a real D to an interval D. For the
tridiagonal LDL fast path (added in Phase 6), D is a 5000×5000
**dense Float64 matrix** whose only nonzero entries are on the
diagonal — so this broadcast does 25 million `Interval{Float64}`
constructions for n=5000 entries that matter. That's the 14 %
inertia hot spot.

Two changes:
1. `_verified_ldl_tridiag` now returns `D::Diagonal{Float64}` and
   `L::SparseMatrixCSC{Float64}` (sparse bidiagonal), instead of dense
   matrices.
2. New `inertia(D::Diagonal; ...)` method iterates D's diagonal
   directly, no broadcast.

### Fix 2 — Redundant interval-matrix allocations

Three lines were doing extra O(nnz) allocation for arithmetic
identities:

- `opnorm(_sup_abs_matrix(.-ΔM), Inf)` → `opnorm(_sup_abs_matrix(ΔM), Inf)`.
  `mag(-x) = mag(x)` for any interval `x`.
- `err_test * Bi .- (.-ΔM)` → `err_test * Bi .+ ΔM`. Identity in
  interval arithmetic.
- `_sup_abs_matrix(M::AbstractMatrix{<:Interval}) = Float64.(mag.(M))`
  → `_sup_abs_matrix(M) = mag.(M)`. The outer `Float64.` was a
  no-op broadcast pass since `mag(::Interval{Float64})::Float64`.

### v8 measurement

| n | v7 | **v8** | savings | MATLAB+INTLAB | ratio |
|---|--:|--:|--:|--:|--:|
| 1000 | 103 ms | **20 ms** | 5× | 93 ms | **Julia 4.7× faster** |
| 2000 | 191 ms | **62 ms** | 3× | 147 ms | **Julia 2.4× faster** |
| 5000 | 758 ms | **441 ms** | 1.7× | 311 ms | MATLAB 1.4× |

The v8 numbers are the first time Julia *beats* MATLAB+INTLAB on
this benchmark. At n=1000 the difference is dominated by the
verified-bounds work being called 5–10 times during cluster
expansion — each time avoiding a 25M-entry interval broadcast adds up.

Bound widths still ULP-tight (7.93e-16 at n=5000). All 723 tests pass.

### Why this win was hiding

The sparse plumbing in Phase 7 was the biggest single change, but it
exposed a previously-hidden bottleneck: the `inertia(::Matrix)` path
had been wasteful since Phase 6 (when we started returning a dense D
from the tridiagonal fast path). Before Phase 7, `verified_ldl` was
12 s and `inertia` was a small fraction; after Phase 7, factorization
dropped to milliseconds and the wasteful interval broadcast in
`inertia` became the new dominant cost. Iter-on-iter profiling
caught this — the v8 fix wouldn't have been worth doing before
Phase 7 because it would have been swamped by the LDL cost.

### What's left

Julia is now within 1.4× of MATLAB at n=5000. The remaining gap is
generic interval-arithmetic constant factor inside `lehmann_behnke`'s
small-matrix interval matmul (the `_fastmul!` and broadcast hot spots
that didn't go away in v8). Could be improved by either specialising
`Interval` arithmetic for the small (typically 3×3 to 30×30)
cluster-matrix case, or by upstream optimisation in
`IntervalArithmetic.jl` v1.x. Diminishing returns from here.

## Cumulative result (2026-05-02)

| version | change | n=1000 | n=5000 | speedup vs v1 (n=1000) |
|---------|--------|------:|------:|--------:|
| v1 | interval matmul + Rump σ_min(R)         | 113.2 s | n/a    | 1× |
| v2 | float gemm + per-entry Wilkinson + σ_min(R) | 31.0 s | (cancelled) | 3.7× |
| v3 | LDL + inertia + shift (Weyl)            |  1.03 s | 80.5 s | 110× |
| v4 | Rump 2006 single Cholesky               |  0.68 s | 64.8 s | 166× |
| v5 | + sym_hull tight loop + KrylovKit shift-invert | 0.72 s | 40.9 s | 157× |
| v6 | + tridiag fast path in verified_ldl & rump2006 | 0.44 s | 19.4 s | 257× |
| v7 | + sparse-end-to-end plumbing | 0.103 s | 0.758 s | 1099× |
| **v8** | **+ Diagonal inertia + redundant-broadcast cleanup** | **0.020 s** | **0.441 s** | **5660×** |

`:rump2006` is now the default. `:ldl_shift` is retained as
`method = :ldl_shift` (or `Veigs.DEFAULT_ISSPD_METHOD[] = :ldl_shift`)
for cross-validation. The two algorithms have *different* soundness
proofs (Rump's diagonal-only `Δ(A)` bound vs. Weyl's perturbation
inequality on the LDL residual), so a divergence between them on any
input is informative — useful for catching algorithmic bugs.

## What's next (iter 8)

The remaining gap to MATLAB+INTLAB on truly sparse inputs (n=5000:
13.1 s vs 64.8 s, ~5×) is structural: Julia's `verified_ldl`
densifies sparse input at the doorway, and `_verified_isspd_rump2006`
is sparse-friendly *only* if the caller passes a `SparseMatrixCSC`
through to it. Plumbing sparse interval matrices end-to-end requires:

1. Keep `SparseMatrixCSC` at `verified_ldl` entry.
2. Use SuiteSparse's sparse Cholesky for the midpoint factorisation.
3. Compute the interval residual on a sparse representation without
   densifying.

With v4 already preferring Cholesky over Bunch-Kaufman, step 2 is
much cleaner than it would have been under v3.

## References

- S. M. Rump, "Verification of Positive Definiteness," *BIT Numerical
  Mathematics*, 46:433–452, 2006.
  [Springer link](https://link.springer.com/article/10.1007/s10543-006-0056-1) ·
  [Preprint PDF](https://www.tuhh.de/ti3/paper/rump/Ru06c.pdf)
- N. J. Higham, *Accuracy and Stability of Numerical Algorithms*,
  2nd ed., SIAM, 2002, §10.
- J. Demmel, "On floating-point errors in Cholesky," LAPACK Working
  Note 14, 1989. [SINUM 24:5](https://doi.org/10.1137/0720095).
- T. Yamamoto, T. Ogita et al., "Verified Error Bounds for Matrix
  Decompositions," *SIAM J. Matrix Anal. Appl.*, 2024+.
  [DOI](https://doi.org/10.1137/24M165096X). Modern follow-up that
  extends Rump 2006 to LU/QR/LDLᵀ.

## Phase 9 — apples-to-apples MATLAB re-bench + GetInertia.m patch (2026-05-02)

### User observation

> "the get inertia for matlab version is not efficient. so it is a
> cheap-win."

The user noticed that `veigs/GetInertia.m` line 24-26 calls `D =
intval(D)` upfront before walking the block-diagonal entries, and
hypothesised that this is the same trap we just fixed in Julia at v8
(materialising an n×n intval matrix when only ~3n entries are actually
read). Patched MATLAB-side and ran a clean A/B bench.

### Patch

`veigs/GetInertia.m`: extract `diag(D)`, `diag(D, 1)`, `diag(D, -1)`
once, iterate those vectors in the loop, intval-promote the four
scalars per 2×2 block on demand. Original kept verbatim as
`veigs/GetInertia_orig.m` for direct A/B (function symbol renamed so
both can live on path simultaneously).

### Repeatable bench harness

`veigs/bench_inertia_apples.m` runs three blocks in one MATLAB session:

1. Full `veigs(A,B,1,'sa')` wall-time on sparse 1D Laplacian + I, n ∈
   {1000, 2000, 5000}, 5 reps after 2 warm-ups.
2. `GetInertia` (patched) per-call: 100 reps for real D and intval D
   inputs at n ∈ {1000, 5000}.
3. `GetInertia_orig` (baseline) per-call: same 100×.

Mirror Julia harness: `Veigs.jl/bench/bench_laplacian_smallest.jl`
(BenchmarkTools, 5 samples, 2 warm-ups). Run each side twice for
stability.

### Per-call result (real D from sparse LDL, n=5000, 100 reps, warmed)

| variant | median | trim mean | speedup |
|---|--:|--:|--:|
| `GetInertia_orig` (baseline) | 174–223 ms | 175–236 ms | 1× |
| **`GetInertia` (patched)** | **5.7 ms** | **5.7 ms** | **~30×** |

The patch is a real ~30× per-call win.

### Wall-clock result

| n | MATLAB+INTLAB (median 5 reps) | Julia v8 (median 5 reps) | Julia speedup |
|---|--:|--:|--:|
| 1000 | 0.337 s | 0.018 s | ~14× |
| 2000 | 1.064 s | 0.061 s | ~17× |
| 5000 | 15.5 s (range 11–26 s) | 0.46 s | ~30× |

Bound widths identical at every n (~8e-16). Soundness preserved.

### What this revealed

1. **The cheap-win the user pointed at exists, but doesn't move
   wall-clock.** `GetInertia` is called once per `veigs` run, so
   saving 168 ms / call × 1 call = 168 ms inside an 11–19 s total is
   <1 % of wall time. INTLAB's `intval(sparse)` already preserves
   sparsity, so the original wasn't actually allocating n²
   intervals — the slow part was per-element scalar indexing into a
   sparse intval matrix inside the loop. The patch's
   diagonal-extraction step bypasses that path; the algorithmic
   improvement is real but its target is not on the wall-clock
   critical path.

2. **The previous "MATLAB 0.311 s at n=5000" baseline in `report.html`
   and §4.4 tables was wrong.** The clean A/B harness on this same
   machine measures 11–19 s with original `GetInertia` and
   essentially the same with patched `GetInertia` (variance >
   patch-savings).

3. **MATLAB's actual bottleneck on this benchmark is `eigs`'s
   convergence failure on λ<sub>min</sub> ≈ 4×10⁻⁷.** When `eigs`
   fails, `veigs.m:127` falls back to `eig(full(mid(A)),
   full(mid(B)))`, which is dense generalized eigendecomposition on
   5000×5000 — that's the multi-second cost. Julia uses KrylovKit +
   sparse-LU shift-invert (Phase 5), which converges in ~6 ms.

4. **Honest takeaway about the prior comparison.** The "Julia 1.4×
   slower than MATLAB at n=5000" claim from Phase 8 was incorrect.
   Julia is actually ~30× faster on this exact harness. The win at
   v8 is genuine but the point of comparison was off.

### What would actually be a cheap-win on the MATLAB side

Switching `eigs` in `veigs.m:127-130` to shift-invert mode (factor
`(A − σ·B)` via sparse LU, run Lanczos against `x ↦ (A − σB)⁻¹·(B·x)`,
σ near the target) — same trick Julia uses — should bring MATLAB
n=5000 wall time from ~15 s to sub-second. That's the actual
algorithmic improvement available on the MATLAB side. The
`GetInertia.m` patch is correct but not on the critical path.

### Updated cumulative comparison

| version | n=1000 | n=5000 (Julia) | MATLAB n=5000 (corrected) |
|--|--:|--:|--:|
| v1 | 113.2 s | n/a | — |
| v4 | 0.68 s | 64.8 s | ~15 s (was reported as 13.1 s) |
| v6 | 0.44 s | 19.4 s | ~15 s |
| v7 | 0.103 s | 0.758 s | ~15 s |
| **v8** | **0.020 s** | **0.46 s** | **~15 s** |

So at v8, Julia ranges from ~14× faster (n=1000) to ~30× faster
(n=5000) than MATLAB+INTLAB on this exact harness — not the "1.4×
behind" or "within 1.4×" stated in earlier Phase 8 narrative. The
correction makes Julia look better, not worse.

## Phase 9b — `'sm'` vs `'sa'` and the corrected comparison (2026-05-02)

### User finding

After exporting `laplacian_n5000.mat`, the user ran their own
interactive measurement:

```matlab
>> tic; veigs(A,B,1,'sm'); toc;
経過時間は 0.323894 秒です。
>> tic; veigs(A,B,1,'sm'); toc;
経過時間は 0.328862 秒です。
>> tic; veigs(A,B,1,'sm'); toc;
経過時間は 0.332584 秒です。
```

Three consecutive calls, ~0.33 s each. That is **two orders of
magnitude** faster than the 11–19 s we measured with target
`'sa'` (smallest *real* eigenvalue) in the Phase-9 bench.

### Why the targets diverge

`'smallestabs'` (`'sm'`) and `'smallestreal'` (`'sa'`) return the
same eigenvalue mathematically for SPD A and B (all eigenvalues
positive), but they trigger different paths inside MATLAB's
`eigs`:

- `'sm'` → shift-invert mode with σ=0: factor `(A − 0·B) = A` via
  sparse LU, run Lanczos against `x ↦ A⁻¹·(B·x)`. Lanczos on the
  inverse converges fast because the smallest λ becomes the dominant
  eigenvalue of A⁻¹ — well-separated.
- `'sa'` → plain Lanczos on A directly. For our problem
  (λ_min ≈ 4×10⁻⁷, poorly separated from machine zero), this fails
  inside the default iteration cap and `veigs.m:127` falls back to
  `eig(full(mid(A)), full(mid(B)))` — multi-second dense work.

Julia's `veigs_driver.jl` already maps both `:sm` and `:sa` to
shift-invert with σ=0 (lines 318–319), so the Julia time is the
same under both targets.

### Re-bench with apples-to-apples target

`bench_inertia_apples.m` and `bench_laplacian_smallest.jl` both
switched from `'sa'` / `:sa` to `'sm'` / `:sm`. Same 5-rep median
protocol.

| n | MATLAB (`'sm'`) | Julia (`:sm`) | winner |
|---|--:|--:|--:|
| 1000 | 0.034 s | 0.024 s | Julia 1.4× |
| 2000 | 0.030 s | 0.087 s | MATLAB 2.9× |
| 5000 | 0.039 s | 0.55 s | **MATLAB ~14×** |

User-reported interactive MATLAB times for n=5000 (0.32–0.33 s)
sit ~8× higher than our scripted bench (0.039 s). The difference
is presumably JIT/warm-state between command-line and
function-scope MATLAB; both are sub-second and confirm
shift-invert succeeds.

### Where the Phase-9 numbers go

The "Julia 30× faster than MATLAB at n=5000" claim in Phase 9 is
**wrong** — it was measured against MATLAB's `'sa'` failure-fallback
path. Under apples-to-apples `'sm'`:

- Julia is faster than MATLAB **only at n ≤ 1000** on this benchmark.
- MATLAB is faster than Julia for n ≥ 2000.
- At n=5000, MATLAB is ~14× faster (scripted) or ~1.7× faster (using
  user's interactive figure).

The crossover sits around n=1000–2000.

### Where the residual gap lives (corrected)

Now that the comparison is apples-to-apples, the gap is structural
and lives in interval-arithmetic constant factor:

- INTLAB's `intval` class has hand-tuned sparse and dense kernels
  with ahead-of-time rounding-mode switching (per-block, not
  per-op).
- Julia's `IntervalArithmetic.jl` validates decoration on every op
  and goes through generic dispatch for sparse interval arithmetic
  — there is no specialised sparse-interval kernel.

Both contribute O(n²)-region constant overhead. Closing this would
need a `SparseIntervalMatrix(inf::SparseMatrixCSC,
sup::SparseMatrixCSC)` type with hand-vectorised primitives, or
upstream optimisation of `IntervalArithmetic.jl` v1.x.

### What stands

- The `GetInertia.m` patch is still a real ~30× per-call win, but
  cosmetic at this benchmark (1 call × ~6 ms saved out of ~40 ms
  total). Keeping the patch in `veigs/` for the regimes where
  `GetInertia` is on the critical path (large clusters, many
  cluster-expansion iters).
- Bound widths identical (~8e-16) across all comparisons.
- Soundness preserved.
- Julia's wins on small dense (Hilbert n=8: 26–307× faster) are
  unaffected — those don't go through `eigs` at all.

### Updated honest summary

| regime | Julia v8 vs MATLAB+INTLAB |
|--|:-:|
| Small dense (n ≤ 100) | Julia 26–307× faster |
| Mid sparse (n ≈ 1000) | Julia 1.4× faster |
| Large sparse (n ≈ 5000) | **MATLAB ~14× faster** |

## Phase 11 — kernel refactor toward DESIGN.md §1 (v9 in progress, 2026-05-02)

Started executing the audit-driven refactor announced in Phase 10:
no n-sized interval matrices outside the ≤30×30 cluster work in
`lehmann_behnke`. Each step is committed only after `Pkg.test()`
passes (723/724) and bound widths match v8 bit-for-bit.

### v9 (#14, #15) — `verified_isspd` / `rough_bounds`

1. **`verified_isspd_rump2006` mid/rad split.** Public
   `verified_isspd(M)` no longer materialises `interval.(M)` for
   float input. New internal `_verified_isspd_rump2006_midrad`
   helper accepts `Mr` as `nothing`, an `AbstractMatrix`, or a
   `Tuple{Float64,Float64}` of precomputed `(‖Mr‖_1, ‖Mr‖_∞)`
   scalar p-norm bounds.
2. **`rough_bounds` no Ai/Bi/Mi.** Replaced `Ai, Bi, M = Ai − λi·Bi`
   with float `M_mid = A_mid − λ·B_mid` plus *scalar* p-norm
   radius bounds `(r1, rinf)` derived analytically from
   `‖M_mid‖_p`, `‖B_mid‖_p`, and any input radii via
   sub-additivity. Wilkinson rounding bounded by
   `u/(1−u)·(‖M_mid‖_p + |λ|·‖B_mid‖_p)`. No interval matrix is
   ever built.

### Cumulative table (with bound-width / precision column)

| version | change | n=1000 | n=5000 | bound width n=5000 |
|---------|--------|------:|------:|------:|
| v1 | interval matmul + Rump σ_min(R) | 113.2 s | n/a | 7.93e-16 |
| v2 | float gemm + per-entry Wilkinson + σ_min(R) | 31.0 s | (cancelled) | 7.93e-16 |
| v3 | LDL + inertia + shift (Weyl) | 1.03 s | 80.5 s | 7.93e-16 |
| v4 | Rump 2006 single Cholesky | 0.68 s | 64.8 s | 7.93e-16 |
| v5 | + sym_hull tight loop + KrylovKit | 0.72 s | 40.9 s | 7.93e-16 |
| v6 | + tridiag fast path | 0.44 s | 19.4 s | 7.93e-16 |
| v7 | + sparse-end-to-end plumbing | 0.103 s | 0.758 s | 7.93e-16 |
| v8 | + Diagonal inertia + redundant-broadcast cleanup | 0.020 s | 0.441 s* | 7.93e-16 |
| **v9** | **+ kernel refactor: verified_isspd / rough_bounds (mid, rad)** | **0.020 s** | **0.470 s** | **7.93e-16** |

> \* The "0.441 s" v8 figure was a single favourable bench reading.
> On the stabilised harness (5 reps, drop fastest+slowest, two
> runs), v8 medians are 0.53–0.55 s. v9 medians are 0.47 s —
> reliably 10–15 % faster than v8 on identical conditions.

**Precision column.** Bound widths have been pinned at ~7.93e-16 for
n=5000 since v3 (when LDL+inertia replaced the original σ_min(R)
form). Every subsequent change including the v9 kernel refactor has
been performance-only; correctness and tightness unchanged. This is
the design contract (DESIGN.md §4): no future refactor widens the
bound without an explicit written rationale.

### v10 — sparse pentadiagonal ΔM in `_verified_ldl_tridiag`

Looked at `_verified_ldl_tridiag` and noticed line 277:
`ΔM = Matrix(Mi)`. For sparse interval input at n=5000, this
densifies to a 25M-entry interval matrix (~400 MB) per call. The
loop only writes pentadiagonal entries; the off-band entries are
just copied from `Mi` (which is itself zero outside the tridiagonal
in this case).

Rewrote to build `ΔM` as `SparseMatrixCSC{Interval{Float64}}` with
≤5n nonzeros directly (using `sparse(rows, cols, vals, n, n)`), no
densification. ~5n vs n² is a 1000× reduction in data movement at
n=5000.

### Performance result

| step | n=1000 | n=2000 | n=5000 | bound width n=5000 |
|--|--:|--:|--:|--:|
| v9 (#14, #15) | 0.020 | 0.068 | 0.470 | 7.93e-16 |
| **v10 (sparse ΔM)** | **0.016–0.020** | **0.045–0.060** | **0.22–0.35** | **7.93e-16** |
| Δ vs v9 | ~ −10 % | ~ −20 % | **~ −40 %** | unchanged |

Multiple runs at v10 show large variance at n=5000 (0.22 s best
case, 0.50 s worst, median ~0.30 s across 8 runs). The
~0.22 s number reported initially was a lucky run; the honest
representative figure is ~0.30 s. The 40 % win vs v9 holds across
runs.

All 723 tests pass; bound widths bit-identical (DESIGN.md §4
preserved).

### Cumulative table (n=5000 since v8 baseline; honest medians)

| version | change | n=5000 (median) | Δ vs v8 |
|--|--|--:|--:|
| v8 | baseline (Phase 8) | 0.55 s | — |
| v9 | kernel refactor (#14, #15) | 0.47 s | −15 % |
| v10 | sparse pentadiag ΔM | **~0.30 s** | **~ −45 %** |

MATLAB+INTLAB at n=5000 (`'sm'` target, scripted bench): 0.039 s.
Julia gap: ~8× (down from 14× at v8).

### Why this win was hiding

The `_verified_ldl_tridiag` path was added in Phase 6 (2026-05-01)
as the tridiagonal fast-path for verified LDL. At that time the
focus was the factor / inertia cost; the residual matrix was almost
an afterthought, and `ΔM = Matrix(Mi)` was the simplest expression.
After Phase 7 (sparse plumbing) and Phase 8 (Diagonal inertia),
the LDL+inertia chain dropped to milliseconds — but the densified
ΔM survived because it was conceptually "just the residual,
shouldn't matter". It mattered. ~30% of n=5000 wall time was
allocating + initialising 400 MB of dense interval entries that
were never read.

This is a recurring pattern: optimisation reveals waste that was
hidden inside cheaper operations. The lesson for the audit-driven
refactor: every "Matrix(...)" or "Float64.(...)" call on a sparse
input is suspicious until proven otherwise.

### v11 — sparse fast-path for `_is_tridiagonal_midpoint`

Profiled v10 at n=5000 and found `_is_tridiagonal_midpoint` was
14% of total time. The function did `for i, j in 1:n; Mm[i,j]; end`
— for sparse Mm, every `getindex` is `searchsortedfirst` over the
column's `rowval` slice. So O(n²·log nnz) per call.

Added a `SparseMatrixCSC` method that walks `colptr` / `rowval`
directly: O(nnz) per call (~5n vs 25M log-probes at n=5000).

Result: n=5000 wall time 0.30 s → 0.13 s, another **−56 %**.
Bound widths unchanged (7.93e-16). 723 tests pass.

### v12 — lehmann_behnke matmul reuse + interval×float dispatch

Profile after v11 showed `lehmann_behnke` was now 75% of veigs
time. Inside, every `Bi·Vi`, `Ai·Vi`, `Bi·nVi`, etc. went through
`interval × interval` matmul.

Two changes:

1. **Reuse the n×m intermediates.** The original code computed
   `Bi·Vi` and `Ai·Vi` *twice* each (once for `A0`, `A1`; again
   inside `err_vec`, `A2`). Six redundant n×m matmuls per call.
   Captured them once into `BV`, `AV`, `BnV`, `AnV`.

2. **Pass V (and nV) as float, not as `interval.(Vf)`**. The
   matmul then dispatches to `interval × float` instead of
   `interval × degenerate-interval`. In `IntervalArithmetic.jl`'s
   linear-algebra extension, the float-operand path skips a round
   of decoration validation per scalar op and produces tighter
   intervals.

Result at n=5000 sparse 1D Laplacian: 0.13 s → **0.085 s** (~36 %
faster), and **bound width 7.93e-16 → 6.18e-16** (22 % tighter).
All sizes show similar precision improvement (n=1000: 7.76e-16 →
6.10e-16, n=2000: 7.46e-16 → 5.95e-16). 723 tests pass.

This is the first change in the session that improved *both* speed
and bound precision.

### Cumulative session result (v8 → v12)

| n | v8 baseline | v9 | v10 | v11 | v12 | Δ vs v8 | bound v12 |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 1000 | 0.020 s | 0.020 | 0.018 | 0.018 | **0.007 s** | −65 % | 6.10e-16 |
| 2000 | 0.080 s | 0.080 | 0.045 | 0.045 | **0.019 s** | −76 % | 5.95e-16 |
| 5000 | 0.55 s | 0.47 | 0.30 | 0.13 | **0.085 s** | **−85 %** | 6.18e-16 |

MATLAB+INTLAB at n=5000 (`'sm'`, scripted): 0.039 s. **Julia gap
closed to ~2.2× at n=5000** (was 14× at v8). At n ≤ 1000, Julia is
now ~5× faster than MATLAB.

### v13 — precision kwarg for `lehmann_behnke`

User asked: "add an option to use multiple precision computing to
see the possibility of improvement of precision." Added
`precision::Type{<:AbstractFloat} = Float64` kwarg to
`lehmann_behnke`. When `precision !== Float64`, every interval op
inside the routine (including the outer n×m matmuls) runs at the
requested precision.

**Probe result** (`bench/precision_probe.jl`, n=50 1D Laplacian,
smallest eigenvalue, exact analytical eigenvector):

| precision | bound width | observation |
|--|--:|--|
| Float64 | 1.54×10⁻¹⁶ | ~ULP floor |
| BigFloat(256 bits) | 2.47×10⁻²⁹ | ~10¹³× tighter |
| BigFloat(512 bits) | 2.47×10⁻²⁹ | same as 256 — saturated |
| BigFloat(1024 bits) | 2.47×10⁻²⁹ | same — confirms floor |

The saturation across BigFloat precision-bits settings tells us the
bound floor is set by the **algorithm's residual term**
(`Err = ‖B·nV − A·V‖² / λ_B_min`) and the input separator gap, not
by arithmetic precision. Exactly the experiment DESIGN.md §3
specified — and a useful diagnostic for any user investigating
whether their bounds are precision- or algorithm-limited.

Cost: BigFloat ops are ~100× slower than Float64 at large n. The
kwarg is for small problems and diagnostic use, not for production
sparse n=5000 work.

### v14 — `lehmann_behnke` outer matmul: pure float gemm + scalar Wilkinson

**Profiling discovery.** At v12, profiled `Bi * Vf` (the n×m
interval × float matmul `IntervalArithmetic.jl` provides) versus
pure float `mid(Bi) * Vf` on the n=5000 sparse 1D Laplacian:

```
n=5000 m=1: Bi*V = 15.26 ms, B_mid*V = 0.013 ms, ratio = 1172×
n=5000 m=10: Bi*V = 54.08 ms, B_mid*V = 0.133 ms, ratio = 407×
n=1000 m=1: Bi*V =  0.82 ms, B_mid*V = 0.003 ms, ratio = 269×
```

`IntervalArithmetic.jl`'s sparse interval×float kernel is ~1000×
slower than pure float sparse·dense for our regime. That's the
v12 bottleneck.

**Fix (v14):** replace each of the four n×m interval×float matmuls
(`BV`, `AV`, `BnV`, `AnV`) with float gemm + per-entry Wilkinson
scalar bound, then pack to `Matrix{Interval{T}}` for the m×m chain.

**Wilkinson constant.** For sparse `(M·X)[i,j]`, the chain length
is `max-nnz-per-row(M)`, *not* the inner dimension n. Zero terms
contribute zero exactly. For 1D Laplacian, max-nnz-per-row = 3 →
γ_3 = 3u/(1−3u) ≈ 3.3×10⁻¹⁶. An earlier draft used γ_n and saw
bound widths balloon ~1500×; tightening to γ_max-nnz brought it
back into the 10⁻¹⁵ regime.

### Performance

| step | n=1000 | n=2000 | n=5000 | n=5000 bound width |
|--|--:|--:|--:|--:|
| v12 | 0.007 s | 0.019 s | 0.085 s | 6.18e-16 |
| **v14** | **0.0029 s** | **0.0055 s** | **0.016 s** | 3.11e-15 |
| Δ vs v12 | 2.4× faster | 3.5× faster | **5.3× faster** | 5× wider |

### Bound width is problem-dependent in v14

Compared to v8's interval-matmul-based bounds:

| problem | v8 width | v14 width | δ |
|--|--:|--:|--|
| Hilbert n=8 (Ex 1a, bound 1) | 4.13e-13 | **1.55e-15** | **270× tighter** |
| Hilbert n=8 (Ex 1a, bound 3) | 1.82e-09 | **3.12e-12** | **580× tighter** |
| 1D Laplacian n=5000 | 7.93e-16 | 3.11e-15 | 4× wider |

The widening on the tridiagonal problem comes from using a
worst-case Wilkinson scalar bound where v12 used IntervalArithmetic's
directed-rounding (per-element optimal). The tightening on the
Hilbert example comes from cleaner accumulation: float gemm + a
single γ_n bound vs. multiple chained interval-matmul widenings.

Per DESIGN.md §4, this widens the bound on tridiagonal problems
specifically; rationale: 5× speedup, bound still well under Float64
ULP × |λ|, dramatically tighter on dense problems.

### Cumulative session (v8 → v14, n=5000)

  Speed: 0.55 s → **0.016 s** — ~34× faster
  vs MATLAB+INTLAB ('sm'): MATLAB 0.039 s, Julia 0.016 s
  → **Julia is now ~2.4× FASTER than MATLAB at n=5000**
  (full reversal from 14× behind at v8).

  At n=1000 Julia is ~12× faster than MATLAB
  (Julia 2.9 ms vs MATLAB 34 ms).

  All 723 tests pass at every step.

### What's next

Diminishing returns from here. Remaining audit item #6
(`verified_ldl` dense path mid/rad form) only matters for non-tridiag
inputs. Plus the `lehmann_behnke` refactor could be tightened
further by using directed-rounding float arithmetic for the
radius computation (matching what interval arithmetic does
internally) — that would bring tridiagonal bounds back to v12
levels without losing the v14 speedup. ~half-day of careful work
if precision matters more.

### v15 — `outer_matmul` kwarg: tightness vs speed as user choice

After v14 the per-entry interval width on tridiagonal-style problems
widened ~15× (from 1.7e-16 to 2.7e-15 for n=5000 1D Laplacian) because
γ_k is the worst-case Higham bound. For matrices with integer
multipliers (1D Laplacian, FEM mass) the actual rounding is often
zero — the v12 directed-rounded interval×float matmul captured
that, the v14 Wilkinson bound cannot.

Resolution: keep both. New kwarg in `lehmann_behnke`:
```julia
lehmann_behnke(A, B, eig_list, V, ρ, σ, λ_B_min, r, s;
               outer_matmul = :fast)   # or :tight
```
- `:fast` (default, v14 path): float gemm + Wilkinson scalar bound.
  ~3.3 ms per call on n=5000 1D Laplacian, width ~2.7e-15.
- `:tight` (v12 path): IntervalArithmetic.jl interval×float matmul.
  ~64 ms per call on the same problem, width ~1.7e-16 (15× tighter).

Both are sound; both enclose the truth. The kwarg surfaces the
trade-off explicitly rather than hiding it. Default stays `:fast`
because for cluster soundness the bound is pre-clipped by the
algorithmic floor (separator gaps), so 15× on the inner step
rarely propagates outward — but power users with extreme
precision needs can opt in.

Trade-off table on n=5000 1D Laplacian (smallest eigenvalue):

| Mode      | Width      | Wall (ms) | Notes                            |
|-----------|------------|-----------|----------------------------------|
| `:fast`   | 2.7e-15    | 3.3       | Default. v14 path.               |
| `:tight`  | 1.7e-16    | 63.8      | v12 path. 15× tighter, 19× slower|

Tests: 5 new in `test_lehmann_behnke.jl` testset 11 — `:fast` and
`:tight` both enclose, `:tight` width ≤ `:fast` width (within 1e-14
slack), and `outer_matmul = :bogus` throws ArgumentError.

The kwarg is *not* plumbed through `veigs(A, B, k, sigma)` — that
public API stays unchanged and uses the default. Power users invoke
`lehmann_behnke` directly when they want `:tight`.

**Surprise result: `:fast` is *more* valuable at BigFloat than Float64.**
The `precision` and `outer_matmul` kwargs are orthogonal, so all four
combinations are valid. Bench at BigFloat(256), 1D Laplacian:

| n    | `:fast` (ms) | `:tight` (ms) | speedup |
|------|--------------|---------------|---------|
| 100  | 1.1          | 22.9          | 21×     |
| 500  | 6.1          | 559           | 91×     |
| 1000 | 10.4         | 2 334         | 225×    |
| 2000 | 22           | 9 574         | **436×**|

At Float64 BLAS keeps the interval×float kernel within ~20× of
sparse·dense gemm; at BigFloat there's no BLAS for either path, so
the entire gap shows up as per-element directed-rounding overhead.
Practical implication: BigFloat verification is only feasible at scale
through `:fast`. At n=5000 `:tight`+BigFloat ≈ 1 minute extrapolated;
`:fast`+BigFloat lands near 50–100 ms. Bound width at BigFloat ULP is
~10⁻³⁰, which is well below any algorithmic floor (separator gaps,
residual norm), so `:tight`'s extra rounding tightness becomes noise
at that precision.

Conclusion: `:fast` is the right default at *both* precisions.
`:tight` is a niche tool for ULP-tight Float64 on integer-multiplier
sparse matrices specifically.

### v16 — `verified_ldl` dense-path (mid, rad) refactor (audit #6)

The dense fallback was the last place in the kernel still doing O(N²)
per-entry interval-arithmetic ops. Profile on the 2D Laplacian P1 FEM
problem (5-point K, 7-point M, non-tridiag → dense path) showed:

- `recon_int = interval.(recon_f)` materialized a full
  `Matrix{Interval{Float64}}` (~600 MB at N=5000).
- `ΔM_perm = Mi_perm .- recon_int` densified to another interval matrix.
- Per-entry `ΔM_perm[i,j] = ΔM_perm[i,j] + interval(-b, b)` did O(N²)
  directed-rounded interval ops with no BLAS path.

Refactor: compute ΔM as a (mid, rad) split in float, pack to interval
once at the end via `_pack_midrad` (outward-rounded addition). Keeps
the Wilkinson reconstruction bound `γ_{3n}·(|L|·|D|·|Lᵀ|)` and adds a
float-subtraction rounding bound `u·(|Mm[p,p]| + |recon_f|)` to
account for `fl(a−b)` on the ΔM_mid step.

Bench on 2D Laplacian P1 dense-path (n_side² DOFs):

| N    | before (s) | after (s) | speedup |
|------|------------|-----------|---------|
| 400  | 0.03       | 0.02      | 1.5×    |
| 1600 | 0.91       | 0.37      | 2.5×    |
| 3600 | 8.28       | 2.65      | 3.1×    |
| 6400 | (~50s est) | 17.49     | ~3×     |

Speedup grows with N — confirms the per-entry interval loop was
linear-in-N²-dominated. Tridiag fast-path (1D Laplacian etc.)
unchanged at ~3 ms for n=5000.

Soundness regression test added (`testset 13`, 2D Laplacian for
n_side ∈ {5,10,15}): every entry of `ΔM[p,p]` rigorously encloses
`Mm[p,p] − fl(L·D·Lᵀ)`. Plus testset 14: float-input ΔM and
interval-promoted-input ΔM agree (intervals overlap).

This closes the kernel-refactor arc that started with v9. Every
hot path now uses the (mid, rad) form: verified_isspd (v9),
rough_bounds (v9), verified_ldl tridiag (v10) and dense (v16),
lehmann_behnke (v12-v15). 70182 tests passing.

### v17 — sparse no-pivot LDL fast path (audit #7)

**Apples-to-apples MATLAB+INTLAB bench** revealed a 47× gap on the
2D Laplacian P1 FEM problem (n_side=60, N=3600): Julia 6.24 s vs
MATLAB 0.13 s. Diagnosis: `verified_ldl` fell back to dense BK
(O(N³)) on non-tridiag inputs because Julia stdlib has no sparse
symmetric indefinite LDL. MATLAB+INTLAB calls MA57 from HSL.

Fix: add `LDLFactorizations.jl` as a runtime dep, route sparse
non-tridiag inputs through its sparse symbolic-then-numerical
LDL with AMD fill-reducing ordering. It's no-pivot, so a zero
pivot causes failure — we detect that (zero entries in `F.d`)
and fall through to dense BK. New kwarg
`ldl_backend ∈ (:auto, :dense, :ldlfac, :mumps)`:

- `:auto` (default) — sparse first, dense fallback.
- `:dense` — force dense BK (legacy / debugging).
- `:ldlfac` — force sparse, error on zero pivot.
- `:mumps` — placeholder seam; throws with implementation guidance.

Bench on 2D Laplacian P1 FEM smallest eigenvalue:

| n_side | N      | v16     | v17     | MATLAB  | v17 vs MATLAB         |
|--------|--------|---------|---------|---------|-----------------------|
| 20     | 400    | 0.047 s | 0.008 s | 0.037 s | Julia 4.6× faster     |
| 30     | 900    | 0.246 s | 0.020 s | 0.040 s | Julia 2.0× faster     |
| 40     | 1600   | 0.741 s | 0.069 s | 0.049 s | Julia 1.4× slower     |
| 50     | 2500   | 2.04 s  | 0.109 s | 0.078 s | Julia 1.4× slower     |
| 60     | 3600   | 6.24 s  | 0.180 s | 0.137 s | Julia 1.3× slower     |
| 80     | 6400   | (~30s)  | 0.439 s | 0.302 s | Julia 1.5× slower     |
| 100    | 10000  | (~80s)  | 0.785 s | 0.612 s | Julia 1.3× slower     |

Gap closed from **47×** to **1.3–1.5× at large N**, and Julia is
**faster than MATLAB at small N**. The remaining gap likely traces
to MA57's BK pivoting (no fallback overhead) and INTLAB's tuned
verified-residual kernel.

Bound widths unchanged from v16 (algorithm structure preserved).

Tests: testset 8 loosened (sparse and dense now use different
algorithms); new testset 15 covers `ldl_backend` dispatch with all
four values plus an invalid-symbol arg test. 70187 passing total.

The `:mumps` seam is intentionally a stub — MUMPS doesn't naturally
fit the `verified_ldl → (L, D, p, ΔM, ok)` interface (it's a solver,
not a factor exposer). The honest path is documented in code: a
package extension (`ext/VeigsMUMPSExt.jl`) plus a refactor of the
caller toward "inertia at λ" as the primitive. Contributions welcome.

### v18 — Higher-order Pk (Lagrange) via Ferrite.jl + MATLAB cross-bench

User asked to test 2D P2 and P3 elements (after the P1 test with the
hand-rolled Friedrichs-Keller helper). Picked option 4: add Ferrite.jl
as a dep, write a small assembler, validate eigenvalue convergence.

`laplacian2d_pk(n_cells, k)` for `k ∈ {1, 2, 3}` returns triangular
Pk Lagrange (K, M, ndof) for the 2D Dirichlet Laplacian on the unit
square. Convergence rates verified at the Strang-Fix predicted O(h^{2k}):
P1 ratio ≈ 4 per refinement, P2 ≈ 16, P3 ≈ 64.

**Apples-to-apples MATLAB+INTLAB comparison** (after exporting K, M
to .mtx via `EXPORT=true` in `bench/bench_laplacian2d_pk.jl` and
running `veigs/bench_laplacian2d_pk.m`):

| Element | ndof   | Julia v18 (s) | MATLAB+INTLAB (s) | Julia / MATLAB |
|---------|--------|---------------|-------------------|----------------|
| P1      | 225    | 0.004         | 0.034             | **8.5× faster**|
| P1      | 961    | 0.032         | 0.043             | **1.3× faster**|
| P1      | 3 969  | 0.256         | 0.152             | 1.7× slower    |
| P1      | 9 025  | 0.767         | 0.516             | 1.5× slower    |
| P2      | 961    | 0.028         | 0.032             | **1.1× faster**|
| P2      | 3 969  | 0.180         | 0.147             | 1.2× slower    |
| P2      | 16 129 | 1.211         | 1.081             | 1.1× slower    |
| P3      | 2 209  | 0.112         | 0.065             | 1.7× slower    |
| P3      | 9 025  | 0.742         | 0.563             | 1.3× slower    |
| P3      | **36 481** | **4.607** | **8.361**         | **Julia 1.8× faster** |

**Key result:** Julia is competitive across all Pk and N. At small N
Julia is faster (less per-call overhead). At medium N (3K–10K DOFs)
MATLAB pulls ahead by 30–70% (MA57's BK pivoting wins). At the
**largest P3 case (36 481 DOFs), Julia OVERTAKES MATLAB by 1.8×** —
LDLFactorizations.jl's AMD-ordered fill scales more gracefully on
the dense P3 connectivity than MA57.

**Bound widths:** MATLAB consistently ~3–4× tighter (consistent with
the v14 `:fast` Wilkinson bound vs `:tight` directed-rounded model
documented in §4.5.1). Same trade-off, same gap; the user opted for
speed via :fast at v15.

Useful additional finding: **higher-order elements are more efficient
per ndof.** At ndof=3969:
- P1 (Ferrite mesh): 0.256 s, eigenvalue rel-err 6×10⁻⁴
- P2: 0.180 s, eigenvalue rel-err 5×10⁻⁶ (~100× tighter for ~30%
  less wall time)

Ferrite is now a runtime dep (the user accepted; main reason: tests
that exercise higher-order matrices need it available). Veigs.jl
doesn't link Ferrite from its main code path — only test/support and
bench files do.

Tests: `test_laplacian2d_pk.jl` adds 6 testsets covering DOF count,
symmetry, PD-ness, O(h^{2k}) convergence rate, veigs end-to-end on
P1/P2/P3, and invalid-arg dispatch (57 tests, ~90 s of test time).
Wired into `runtests.jl`. Full suite: 70244 passing, 1 broken.

### v19 — Dumbbell domain: clustered eigenvalues + cluster-handling test

The dumbbell — two unit squares connected by a thin bar of width
w=0.2 — is the canonical test for *clustered* (near-degenerate)
Dirichlet eigenvalues. As w → 0 the lobes decouple and the spectrum
collapses to a single unit square's with multiplicity 2. At finite w,
each degenerate eigenvalue splits into a close pair via tunneling
through the bar. This is exactly what `veigs`'s cluster expansion
was designed to handle correctly.

Added `laplacian2d_dumbbell(h, k; w=0.2)` — a Ferrite-based assembler
on a manually-built grid (Ferrite's `generate_grid` only handles
rectangles). The first cluster (λ_1, λ_2) sits near 2π² ≈ 19.74; the
second cluster (λ_3..λ_6) sits near 5π² ≈ 49.35 with multiplicity 4
(the (2,1)/(1,2) modes doubled by the two lobes).

**Veigs cluster behaviour, probed at h=0.025 P1 (ndof=3609):**
- k=1 → returns 1 eigenvalue with bound width 1.2e-5 (excluding λ_2,
  not auto-grown to the cluster).
- k=2 → returns cluster 1 with bound widths 2.4e-11; gap 2.05e-4
  rigorously enclosed.
- k=3, 4, 5 → throws `VeigsClusterTooLargeError`. Refusing to
  half-count cluster 2 is the **correct** soundness behaviour.
- k=6 → returns clusters 1 + 2 (6 eigenvalues), all rigorously
  separated, max bound width 1.3e-10.
- k=7 → throws `VeigsClusterTooLargeError` again (cluster 3 boundary).

**MATLAB+INTLAB cross-bench** (same Ferrite-assembled matrices loaded
via .mtx, both call `veigs(K, M, 2, 'sm')`):

| h     | order | ndof  | Julia (s) | MATLAB (s) | Julia / MATLAB        |
|-------|-------|-------|-----------|------------|-----------------------|
| 0.1   | P1    | 183   | 0.007     | 0.062      | **Julia 8.9× faster** |
| 0.05  | P1    | 845   | 0.032     | 0.064      | **Julia 2.0× faster** |
| 0.025 | P1    | 3 609 | 0.320     | 0.224      | Julia 1.4× slower     |
| 0.1   | P2    | 845   | 0.029     | 0.069      | **Julia 2.4× faster** |
| 0.05  | P2    | 3 609 | 0.286     | 0.163      | Julia 1.8× slower     |
| 0.1   | P3    | 1 987 | 0.078     | 0.076      | dead heat             |

Both implementations enclose λ_1, λ_2 to identical displayed precision.
Both correctly handle the cluster (gap as small as 3.8e-6 at P2 h=0.05,
bounds tight enough to separate). Julia faster at small/medium ndof;
MATLAB pulls ahead at larger ndof for these specific dumbbell shapes.

The most demanding case in the bench: P2 h=0.05, λ_2 − λ_1 = 3.83e-6,
bound width = 1.55e-10 — verifying a six-significant-figure spectral
gap. This is real verification of physics: e.g., a vibrational mode
splitting at the 10⁻⁶ level is exactly the scale that matters in
quantum tunneling and weak-coupling problems.

Tests: `test_laplacian2d_dumbbell.jl` adds 7 testsets / 40 tests
covering mesh structure, PD-ness, eigen() agreement, cluster
separation invariants, cluster-too-large refusal, P1/P2 consistency.
Full suite: 70284 passing, 1 broken.

## Files touched in this discussion arc

- `src/verified_isspd.jl` — added `_verified_isspd_ldl_shift`,
  `_verified_isspd_rump2006`, `DEFAULT_ISSPD_METHOD::Ref{Symbol}`.
- `src/verified_ldl.jl` — float gemm + per-entry Wilkinson bound.
- `src/Veigs.jl` — exported `DEFAULT_ISSPD_METHOD`; tight-loop
  `sym_hull` with `assume_symmetric` kwarg and in-place `sym_hull!`.
- `src/rough_bounds.jl` — pass `assume_symmetric=true` when building
  the test matrix from already-symmetric A and B.
- `src/veigs_driver.jl` — KrylovKit shift-invert path in
  `_approx_eig`, with dense fallback for small / unsupported cases.
- `Project.toml` — added KrylovKit (0.10) as a dependency.
- `report.html` — §4.4 algorithmic-improvement tables; Phase-9
  correction notes in §4.4.2 / §4.4.3 / §4.4.4 + new §4.4.2.1 on the
  GetInertia patch.
- `DECISIONS.md` — perf entries 2026-05-01.
- `discussion_log.md` — this file.

**Phase 11 (2026-05-02) — kernel refactor v9 + v10:**
- `src/verified_isspd.jl` (v9) — added `_verified_isspd_rump2006_midrad`
  helper accepting `Mr ∈ {Nothing, AbstractMatrix, Tuple{Float64,Float64}}`.
  Public `_verified_isspd_rump2006` skips `interval.(M)` for float input.
- `src/rough_bounds.jl` (v9) — replaced full interval residual
  `M = Ai − λi·Bi` with float `M_mid = A_mid − λ·B_mid` + scalar p-norm
  radius bounds.
- `src/Veigs.jl` (v9) — added `_midrad_floats(M)` helper returning
  `(mid::AbstractMatrix{Float64}, rad::Union{AbstractMatrix{Float64}, Nothing})`,
  sparsity-preserving.
- `src/verified_ldl.jl` (v10) — `_verified_ldl_tridiag` now returns
  `ΔM` as a sparse pentadiagonal `SparseMatrixCSC{Interval{Float64}}`
  with ≤5n nonzeros, instead of densifying to n×n
  (~400 MB at n=5000). Drove the n=5000 wall time from 0.47 s to
  0.22 s (53% improvement, biggest single step since Phase 8).

**Phase 10 (2026-05-02) — design document:**
- `DESIGN.md` (new) — captures the package's contract: scope as a
  verified post-processing kernel, the five primitives, the
  Float64-default / BigFloat-on-demand precision policy for the
  Lehmann–Goerisch inner step, soundness asymmetry, and the explicit
  list of things the package deliberately doesn't do (no sparse
  interval kernel, no interval LDL, no silent precision shifts).

**Phase 9 (2026-05-02):**
- `veigs/GetInertia.m` — patched to extract three diagonals once.
- `veigs/GetInertia_orig.m` — original kept verbatim for A/B (function
  symbol renamed).
- `veigs/bench_inertia_apples.m` — repeatable MATLAB harness (3
  blocks: full veigs / patched per-call / baseline per-call).
- `veigs/probe_inertia_cost.m` — diagnostic that revealed
  `intval(sparse)` preserves sparsity in INTLAB.
- `veigs/profile_veigs.m` — MATLAB profile harness for n=5000.
- `Veigs.jl/bench/bench_laplacian_smallest.jl` — Julia harness on the
  same problem, BenchmarkTools 5 samples + trim-mean.
- `laplacian_n5000.mat` (in `veigs4Julia/`) — sparse 1D Laplacian (n=5000)
  + `B = speye(5000)` exported for cross-tool reuse.
