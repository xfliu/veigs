# Design principles for `Veigs.jl`

Last revision: 2026-05-02.

This document captures the design principles that govern what
`Veigs.jl` is — and isn't — and the rules each algorithmic primitive
must follow. These are not implementation notes; they are the
package's contract. When in doubt about whether a change belongs in
the package, check it against this document.

---

## 1. Scope: verified post-processing, not interval linear algebra

`Veigs.jl` is a **verified post-processing kernel** for symmetric
generalized eigenvalue problems. We do *not* try to be a
general-purpose interval-matrix solver:

- Heavy lifting (Cholesky, LDL, eigensolves, sparse factorisation) is
  delegated to mature **non-interval** routines: SuiteSparse,
  LAPACK/BLAS, KrylovKit, Arpack.
- Interval arithmetic is used **only** for the *certificate* — i.e.,
  the rigorous bound on the answer's distance from truth.

Why this scope:

- `IntervalArithmetic.jl` deliberately does not provide a
  sparse-pattern-aware interval matrix type, and its maintainers have
  no plans to (see issue
  [JuliaIntervals/IntervalArithmetic.jl#628](https://github.com/JuliaIntervals/IntervalArithmetic.jl/issues/628)
  — a fix landed for `fill!` correctness, but no specialised sparse
  kernel was added). Building one ourselves would mean re-inventing
  ~25 years of INTLAB's hand-tuned work.
- Traditional linear-algebra libraries are *much* faster than any
  generic-Julia interval matrix ops we could write, and they're
  already optimised for sparsity, BLAS, threading, and modern
  hardware.
- A post-processing certificate is short — typically a few lines of
  scalar interval arithmetic plus a Wilkinson-style residual bound —
  and that's the only place soundness lives. Move the certificate to
  the boundary; let everything else stay in float.

**Practical rule.** A routine in `Veigs.jl` should never construct an
interval matrix of dimension > 30 unless it represents the *result*
(a returned bound). The `m × m` cluster matrices A0, A1, A2 in
Lehmann–Goerisch are the only working interval matrices the package
needs.

---

## 2. The five-primitive kernel

Every public verified result in `Veigs.jl` is built from these
primitives. New routines must be expressible as a composition of
these; if not, we widen the kernel deliberately and update this
document.

| # | Primitive | Inputs (all float / sparse OK) | Output | What's verified |
|---|---|---|---|---|
| 1 | `verified_residual_norm(A, X, λ, B)` | float A (n×n), float X (n×k), float λ, float B (n×n) | `Interval{Float64}` bounding ‖A·X − λ·B·X‖₂ | Rounding error of one float `gemm`, bounded by Wilkinson γ_n + interval-radius contribution from any input ranges |
| 2 | `verified_isspd(M; method=:rump2006)` | float-or-(mid, radius) M (n×n) | `Bool` (`true` ⇒ provably SPD) | Rump 2006 single-Cholesky; soundness asymmetric (allowed to return `false` on borderline-PD inputs) |
| 3 | `verified_inertia(D)` | block-diagonal D from any LDL factorisation (1×1 + 2×2 blocks) | `(neg, pos, zero, F::Bool)` | Sign of each block's determinant / trace under interval arithmetic; `F == false` ⇒ counts certified |
| 4 | `verified_eig_inclusion(A, B, X̃, λ̃, lower_bd; precision)` | A, B (n×n), X̃ (n×m), λ̃ (m), lower_bd (m+1) | `Vector{Interval}` of m verified bounds | Lehmann–Goerisch on the m × m cluster matrices |
| 5 | `verified_rough_bound(A, B, λ_trial)` | A, B (n×n), λ_trial | `Bool` (true ⇒ all eigenvalues > λ_trial, given direction) | Inertia of `A − λ_trial·B` via primitives (3) + (2) |

All primitive signatures are designed so the n-sized inputs can stay
in their natural representation (float `Matrix`, `SparseMatrixCSC`,
or interval `Matrix`/`SparseMatrixCSC` if the caller already has
one). The implementation of each primitive must dispatch
appropriately and never widen sparse → dense unless the underlying
factorisation forces it.

---

## 3. Float64 is the default; precision flows into the inner step

Most of the work — including all O(n) and O(n²) operations — runs in
`Float64` with rigorous Wilkinson bounds. The arithmetic is correct
to within ~γ_n·‖input‖, and that bound is cheap to compute as a
single scalar interval.

The **inner step of Lehmann–Goerisch** is the exception. The width
of the final eigenvalue bound is gated by the accuracy of the inner
small EVP `veig(SA, SB)` on the cluster matrices A0, A1, A2.
Concretely:

- The final bound width has a floor of ~ULP × |λ| × κ, where κ is
  the conditioning of the cluster matrices. For well-conditioned
  problems this is small enough that `Float64` is more than enough.
- For tightly clustered eigenvalues (relative gap < 1e-10), or for
  problems where the user specifically wants bounds tighter than
  `Float64` allows, the inner step's `Float64` arithmetic becomes the
  bottleneck. *No amount* of outer-loop tightening will produce a
  tighter final bound if `veig` itself can't resolve the cluster.

**Policy.** `verified_eig_inclusion` accepts a `precision` keyword:

```julia
verified_eig_inclusion(A, B, X, λ, lower_bd;
                       precision = Float64)         # default
verified_eig_inclusion(...; precision = DoubleFloat)
verified_eig_inclusion(...; precision = BigFloat)
```

The keyword controls the eltype of A0, A1, A2 and of the small EVP
inside `veig`; outer products (`B·V`, `A·V`, etc.) stay in `Float64`
because their rounding is rigorously bounded — promoting them to
`BigFloat` would not tighten anything and would cost real wall-time
at large n.

The default `Float64` is correct for ~99 % of practical problems.
The `BigFloat` knob is for users who genuinely need bounds below the
`Float64` ULP floor (e.g., quasi-degenerate eigenvalues in PDE
discretisations, Schrödinger eigenproblems with near-crossings,
ill-conditioned generalised problems).

The package will *not* auto-promote silently. Users get whatever
precision they ask for; if `Float64` produces a bound that's wider
than they wanted, they're expected to retry at higher precision
themselves. Silent promotion would mask cases where the underlying
problem is genuinely ill-posed at the requested precision.

> **Naming note.** The literature attributes the inclusion theorem
> to Lehmann (1949), with substantial generalisations by Goerisch
> (1980s) and Behnke (1991, *Computing*). The Julia file is
> `src/lehmann_behnke.jl` to match the MATLAB reference toolbox; the
> mathematical method itself is "Lehmann–Goerisch" in modern usage.

---

## 4. Soundness asymmetry

Every primitive in §2 has the property:

- `true` / a returned interval ⇒ **mathematically certified**.
- `false` / a thrown error ⇒ **the routine could not certify**, *not*
  that the property fails to hold.

This asymmetry is load-bearing. A routine that returns `false` on a
borderline-PD input is correct; a routine that returns `true` on a
non-PD input is a soundness bug and must be fixed before any other
work continues.

Consequence: tests cover both the "must succeed on this true input"
path and the "must fail closed on this constructed counter-example"
path. See the corner-case taxonomy at the top of every
`test_*.jl` file.

---

## 5. What we deliberately don't do

This list exists so reviewers can quickly recognise things that
*don't* belong in the package without re-litigating each one.

- **No sparse-interval matrix kernel.** We don't write a
  `SparseIntervalMatrix(inf, sup)` type. If we hit a use case that
  wants one, the answer is to re-express the use case as a
  certificate over float operations (§1, §2).
- **No interval LDL or interval Cholesky on n×n inputs.** These were
  in the v1 / v2 code paths and removed — all factorisations run in
  float, certified by post-processing. (Rump 2006, Yamamoto 2024 et
  al. all agree: you don't need interval LDL.)
- **No reimplementation of mature solvers.** We don't ship our own
  LU, QR, eigensolver, Krylov method. We wrap mature implementations
  and certify their output.
- **No silent precision changes.** Every precision shift is an
  explicit kwarg. See §3.
- **No "best-effort" or "approximate" bounds.** Every public function
  either returns a bound that's mathematically rigorous or it
  signals failure. There is no middle ground. ("A fast wrong answer
  has zero value" — `guide.md` §11.)
- **No global state for arithmetic mode.** `IntervalArithmetic.jl`'s
  configuration is set once at module load (D-004); we don't toggle
  rounding modes mid-computation. INTLAB does this for performance,
  but it makes thread safety hard and bug surface area large.

---

## 6. When this document needs updating

If you find yourself doing any of the following, stop and update
this document first (then code):

- Adding a 6th primitive to §2 (the kernel widens).
- Constructing an interval matrix of dimension > 30 outside
  `lehmann_behnke.jl`'s working set.
- Adding a code path that depends on a sparse interval kernel.
- Promoting precision automatically anywhere.
- Returning a bound that the routine can't prove is rigorous.

Each of these is a deliberate departure from §1–§5 and deserves a
written rationale before code review.

---

## 7. Pointers

- `guide.md` — operational guide for working in this repo.
- `DECISIONS.md` — running diary of design choices (D-001 …).
- `discussion_log.md` — phase-by-phase optimisation history (Phase 1
  through Phase 9b).
- `report.html` — current performance and bound-quality state.
- `src/inertia.jl`, `src/verified_isspd.jl`, `src/verified_ldl.jl`
  — the three primitives that already align cleanly with this
  design.

If a future routine looks like it can't be expressed inside §2,
that's the signal to widen the kernel — but only after writing why
in this document.
