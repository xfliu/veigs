# agent.md — Veigs.jl structure

A 2-minute orientation for a fresh agent (or human) dropping into this package. For *why* decisions were made, read `DECISIONS.md`. For the auto-loop / migration plan, read `guide.md`. This file just describes *what is here*.

## Purpose

Verified eigenvalue solver for the symmetric generalized eigenproblem `A x = λ B x`. Julia port of the MATLAB `veigs` toolbox (sibling tree at `../veigs/`). Replaces INTLAB with `IntervalArithmetic.jl` + a hand-rolled verified `isspd`.

## Top-level layout

```
Veigs.jl/
├── Project.toml          # package metadata + [deps] + [compat]
├── Manifest.toml         # generated; pinned dep versions
├── setup.jl              # one-time bootstrap (`julia --project=. setup.jl`)
├── DECISIONS.md          # decision diary (read before changing conventions)
├── guide.md              # migration guide / auto-loop instructions
├── agent.md              # this file
├── issue_on_publishing.html  # Q-001 / Q-002 detail (LICENSE + registry decisions)
├── src/                  # one source file per algorithmic routine
├── test/                 # one test file per routine + shared support/
└── bench/                # one bench file per routine, excluded from Pkg.test
```

No `LICENSE` file yet — see `issue_on_publishing.html`.

## Source files (`src/`)

| File | Public name(s) | Role |
|---|---|---|
| `Veigs.jl` | (module) | Module root: `using` imports, `include`s, error hierarchy (`VeigsError` and 5 subtypes), `sym_hull` helper, `_check_finite_midpoint` helper, `export` list. |
| `inertia.jl` | `inertia` | Count signed inertia (neg, pos, zero, F-flag) of a block-diagonal symmetric matrix made of 1×1 and 2×2 blocks — the `D` factor of Bunch-Kaufman. |
| `verified_isspd.jl` | `verified_isspd` | Rump-style verified positive-definiteness test for interval matrices. Replaces INTLAB's `isspd`. |
| `verified_ldl.jl` | `verified_ldl` | Wrapper around `bunchkaufman(Symmetric(mid.(M), :L))` returning `(L, D, p, ΔM, ok)` where `ΔM` is the verified interval residual `M - P L D Lᵀ Pᵀ`. |
| `rough_bounds.jl` | `rough_lower`, `rough_upper` | Exponential-backoff search for a `λ` with `A − λB` (or `λB − A`) verified PD. Used to seed the cluster expansion with crude spectrum bounds. |
| `veig.jl` | `veig` | Yamamoto-style verified solver for small dense `(A, B)` (n ≤ 100). Returns interval bounds for each requested eigenvalue index. Workhorse for the Lehmann subproblem. |
| `lehmann_behnke.jl` | `lehmann_behnke` | Lehmann-Behnke complementary variational principle: tightens cluster bounds for `(λ_r, …, λ_s)` given separators ρ, σ. Calls `veig` for cluster cases, scalar ratio for `s == r`. |
| `veigs_driver.jl` | `veigs` | Top-level cluster-aware driver. Multiple dispatch on `(A, B[, k][, sigma])`. Cluster expansion loop. Falls back to Rayleigh quotients at spectrum extremes when LDL separation fails. **File is named `veigs_driver.jl` rather than `veigs.jl` to avoid case-insensitive collision with `Veigs.jl` on macOS APFS — see D-018.** |

## Module dependency DAG

```
inertia              ── no in-package deps
verified_isspd       ── no in-package deps  (uses LinearAlgebra.cholesky internally)
verified_ldl         ── no in-package deps  (uses LinearAlgebra.bunchkaufman)
        │
        ├─→ rough_bounds        ── verified_isspd, sym_hull
        │
        ├─→ veig                ── verified_ldl, inertia, verified_isspd, sym_hull
        │           │
        │           └─→ lehmann_behnke   ── veig (cluster case), sym_hull, verified_isspd
        │
        └─→ veigs_driver        ── all of the above
```

`include` order in `src/Veigs.jl` matches this DAG (dependencies first). Don't reorder without checking.

## Test files (`test/`)

| File | Tests |
|---|---|
| `runtests.jl` | Top-level entry. Just `include`s the per-routine files in DAG order. |
| `support/matrix_factory.jl` | Shared random-matrix generators: `spd_random`, `indef_random`, `near_singular_spd`, `barely_indef`, `hilbert`, `banded_test`, `interval_perturb`. Imported by every `test_*.jl`. |
| `test_inertia.jl` | 14 corner cases (one `@test_skip` for case 9 — branch is structurally unreachable; see comment in file). |
| `test_isspd.jl` | 11 batteries incl. adversarial soundness (case 5: `λ_min = -ε` MUST return false). |
| `test_verified_ldl.jl` | 12 cases incl. 50-matrix reconstruction regression for both `uplo=:L` and `:U`. |
| `test_rough_bounds.jl` | 11 cases. |
| `test_veig.jl` | 11 cases incl. README example 1 containment. |
| `test_lehmann_behnke.jl` | 10 cases incl. cluster-of-2 / cluster-of-5 (which depend on `veig`). |
| `test_veigs.jl` | 16 cases across plan §7.7's 5 sub-batteries (dispatch, numerical correctness, soundness, failure modes, performance). Uses ULP-tolerant `_enclosed` helper to compare against LAPACK's `eigvals` reference. |

Every test file starts with `Random.seed!(20260501)` (D-011). Imports use `import Veigs` + explicit `using Veigs: name1, name2` to avoid the Julia 1.12 `LinearAlgebra.inertia` clash.

Current totals: **723 passing, 1 broken-skip, 0 failing.**

## Bench files (`bench/`)

`bench_<routine>.jl`, one per source routine. Each is independently runnable via `julia --project=. bench/bench_<routine>.jl`. They use `BenchmarkTools.@benchmark` and print medians at a few sizes. Not included in `Pkg.test()` (D-010).

## Public API surface

Exported from `src/Veigs.jl`:

```
veig(A, B[, ind])                          → (Vector{Interval}, UnitRange)
veigs(A, B[, k][, sigma])                  → (Vector{Interval}, UnitRange)
inertia(D; abort_on_err=false)             → (neg, pos, zer, F)
verified_isspd(M; method=:rump)            → Bool
verified_ldl(M; uplo=:L)                   → (L, D, p, ΔM, ok)
rough_lower(A, B, λ_seed, λ_floor)         → Float64
rough_upper(A, B, λ_seed, λ_ceiling)       → Float64
lehmann_behnke(A, B, eig_list, V, ρ, σ,
               λ_B_min, r, s; do_shift)    → Vector{Interval}
sym_hull(M)                                → Matrix{Interval}
```

Plus error types: `VeigsError` (abstract) and `VeigsLDLFailureError`, `VeigsClusterTooLargeError`, `VeigsLehmannBehnkeError`, `VeigsRoughBoundError`, `VeigsSizeError`.

## Key conventions (one-liners; full list in DECISIONS.md)

- D-004: import `inf, sup` from IntervalArithmetic 1.0.8 (NOT `infimum`/`supremum`).
- D-005: always `bunchkaufman(Symmetric(M, :L))` to match MATLAB's `[L,D,P]=ldl(M)` lower-triangular convention.
- D-008: throw typed `VeigsError` subtypes; every concrete type has at least one test that triggers it.
- D-011: every `test_*.jl` starts with `Random.seed!(20260501)`.
- D-013: use `sym_hull(M)` wherever MATLAB does `hull(M, M')`.
- D-014: `verified_isspd` returns `false` on midpoint Cholesky failure (does not throw); only `Inf`/`NaN` triggers `DomainError`.
- D-016, D-017, D-019: see DECISIONS.md.

## Run commands

Julia binary: `/Volumes/DropboxDisk/AppDisk/Julia/bin/julia`. Project root: this directory.

```bash
JULIA=/Volumes/DropboxDisk/AppDisk/Julia/bin/julia

$JULIA --project=. setup.jl                        # one-time bootstrap
$JULIA --project=. -e 'using Pkg; Pkg.test()'      # full suite
$JULIA --project=. test/test_<routine>.jl          # single test file
$JULIA --project=. bench/bench_<routine>.jl        # single benchmark
```

## What's not yet here (iter 8, optional)

- `docs/` skeleton with Documenter.jl
- `.github/workflows/ci.yml`
- `src/verified_isspd_via_solve.jl` — alternate `verified_isspd` path via `IntervalLinearAlgebra.solve` (cross-check)
- Sparse / Arpack path for `veigs`'s approximate eigensolve (currently dense-only via `LinearAlgebra.eigen`)
- `bench/baselines/julia-1.10.json` — committed performance baseline
- `LICENSE` file — see `issue_on_publishing.html` for the recommendation

These are polish/infrastructure work, not core algorithms.
