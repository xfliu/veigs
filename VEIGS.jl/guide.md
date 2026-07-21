# Veigs.jl Migration Guide — for Claude Code

You are continuing an in-progress port of a verified-eigenvalue MATLAB toolbox to Julia. This guide is the single entry point. Read it through once, then follow the auto-loop in §3.

---

## 0. Mission in one paragraph

Port the MATLAB `veigs` toolbox (rigorous bounds for the symmetric generalized eigenproblem `Ax = λBx`) to a stand-alone Julia package `Veigs.jl`. The package replaces the proprietary INTLAB dependency with `IntervalArithmetic.jl` + a hand-rolled verified `isspd` test. The port is partially complete: package scaffolding and 2 of 7 routines are done. Your job is to drive the remaining iterations to completion under a strict per-routine test discipline.

---

## 1. Where to find context (read in this order)

| File | Why read it |
|---|---|
| `/Volumes/DropboxDisk/Dropbox/Document/ReviewPaper/2026_03_SheYuanfeng/veigs_julia_port_plan.md` | The strategic plan. §3 has the INTLAB→Julia primitive map. §4 has the four hard parts. **§7 has the per-routine corner-case taxonomy you must implement** — this is the spec for every test file. |
| `./DECISIONS.md` (sibling of this file) | Running diary. The status table at the top tells you which iter is next. The "Standing decisions" list (D-001…D-015) are the conventions every new file must follow. |
| `../veigs/veigs.m`, `../veigs/veig.m`, `../veigs/GetInertia.m` | The MATLAB source. Read the routine you are about to port BEFORE you implement it. |
| `./src/Veigs.jl`, `./src/inertia.jl`, `./src/verified_isspd.jl` | The two completed routines. Use them as the style template — same docstring shape, same error-type usage, same testing pattern. |
| `./test/test_inertia.jl`, `./test/test_isspd.jl` | The two completed test files. The corner-case-taxonomy comment block at the top is mandatory for every new test file. |

The MATLAB README (`../veigs/README.md`) has the two worked examples (Hilbert n=8, banded n=10). Their outputs are your golden regression target.

---

## 2. The non-negotiable project rule

> **Every routine — public or internal, helper or core — has its own test file. No iter is "done" until its test file is in place AND passes locally.** The test file must cover (a) typical correct inputs, (b) corner cases enumerated **in code at the top of the file** as a comment block, (c) the rigorous mathematical contract (e.g., "returns `true` ⇒ matrix is PD"), and (d) at least one efficiency benchmark with a regression threshold.

This rule was set by the user explicitly. Do not skip it for any routine, even "trivial" ones.

Operational consequences:
1. Test contract goes in code FIRST (as the comment block), then implementation, then iterate until green.
2. One test file per routine, lumped under `test/runtests.jl` via `include`.
3. Soundness contracts get adversarial tests. The classic example is `verified_isspd` — there is a test that constructs matrices with `λ_min = -ε` and asserts the routine returns `false`. Repeat this pattern for any soundness-critical routine.
4. Every typed exception gets a test that triggers it.
5. Performance budgets live in tests as `@test t < threshold` — they exist to flag regressions, not to be exact.

---

## 3. The auto-loop

Repeat for each ⚪ row in `DECISIONS.md`'s loop status table, in order:

```
1.  TaskWrite "iter <N>: <routine>" and mark in_progress
2.  Read MATLAB source for the routine (../veigs/<file>.m)
3.  Read plan §7.<N+0> for the corner-case taxonomy
4.  Read DECISIONS.md standing decisions for any clauses that constrain this routine
5.  Write src/<routine>.jl (follow the inertia.jl docstring/error-type style)
6.  Write test/test_<routine>.jl with the corner-case taxonomy as a leading comment block
7.  Write bench/bench_<routine>.jl (small file, 5-10 sizes, BenchmarkTools)
8.  Add `include("<routine>.jl")` to src/Veigs.jl
9.  Add `include("test_<routine>.jl")` to test/runtests.jl
10. Run: julia --project=. -e 'using Pkg; Pkg.test()'
11. Iterate steps 5-10 until tests are green (THIS is the inner TDD loop)
12. Update DECISIONS.md:
    - Mark the iter row 🟢
    - Add a "Iter <N> — <routine>" entry to the per-iteration journal
    - Append any new [PENDING-CONFIRM] decisions you made
13. git add -A && git commit -m "iter <N>: <routine> green"
14. Move to next iter (back to step 1)
```

If you get stuck in the inner TDD loop and have tried the obvious fixes, **stop and document the blocker in DECISIONS.md under "Open questions for you"** rather than thrashing. The user reviews open questions out-of-band.

---

## 4. Remaining iteration queue with per-routine notes

### Iter 3 — `verified_ldl`
- **MATLAB source:** the LDL-related code is inline in `veigs.m` (lines ~167-185 and ~322-326 and ~400-414) and `veig.m` (lines ~119-120, ~151-153, ~169-170, ~205-206). There is no standalone MATLAB file; you are factoring out a reusable wrapper.
- **What to write:** `src/verified_ldl.jl` exposing
  ```julia
  verified_ldl(M; uplo=:L) -> (L, D, p, ΔM, ok)
  ```
  where `M` is a real or interval matrix, `(L, D, p)` come from `bunchkaufman(Symmetric(mid.(M), uplo))`, `ΔM = M - PLDLᵀPᵀ` is an interval matrix, and `ok::Bool` indicates factorization success. Return `ok=false` (don't throw) if the midpoint factorization fails — the cluster-expansion logic relies on this fall-through.
- **Corner-case taxonomy** (plan §7.3): SPD inputs, NPD inputs, indefinite forced-2×2 block (`[0 1; 1 0]`), upper- vs. lower-triangle reconstruction regression (50 random matrices, assert `‖M[p,p] - L*D*L'‖ < 1e-10`), singular `M`, permutation correctness, extreme conditioning, sparse `M`.
- **Watch out for:** D-005 forces `uplo=:L` to match MATLAB. The 50-matrix reconstruction test is the silent-bug guard from plan §4.2; do not weaken it.

### Iter 4 — `rough_bounds`
- **MATLAB source:** `veigs_RoughLower` and `veigs_RoughUpper` subfunctions in `veigs.m` (lines ~518-570).
- **What to write:** `src/rough_bounds.jl` with two functions:
  ```julia
  rough_lower(A, B, λ_seed, λ_floor) -> Float64    # rough lower bound for λ_min
  rough_upper(A, B, λ_seed, λ_ceiling) -> Float64  # rough upper bound for λ_max
  ```
  Both iterate exponential backoff (powers of 2) from the seed, calling `verified_isspd` each step. Throw `VeigsRoughBoundError` if no valid bound found before the floor/ceiling.
- **Corner-case taxonomy** (plan §7.4): seed already valid, seed fails / floor succeeds, floor fails (must throw), pathological `λ_seed = 0`, barely-PD `B`, identity-shaped problems (verify tightness), iter-count cap.
- **Watch out for:** the MATLAB code uses `eig_test - delta*2^(k-1)` with `delta = (eig_test - eig_test_min)/2^32`. Reproduce exactly; the schedule is delicate near the spectrum extremes.

### Iter 5 — `lehmann_behnke`
- **MATLAB source:** `Lehmann_Behnke__` subfunction in `veigs.m` (lines ~575-635).
- **What to write:** `src/lehmann_behnke.jl` with
  ```julia
  lehmann_behnke(A, B, V, ρ, σ, λ_B_min, r, s; do_shift=true) -> Interval
  ```
  Returns the verified enclosure of the cluster `(λ_r, …, λ_s)` using complementary variational principles. Calls `veig` (forward declaration — implement an `error("not yet implemented")` stub if iter 6 not done; or implement them in order, which is what the plan assumes).
- **Corner-case taxonomy** (plan §7.5): scalar case `s == r`, cluster of 2, cluster of 5, `do_shift=0` vs `=1`, `SB` barely PD, `SB` not PD (must throw `VeigsLehmannBehnkeError`), orthogonal vs non-orthogonal `V`, interval `A` with non-zero radius.
- **Watch out for:**
  - The shift step `A := A − λ_shift · B` and the matching `ρ`/`σ` adjustments. Easy to off-by-one.
  - `nV = mid(B) \ (mid(A) * mid(V))` is a *float* solve used as a preconditioner; the verified residual `Err = (err_vec' * err_vec) / c` corrects for the rough inverse.
  - `A2 = V' * A * intval(nV) - intval(nV)' * (B * intval(nV) - A * V) + Err` — the order of operations matters; reproduce the MATLAB expression exactly.

### Iter 6 — `veig`
- **MATLAB source:** `veig.m` (full file, ~250 lines).
- **What to write:** `src/veig.jl` implementing the Yamamoto-style verified solver for small dense generalized problems.
- **Corner-case taxonomy** (plan §7.6): diagonal known-spectrum, Hilbert + I (README example #1 segment), banded (README example #2), `n=1`, `n=100` (boundary), `n=101` (must throw `VeigsSizeError`), single index vs full range, asymmetric input (must throw), `B` not PD (must throw or fail gracefully), tightly clustered (gap ≤ 1e-8), sparse path.
- **Watch out for:**
  - The cap `n > 100` is hard-coded in the MATLAB; D-Q-004 says keep it at 100 for parity unless the user overrides.
  - The verified-bounds loops (lines ~166-235 in `veig.m`) double `err_est` until `isspd(err·B − ΔM)` AND `isspd(err·B + ΔM)` both pass. Both checks are required; do not drop one. Cap iterations at 53 (matches MATLAB).
  - Index range `(neg_num_low+1):neg_zero_num_upper` is the cluster the bound covers; may extend beyond the requested `ind`.

### Iter 7 — `veigs` (the driver)
- **MATLAB source:** `veigs.m` outer function + the three nested subfunctions.
- **What to write:** `src/veigs.jl` with:
  - Four method dispatches (D-002): `veigs(A, B)`, `veigs(A, B, k)`, `veigs(A, B, sigma)`, `veigs(A, B, k, sigma)`.
  - Symbol/string normalization (`:lm` → `:largestabs`, etc.).
  - The cluster state machine (use a `mutable struct ClusterState` per plan §4.4 — DO NOT translate the `para` vector literally).
  - Calls to `verified_ldl`, `verified_isspd`, `rough_bounds`, `lehmann_behnke`, `veig`.
  - Fall-back Rayleigh-quotient bounds at spectrum extremes when LDL separation fails.
- **Corner-case taxonomy** (plan §7.7): broken into 5 sub-batteries — dispatch (§7.7.1, 7 cases), numerical correctness incl. README examples (§7.7.2, 7 cases), soundness (§7.7.3, 3 properties), failure modes (§7.7.4, one per typed exception), performance budgets (§7.7.5, 3 envelopes).
- **Watch out for:**
  - The cluster expansion's `while do_next` loop has subtle termination conditions involving `direction_left`/`direction_right` flags that get reset by the body. Translate one branch at a time and cover each with a focused test BEFORE moving to the next.
  - `eigs` may diverge differently from MATLAB; `lambda_ratio = 1e-5` absorbs typical differences. If a test fails because Arpack returns slightly different approximate eigenvalues, do NOT loosen the soundness assertions — soundness must hold regardless of starting data.
  - `EigNum = min(size(A,1), NumOfEigs+2)` (or `min(10, n)` if `NumOfEigs == 1`). Off-by-one here is a documented historical bug source.

### Iter 8 — docs + CI yaml + alternate `verified_isspd_via_solve`
- Create `docs/` skeleton with Documenter.jl (API reference, the two README examples, test philosophy, benchmark page auto-generated from `bench/baselines/*.json`).
- Create `.github/workflows/ci.yml` (Julia 1.10 LTS + current; Linux + macOS).
- Implement `src/verified_isspd_via_solve.jl` (alternate path using `IntervalLinearAlgebra.solve`, gated behind `method=:solve` kwarg) ONLY IF `IntervalLinearAlgebra.jl` installs cleanly on the user's IntervalArithmetic version. Add the cross-check test deferred from `test_isspd.jl` case 12.
- Final pass: regenerate `bench/baselines/julia-1.10.json` and commit.

---

## 5. Coding conventions (recap of D-001 … D-015)

- **Module layout:** one routine per file under `src/`; `src/Veigs.jl` is just `using` declarations + `include`s + helper functions + exports.
- **Imports:** be explicit. `using IntervalArithmetic: Interval, interval, mid, hull, radius, mag, infimum, supremum`. Don't `using IntervalArithmetic` bare — pollutes the namespace.
- **Errors:** typed exceptions only. Sub-types of `VeigsError` already exist; add new ones in `src/Veigs.jl` if you need a new failure mode, and document the trigger condition in a test.
- **Docstrings:** every exported function gets a docstring. Follow the inertia.jl shape: signature, contract, examples block. Mention soundness explicitly when applicable.
- **No `using IntervalArithmetic` in test files** — use the explicit-name form to mirror src.
- **Random tests use `Random.seed!(20260501)` at file top** (D-011) — failures must be reproducible.
- **Helpers stay in `src/Veigs.jl`** unless they grow beyond ~40 lines, then split out.
- **Stdlib deps must be in `Project.toml` `[deps]`** (LinearAlgebra, SparseArrays, Random, Test). They look optional but `using` will fail without them.

---

## 6. Common pitfalls

### IntervalArithmetic.jl ≥ 1.0 quirks
- `inf` and `sup` were removed; use `infimum` / `supremum`.
- `Interval(a, b)` constructor is discouraged; use `interval(a, b)` (lowercase factory).
- Some operations on degenerate intervals (e.g., `interval(NaN, NaN)`) return `NaI` (Not-an-Interval). Defend at entry with `_check_finite_midpoint`.
- Decorations are on by default; `interval(...)` returns `Interval{Float64}` which behaves as expected. If you ever see a `BareInterval` type, you've used `bareinterval()` by accident.

### `bunchkaufman` orientation (D-005)
- MATLAB `[L,D,P]=ldl(M)` gives `P L D L' P' = M` with **lower-triangular L**.
- Julia default is **upper-triangular `U D U'`**. Always pass `Symmetric(M, :L)` or use `bunchkaufman(M, :L)` to force lower form.
- The reconstruction sanity test (`‖M[p,p] - L*D*L'‖ < 1e-10` over 50 random matrices) is in `test_verified_ldl.jl` — do not delete it.

### Interval matrix arithmetic performance
- `Matrix{Interval{Float64}} * Matrix{Interval{Float64}}` does NOT call BLAS; it's a generic loop ~100× slower than float matmul.
- For dense `n ≲ 200` this is fine. For larger, migrate hot products to mid-rad form (see plan §4.3). The `Lehmann_Behnke` step's `A2` computation is the prime candidate.
- Only intervalify what must be interval — keep `V` (eigenvector basis) as `Matrix{Float64}` until the moment of forming `V' * intval(A) * V` etc.

### Soundness vs completeness
- A *sound* test never returns `true` for an invalid input. A *complete* test always returns `true` for a valid input.
- Veigs cares about soundness, not completeness. If `verified_isspd` returns `false` on a borderline-PD matrix, that's acceptable. If it returns `true` on an indefinite matrix, that's a critical bug.
- Tests reflect this asymmetry: the adversarial battery is `@test ... == false` (must fail closed); the borderline tests are conditional (`if got @test isposdef(M) end`).

### Don't paraphrase MATLAB expressions
- The Lehmann-Behnke construction `A2 = V'*A*intval(nV) - intval(nV)'*(B*intval(nV) - A*V) + Err` looks symmetric on paper but isn't reordered safely. Translate the MATLAB expression literally, then rely on `sym_hull` if the algorithm requires symmetric output.

---

## 7. How to run things

Julia binary: `/Volumes/DropboxDisk/AppDisk/Julia/bin/julia`

Project root: `/Volumes/DropboxDisk/Dropbox/LIU&YUUKA/01_MATLAB_code/veigs4Julia/Veigs.jl`

| Task | Command |
|---|---|
| One-time setup (or after editing `[deps]`) | `julia --project=. setup.jl` (or `julia --project=. -e 'using Pkg; Pkg.resolve(); Pkg.precompile()'` for a quick fix) |
| Run all tests | `julia --project=. -e 'using Pkg; Pkg.test()'` |
| Run a single test file | `julia --project=. test/test_<routine>.jl` |
| Run a single benchmark | `julia --project=. bench/bench_<routine>.jl` |
| REPL with package loaded | `julia --project=. -e 'using Veigs'` then continue interactively |
| Quick syntax check on a src file | `julia --project=. -e 'include("src/<routine>.jl")'` (won't load the module; use sparingly) |

For a tight TDD loop, `Revise.jl` is your friend — `using Revise; using Veigs; …` lets you re-run tests without restarting Julia. Optional install: `Pkg.add("Revise")` and add to `~/.julia/config/startup.jl`.

---

## 8. End-of-iteration checklist (run before declaring iter green)

- [ ] `julia --project=. -e 'using Pkg; Pkg.test()'` exits with `Test Summary: PASS`.
- [ ] The bench file runs without error (timings recorded for reference, not yet enforced as CI gate).
- [ ] `src/<routine>.jl` has a top-of-file comment block explaining the routine's contract, soundness guarantees, and any error conditions.
- [ ] `test/test_<routine>.jl` has the corner-case taxonomy as a comment block at the top, derived from plan §7.X.
- [ ] All adversarial / soundness-critical tests pass with `==` (not approximate).
- [ ] `src/Veigs.jl` `include` line added, in the correct order (dependencies before dependents).
- [ ] `test/runtests.jl` `include` line uncommented.
- [ ] `DECISIONS.md`:
  - Loop status row marked 🟢 with file list and "passing locally".
  - "Iter N — <routine>" entry added to the per-iteration journal (one paragraph: what was implemented, any deviations from plan, any new decisions logged).
  - Any new convention you adopted appended to the "Standing decisions" list with `[PENDING-CONFIRM]` tag.
- [ ] `git add -A && git commit -m "iter <N>: <routine> green"` (one commit per iter; keeps history bisectable).
- [ ] If you hit a real ambiguity in the MATLAB code that you resolved by guessing, log the guess as an Open Question in DECISIONS.md so the user can review it.

---

## 9. When NOT to proceed

Pause the loop and surface the issue (in chat or as an Open Question in DECISIONS.md) if:

1. A plan §7 corner case turns out to be impossible to construct (this happened for `inertia` case 9 — see `test_inertia.jl`). Don't fake the test; mark it `@test_skip` with a comment explaining why.
2. A MATLAB expression's intent is genuinely unclear (e.g., a `% TODO` comment in the source). Don't guess silently.
3. A test you wrote per the plan fails not because of a bug but because the plan itself is over-specified (e.g., a performance budget that's unrealistic on this machine). Document the discovered envelope in DECISIONS.md and bump the threshold with a note, rather than removing the test.
4. `verified_isspd` returns `true` in any soundness test. Stop everything. This is a critical-class bug; the routine is unsound and every downstream result is suspect. Hunt the root cause before continuing any other iter.

---

## 10. Final deliverable definition (for "we're done")

When all 8 iters are 🟢:

1. `Pkg.test()` passes on Julia 1.10 LTS and current stable.
2. The two README MATLAB examples (Hilbert n=8 and banded n=10) reproduce, with the Julia interval enclosures provably containing the MATLAB intervals (use `issubset` with a small radius slop, not bit-equality).
3. `verified_isspd`'s adversarial soundness battery passes with `==` on all `λ_min = -ε` cases for `ε ∈ {1e-2, 1e-8, 1e-14}`.
4. `bench/baselines/julia-1.10.json` exists with median timings for all routines.
5. `DECISIONS.md` has every standing decision either accepted (no `[PENDING-CONFIRM]`) or carried forward as a known limitation.
6. `docs/` builds with Documenter.jl (`julia --project=docs docs/make.jl`).
7. `git log --oneline` shows ~8 clean iter commits.
8. The package is ready for `Pkg.activate(".") ; Pkg.test()` from a fresh checkout — no manual setup steps beyond `julia --project=. setup.jl`.

At that point, ping the user. Do not register on JuliaRegistries without their explicit OK (Q-001).

---

## 11. One-liner mental model

> *Read MATLAB → log decisions → write test contract → write code → run → iterate → commit → next.*

The discipline is the deliverable. The user explicitly elevated test discipline to a project rule because this is a verified-numerics library — a "fast" implementation that returns the wrong sign once in a million has zero value. Slow and right beats fast and probably-right.

Good luck. The hard parts are mostly behind us; the rest is methodical translation under the rule.
