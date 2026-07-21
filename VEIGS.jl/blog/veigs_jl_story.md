# Catching Up to the Standard: A Verified Eigenvalue Solver's Journey from 14× Behind to Side-by-Side

*By Xuefeng Liu, with an AI coding agent as collaborator. A story about porting MATLAB code to Julia, the surprises that came along the way, and what we learned about honesty in benchmarking — written in respectful conversation with the work of Siegfried Rump and the INTLAB community, on whose shoulders this project stands.*

---

## The problem nobody likes to think about

Imagine you're a quantum chemist. You discretize a Schrödinger equation, you solve a giant generalized eigenvalue problem `K x = λ M x`, and out pops a number — the ground-state energy of your molecule, or the lowest vibrational frequency of a beam, or the principal mode of a bridge.

How accurate is that number?

Most of the time, nobody asks. The eigensolver returned 19.7392, IEEE-754 floating point gave us 16 digits, the question is settled. We move on.

But "most of the time" is doing a lot of work in that sentence. There are entire subfields — verified computing, certified numerics, rigorous bounds on PDE solutions — built around the cases where it isn't enough. When you're proving that a beam *cannot* resonate at a certain frequency, or that a quantum system has *exactly* this many bound states below an energy threshold, "the answer is approximately 19.7392" is not a proof. You need an interval `[a, b]` containing the true answer, with mathematical certainty, accounting for every rounding error from input to output.

This is what `veigs` does — both the original MATLAB version, built on Siegfried Rump's INTLAB, and the Julia port we'll talk about here. Rump began publishing on directed-rounded interval arithmetic in the 1980s; INTLAB has been the gold standard of practical verified numerics ever since. If you have done any serious work in this area, you know the toolkit, and you know the name. The Julia port could not exist without that foundation — both because we ported algorithms that originated there, and because we benchmarked our work against an incumbent whose maturity set our quality bar.

The verified-computing community is small. The tooling is older than most production scientific software. And until very recently, MATLAB + INTLAB was the only practical way to do this work at scale.

This is a story about bringing that capability to Julia, where things didn't go quite as expected.

---

## The starting point: 14× behind

The MATLAB `veigs` toolbox solves verified eigenvalue problems via Lehmann–Behnke complementary variational principles, with LDL inertia counts and Bunch–Kaufman factorizations doing the careful work underneath. It is clever, it is mature, and it has years of careful tuning behind it. The Julia port we set out to write was a translation, not a replacement.

Our Julia port — call it Veigs.jl, version 8 — was algorithmically faithful, MATLAB-output-matching, fully tested. And on a 1D Laplacian eigenvalue problem at n=5000, it was 14× slower than MATLAB.

This was not surprising and not shameful. MATLAB's `intval` arithmetic is built on directed-rounding kernels Rump has been refining for forty years. INTLAB's sparse interval × float matmul is the result of an enormous accumulation of insight about what interval arithmetic actually demands of a CPU. Julia's `IntervalArithmetic.jl` is excellent at correctness, but younger and more conservative about performance. We were comparing a first-pass translation to a polished original. The right framing for the work ahead was *catch up*, not *beat*.

Still — 14× is a lot. And for the people who actually use this kind of solver, the choice between "minutes in MATLAB" and "tens of minutes in Julia" matters. Sometimes it is the difference between running an experiment overnight and running it during a coffee break.

So we started looking.

---

## Round 1: the mid/rad refactor

Interval arithmetic stores each number as a pair `[lo, hi]`. Most numerical operations on such intervals are slow not because the math is hard, but because every elementary `+ × ÷` requires *directed rounding* — round one endpoint down and the other endpoint up, separately. CPUs hate this. The branch predictors hate it. Vectorization gives up.

The classic trick — Rump's own — is to represent intervals as a *midpoint plus a radius*: `(m, r)` where the interval is `[m - r, m + r]`. Now you can do arithmetic on the midpoints in plain hardware-accelerated float, accumulate the rounding error into a radius bound (the Wilkinson γ formula), and only "pack" back to interval form at the end.

This is well-known to interval-computing experts, and standard in INTLAB. But our Julia port wasn't using it. We were calling `IntervalArithmetic.jl`'s element-wise interval×float kernel everywhere — pretty, simple, slow.

Over a series of refactors (we called them v9 through v15), every hot path in the kernel got the mid/rad treatment:

- `verified_isspd` (the routine that proves a matrix is positive definite)
- `rough_bounds` (initial spectrum separation)
- `verified_ldl` (the central LDL factorization with verified residual)
- `lehmann_behnke` (the variational tightening step, where most cycles go)

For the curious: the `lehmann_behnke` refactor alone replaced four interval×float matrix multiplications (each ~15 ms at n=5000) with float-only matrix multiplications plus a per-element scalar bound (each ~0.013 ms). That's roughly a thousand-fold acceleration on those particular calls.

By v14, we ran the bench again on the 1D Laplacian at n=5000.

**Julia: 0.16 seconds. MATLAB+INTLAB: 0.39 seconds.**

This was the moment we caught up. Not "beat" — caught up. The MATLAB number is what a forty-year-tuned production verified-arithmetic library does on a sparse 1D problem; reaching parity with a much younger Julia ecosystem felt like a real milestone, not a victory at someone else's expense.

---

## A philosophical detour: speed versus tightness

There's a wrinkle. The mid/rad refactor uses Wilkinson's *worst-case* bound on rounding error — the bound says "the error in `fl(M·X)[i,j]` is at most `γ_k · (|M|·|X|)[i,j]`" where `γ_k = k·u/(1−k·u)` and `k` is the chain length of the dot product.

Worst case. For most matrices, the actual rounding is smaller — sometimes much smaller. For matrices with integer multipliers (a 1D Laplacian with stencil `[-1, 2, -1]`, for instance), many of the individual multiplies are *exact*, and the actual rounding is essentially zero.

INTLAB doesn't pay this cost. Its directed-rounded matmul gives you the *actual* rounding, ULP-tight. So while we got the speedup, we also got bound widths about 5× wider on tridiagonal-style problems.

This is the kind of trade-off that makes interval-arithmetic work uncomfortable. There's no "right" answer. So we did what good API designers do: we surfaced it.

```julia
lehmann_behnke(A, B, ...; outer_matmul = :fast)   # default: speed
lehmann_behnke(A, B, ...; outer_matmul = :tight)  # tightness
```

Both modes are correct — both produce intervals that mathematically contain the true eigenvalue. The user picks based on their problem. For most use cases the `:fast` bound is well below any algorithmic precision floor (separator gaps, residual norms). For users who genuinely need ULP-tight bounds — say, on integer-multiplier matrices used in research-grade verification — `:tight` is one kwarg away.

There's a delightful side-finding from this. The `:fast` trick turns out to be *more* valuable at higher precision. At regular Float64 the gap between `:fast` and `:tight` is ~20×; at BigFloat (256-bit) it's ~440× at n=2000. The reason: at BigFloat there's no BLAS to optimize either kernel, so the entire gap shows up as per-element directed-rounding overhead. Practical implication: if you ever need verified eigenvalues at >Float64 precision (a real need in mathematical-physics certification), you basically *must* use `:fast`.

If you're skipping the tech, here's the takeaway: **we made some bounds 5× wider in exchange for 5× speed, and made it user-selectable**. Same algorithm, different rounding model.

---

## The 47× surprise

We were feeling good about 1D Laplacian. The natural next test was 2D — a P1 finite-element discretization of the Laplacian on the unit square, in the spirit of every numerical-PDE benchmark since 1980.

```
n_side = 60       # 60 × 60 grid → N = 3600 interior DOFs
K, M = laplacian2d_p1(n_side)
veigs(K, M, 1, :smallestabs)  # smallest eigenvalue
```

Julia: **6.24 seconds.**

We ran MATLAB on the same matrices.

MATLAB: **0.137 seconds.**

The Julia version was forty-seven times slower.

This was, to put it mildly, not what we expected.

We had spent multiple iterations refactoring kernels, watching dense-matmul kernels in our profiler, optimizing interval×float dispatch — and the moment we touched a more realistic test case (2D, not just 1D), our painstakingly-won lead evaporated. Worse than evaporated. We were further behind than we'd been at v8.

At first this felt like a bug. Was the 2D matrix somehow different? Was it triggering a slow path? After ten minutes of disbelief, we ran a profile.

The profile was unambiguous. **`verified_ldl`, on a single call, was eating 2.97 seconds.**

The reason was structural. The 1D Laplacian is *tridiagonal*. We had a fast path for that — `_verified_ldl_tridiag` — that ran in 3 milliseconds at n=5000. For tridiagonal matrices, the LDL factor is itself banded, the residual is pentadiagonal, everything stays sparse and small.

The 2D Laplacian is *not* tridiagonal. It's 5-point. Once you cross that line, our code took the dense fallback. Dense Bunch–Kaufman at N=3600 is a couple of cubic-time matrix products — 6 seconds, then again, then again. Two LDL calls per `veigs`, multiple matrix multiplications each, all on dense N×N matrices.

Meanwhile MATLAB's `[L, D, P] = ldl(M)` for sparse symmetric indefinite matrices calls **MA57** from the HSL library — a battle-tested 1980s sparse symmetric indefinite LDL with 1×1/2×2 Bunch–Kaufman pivoting. MA57 was solving the 2D Laplacian factorization in milliseconds. We were doing it in seconds.

We didn't have an MA57 equivalent in Julia. *Stdlib has no sparse symmetric indefinite LDL.* You can do sparse Cholesky (positive definite only), or sparse LU (general but loses symmetry), but the specific thing MA57 does — pivoted symmetric indefinite — Julia stdlib does not provide.

This is the kind of moment that separates ecosystems with mature numerical libraries from younger ones. INTLAB-on-MATLAB inherited a working tool. Veigs.jl, ported to Julia, did not.

We had options.

---

## Choosing the right tool

We laid them out, in increasing order of effort:

1. **`LDLFactorizations.jl`** — pure-Julia sparse LDL with AMD fill-reducing ordering. Lightweight, easy to install. Doesn't pivot on the fly, so a zero pivot causes failure — but for our use case (matrices `K - λ M` with λ off the spectrum), zero pivots are rare. Build a try/catch fallback to dense BK and you've covered the corner cases.

2. **`MUMPS.jl`** — bindings to the heavyweight MUMPS C++ sparse solver, the equivalent of MA57 (and arguably better for very large problems via out-of-core mode). Would close the gap fully, but it's a heavy dependency, and MUMPS isn't designed to expose its L and D factors as `SparseMatrixCSC` objects — it's a solver, not a factor library. Integrating it cleanly would mean refactoring our caller from "give me the LDL factorization" toward "give me the inertia and verified solves".

3. **A custom banded LDL** with hand-written 1×1/2×2 BK pivoting. Three hundred lines of careful code, no new dependencies, would handle banded problems but not arbitrary sparsity.

4. **Live with the gap. Document it. Move on.**

We picked option 1. It's the kind of choice that I think is underrated in software engineering: when a problem looks daunting, ask whether you can solve 80% of it cheaply before investing in solving 100% of it expensively.

The integration was straightforward — about a hundred lines of new code. We added LDLFactorizations as a runtime dependency, wired it in as a sparse-non-tridiag fast path in `verified_ldl`, with a fall-through to the dense Bunch–Kaufman if a zero pivot is encountered. We also added a `ldl_backend` keyword (`:auto`, `:dense`, `:ldlfac`, `:mumps`) so users can force a specific backend, and so the seam is visible for whoever wants to do a proper MUMPS integration later. (We documented exactly what that integration would look like, in the source code, near where it would go. There's nothing more frustrating than reading "future work" without a sketch of the future.)

We re-ran the bench.

| n_side | N | Old Julia | New Julia | MATLAB |
|--------|------|-----------|-----------|--------|
| 60 | 3 600 | 6.24 s | **0.18 s** | 0.137 s |
| 80 | 6 400 | (~30 s est.) | **0.44 s** | 0.302 s |
| 100 | 10 000 | (~80 s est.) | **0.79 s** | 0.612 s |

A 35× speedup at N=3600, and the gap to MATLAB was now 30%, not 47×.

We were back in the game.

---

## The crossover

At this point a reasonable person might have stopped. But the user — who, it turned out, was looking ahead to a 3D Schrödinger problem with millions of DOFs — wanted higher-order finite elements tested. P2 and P3 (quadratic and cubic Lagrange) on the same 2D unit square.

Triangular P2 and P3 are not trivial to assemble by hand. We'd done P1 from scratch — 70 lines of careful index management on the Friedrichs–Keller triangulation. P2 has 6 DOFs per element, P3 has 10, with edge midpoints and face centers and barycentric basis functions and quadrature rules. Hundreds of lines, easy to get wrong.

Rather than re-derive it, we leaned on **Ferrite.jl**, a mature Julia FEM package, as a test-only dependency. About fifty lines of `Ferrite` calls — generate a triangular mesh, declare a Pk Lagrange interpolation, loop over cells, accumulate stiffness and mass matrices, apply Dirichlet boundary conditions, extract the interior submatrix. Trivial.

We checked the convergence rates first — the eigenvalue error should decay as O(h^{2k}), where h is the mesh size and k the polynomial degree. Strang–Fix theorem. Halve h, expect the error to drop by 2^{2k}.

```
P1: ratios of 4.08, 4.02, 4.01  →  expected 4    ✓
P2: ratios of 14.9, 15.7, 15.9  →  expected 16   ✓
P3: ratios of 63.2, 64.4, 63.9  →  expected 64   ✓
```

Good. The assembler is correct.

Now the bench. We ran Julia and MATLAB on identical matrices (we exported the matrices from Julia in MatrixMarket format, wrote a small reader for MATLAB, and ran both `veigs` calls on the same data).

The result, in one table, sorted by problem size:

| Element | DOFs | Julia | MATLAB | Julia / MATLAB |
|---------|--------|---------|---------|------|
| P1 | 225 | 0.004 s | 0.034 s | **Julia 8.5× faster** |
| P1 | 961 | 0.032 s | 0.043 s | Julia 1.3× faster |
| P2 | 961 | 0.028 s | 0.032 s | Julia 1.1× faster |
| P3 | 2 209 | 0.112 s | 0.065 s | Julia 1.7× slower |
| P1 | 3 969 | 0.256 s | 0.152 s | Julia 1.7× slower |
| P2 | 3 969 | 0.180 s | 0.147 s | Julia 1.2× slower |
| P1 | 9 025 | 0.767 s | 0.516 s | Julia 1.5× slower |
| P3 | 9 025 | 0.742 s | 0.563 s | Julia 1.3× slower |
| P2 | 16 129 | 1.211 s | 1.081 s | Julia 1.1× slower |
| **P3** | **36 481** | **4.607 s** | **8.361 s** | **Julia 1.8× faster** |

Read the bottom row carefully. **At the largest case we tested — P3 elements with 36,481 degrees of freedom, the kind of problem real users actually have — Julia is 1.8× *faster* than MATLAB+INTLAB.**

The reason is not that we wrote better numerical code than Rump. The reason is that LDLFactorizations.jl's AMD fill-reducing ordering happens to scale more gracefully than MA57's on the dense connectivity of cubic finite elements. MA57 is from 1985. The HSL Fortran code is among the most-trusted sparse symmetric indefinite solvers in scientific computing — and it remains so. AMD ordering is from 1996 (also Davis et al.) and was tuned with somewhat different problem classes in mind. At small and medium sizes, MA57's careful pivoting wins; at very large P3 sizes, the modern ordering wins. Both are right answers for different operating points.

This is not a "Julia beats MATLAB" story. It is a "for one particular operating point, two excellent tools cross over, and the modern one happens to be the one that hands the result back faster" story. Those are very different stories. The classical tool — and the half-century of work behind it — still wins at small N (less per-call overhead), wins on bound tightness (~3–4× tighter intervals everywhere; INTLAB's directed rounding is hard to beat and we are not embarrassed by that), and remains the right choice for many users.

But for a Julia user who six months ago would have had to fall back to MATLAB to do verified eigenvalue computation, the picture has shifted. Veigs.jl is competitive across the board, and at the sizes that actually matter for serious 3D PDE work, it has joined the conversation.

---

## A surprising bonus

While we were at it, we noticed something else.

At 3,969 DOFs:
- **P1**: 0.256 s, eigenvalue relative error ~6×10⁻⁴
- **P2**: 0.180 s, eigenvalue relative error ~5×10⁻⁶

P2 is 30% faster than P1 and gives a hundred times more accuracy. For smooth solutions — which Schrödinger eigenvalues mostly are — there is essentially no reason to use P1.

This is well-known FEM lore but worth restating. Higher-order elements aren't a luxury. For smooth problems they're a free lunch.

---

## What we learned

A few things, ordered by importance:

**Honest benchmarking is the only kind that matters.** The 47× surprise wasn't a bug — it was discovering that our 1D-Laplacian bench had been measuring the wrong thing for our purposes. As soon as we switched to a slightly more realistic problem, our victory evaporated. The correct response wasn't to be defensive about the 1D result; it was to take the new measurement seriously and figure out what it implied.

**Profile before refactoring.** Every single performance improvement in this story came from looking at the profile and following the largest number. We never saved more than a few percent by guessing at hot spots. The wins came from observation.

**Mature ecosystems show up exactly where you'd expect.** MATLAB has had MA57 for nearly forty years. INTLAB has had its tuned interval kernels almost as long. Julia has had LDLFactorizations.jl for about ten. The gap of accumulated expertise — pivoting strategies, fill-reducing orderings, knowledge of pathological inputs, decisions about *where the operating-point boundaries really are* — is the kind of thing you can close in some places and not others. Some of it cannot be closed at all without simply reproducing the work. The right attitude is humility: catch up where you can, learn from the parts you cannot replicate, and acknowledge what you stand on top of.

**Visible seams beat invisible cleverness.** Every place we made an algorithmic trade-off, we surfaced it as a keyword argument with documentation. `outer_matmul = :fast | :tight`. `ldl_backend = :auto | :dense | :ldlfac | :mumps`. Future maintainers and users see the choices that were made. There's nothing hidden.

**The question worth asking is: who will use this?** This whole story makes sense because there are users. Quantum chemists who need verified ground states. Mechanical engineers who need certified resonance bounds. Mathematical physicists who need to *prove* things about operators, not just approximate them. The pleasure of the work is real, but the *importance* of the work depends on those people. Without them, none of this matters.

---

## What's next: verified FEM for the 3D Schrödinger ground state

The story above is the prologue. The real target — the problem that motivated this whole optimization arc — is a **verified finite-element method for the ground-state eigenvalue of a 3D Schrödinger operator**.

The Schrödinger eigenvalue problem in three dimensions, with a confining or many-body potential, sits at the heart of computational chemistry, condensed-matter physics, and quantum many-body theory. Standard practice is to discretize, solve the resulting matrix eigenvalue problem to numerical convergence, and report the lowest eigenvalue as the ground-state energy. *Verified* practice is to additionally produce a mathematically rigorous interval `[E_lo, E_hi]` containing the true continuum ground-state energy — accounting not only for floating-point rounding, but for the discretization error of the FEM itself.

This combination — *FEM error bounds + interval-arithmetic rounding bounds, jointly verified* — is the natural next layer on top of Veigs.jl. The pieces are recognizable from the work above:

- **Higher-order Lagrange elements (P2, P3 — and their tetrahedral counterparts in 3D)** for fast convergence in DOFs. The Ferrite.jl integration in v18 was the warm-up.
- **A posteriori error estimators** for FEM, combined with verified residual bounds, to certify the discretization gap.
- **Veigs.jl's Lehmann–Behnke kernel** to tighten the discrete eigenvalue interval.
- **Cluster handling** for the (often nearly-degenerate) lowest few eigenvalues that physicists actually care about.

Several open challenges sit on this path. Three-dimensional sparse LDL is harder than 2D — fill-in scales worse, AMD ordering hits its limits sooner, and the threshold above which a single workstation cannot finish a factorization arrives at smaller `N` than in 2D. We will likely meet MUMPS — really meet it, not just leave a documented seam — somewhere along this road. We will likely meet iterative verified eigensolvers, contour-integration methods (FEAST and friends), and domain-decomposition approaches for problems that no single LDL factorization can hold. Some of those are research, not engineering. They will be other stories.

For now, the milestone is this: the eigenvalue verification kernel at the heart of all of this is on solid ground in Julia. The next chapter — the verified-FEM-for-Schrödinger chapter — can be written with that foundation.

## A note on collaboration

This whole arc — the v9–v18 series of refactors, the benchmarking, the surprise on 2D, the integration of LDLFactorizations.jl and Ferrite.jl, and the long writeup you've just read — was built in close cooperation between a human researcher (myself, Xuefeng Liu) and an AI coding agent. Decisions were mine; many of the implementations, profiling traces, and exploratory benches were done together in dialogue. The agent did not replace expertise; it amplified the rate at which an experienced numerical analyst could try ideas, fail fast, and iterate. I think this kind of cooperation has a real future in scientific software, and the project here is one small piece of evidence for that.

I want to also re-state, plainly: nothing in this work would exist without the decades of effort that produced INTLAB and the MATLAB veigs toolbox. We learned by reading. We benchmarked against the standard. We are grateful for both.

---

*If you found this interesting and want to play with verified eigenvalue computation in Julia, the package is at [Veigs.jl](https://github.com/), alongside the original [veigs MATLAB toolbox](https://github.com/) and Rump's [INTLAB](http://www.ti3.tuhh.de/intlab/). The story above corresponds to commits v8 through v18 in the development log.*
