# Veigs.jl — verified eigenvalue bounds for symmetric generalized eigenproblems

`Veigs.jl` computes **mathematically rigorous enclosures** of eigenvalues of the
symmetric generalized eigenvalue problem

    A x = λ B x,        A = Aᵀ,   B = Bᵀ positive definite,

for dense and sparse matrices, including matrices whose entries are intervals.
Every returned bound is an `Interval{Float64}` that is guaranteed to contain the
exact eigenvalue of every matrix pair inside the input enclosure; all rounding
errors of the floating-point computation are accounted for.

The package is the Julia port of the MATLAB toolbox `veigs` (X. Liu) that has been
used for computer-assisted proofs in the verified finite element method
([`VFEM.jl`](https://github.com/xfliu/vfem), [`VFEM2D`](https://github.com/xfliu/VFEM2D)).
It replaces the INTLAB dependency of the MATLAB version with
[`IntervalArithmetic.jl`](https://github.com/JuliaIntervals/IntervalArithmetic.jl)
and standard LAPACK / SuiteSparse factorizations: the heavy linear algebra runs in
floating point, and interval arithmetic is used only for the short certificate at
the end (Rump's verified positive-definiteness test, Bunch–Kaufman inertia counts,
and Lehmann–Behnke-type enclosures for clusters).

The Julia package lives in the sub-directory [`VEIGS.jl`](VEIGS.jl).

## Features

* `veigs(A, B, k, sigma)` — cluster-aware driver: verified bounds for `k`
  eigenvalues selected by `sigma` (`:smallestabs`, `:largestabs`, `:smallestreal`,
  `:largestreal`, or a numeric shift), for dense or **sparse** matrices with tens of
  thousands of unknowns (40 000 in the tutorial below). Clusters of close eigenvalues are enclosed jointly.
* `veig(A, B[, ind])` — full-spectrum verified solver for small dense pencils
  (n ≤ 100), Yamamoto-style with LDLᵀ inertia counting.
* Building blocks, exported for reuse: `verified_isspd` (Rump 2006 test, the
  algorithm of INTLAB's `isspd`), `verified_ldl` + `inertia` (certified inertia of
  `A − λB`), `rough_lower` / `rough_upper`, `lehmann_behnke`, `sym_hull`.
* Interval-valued `A` and `B` are accepted everywhere; results are then valid for
  every matrix in the enclosure.
* Soundness first: the test suite contains adversarial cases (`λ_min = −ε` for
  `ε` down to 1e-14) that must fail closed; the package prefers to return a wider
  bound or an error over an unjustified certificate.

## Installation

`Veigs` is not in the General registry. Install it from GitHub (Julia ≥ 1.10):

```julia
using Pkg
Pkg.add(url = "https://github.com/xfliu/veigs", subdir = "VEIGS.jl")
```

To run the tests and benchmarks from a checkout:

```bash
git clone https://github.com/xfliu/veigs.git
cd veigs/VEIGS.jl
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
julia --project=. bench/bench_veigs.jl
```

## Quick start

```julia
using LinearAlgebra, Veigs
using IntervalArithmetic: inf, sup

A = Matrix(SymTridiagonal(fill(2.0, 4), fill(-1.0, 3)))   # 1D Laplacian, n = 4
B = Matrix{Float64}(I, 4, 4)

bounds, ir = veig(A, B)              # all eigenvalues, verified
for (i, b) in zip(ir, bounds)
    println("λ_$i ∈ [", inf(b), ", ", sup(b), "]")
end
```

```
λ_1 ∈ [0.3819660112500948, 0.3819660112883118]
λ_2 ∈ [1.3819660111118963, 1.381966011250118]
λ_3 ∈ [2.618033988488079, 2.6180339887499073]
λ_4 ∈ [3.618033988388082, 3.618033988749905]
```

The exact eigenvalues `2 − 2cos(kπ/5)` = 0.38196601125010…, 1.38196601125010…,
2.61803398874989…, 3.61803398874989… lie inside the enclosures.

## Tutorial

All outputs below were produced with Julia 1.12.6 (`VEIGS.jl` at the current
commit); the code is in the tutorial section of this file and can be pasted into
a REPL after `using LinearAlgebra, SparseArrays, Veigs` and
`using IntervalArithmetic: inf, sup, interval`.

### 1. A few eigenvalues of a dense pencil: `veigs`

`veigs` returns the bounds and the index range they belong to. The classical
example of the MATLAB toolbox is the Hilbert matrix: `B = H₈`, `A = I`, and the
three eigenvalues closest to the shift 3.5.

```julia
n = 8
H = [1 / (i + j - 1) for i in 1:n, j in 1:n]
bounds, ir = veigs(Matrix{Float64}(I, n, n), H, 3, 3.5)   # 3 eigenvalues nearest 3.5
for (i, b) in zip(ir, bounds)
    println("λ_$i ∈ [", inf(b), ", ", sup(b), "]")
end
```

```
λ_1 ∈ [0.589643850288809, 0.5896438502888105]
λ_2 ∈ [3.3542953163290723, 3.354295316329102]
λ_3 ∈ [38.149237682656455, 38.14923768265958]
```

The MATLAB + INTLAB toolbox gives `λ_1 ∈ [0.589643850288603, 0.589643850289016]`
for the same problem; the Julia enclosure is contained in it and more than 100 times
narrower. Selecting by position instead of by shift:

```julia
veigs(Matrix{Float64}(I, n, n), H, 1, :smallestreal)   # smallest eigenvalue
veigs(A, B, 2, :largestabs)                             # two largest in magnitude
```

`sigma` also accepts the legacy MATLAB names `:sm`, `:lm`, `:sa`, `:la`.

### 2. Large sparse problems: finite element Laplacian

`veigs` works directly on `SparseMatrixCSC` input; the factorizations are done
by SuiteSparse and the approximate eigenpairs by KrylovKit, while the certificate
stays in interval arithmetic. The 2D Dirichlet Laplacian on the unit square with
the 5-point stencil:

```julia
m = 200; h = 1 / (m + 1)                     # 40 000 unknowns
T = spdiagm(-1 => fill(-1.0, m - 1), 0 => fill(2.0, m), 1 => fill(-1.0, m - 1))
K = (kron(I(m), T) + kron(T, I(m))) / h^2
M = sparse(1.0 * I, m^2, m^2)

bounds, ir = veigs(K, M, 3, :smallestreal)   # three smallest eigenvalues
```

```
λ_1 ∈ [19.738806962477916, 19.738806962894916]
λ_2 ∈ [49.344606447883486, 49.34460644916812]
λ_3 ∈ [49.34460644788366,  49.34460644916829]
```

The discrete eigenvalues are `4(m+1)²(1 − cos(kπ/(m+1)))` combinations, e.g.
`λ_1 = 19.738806962711…`, and the double eigenvalue `λ_2 = λ_3` is returned as a
cluster with two (nearly identical) enclosures. This call takes about 6 s on a
Xeon workstation; `n = 10 000` takes 1.3 s.

### 3. Clustered eigenvalues

Close eigenvalues are the hard case for verification: a certificate for a single
eigenvalue needs a gap to its neighbours. `veigs` detects clusters automatically,
encloses them jointly with a Lehmann–Behnke bound and reports the whole index
range. The test suite contains a textbook example, the Dirichlet Laplacian on a
dumbbell domain (two unit squares joined by a thin bar), whose first two
eigenvalues differ by a tunnelling-small amount:

```julia
include(joinpath(dirname(pathof(Veigs)), "..", "test", "support", "laplacian2d_dumbbell.jl"))
K, M, ndof, _ = laplacian2d_dumbbell(0.05, 2)        # P2 elements, h = 0.05, 3609 unknowns (needs Ferrite.jl)
bounds, ir = veigs(K, M, 2, :smallestabs)
```

```
λ_1 ∈ [19.714345…]   λ_2 ∈ [19.714349…]     gap λ₂ − λ₁ = 3.8e-6, enclosure widths 1.6e-10
```

Both eigenvalues are separated rigorously even though their distance is only
`3.8e-6` (see the benchmark table below for other mesh sizes and orders).

### 4. Interval-valued matrices

If the matrices are themselves only known up to rounding, pass interval
matrices; the bounds then hold for every matrix in the enclosure:

```julia
Ai = interval.(A .- 1e-12, A .+ 1e-12)     # A from the quick start, radius 1e-12
bounds, ir = veig(Ai, B)
```

```
λ_1 ∈ [0.3819660112420943, 0.3819660112963123]   width 5.4e-11
λ_2 ∈ [1.3819660111038956, 1.3819660112581187]   width 1.5e-10
λ_3 ∈ [2.6180339884800787, 2.6180339887579076]   width 2.8e-10
λ_4 ∈ [3.6180339883800814, 3.6180339887579054]   width 3.8e-10
```

### 5. The building blocks

The verified primitives can be used on their own, for example to certify that a
shifted matrix is positive definite (this is exactly what a Lehmann–Goerisch
eigenvalue bound in VFEM needs):

```julia
verified_isspd(A - 0.38I)     # true   (λ_min(A) = 0.3819…)
verified_isspd(A - 0.39I)     # false  (0.39 > λ_min: not positive definite, correctly refused)
verified_isspd(Matrix{Float64}(I, 3, 3) - 0.5 * ones(3, 3))   # false
```

`verified_isspd` is one-sided: `true` is a proof, `false` only means "could not
certify". The inertia of `A − λB`, i.e. the number of eigenvalues below `λ`, is
obtained from a Bunch–Kaufman factorization with a verified residual:

```julia
L, D, p, ΔM, ok = verified_ldl(A - 1.0I)
Veigs.inertia(D)          # (neg, pos, zero, failed) = (1, 3, 0, false): exactly one eigenvalue below 1
```

(`inertia` is also a name in `LinearAlgebra`; qualify it as `Veigs.inertia` when
both are loaded.) `rough_lower` / `rough_upper` search separators with these
tests, and `lehmann_behnke(A, B, eig_list, V, ρ, σ, λ_B_min, r, s)` turns
approximate eigenvectors `V` of a cluster `r:s` plus separators `ρ < λ_r`,
`λ_s < σ` into verified bounds — the same construction that `veigs` uses
internally.

## API summary

| Function | Purpose | Returns |
| --- | --- | --- |
| `veigs(A, B[, k][, sigma])` | `k` verified eigenvalues selected by `sigma`; dense or sparse; clusters handled | `(Vector{Interval}, ind_range)` |
| `veig(A, B[, ind])` | verified enclosures of all (or selected) eigenvalues, dense, `n ≤ 100` | `(Vector{Interval}, ind_range)` |
| `verified_isspd(M; method)` | proof that the (interval) matrix `M` is positive definite; `:rump2006` (default) or `:ldl_shift` | `Bool` |
| `verified_ldl(M)` | Bunch–Kaufman factors of `mid(M)` and interval residual | `(L, D, p, ΔM, ok)` |
| `inertia(D)` | certified inertia of a block-diagonal `D` | `(neg, pos, zero, failed)` |
| `rough_lower`, `rough_upper` | verified separators below/above the spectrum region | `Float64` |
| `lehmann_behnke(...)` | Lehmann–Behnke enclosure of an eigenvalue cluster | `Vector{Interval}` |
| `sym_hull(M)` | symmetric hull of an interval matrix | matrix |

Errors are typed (`VeigsSizeError`, `VeigsLDLFailureError`,
`VeigsClusterTooLargeError`, `VeigsLehmannBehnkeError`, `VeigsRoughBoundError`),
so a failed certificate is never silent.

## Benchmarks

Measured on a workstation with two Intel Xeon Gold 5318Y (2.1 GHz), Julia 1.12.6,
default BLAS threads, with the scripts in `VEIGS.jl/bench/`. Times are per call
after compilation.

**Dense, full spectrum — `veig(A, B)`, random SPD `A`, `B = I`** (`bench_veig.jl`)

| n | median time |
| --- | --- |
| 5 | 0.16 ms |
| 10 | 1.0 ms |
| 25 | 15 ms |
| 50 | 134 ms |

**Dense, one eigenvalue — `veigs(A, B, 1, :la)`** (`bench_veigs.jl`)

| n | median time |
| --- | --- |
| 8 | 0.1 ms |
| 30 | 0.7 ms |
| 100 | 15 ms |

**Sparse 1D Laplacian, `B = I`, smallest eigenvalue — `veigs(A, B, 1, :sm)`** (`bench_laplacian_smallest.jl`)

| n | median time | enclosure width |
| --- | --- | --- |
| 1 000 | 7.8 ms | 3.1e-15 |
| 2 000 | 14 ms | 3.1e-15 |
| 5 000 | 36 ms | 3.1e-15 |

For comparison, the MATLAB + INTLAB toolbox needs 34, 30 and 39 ms for the same
three problems on the same machine.

**Sparse 2D Laplacian (5-point stencil), smallest eigenvalue — `veigs(K, M, 1, :sm)`**

| n | time |
| --- | --- |
| 10 000 | 1.3 s |
| 40 000 | 4.6 s |

**Finite element eigenvalues on the dumbbell domain, clustered first pair —
`veigs(K, M, 2, :smallestabs)`** (`bench_dumbbell.jl`; the last column is the
MATLAB + INTLAB toolbox on the same matrices, exported as MatrixMarket files)

| h | order | ndof | λ₂ − λ₁ | enclosure width | Julia | MATLAB + INTLAB |
| --- | --- | --- | --- | --- | --- | --- |
| 0.1 | P1 | 183 | 1.9e-3 | 1.5e-12 | 0.015 s | 0.062 s |
| 0.05 | P1 | 845 | 7.1e-4 | 6.1e-12 | 0.11 s | 0.064 s |
| 0.025 | P1 | 3 609 | 2.1e-4 | 2.4e-11 | 0.58 s | 0.22 s |
| 0.1 | P2 | 845 | 6.3e-5 | 3.9e-11 | 0.042 s | 0.069 s |
| 0.05 | P2 | 3 609 | 3.8e-6 | 1.6e-10 | 0.49 s | 0.16 s |
| 0.1 | P3 | 1 987 | 7.8e-6 | 2.5e-10 | 0.20 s | 0.076 s |

Both implementations enclose the cluster to the same displayed precision. Julia
is faster on small problems; for the larger FEM matrices INTLAB's hand-tuned
sparse interval kernels still win by a factor of two to three (see
`VEIGS.jl/discussion_log.md` for the profiling history).

## Testing and design

* `Pkg.test()` runs 70 285 assertions (70 284 pass, one is a documented `@test_broken` corner case) in about 3 minutes: one test file per routine, each with an
  explicit corner-case list, adversarial soundness cases, and a performance
  budget.
* [`VEIGS.jl/DESIGN.md`](VEIGS.jl/DESIGN.md) — the design contract (verified
  post-processing, not interval linear algebra; the five verified primitives).
* [`VEIGS.jl/DECISIONS.md`](VEIGS.jl/DECISIONS.md) — the decision log of the
  port, [`VEIGS.jl/guide.md`](VEIGS.jl/guide.md) — the porting guide, and
  [`VEIGS.jl/blog/veigs_jl_story.md`](VEIGS.jl/blog/veigs_jl_story.md) — the
  development story with the performance work from the first port to the
  current version.

## Known limitations

* `veig` is capped at `n ≤ 100` (as in the MATLAB toolbox); use `veigs` for
  larger problems.
* When a floating-point approximate eigenvalue coincides *exactly* with an
  eigenvalue (for instance `λ = 2` of the 3×3 matrix `tridiag(−1, 2, −1)`), the
  rough-bound search of `veig` can return a very loose but still valid lower
  bound; `veigs` handles such cases tightly.
* Interval matrix products for the small cluster matrices use generic Julia
  loops, which is the dominant cost for clusters of size above about 20.

## Citation

```bibtex
@misc{liu2026veigs,
  author       = {Liu, Xuefeng},
  title        = {{Veigs}: a {MATLAB} and {Julia} library for verified eigenvalue computation for matrices},
  year         = {2026},
  howpublished = {GitHub repository},
  url          = {https://github.com/xfliu/veigs}
}
```

## License

MIT, see [`LICENSE`](LICENSE).
