# bench/bench_laplacian_smallest.jl
#
# Apples-to-apples bench mirroring the MATLAB harness:
#   sparse 1D Laplacian (tridiagonal), B = I, smallest eigenvalue.
#
# Run: julia --project=. bench/bench_laplacian_smallest.jl

using BenchmarkTools, LinearAlgebra, SparseArrays, Random, Statistics
using IntervalArithmetic: inf, sup
using Veigs: veigs

Random.seed!(20260501)

function laplacian1d(n)
    spdiagm(-1 => fill(-1.0, n - 1),
             0 => fill( 2.0, n),
             1 => fill(-1.0, n - 1))
end

println("Sparse 1D Laplacian, B = I, smallest real eigenvalue")
println("="^88)
println(rpad("n", 6), " | ", rpad("median (s)", 12), " | ",
        rpad("min (s)", 12), " | ", rpad("max (s)", 12), " | ",
        rpad("trim mean", 12), " | bound width")
println(repeat("-", 88))

trim_mean(xs) = length(xs) >= 3 ? mean(sort(xs)[2:end-1]) : mean(xs)

for n in (1000, 2000, 5000)
    A = laplacian1d(n)
    B = sparse(1.0 * I, n, n)
    bnd, _ = veigs(A, B, 1, :sm)               # warm
    veigs(A, B, 1, :sm)                         # 2nd warm
    width = sup(bnd[1]) - inf(bnd[1])
    b = @benchmark veigs($A, $B, 1, :sm) samples = 5 evals = 1 seconds = 120
    ts  = sort(b.times) ./ 1e9
    med = ts[(length(ts)+1) ÷ 2]
    mn  = minimum(ts)
    mx  = maximum(ts)
    tm  = trim_mean(ts)
    println(lpad(n, 6), " | ",
            lpad(round(med, digits=4), 12), " | ",
            lpad(round(mn,  digits=4), 12), " | ",
            lpad(round(mx,  digits=4), 12), " | ",
            lpad(round(tm,  digits=4), 12), " | ",
            round(width; sigdigits=3))
end
