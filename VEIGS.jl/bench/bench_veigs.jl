# bench/bench_veigs.jl
#
# Run with: julia --project=. bench/bench_veigs.jl

using BenchmarkTools, LinearAlgebra, Random
using Veigs: veigs

include(joinpath(@__DIR__, "..", "test", "support", "matrix_factory.jl"))

Random.seed!(20260501)

println("veigs benchmarks (single largest eigenvalue, A = SPD, B = I)")
println("="^60)

for n in (8, 30, 100)
    A = Matrix(spd_random(n))
    B = Matrix{Float64}(I, n, n)
    veigs(A, B, 1, :la)                               # warm
    b = @benchmark veigs($A, $B, 1, :la)
    println("n = ", lpad(n, 4),
            "  median = ", round(median(b).time / 1e6, digits = 1), " ms")
end
