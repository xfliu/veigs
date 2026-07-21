# bench/bench_verified_ldl.jl
#
# Run with: julia --project=. bench/bench_verified_ldl.jl

using BenchmarkTools, LinearAlgebra, Random
using Veigs: verified_ldl

include(joinpath(@__DIR__, "..", "test", "support", "matrix_factory.jl"))

Random.seed!(20260501)

println("verified_ldl benchmarks (dense SPD inputs)")
println("="^60)

for n in (10, 50, 100, 200, 500)
    M = Matrix(spd_random(n))
    verified_ldl(M)                                # warm
    b = @benchmark verified_ldl($M)
    println("n = ", lpad(n, 4),
            "  median = ", round(median(b).time / 1e6, digits = 3), " ms",
            "   minimum = ", round(minimum(b).time / 1e6, digits = 3), " ms")
end

println("\nBudget reminder: n=100 should be under 100 ms.")
