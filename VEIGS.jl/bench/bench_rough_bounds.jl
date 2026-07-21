# bench/bench_rough_bounds.jl
#
# Run with: julia --project=. bench/bench_rough_bounds.jl

using BenchmarkTools, LinearAlgebra, Random
using Veigs: rough_lower, rough_upper

include(joinpath(@__DIR__, "..", "test", "support", "matrix_factory.jl"))

Random.seed!(20260501)

println("rough_bounds benchmarks (SPD A, B = I)")
println("="^60)

for n in (10, 50, 100)
    A = Matrix(spd_random(n))
    B = Matrix{Float64}(I, n, n)
    λ_min = minimum(eigvals(Symmetric(A)))
    seed = λ_min - 0.05; floor = λ_min - 5.0
    rough_lower(A, B, seed, floor)                 # warm
    b = @benchmark rough_lower($A, $B, $seed, $floor)
    println("rough_lower  n = ", lpad(n, 4),
            "  median = ", round(median(b).time / 1e6, digits = 3), " ms")
end
