# bench/bench_inertia.jl
#
# Run with: julia --project=. bench/bench_inertia.jl

using BenchmarkTools, LinearAlgebra, Random
using Veigs: inertia

Random.seed!(20260501)

println("inertia benchmarks")
println("="^60)

for n in (10, 50, 100, 200, 500)
    D = Matrix(Diagonal(randn(n)))
    b = @benchmark inertia($D)
    println("n = $(lpad(n, 4))  median = $(round(median(b).time/1e3, digits=2)) μs   minimum = $(round(minimum(b).time/1e3, digits=2)) μs")
end

println("\nBudget reminder: n=100 should be under 5 ms (5000 μs).")
