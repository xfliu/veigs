# bench/bench_veig.jl
#
# Run with: julia --project=. bench/bench_veig.jl

using BenchmarkTools, LinearAlgebra, Random
using Veigs: veig

include(joinpath(@__DIR__, "..", "test", "support", "matrix_factory.jl"))

Random.seed!(20260501)

println("veig benchmarks (full spectrum, A=SPD, B=I)")
println("="^60)

for n in (5, 10, 25, 50)
    A = Matrix(spd_random(n))
    B = Matrix{Float64}(I, n, n)
    veig(A, B)                                      # warm
    b = @benchmark veig($A, $B)
    println("n = ", lpad(n, 4),
            "  median = ", round(median(b).time / 1e6, digits = 2), " ms",
            "   minimum = ", round(minimum(b).time / 1e6, digits = 2), " ms")
end
