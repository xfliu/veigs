# bench/bench_lehmann_behnke.jl
#
# Run with: julia --project=. bench/bench_lehmann_behnke.jl

using BenchmarkTools, LinearAlgebra, Random
using Veigs: lehmann_behnke

include(joinpath(@__DIR__, "..", "test", "support", "matrix_factory.jl"))

Random.seed!(20260501)

println("lehmann_behnke benchmarks (diagonal SPD, B = I, single-eig case)")
println("="^60)

for n in (5, 10, 20, 50)
    A = Matrix(Diagonal(0.5 .+ rand(n) .* 5))
    B = Matrix{Float64}(I, n, n)
    F = eigen(Symmetric(A), Symmetric(B))
    perm = sortperm(real.(F.values))
    ev = real.(F.values[perm])
    V  = real.(F.vectors[:, perm])
    r = 1
    ρ = ev[1] - 1.0
    σ = (ev[r] + ev[r+1]) / 2
    Vc = V[:, r:r]
    lehmann_behnke(A, B, ev, Vc, ρ, σ, 1.0, r, r)       # warm
    b = @benchmark lehmann_behnke($A, $B, $ev, $Vc, $ρ, $σ, 1.0, $r, $r)
    println("n = ", lpad(n, 4),
            "  median = ", round(median(b).time / 1e6, digits = 2), " ms")
end
