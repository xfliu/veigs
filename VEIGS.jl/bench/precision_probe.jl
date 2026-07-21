# bench/precision_probe.jl
#
# Probe whether higher precision in `lehmann_behnke` actually tightens
# the verified bound. Compares Float64, BigFloat at default 256-bit
# precision, and BigFloat at 1024-bit precision on a small banded
# eigenvalue problem (m = 1, single eigenvalue) where the inner
# `_lehmann_solve` falls through to a scalar interval division.
#
# Per DESIGN.md §3, this is the kind of experiment the user can run
# to "see" whether the bound floor is set by the algorithm or by the
# Float64 arithmetic.

using LinearAlgebra
using IntervalArithmetic: Interval, interval, mid, inf, sup, radius
using Veigs: lehmann_behnke

# Small test: 1D Laplacian n = 50, smallest eigenvalue.
n = 50
A = SymTridiagonal(fill(2.0, n), fill(-1.0, n - 1))
B = Matrix{Float64}(I, n, n)
A_f = Matrix(A)

# Approximate eigenvalue + vector (smallest, exact closed form).
λ_true = 2.0 * (1.0 - cos(π / (n + 1)))
v_true = sin.(π * (1:n) / (n + 1))   ; v_true ./= norm(v_true)
V = reshape(v_true, n, 1)
eig_list = [λ_true]                  # one approximate eigenvalue
ρ        = 0.0                       # left separator (below smallest)
σ        = 0.05                      # right separator (below 2nd smallest ≈ 0.082)
λ_B_min  = 1.0                       # smallest eig of identity B

println("n=$n 1D Laplacian, smallest eigenvalue λ_true = $(λ_true)")
println("---")
for prec in (Float64, BigFloat)
    if prec === BigFloat
        # Default BigFloat precision is 256 bits; let's also probe a higher one
        for bits in (256, 512, 1024)
            setprecision(BigFloat, bits)
            t = @elapsed begin
                bnd = lehmann_behnke(A_f, B, eig_list, V, ρ, σ, λ_B_min, 1, 1;
                                     precision = BigFloat)
            end
            bnd = lehmann_behnke(A_f, B, eig_list, V, ρ, σ, λ_B_min, 1, 1;
                                 precision = BigFloat)
            println("  precision = BigFloat($bits)  time = $(round(t*1e3, digits=2)) ms")
            println("    bound        = [$(inf(bnd[1])), $(sup(bnd[1]))]")
            println("    bound width  = $(BigFloat(sup(bnd[1]) - inf(bnd[1])))")
            println("    Δ from true  = $(abs(BigFloat(mid(bnd[1])) - λ_true))")
        end
    else
        t = @elapsed begin
            bnd = lehmann_behnke(A_f, B, eig_list, V, ρ, σ, λ_B_min, 1, 1)
        end
        bnd = lehmann_behnke(A_f, B, eig_list, V, ρ, σ, λ_B_min, 1, 1)
        println("  precision = Float64    time = $(round(t*1e3, digits=2)) ms")
        println("    bound        = [$(inf(bnd[1])), $(sup(bnd[1]))]")
        println("    bound width  = $(sup(bnd[1]) - inf(bnd[1]))")
        println("    Δ from true  = $(abs(mid(bnd[1]) - λ_true))")
    end
end
