# test/runtests.jl
#
# Top-level test entry. Includes one file per routine. As routines come online
# in the auto-loop, uncomment the corresponding `include` line.

using Test
import Veigs

@testset "Veigs.jl" begin
    include("test_inertia.jl")
    include("test_isspd.jl")
    include("test_verified_ldl.jl")
    include("test_rough_bounds.jl")
    include("test_veig.jl")
    include("test_lehmann_behnke.jl")
    include("test_veigs.jl")
    include("test_laplacian2d_pk.jl")
    include("test_laplacian2d_dumbbell.jl")
    # include("test_lehmann_behnke.jl")  # iter 5
    # include("test_veig.jl")            # iter 6
    # include("test_veigs.jl")           # iter 7
end
