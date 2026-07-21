# setup.jl — one-time bootstrap.
#
# Run from the package root with the project active:
#     /Volumes/DropboxDisk/AppDisk/Julia/bin/julia --project=. setup.jl
#
# This adds runtime + test deps to Project.toml and downloads them into the
# project's local depot. Subsequent runs are no-ops once `Manifest.toml` exists.

using Pkg

@info "Adding runtime dependencies..."
Pkg.add([
    "IntervalArithmetic",
    "Arpack",
    "KrylovKit",         # iterative eigen approx for shift-invert (used by veigs_driver)
    "LDLFactorizations", # sparse LDL fast path for verified_ldl (v17)
    "LinearAlgebra",     # stdlib, but must appear in [deps] for `using LinearAlgebra` inside the package
    "SparseArrays",      # ditto
])

@info "Optional: MUMPS.jl for higher-performance sparse symmetric indefinite LDL..."
@info "(Not auto-installed — heavy build. Add manually with `Pkg.add(\"MUMPS\")` if desired.)"

@info "Adding optional dependencies (alternate verified-solve path)..."
try
    Pkg.add("IntervalLinearAlgebra")
catch err
    @warn "IntervalLinearAlgebra failed to install — the alternate verified_isspd path will be unavailable but the core package still works." exception=err
end

@info "Adding test/benchmark deps..."
Pkg.add([
    "Test",
    "BenchmarkTools",
    "Random",
])

@info "Resolving and precompiling..."
Pkg.resolve()
Pkg.precompile()

@info "Setup complete. Try: julia --project=. -e 'using Pkg; Pkg.test()'"
