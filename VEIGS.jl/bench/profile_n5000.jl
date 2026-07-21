# bench/profile_n5000.jl
#
# Sample-profile veigs at n=5000 sparse Laplacian to localise where v10 spends
# its ~300 ms. Used to pick the v11 target.

using Profile, LinearAlgebra, SparseArrays, Random
using IntervalArithmetic: inf, sup
using Veigs: veigs

function laplacian1d(n)
    spdiagm(-1 => fill(-1.0, n - 1),
             0 => fill( 2.0, n),
             1 => fill(-1.0, n - 1))
end

n = 5000
A = laplacian1d(n)
B = sparse(1.0 * I, n, n)

# Warm
veigs(A, B, 1, :sm)
veigs(A, B, 1, :sm)

# Profile across N runs to get good sample density
Profile.clear()
@profile for _ in 1:8
    veigs(A, B, 1, :sm)
end

# Bucket samples by top frame
function bucket_samples()
    data, lidict = Profile.retrieve()
    bucket = Dict{String,Int}()
    total = 0
    for d in data
        info = get(lidict, d, nothing)
        info === nothing && continue
        total += 1
        # `info` may be a Vector of stack frames (newer Julia) or a single
        # StackFrame. We collect all unique function names from the stack.
        frames = info isa AbstractVector ? info : (info,)
        for fr in frames
            # `func` is a Symbol; show just the short name.
            name = string(fr.func)
            # Skip internal Julia book-keeping frames.
            startswith(name, "_jl_") && continue
            bucket[name] = get(bucket, name, 0) + 1
            break  # only count the top frame per sample
        end
    end
    return bucket, total
end

bucket, total = bucket_samples()

# Print top 25 buckets by count (descending).
println("Total samples: $total")
println()
println(rpad("function", 60), " | samples | %")
println("-"^80)
sorted = sort(collect(bucket); by = x -> -x[2])
for (name, count) in sorted[1:min(25, end)]
    pct = round(100 * count / total, digits=1)
    println(rpad(name, 60), " | ", lpad(count, 7), " | ", lpad("$(pct)%", 6))
end

# Also show flat profile via Julia's native printer (cleaner attribution).
println("\n\n== Profile.print(format=:flat, sortedby=:count) (top 30) ==\n")
io = IOBuffer()
Profile.print(IOContext(io, :displaysize=>(40,200)),
              format=:flat, sortedby=:count, mincount=80)
println(String(take!(io)))
