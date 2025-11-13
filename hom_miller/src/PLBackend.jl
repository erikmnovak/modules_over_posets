module PLBackend
using ..PosetModules
using ..PosetEncodings   # if you split encodings out; otherwise import from your Phase-2 file
using ..IndicatorResolutions

# ─────────────── Axis-aligned PL up/down shapes in R^n ────────────────────────

"Axis-aligned **upset**: {x ∈ ℝⁿ | x ≥ ℓ coordinatewise}."
struct BoxUpset
    ℓ::Vector{Float64}   # lower thresholds
end

"Axis-aligned **downset**: {x ∈ ℝⁿ | x ≤ u coordinatewise}."
struct BoxDownset
    u::Vector{Float64}   # upper thresholds
end

"Membership predicates."
contains(U::BoxUpset, x::Vector{Float64})  = all(x .>= U.ℓ)
contains(D::BoxDownset, x::Vector{Float64})= all(x .<= D.u)

# ─────────────── Finite encoding by threshold grid (uptight poset) ────────────
# We take all coordinates from the up/down thresholds, insert ±∞ sentinels, and
# form rectangular cells (product of intervals). Each cell is a vertex in a finite
# product-of-chains poset P. A representative sample point per cell evaluates Φ.

"""
    encode_fringe_boxes(Ups::Vector{BoxUpset}, Downs::Vector{BoxDownset}, Φ::Matrix{T}) -> (P, H)

Finite **PL encoding** of a box-fringe on ℝⁿ:
- `Ups` are birth upsets (axis-aligned lower thresholds),
- `Downs` are death downsets (axis-aligned upper thresholds),
- `Φ` is the monomial scalar matrix (rows match `Downs`, cols match `Ups`).

The encoding poset `P` is a product of finite chains induced by the threshold grid.
The returned `H` is a `PModule` whose fiber at each cell is `im(Φ_cell)`.
"""
function encode_fringe_boxes(Ups::Vector{BoxUpset}, Downs::Vector{BoxDownset}, Φ::Matrix{T}) where {T<:Number}
    @assert !isempty(Ups) "need at least one upset"
    n = length(Ups[1].ℓ)

    # 1) coordinates: all thresholds
    coords = [Float64[] for _ in 1:n]
    for i in 1:n
        for U in Ups; push!(coords[i], U.ℓ[i]); end
        for D in Downs; push!(coords[i], D.u[i]); end
        sort!(coords[i]); unique!(coords[i])
    end

    # 2) build cells as products of adjacent intervals (including extreme rays)
    # Encode each axis by indices 0..m, where segment k corresponds to (coords[k], coords[k+1])
    axes = [collect(0:length(coords[i])) for i in 1:n]
    # Elements of P are tuples (i1,...,in) indexing these segments. Partial order is product order.
    function cell_rep(idx::NTuple{N,Int}) where {N}
        # pick a point strictly inside each segment; we choose midpoints with paddings
        x = Vector{Float64}(undef, N)
        for j in 1:N
            if idx[j] == 0
                x[j] = coords[j][1] - 1.0
            elseif idx[j] == length(coords[j])
                x[j] = coords[j][end] + 1.0
            else
                a = coords[j][idx[j]]; b = coords[j][idx[j]+1]
                x[j] = (a+b)/2
            end
        end
        x
    end

    # enumerate all cells
    using Base.Iterators: product
    cells = collect(product((axes[i] for i in 1:n)...))
    m = length(cells)
    elements = collect(1:m)

    # Hasse edges: cover if tuples differ by +1 in exactly one coordinate
    edges = Tuple{Int,Int}[]
    idxmap = Dict{NTuple{n,Int},Int}()
    for (k, t) in enumerate(cells); idxmap[t]=k; end
    for (k, t) in enumerate(cells)
        for j in 1:n
            if t[j] < length(coords[j])
                t2 = ntuple(i -> i==j ? t[i]+1 : t[i], n)
                push!(edges, (k, idxmap[t2]))
            end
        end
    end
    P = PosetModules.Poset(elements, edges)

    # 3) For each cell, decide which Ups/Downs are active, form Φ_cell, and store im(Φ_cell)
    dims  = Int[]                        # dim H_cell
    basis = Vector{Matrix{Rational{BigInt}}}()  # image bases at each cell
    for t in cells
        x = cell_rep(t)
        activeU = findall(U -> contains(U, x), Ups)
        activeD = findall(D -> contains(D, x), Downs)
        if isempty(activeU) || isempty(activeD)
            push!(dims, 0); push!(basis, zeros(Rational{BigInt}, 0, 0)); continue
        end
        Φcell = map(Rational{BigInt}, Φ[activeD, activeU])
        B = colspace_basis(Φcell)
        push!(dims, size(B,2)); push!(basis, B)
    end

    # 4) Structure maps across covers: project death rows + include birth cols (as in finite case)
    emaps = Dict{Tuple{Int,Int}, Matrix{Rational{BigInt}}}()
    nb, nd = length(Ups), length(Downs)
    # Precompute membership signatures to know projection/inclusion pattern
    sigU = [BitVector([contains(U, cell_rep(t)) for U in Ups])   for t in cells]
    sigD = [BitVector([contains(D, cell_rep(t)) for D in Downs]) for t in cells]
    for (u,v) in P.hasse_edges
        Bu, Bv = basis[u], basis[v]
        ru, rv = size(Bu,2), size(Bv,2)
        # project deaths
        Du = findall(identity, collect(sigD[u])); Dv = findall(identity, collect(sigD[v]))
        P_d = zeros(Rational{BigInt}, length(Dv), length(Du))
        posDu = Dict{Int,Int}(); for (i,g) in enumerate(Du); posDu[g]=i; end
        for (i,g) in enumerate(Dv)
            if haskey(posDu,g); P_d[i, posDu[g]] = 1//1; end
        end
        # include births
        Uu = findall(identity, collect(sigU[u])); Uv = findall(identity, collect(sigU[v]))
        # (inclusion is implicit via column selection—already encoded by restricting columns)
        # transport Bu through projection and express in Bv
        S = zeros(Rational{BigInt}, rv, ru)
        for c in 1:ru
            w = P_d * Bu[:,c]
            if rv == 0
                # both sides zero -> map is zero
            else
                S[:,c] = solve_fullcolumn(Bv, w)
            end
        end
        emaps[(u,v)] = S
    end

    H = PModule(P, dims, emaps)
    return P, H
end

export BoxUpset, BoxDownset, encode_fringe_boxes

end # module
