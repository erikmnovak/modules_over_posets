module ZnEncoding

using LinearAlgebra
using SparseArrays

using ..CoreModules: QQ
using ..ExactQQ: colspaceQQ, solve_fullcolumnQQ
using ..FiniteFringe: FinitePoset, cover_edges
using ..IndicatorResolutions: PModule
using ..FlangeZn: Flange, in_flat, in_inj

# Build the finite grid poset on [a,b] subset Z^n, ordered coordinatewise.
# Returns (Q, coords) where coords[i] is an NTuple{n,Int}.
function grid_poset(a::Vector{Int}, b::Vector{Int})
    n = length(a)
    @assert length(b) == n
    lens = [b[i] - a[i] + 1 for i in 1:n]
    if any(lens .<= 0)
        error("grid_poset: invalid box")
    end

    # enumerate all lattice points in the box
    coords = NTuple{Int, Int}[]  # placeholder, will be retyped below
    coords_any = Vector{Any}()

    # Iterators.product over ranges
    ranges = [a[i]:b[i] for i in 1:n]
    for tup in Iterators.product(ranges...)
        push!(coords_any, tup)
    end
    # materialize as concrete NTuple{n,Int}
    coords = Vector{NTuple{n, Int}}(undef, length(coords_any))
    for i in 1:length(coords_any)
        coords[i] = coords_any[i]
    end

    N = length(coords)
    leq = BitMatrix(false, N, N)
    for i in 1:N
        gi = coords[i]
        for j in 1:N
            gj = coords[j]
            ok = true
            for k in 1:n
                if gi[k] > gj[k]
                    ok = false
                    break
                end
            end
            leq[i, j] = ok
        end
    end
    Q = FinitePoset(leq)
    return Q, coords
end

# Construct the PModule on the grid box induced by a flange presentation:
# M_g = im(Phi_g : F_g -> E_g), and maps are induced from the E-structure maps (projections).
#
# This returns a PModule over the finite grid poset. It is the object you want for Ext/Tor on that layer.
function pmodule_on_box(FG::Flange{QQ}; a::Vector{Int}, b::Vector{Int})
    Q, coords = grid_poset(a, b)
    N = length(coords)

    r = length(FG.E)
    c = length(FG.F)
    Phi = FG.Phi

    # For each vertex, compute:
    # - active injectives (rows) in E_g
    # - active flats (cols) in F_g
    # - B_g = basis matrix for im(Phi_g) inside E_g coordinates
    active_rows = Vector{Vector{Int}}(undef, N)
    B = Vector{Matrix{QQ}}(undef, N)
    dims = zeros(Int, N)

    for i in 1:N
        g = collect(coords[i])

        rows = Int[]
        for rr in 1:r
            if in_inj(FG.E[rr], g)
                push!(rows, rr)
            end
        end
        cols = Int[]
        for cc in 1:c
            if in_flat(FG.F[cc], g)
                push!(cols, cc)
            end
        end

        active_rows[i] = rows

        if isempty(rows) || isempty(cols)
            B[i] = zeros(QQ, length(rows), 0)
            dims[i] = 0
        else
            Phi_g = Phi[rows, cols]
            Bg = colspaceQQ(Phi_g)   # rows x dim(im)
            B[i] = Bg
            dims[i] = size(Bg, 2)
        end
    end

    # Build edge maps along cover edges in the grid poset using induced maps from E (projection).
    C = cover_edges(Q)
    edge_maps = Dict{Tuple{Int, Int}, Matrix{QQ}}()

    # Helper: build projection E_g -> E_h by selecting the common injective summands (rows_h subset rows_g).
    function projection_matrix(rows_g::Vector{Int}, rows_h::Vector{Int})
        Pg = length(rows_g)
        Ph = length(rows_h)
        P = zeros(QQ, Ph, Pg)
        # rows_* are sorted by construction
        j = 1
        for i in 1:Ph
            target = rows_h[i]
            while j <= Pg && rows_g[j] < target
                j += 1
            end
            if j > Pg || rows_g[j] != target
                error("pmodule_on_box: projection mismatch; expected rows_h subset rows_g")
            end
            P[i, j] = one(QQ)
        end
        return P
    end

    for u in 1:N
        for v in 1:N
            if C[u, v]
                # u < v is a cover edge in grid poset, so coordwise u <= v.
                # Injectives are downsets: active_rows[v] subset active_rows[u].
                rows_u = active_rows[u]
                rows_v = active_rows[v]

                du = dims[u]
                dv = dims[v]

                if dv == 0 || du == 0
                    edge_maps[(u, v)] = zeros(QQ, dv, du)
                    continue
                end

                Pu = projection_matrix(rows_u, rows_v)     # E_u -> E_v
                Im = Pu * B[u]                             # E_v x du
                X = solve_fullcolumnQQ(B[v], Im)          # dv x du
                edge_maps[(u, v)] = X
            end
        end
    end

    return PModule{QQ}(Q, Vector{Int}(dims), edge_maps)
end

end
