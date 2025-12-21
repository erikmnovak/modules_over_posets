module PLBackend
# =============================================================================
# Axis-aligned PL backend (no external deps).
#
# Shapes:
#   BoxUpset  : { x in R^n : x[i] >= ell[i] for all i }
#   BoxDownset: { x in R^n : x[i] <= u[i]   for all i }
#
# Encoding algorithm:
#   1) Collect all coordinate thresholds from ell's and u's.
#   2) Form the product-of-chains cell complex (rectangular grid cells).
#   3) Each cell gets a representative point x (strictly inside).
#   4) Y-signature of a cell is computed from contains(U_i, x) and the
#      complement for D_j. Cells with equal signatures are the uptight
#      regions (Defs. 4.12-4.17).
#   5) Build P on signatures by inclusion. Build Uhat,Dhat images on P.
#   6) Enforce monomial condition and return FringeModule + classifier pi.
#
# Complexity knobs:
#   - max_regions caps the number of regions (early stop if too large).
# =============================================================================

using ..FiniteFringe
using ..CoreModules: QQ

# ------------------------------- Shapes ---------------------------------------

"Axis-aligned upset: x[i] >= ell[i] for all i."
struct BoxUpset
    ell::Vector{Float64}
end

"Axis-aligned downset: x[i] <= u[i] for all i."
struct BoxDownset
    u::Vector{Float64}
end

"Membership predicates."
contains(U::BoxUpset, x::Vector{Float64})   = all(x[i] >= U.ell[i] for i in 1:length(x))
contains(D::BoxDownset, x::Vector{Float64}) = all(x[i] <= D.u[i]   for i in 1:length(x))

# ---------------------------- Encoding map type -------------------------------

"""
PLEncodingMapBoxes

Classifier pi : R^n -> P (regions) for the axis-aligned backend.
Stores:
  - the region signatures (y,z),
  - the cell representative interior points,
  - per-axis threshold arrays so we can place new x inside a cell
    and then map to the region by signature.

The 'locate' here computes signature(y,z) for x and looks up the
first region with the same signature. For speed, you can also snap
x to a cell and then use an index map, but we keep it simple.
"""
struct PLEncodingMapBoxes
    n::Int
    # thresholds per axis (sorted unique)
    coords::Vector{Vector{Float64}}
    # region signatures and representatives
    sig_y::Vector{BitVector}
    sig_z::Vector{BitVector}
    reps::Vector{Vector{Float64}}
    # original shapes to evaluate signatures
    Ups::Vector{BoxUpset}
    Downs::Vector{BoxDownset}
end

"Compute Y-signature (y,z) for a point x."
function _signature(x::Vector{Float64}, Ups::Vector{BoxUpset}, Downs::Vector{BoxDownset})
    m = length(Ups); r = length(Downs)
    y = falses(m); z = falses(r)
    for i in 1:m
        y[i] = contains(Ups[i], x)
    end
    for j in 1:r
        z[j] = !contains(Downs[j], x)  # complement
    end
    return y, z
end

"Locate by recomputing the signature and finding a matching region."
function locate(pi::PLEncodingMapBoxes, x::Vector{Float64})
    yx, zx = _signature(x, pi.Ups, pi.Downs)
    for (k, yk) in enumerate(pi.sig_y)
        if all(yx .== yk) && all(zx .== pi.sig_z[k]); return k; end
    end
    return 0
end

# ------------------------------- Encoding -------------------------------------

"""
encode_fringe_boxes(Ups, Downs, Phi; max_regions=200000)

Axis-aligned finite encoding with return (P, H_hat, pi).
"""
function encode_fringe_boxes(Ups::Vector{BoxUpset},
                             Downs::Vector{BoxDownset},
                             Phi_in::AbstractMatrix{QQ};
                             max_regions::Int=200_000)

    n = length(Ups) > 0 ? length(Ups[1].ell) : (length(Downs) > 0 ? length(Downs[1].u) : 0)

    # 1) thresholds per axis
    coords = [Float64[] for _ in 1:n]
    for i in 1:n
        for U in Ups; push!(coords[i], U.ell[i]); end
        for D in Downs; push!(coords[i], D.u[i]); end
        sort!(coords[i]); unique!(coords[i])
    end

    # 2) cells are products of segments between consecutive thresholds plus two unbounded sides
    #    We encode a cell by integer tuple (k1,...,kn) where k_i in 0..length(coords[i])
    #    and pick a point strictly inside that interval.
    function cell_rep(idx::NTuple{N,Int}) where {N}
        x = Vector{Float64}(undef, N)
        for j in 1:N
            if idx[j] == 0
                # below the first threshold
                x[j] = coords[j][1] - 1.0
            elseif idx[j] == length(coords[j])
                # above the last threshold
                x[j] = coords[j][end] + 1.0
            else
                a = coords[j][idx[j]]; b = coords[j][idx[j]+1]
                # strict interior point
                x[j] = (a + b) / 2.0
            end
        end
        x
    end

    # enumerate all cells, collecting DISTINCT region signatures.
    # max_regions is a cap on the number of DISTINCT signatures (regions),
    # not on the number of cells visited.
    axes = [collect(0:length(coords[i])) for i in 1:n]
    cells_iter = Base.Iterators.product((axes[i] for i in 1:n)...)

    seen = Set{Tuple{Tuple{Vararg{Bool}},Tuple{Vararg{Bool}}}}()
    sigY = BitVector[]              # unique y-signatures
    sigZ = BitVector[]              # unique z-signatures
    Reps = Vector{Float64}[]        # representative points (one per signature)

    for t in cells_iter
        x = cell_rep(t)
        y, z = _signature(x, Ups, Downs)
        key = (Tuple(y), Tuple(z))
        if !(key in seen)
            push!(seen, key)
            push!(sigY, y)
            push!(sigZ, z)
            push!(Reps, x)

            if length(sigY) > max_regions
                error("encode_fringe_boxes: requires more than max_regions=$(max_regions) regions; increase max_regions")
            end
        end
    end


    # 4) build uptight poset P by inclusion of signatures
    function uptight(sigY::Vector{BitVector}, sigZ::Vector{BitVector})
        rN = length(sigY)
        leq = falses(rN, rN)
        for i in 1:rN; leq[i,i] = true; end
        for i in 1:rN, j in 1:rN
            yi, zi = sigY[i], sigZ[i]
            yj, zj = sigY[j], sigZ[j]
            leq[i,j] = all(yi .<= yj) && all(zi .<= zj)
        end
        for k in 1:rN, i in 1:rN, j in 1:rN
            leq[i,j] = leq[i,j] || (leq[i,k] && leq[k,j])
        end
        return FiniteFringe.FinitePoset(leq)
    end
    P = uptight(sigY, sigZ)

    # 5) image upsets / downsets on P
    m = length(Ups); r = length(Downs)
    Uhat = Vector{FiniteFringe.Upset}(undef, m)
    Dhat = Vector{FiniteFringe.Downset}(undef, r)
    for i in 1:m
        mask = BitVector([ sigY[t][i] == 1 for t in 1:P.n ])
        Uhat[i] = FiniteFringe.upset_closure(P, mask)
    end
    for j in 1:r
        mask = BitVector([ sigZ[t][j] == 0 for t in 1:P.n ])
        Dhat[j] = FiniteFringe.downset_closure(P, mask)
    end

    # 6) monomial condition
    Phi = Matrix{QQ}(Phi_in)
    for j in 1:r, i in 1:m
        if !FiniteFringe.intersects(Uhat[i], Dhat[j])
            Phi[j,i] = zero(QQ)
        end
    end
    H = FiniteFringe.FringeModule{QQ}(P, Uhat, Dhat, Phi)

    # 7) return classifier
    pi = PLEncodingMapBoxes(n, coords, sigY, sigZ, Reps, Ups, Downs)
    return P, H, pi
end

export BoxUpset, BoxDownset, PLEncodingMapBoxes, locate, encode_fringe_boxes

end # module
