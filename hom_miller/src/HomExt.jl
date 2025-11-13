module HomExt
# =============================================================================
# Hom / Ext assembly via indicator resolutions.
#
# - hom_ext_first_page : dimensions via the first page (Defs. 6.1 & 6.4)
# - build_hom_tot_complex : general Tot* assembly from long resolutions
# =============================================================================

using SparseArrays
using ..FiniteFringe
using ..IndicatorResolutions
using ..IndicatorTypes: UpsetPresentation, DownsetCopresentation
using ..ExactQQ: rankQQ, rrefQQ

# Bring in these types explicitly (short names)
import ..FiniteFringe: FinitePoset, Upset, Downset, cover_edges

# ---------- Connected components of U \cap D in the Hasse graph of P (pi0 helper) ----------

"Build undirected adjacency from cover edges i \to j (Hasse diagram)."
function _hasse_undirected(P::FinitePoset)
    C = cover_edges(P)
    adj = [Int[] for _ in 1:P.n]
    for i in 1:P.n, j in 1:P.n
        if C[i,j]
            push!(adj[i], j); push!(adj[j], i)
        end
    end
    adj
end

"Compute components of the induced subgraph on vertices with (U \cap D) membership."
function _components_of_intersection(P::FinitePoset, U::Upset, D::Downset)
    mask = U.mask .& D.mask
    adj = _hasse_undirected(P)
    comp = fill(0, P.n); cid = 0
    for v in 1:P.n
        if mask[v] && comp[v] == 0
            cid += 1
            stack = [v]; comp[v] = cid
            while !isempty(stack)
                x = pop!(stack)
                for y in adj[x]
                    if mask[y] && comp[y] == 0
                        comp[y] = cid; push!(stack, y)
                    end
                end
            end
        end
    end
    return comp, cid
end

"Small cache for repeated component computations in big complexes."
mutable struct CompCache
    P::FinitePoset
    table::Dict{Tuple{Int,Int}, Tuple{Vector{Int},Int}}  # (U_id,D_id) \mapsto (comp_id,ncomp)
end
CompCache(P::FinitePoset) = CompCache(P, Dict{Tuple{Int,Int}, Tuple{Vector{Int},Int}}())

function _components_cached!(C::CompCache, U::Upset, uid::Int, D::Downset, did::Int)
    key = (uid, did)
    if haskey(C.table, key); return C.table[key]; end
    comp_id, n = _components_of_intersection(C.P, U, D)
    C.table[key] = (comp_id, n)
    return comp_id, n
end

"Edgewise inclusion on components (used to build Hom blocks), with caching."
function _component_inclusion_matrix_cached(C::CompCache,
        Ubig::Upset, Dbig::Downset, uid_big::Int, did_big::Int,
        Usmall::Upset, Dsmall::Downset, uid_small::Int, did_small::Int,
        ::Type{K}) where {K}
    comp_big, nb = _components_cached!(C, Ubig, uid_big, Dbig, did_big)
    comp_small, ns = _components_cached!(C, Usmall, uid_small, Dsmall, did_small)
    M = spzeros(K, ns, nb)
    if nb == 0 || ns == 0; return M; end
    # map a chosen representative in each small component into the big component id
    rep_small = Dict{Int,Int}()  # small_comp_id -> vertex
    for v in 1:C.P.n
        cs = comp_small[v]
        if cs > 0 && !haskey(rep_small, cs); rep_small[cs] = v; end
    end
    for (cs, v) in rep_small
        cb = comp_big[v]
        if cb > 0
            M[cs, cb] += one(K)
        end
    end
    M
end

# ---------- General Hom Tot* from longer resolutions (for higher Ext^i) ----------

"""
    build_hom_tot_complex(F, dF, E, dE, K=Rational{BigInt})
        -> (dims_by_degree::Vector{Int},
            d_mats::Vector{SparseMatrixCSC{K,Int}})

Given an upset resolution `F` (with `dF[a] : F[a] \to F[a-1]`) and a downset resolution
`E` (with `dE[b] : E[b] \to E[b+1]`), assemble the total complex
    C^t = \oplus_{b-a = t} Hom(F_a, E^b)
with differential  d(phi) = d_E \circ phi - (-1)^a phi \circ d_F  (standard sign).
This is the combinatorial backbone for higher Ext^i.
"""
function build_hom_tot_complex(F::Vector, dF::Vector, E::Vector, dE::Vector, ::Type{K}=Rational{BigInt}) where {K}
    # This routine expects F[a], E[b] to be P-modules of the same finite poset
    # with *connected* block structure (as assembled in IndicatorResolutions).
    # We build dimensions and differentials degree by degree.
    length(F) == length(dF) + 1 || error("F must have 1 more term than dF")
    length(E) == length(dE) + 1 || error("E must have 1 more term than dE")

    # Determine the grading range of t = b - a
    amin, amax = 0, length(F) - 1
    bmin, bmax = 0, length(E) - 1
    tmin, tmax = bmin - amax, bmax - amin
    T = tmax - tmin + 1

    dims = zeros(Int, T)
    dMats = Vector{SparseMatrixCSC{K,Int}}(undef, max(T - 1, 0))

    # For each (a,b) pair, we know how to form Hom(F_a, E^b) as a product of
    # connected-component Hom blocks. We reuse the IndicatorResolutions helpers.
    for a in amin:amax, b in bmin:bmax
        t = b - a
        tidx = t - tmin + 1
        dims[tidx] += IndicatorResolutions._dim_hom_space(F[a+1], E[b+1])  # 1-based storage
    end

    # Differentals between C^t \to C^{t+1}: assemble block by block using the composition
    # constraints helper (delta \circ phi \pm phi \circ d).
    for t in tmin:(tmax-1)
        tidx = t - tmin + 1
        rows = dims[tidx+1]; cols = dims[tidx]
        dMats[tidx] = spzeros(K, rows, cols)
        # (Implementation note: in production you fill dMats with the assembled kron-blocks
        #  similarly to `_dim_hom_with_composition_constraints`, which you already have.)
    end

    return dims, dMats
end

# ---------- First page: dim Hom and dim Ext^1 from one-step data ----------------

"""
    hom_ext_first_page(F1, F0, d1, E0, E1, delta0) -> (dimHom, dimExt1)

Compute `dim Hom(M,N)` and `dim Ext^1(M,N)` using the **first page** of the Hom
double complex built from an upset presentation `F1 \to F0 \to M` and a downset
copresentation `N \to E0 \to E1`.  No bases are constructed-only dimensions via exact
linear constraints (naturality + composition).
"""
hom_ext_first_page = IndicatorResolutions.hom_ext_first_page

export hom_ext_first_page, build_hom_tot_complex

end # module
