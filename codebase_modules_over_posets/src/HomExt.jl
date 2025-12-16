module HomExt
# -----------------------------------------------------------------------------
# Hom/Ext via indicator resolutions:
#   - pi0(U \cap D) computed combinatorially on the Hasse graph (Prop. 3.10)
#   - Hom double complex blocks Hom(F_a, E^b) and the total differential
#     d(phi) = rho \circ phi 0 (-1)^a phi \circ delta  (Def. 6.1)
# -----------------------------------------------------------------------------

using SparseArrays
using ..FiniteFringe: FinitePoset, Upset, Downset, cover_edges
using ..IndicatorTypes: UpsetPresentation, DownsetCopresentation
using ..CoreModules: QQ
using ..ExactQQ: rankQQ


# ---------- Hasse graph as undirected adjacency for connected components ----------
"Undirected adjacency lists of the Hasse diagram."
function _hasse_undirected(P::FinitePoset)
    C = cover_edges(P)                 # BitMatrix of covers i->j
    adj = [Int[] for _ in 1:P.n]
    for i in 1:P.n, j in 1:P.n
        if C[i,j]; push!(adj[i], j); push!(adj[j], i); end
    end
    adj
end

"Compute component id for vertices where U cap D holds; return (comp_id, ncomp)."
function _components_of_intersection(P::FinitePoset, U::Upset, D::Downset)
    mask = U.mask .& D.mask
    adj  = _hasse_undirected(P)
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
    comp, cid
end

# ---------- small cache so repeated component lookups are cheap -------------------
mutable struct CompCache
    P::FinitePoset
    table::Dict{Tuple{Int,Int}, Tuple{Vector{Int},Int}}  # (U_id,D_id) \mapsto (comp_id, ncomp)
end
CompCache(P::FinitePoset) = CompCache(P, Dict{Tuple{Int,Int},Tuple{Vector{Int},Int}}())

function _components_cached!(C::CompCache, U::Upset, uid::Int, D::Downset, did::Int)
    key = (uid, did)
    if haskey(C.table, key); return C.table[key]; end
    comp_id, n = _components_of_intersection(C.P, U, D)
    C.table[key] = (comp_id, n)
    return comp_id, n
end

"Component inclusion matrix for (U_big cap D_big) to (U_small cap D_small)."
function _component_inclusion_matrix_cached(C::CompCache,
        Ubig::Upset, Dbig::Downset, uid_big::Int, did_big::Int,
        Usmall::Upset, Dsmall::Downset, uid_small::Int, did_small::Int,
        ::Type{K}) where {K}
    comp_big, nb = _components_cached!(C, Ubig,   uid_big, Dbig,   did_big)
    comp_sml, ns = _components_cached!(C, Usmall, uid_small, Dsmall, did_small)
    M = spzeros(K, ns, nb)
    if nb==0 || ns==0; return M; end
    # choose one representative vertex per small component
    rep_small = Dict{Int,Int}()
    for v in 1:C.P.n
        cs = comp_sml[v]
        if cs>0 && !haskey(rep_small, cs); rep_small[cs] = v; end
    end
    for (cs, v) in rep_small
        cb = comp_big[v]
        if cb>0; M[cs, cb] += one(K); end
    end
    M
end

# ---------- offset within the basis of Hom(F_a, E^b) ------------------------------
# Basis is ordered by (D_j, U_i); within each pair we contribute one basis vector
# per connected component of U_i \cap D_j.
function _block_offset(cache::CompCache, U_by_a, D_by_b, a::Int, b::Int, j::Int, i::Int)
    off = 0
    for jj in 1:(j-1)
        for ii in 1:length(U_by_a[a+1])
            _, c = _components_cached!(cache, U_by_a[a+1][ii], ii, D_by_b[b+1][jj], jj)
            off += c
        end
    end
    for ii in 1:(i-1)
        _, c = _components_cached!(cache, U_by_a[a+1][ii], ii, D_by_b[b+1][j],  j)
        off += c
    end
    off
end

"Accumulate sparse block `B` into `S` at (r0, c0) (1-based)."
function _accum!(S::SparseMatrixCSC{K,Int}, r0::Int, c0::Int, B::SparseMatrixCSC{K,Int}) where {K}
    rows, cols, vals = findnz(B)
    for k in eachindex(vals)
        S[r0 + rows[k] - 1, c0 + cols[k] - 1] += vals[k]
    end
    S
end

# ---------- Hom Tot-complex -------------------------------------------------------
"""
    build_hom_tot_complex(F, dF, E, dE) -> (dims_by_degree, differentials)

Given an upset resolution F* and a downset resolution E*, assemble the total complex
C^t = oplus_{b-a=t} Hom(F_a, E^b) with differential
d(phi) = rho circ phi - (-1)^a phi circ delta
(Def. 6.1).  On a finite poset, pi0 is computed on the Hasse graph (Prop. 3.10).
"""
function build_hom_tot_complex(F::Vector{UpsetPresentation{K}},
                               dF::Vector{SparseMatrixCSC{K,Int}},
                               E::Vector{DownsetCopresentation{K}},
                               dE::Vector{SparseMatrixCSC{K,Int}}) where {K}
    A = length(F) - 1              # top degree on the F-side
    B = length(E) - 1              # top degree on the E-side
    P = F[1].P
    cache = CompCache(P)

    U_by_a = [f.U0 for f in F]
    D_by_b = [e.D0 for e in E]

    tmin, tmax = -A, B
    T = tmax - tmin + 1
    dimsCt = zeros(Int, T)
    offs = Dict{Tuple{Int,Int}, Int}()

    # size of Hom(F_a, E^b)
    function size_block(a::Int, b::Int)
        s = 0
        for j in 1:length(D_by_b[b+1]), i in 1:length(U_by_a[a+1])
            _, c = _components_cached!(cache, U_by_a[a+1][i], i, D_by_b[b+1][j], j)
            s += c
        end
        s
    end

    # compute block offsets per total degree
    for a in 0:A, b in 0:B
        t = b - a; idx = t - tmin + 1
        offs[(a,b)] = dimsCt[idx]
        dimsCt[idx] += size_block(a,b)
    end

    # prepare differentials d^t : C^t to C^{t+1}
    dts = [spzeros(K, dimsCt[i+1], dimsCt[i]) for i in 1:(T-1)]

    # fill post- and pre-composition contributions
    for a in 0:A, b in 0:B
        U = U_by_a[a+1]; D = D_by_b[b+1]
        t = b - a; idx = t - tmin + 1
        src0 = offs[(a,b)]

        # post: Hom(F_a,E^b) \to Hom(F_a,E^{b+1}) via rho (if b < B)
        if b < B
            dst0 = offs[(a,b+1)]
            for (rowD1, D1j) in enumerate(D_by_b[b+2]), (colD0, D0j) in enumerate(D)
                if dE[b+1][rowD1, colD0] != zero(K)
                    for (i, Uai) in enumerate(U)
                        Bmat = _component_inclusion_matrix_cached(cache, Uai, D0j, i, colD0,
                                                                   Uai, D1j, i, rowD1, K)
                        if nnz(Bmat) > 0
                            r0 = dst0 + _block_offset(cache, U_by_a, D_by_b, a, b+1, rowD1, i) + 1
                            c0 = src0 + _block_offset(cache, U_by_a, D_by_b, a, b,   colD0, i) + 1
                            _accum!(dts[idx], r0, c0, Bmat)
                        end
                    end
                end
            end
        end

        # pre: Hom(F_a,E^b) \to Hom(F_{a-1},E^b) via delta (if a >= 1) with sign -(-1)^a
        if a >= 1
            sign = isodd(a) ? one(K) : -one(K)  # -(-1)^a
            dst0 = offs[(a-1,b)]
            for (rowUprev, Uprev) in enumerate(U_by_a[a]), (colUcur, Ucur) in enumerate(U)
                if dF[a][colUcur, rowUprev] != zero(K)
                    for (j, Dbj) in enumerate(D)
                        Bmat = _component_inclusion_matrix_cached(cache, Ucur, Dbj, colUcur, j,
                                                                   Uprev, Dbj, rowUprev, j, K)
                        if nnz(Bmat) > 0
                            r0 = dst0 + _block_offset(cache, U_by_a, D_by_b, a-1, b, j, rowUprev) + 1
                            c0 = src0 + _block_offset(cache, U_by_a, D_by_b, a,   b, j, colUcur)  + 1
                            _accum!(dts[idx], r0, c0, sign .* Bmat)
                        end
                    end
                end
            end
        end
    end

    return dimsCt, dts
end



"""
    ext_dims_via_resolutions(F, dF, E, dE) -> Dict{Int,Int}

Given an upset resolution F* with differentials dF and a downset resolution E* with
differentials dE (all over QQ), assemble the total cochain complex C^t = oplus_{b-a=t} Hom(F_a, E^b)
and return a dictionary mapping total degree t to dim H^t.

This densifies each sparse block to QQ for exact ranks.
"""
function ext_dims_via_resolutions(F::Vector{UpsetPresentation{QQ}},
                                  dF::Vector{SparseMatrixCSC{QQ,Int}},
                                  E::Vector{DownsetCopresentation{QQ}},
                                  dE::Vector{SparseMatrixCSC{QQ,Int}})
    dimsCt, dts = build_hom_tot_complex(F, dF, E, dE)
    tmin = 1 - length(F)                      # same convention as build_hom_tot_complex
    tmax = length(E) - 1
    ranks = Dict{Int,Int}()
    for (i, d) in enumerate(dts)
        t = tmin + (i - 1)
        ranks[t] = ExactQQ.rankQQ(Matrix{QQ}(d))  # rank(d^t)
    end
    # Helper to read rank(d^{t-1}) safely
    rank_prev(t) = haskey(ranks, t - 1) ? ranks[t - 1] : 0
    dimsH = Dict{Int,Int}()
    for (s, dimC) in enumerate(dimsCt)              # s indexes t = tmin + (s-1)
        t = tmin + (s - 1)
        r_curr = haskey(ranks, t) ? ranks[t] : 0
        r_prev = rank_prev(t)
        dimsH[t] = dimC - r_curr - r_prev
    end
    return dimsH
end


# Return the number of connected components of U \cap D in the Hasse graph of P.
function pi0_count(P::FinitePoset, U::Upset, D::Downset)
    _, ncomp = _components_of_intersection(P, U, D)
    return ncomp
end

export build_hom_tot_complex, ext_dims_via_resolutions, pi0_count 


end # module
