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
    # Cache components of U cap D.
    # IMPORTANT: key by the actual masks, not by local indices, since indices
    # are reused across resolution degrees.
    table::Dict{Tuple{BitVector,BitVector}, Tuple{Vector{Int},Int}}  # (U.mask, D.mask) -> (comp_id, ncomp)
end
CompCache(P::FinitePoset) =
    CompCache(P, Dict{Tuple{BitVector,BitVector},Tuple{Vector{Int},Int}}())


function _components_cached!(C::CompCache, U::Upset, uid::Int, D::Downset, did::Int)
    key = (U.mask, D.mask)
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
        if cb == 0
            error("component rep not found in big intersection; check containment direction")
        end
        M[cs, cb] += one(K)
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

Given an upset resolution F* and a downset resolution E*, assemble the total cochain complex
C^t = oplus_{a+b=t} Hom(F_a, E^b) with differential
d(phi) = (rho circ phi) + (-1)^b (phi circ delta)
(see Def. 6.1 for the two resolutions; totalization here is the standard one for Ext).
On a finite poset, pi0 is computed on the Hasse graph (Prop. 3.10).
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

    tmin, tmax = 0, A + B
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
        t = a + b; idx = t - tmin + 1
        offs[(a,b)] = dimsCt[idx]
        dimsCt[idx] += size_block(a,b)
    end

    # prepare differentials d^t : C^t to C^{t+1}
    dts = [spzeros(K, dimsCt[i+1], dimsCt[i]) for i in 1:(T-1)]

    # fill post- and pre-composition contributions
    for a in 0:A, b in 0:B
        U = U_by_a[a+1]; D = D_by_b[b+1]
        t = a + b; idx = t - tmin + 1
        src0 = offs[(a,b)]

        # post: Hom(F_a,E^b) \to Hom(F_a,E^{b+1}) via rho (if b < B)
        if b < B
            dst0 = offs[(a,b+1)]
            for (rowD1, D1j) in enumerate(D_by_b[b+2]), (colD0, D0j) in enumerate(D)
                coeff = dE[b+1][rowD1, colD0]
                if coeff != zero(K)
                    for (i, Uai) in enumerate(U)
                        Bmat = _component_inclusion_matrix_cached(cache,
                            Uai, D0j, i, colD0,
                            Uai, D1j, i, rowD1, K)

                        if nnz(Bmat) > 0
                            r0 = dst0 + _block_offset(cache, U_by_a, D_by_b, a, b+1, rowD1, i) + 1
                            c0 = src0 + _block_offset(cache, U_by_a, D_by_b, a, b,   colD0, i) + 1
                            _accum!(dts[idx], r0, c0, coeff * Bmat)
                        end
                    end
                end
            end
        end

        # pre: Hom(F_a,E^b) \to Hom(F_{a-1},E^b) via delta (if a >= 1) with sign -(-1)^a
        if a < A
            sign = isodd(b) ? -one(K) : one(K)     # (-1)^b
            dst0 = offs[(a+1,b)]

            Unexts = U_by_a[a+2]  # U_{a+1}
            for (rowUnext, Unext) in enumerate(Unexts), (colUcur, Ucur) in enumerate(U)
                coeff = dF[a+1][rowUnext, colUcur]  # delta_a : U_{a+1} -> U_a
                if coeff != zero(K)
                    for (j, Dbj) in enumerate(D)
                        # restriction: (Ucur cap Dbj) -> (Unext cap Dbj)
                        Bmat = _component_inclusion_matrix_cached(cache,
                            Ucur,  Dbj, colUcur,  j,
                            Unext, Dbj, rowUnext, j, K)

                        if nnz(Bmat) > 0
                            r0 = dst0 + _block_offset(cache, U_by_a, D_by_b, a+1, b, j, rowUnext) + 1
                            c0 = src0 + _block_offset(cache, U_by_a, D_by_b, a,   b, j, colUcur)  + 1
                            _accum!(dts[idx], r0, c0, (sign * coeff) * Bmat)
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
differentials dE (all over QQ), assemble the total cochain complex C^t = oplus_{a+b=t} Hom(F_a, E^b)
and return a dictionary mapping total degree t to dim H^t.

This densifies each sparse block to QQ for exact ranks.
"""
function ext_dims_via_resolutions(F::Vector{UpsetPresentation{QQ}},
                                  dF::Vector{SparseMatrixCSC{QQ,Int}},
                                  E::Vector{DownsetCopresentation{QQ}},
                                  dE::Vector{SparseMatrixCSC{QQ,Int}})
    dimsCt, dts = build_hom_tot_complex(F, dF, E, dE)
    
    A = length(F) - 1
    B = length(E) - 1
    tmin, tmax = 0, A + B

    dimsH = Dict{Int,Int}()
    for t in tmin:tmax
        i = t - tmin + 1
        dimC = dimsCt[i]
        r_next = (t < tmax) ? rankQQ(Matrix{QQ}(dts[i])) : 0
        r_prev = (t > tmin) ? rankQQ(Matrix{QQ}(dts[i-1])) : 0
        dimsH[t] = dimC - r_next - r_prev
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
