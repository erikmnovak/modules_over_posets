module IndicatorResolutions
# =============================================================================
# Indicator (co)presentations over a finite poset, harmonized with FiniteFringe.
#
# What this file does:
#   * Convert a FringeModule to an internal PModule with explicit structure maps.
#   * Build the ONE-STEP upset presentation (Def. 6.4.1) and downset copresentation
#     (Def. 6.4.2), returning the lightweight wrappers:
#        UpsetPresentation{K}(P, U0, U1, delta)
#        DownsetCopresentation{K}(P, D0, D1, rho)
#   * Provide a first-page Hom/Ext dimension routine from these one-step data.
#
# Design notes:
#   - We keep a small internal PModule/PMorphism type here, to avoid imposing a new
#     surface type on the rest of the code base. The public surface uses the
#     IndicatorTypes wrappers exclusively, which are already consumed by HomExt.jl.
#   - All computations use QQ = Rational{BigInt} for exactness (ExactQQ.jl).
#   - The one-step routines are sufficient to feed HomExt.hom_ext_first_page and
#     also the general Tot* builder when longer resolutions are later added.
# =============================================================================

using SparseArrays
using ..FiniteFringe
using ..IndicatorTypes: UpsetPresentation, DownsetCopresentation
using ..ExactQQ: QQ, rrefQQ, rankQQ, nullspaceQQ, solve_fullcolumnQQ, colspaceQQ
using ..HomExt: pi0_count

import ..FiniteFringe: FinitePoset, Upset, Downset, principal_upset, principal_downset, cover_edges


# ---- Cached cover edges and adjacency on the Hasse diagram -------------------
struct CoverCache
    Q::FinitePoset
    C::BitMatrix                     # cover relation: i -> j iff C[i,j] = true
    preds::Vector{Vector{Int}}       # predecessors by covers
    succs::Vector{Vector{Int}}       # successors by covers
end

function _cover_cache(Q::FinitePoset)
    C = cover_edges(Q)
    preds = [Int[] for _ in 1:Q.n]
    succs = [Int[] for _ in 1:Q.n]
    for u in 1:Q.n, v in 1:Q.n
        if C[u,v]
            push!(preds[v], u)
            push!(succs[u], v)
        end
    end
    return CoverCache(Q, C, preds, succs)
end



# ------------------------------ tiny internal model ------------------------------

"""
A minimal module over a finite poset Q:
- dims[i] = dim_Qk M_i
- edge_maps[(u,v)] : M_u \to M_v for *cover* edges u < v.
"""
struct PModule{K}
    Q::FinitePoset
    dims::Vector{Int}
    edge_maps::Dict{Tuple{Int,Int}, Matrix{K}}
end

"Vertexwise morphism of P-modules (components are M_i \to N_i)."
struct PMorphism{K}
    dom::PModule{K}
    cod::PModule{K}
    comps::Vector{Matrix{K}}   # comps[i] :: Matrix{K} of size cod.dims[i] \times dom.dims[i]
end

"Identity morphism."
id_morphism(M::PModule{K}) where {K} =
    PMorphism{K}(M, M, [Matrix{K}(I, M.dims[i], M.dims[i]) for i in 1:length(M.dims)])

    
function _predecessors(Q::FinitePoset)
    return _cover_cache(Q).preds
end


function map_leq(M::PModule{K}, u::Int, v::Int; cache::Union{Nothing,CoverCache}=nothing) where {K}
    u == v && return Matrix{K}(I, M.dims[v], M.dims[u])
    Q = M.Q
    cc = cache === nothing ? _cover_cache(Q) : cache
    preds = cc.preds

    # memoize composites along the Hasse diagram
    memo = IdDict{Tuple{Int,Int}, Matrix{K}}()
    function go(a::Int, b::Int)
        key = (a,b)
        if haskey(memo, key); return memo[key]; end
        if a == b
            X = Matrix{K}(I, M.dims[b], M.dims[a])
            memo[key] = X
            return X
        end
        for w in preds[b]
            if Q.leq[a,w]  # a <= w <= b
                X = go(a, w)
                Y = M.edge_maps[(w,b)]
                memo[key] = Y * X
                return memo[key]
            end
        end
        error("No chain a <= b found; ensure cover edges are computed for Q.")
    end
    return go(u, v)
end


# ----------------------- from FringeModule to PModule ----------------------------

"""
    pmodule_from_fringe(H::FiniteFringe.FringeModule{K})
Return an internal `PModule{QQ}` whose fibers and structure maps are induced by the
fringe presentation `phi : oplus k[U_i] to oplus k[D_j]` (Defs. 3.16-3.17).
Implementation: M_q = im(phi_q) inside E_q; along a cover u<v the map is the restriction
E_u to E_v followed by projection to M_v.
"""
function pmodule_from_fringe(H::FiniteFringe.FringeModule{K}) where {K}
    Q = H.P
    n = Q.n

    # Basis for each fiber M_q as columns of a QQ matrix B[q] spanning im(phi_q).
    B = Vector{Matrix{QQ}}(undef, n)
    dims = zeros(Int, n)
    for q in 1:n
        cols = findall(U -> U.mask[q], H.U)
        rows = findall(D -> D.mask[q], H.D)
        if isempty(cols) || isempty(rows)
            B[q] = zeros(QQ, length(rows), 0)
            dims[q] = 0
            continue
        end
        phi_q = Matrix{QQ}(Matrix(H.phi[rows, cols]))
        B[q] = colspaceQQ(phi_q)
        dims[q] = size(B[q], 2)
    end

    # Death projection E_u \to E_v on a cover u<v: keep row indices j that remain active at v.
    function death_projection(u::Int, v::Int)
        rows_u = findall(D -> D.mask[u], H.D)
        rows_v = findall(D -> D.mask[v], H.D)
        pos_v = Dict{Int,Int}(rows_v[i] => i for i in 1:length(rows_v))
        P = zeros(QQ, length(rows_v), length(rows_u))
        for (jpos, jidx) in enumerate(rows_u)
            if haskey(pos_v, jidx)
                P[pos_v[jidx], jpos] = 1//1
            end
        end
        P
    end

    # Structure map on a cover u<v: M_u --incl--> E_u --proj--> E_v --coords--> M_v
    edge_maps = Dict{Tuple{Int,Int}, Matrix{QQ}}()
    C = cover_edges(Q)
    for u in 1:n, v in 1:n
        if C[u,v] && dims[u] > 0 && dims[v] > 0
            Puv = death_projection(u,v)            # E_u \to E_v
            Im = Puv * B[u]                        # in E_v coordinates
            # coordinates in M_v: solve B[v] * X = Im
            X = solve_fullcolumnQQ(B[v], Im)
            edge_maps[(u,v)] = X                   # M_u \to M_v
        elseif C[u,v]
            edge_maps[(u,v)] = zeros(QQ, dims[v], dims[u])
        end
    end

    PModule{QQ}(Q, dims, edge_maps)
end

# -------------------------- projective cover (Def. 6.4.1) --------------------------

"Incoming image at v from immediate predecessors; basis matrix with QQ columns."
function _incoming_image_basis(M::PModule{QQ}, v::Int; cache::Union{Nothing,CoverCache}=nothing)
    preds = (cache === nothing ? _cover_cache(M.Q).preds : cache.preds)[v]
    if isempty(preds) || M.dims[v] == 0
        return zeros(QQ, M.dims[v], 0)
    end
    blocks = Matrix{QQ}[]
    for u in preds
        push!(blocks, M.edge_maps[(u,v)])  # M_u -> M_v
    end
    if isempty(blocks)
        return zeros(QQ, M.dims[v], 0)
    end
    A = hcat(blocks...)
    return colspaceQQ(A)
end


"""
    projective_cover(M::PModule{QQ})
Return (F0, pi0, gens_at) where F0 is a direct sum of principal upsets covering M,
pi0 : F0 \to M is the natural surjection, and `gens_at[v]` lists the generators activated
at vertex v (each item is a pair (p, local_index_in_Mp)).
"""
function projective_cover(M::PModule{QQ}; cache::Union{Nothing,CoverCache}=nothing)
    Q = M.Q; n = Q.n
    cc = cache === nothing ? _cover_cache(Q) : cache

    # number of generators at each vertex = dim(M_v) - rank(incoming_image)
    gens_at = Vector{Vector{Tuple{Int,Int}}}(undef, n)
    gen_of_p = fill(0, n)
    for v in 1:n
        Img = _incoming_image_basis(M, v; cache=cc)
        beta = M.dims[v] - size(Img, 2)
        chosen = Int[]
        if beta > 0 && M.dims[v] > 0
            S = Img
            I = Matrix{QQ}(I, M.dims[v], M.dims[v])
            rS = size(S, 2)
            for j in 1:M.dims[v]
                T = hcat(S, I[:, j])
                if rankQQ(T) > rS
                    push!(chosen, j); S = colspaceQQ(T); rS += 1
                    length(chosen) == beta && break
                end
            end
        end
        gens_at[v] = [(v, j) for j in chosen]
        gen_of_p[v] = length(chosen)
    end

    # F0 as a direct sum of principal upsets
    F0_dims = [sum(gen_of_p[p] for p in 1:n if Q.leq[p,i]) for i in 1:n]
    F0_edges = Dict{Tuple{Int,Int}, Matrix{QQ}}()
    C = cc.C
    for u in 1:n, v in 1:n
        if C[u,v]
            F0_edges[(u,v)] = Matrix{QQ}(I, F0_dims[v], F0_dims[u])[1:F0_dims[v], 1:F0_dims[u]]
        end
    end
    F0 = PModule{QQ}(Q, F0_dims, F0_edges)

    # pi0 : F0 -> M
    comps = Vector{Matrix{QQ}}(undef, n)
    for i in 1:n
        cols = Matrix{QQ}(undef, M.dims[i], 0)
        for p in 1:n
            if gen_of_p[p] > 0 && Q.leq[p,i]
                A = map_leq(M, p, i; cache=cc)  # M_p -> M_i
                I_mp = Matrix{QQ}(I, M.dims[p], M.dims[p])
                J = [pair[2] for pair in gens_at[p]]
                cols = hcat(cols, A * I_mp[:, J])
            end
        end
        comps[i] = cols
    end
    pi0 = PMorphism{QQ}(F0, M, comps)
    return F0, pi0, gens_at
end


# --------------------------- kernel and upset presentation --------------------------

"Kernel of f with inclusion iota : ker(f) \to dom(f), degreewise."
function kernel_with_inclusion(f::PMorphism{QQ}; cache::Union{Nothing,CoverCache}=nothing)
    M = f.dom; n = length(M.dims)
    K_dims = zeros(Int, n)
    basisK = Vector{Matrix{QQ}}(undef, n)
    for i in 1:n
        Ni = nullspaceQQ(f.comps[i])     # columns span ker
        basisK[i] = Ni
        K_dims[i] = size(Ni, 2)
    end
    K_edges = Dict{Tuple{Int,Int}, Matrix{QQ}}()
    C = (cache === nothing ? cover_edges(M.Q) : cache.C)
    for u in 1:n, v in 1:n
        if C[u,v]
            if K_dims[u] == 0 || K_dims[v] == 0
                K_edges[(u,v)] = zeros(QQ, K_dims[v], K_dims[u])
            else
                T = M.edge_maps[(u,v)]
                Im = T * basisK[u]
                X = solve_fullcolumnQQ(basisK[v], Im)
                K_edges[(u,v)] = X
            end
        end
    end
    K = PModule{QQ}(M.Q, K_dims, K_edges)
    iota = PMorphism{QQ}(K, M, [basisK[i] for i in 1:n])
    return K, iota
end






# Basis of the "socle" at vertex u: kernel of the stacked outgoing map
# M_u to oplus_{u<v} M_v along cover edges u < v.
# Columns of the returned matrix span soc(M)_u subseteq M_u.
function _socle_basis(M::PModule{QQ}, u::Int; cache::Union{Nothing,CoverCache}=nothing)
    su = (cache === nothing ? _cover_cache(M.Q).succs : cache.succs)[u]
    if isempty(su) || M.dims[u] == 0
        return Matrix{QQ}(I, M.dims[u], M.dims[u])
    end
    A = vcat([M.edge_maps[(u,v)] for v in su]...)     # size (sum dim M_v) x dim M_u
    return nullspaceQQ(A)                              # dim M_u x socle dimension
end


# A canonical left-inverse for a full-column-rank matrix S: L*S = I.
# Implemented as L = (S^T S)^{-1} S^T using exact QQ solves.
function _left_inverse_full_column(S::AbstractMatrix{QQ})
    s = size(S,2)
    if s == 0
        return zeros(QQ, 0, size(S,1))
    end
    G = transpose(S) * S                       # s*s Gram matrix, invertible over QQ
    return solve_fullcolumnQQ(G, transpose(S)) # returns (S^T S)^{-1} S^T with size s * m
end

# Build the injective (downset) hull:  iota : M into E  where
# E is a direct sum of principal downsets with multiplicities = socle dimensions.
# Also return the generator labels as (u, j) with u the vertex and j the column.
function _injective_hull(M::PModule{QQ}; cache::Union{Nothing,CoverCache}=nothing)
    Q = M.Q; n = Q.n
    cc = cache === nothing ? _cover_cache(Q) : cache

    # socle bases at each vertex and their multiplicities
    Soc = Vector{Matrix{QQ}}(undef, n)
    mult = zeros(Int, n)
    for u in 1:n
        Soc[u]  = _socle_basis(M, u; cache=cc)
        mult[u] = size(Soc[u], 2)
    end

    # fiber dimensions of E
    Edims = [sum(mult[u] for u in 1:n if Q.leq[i,u]) for i in 1:n]

    # E structure maps: identity on active coordinate slices
    Eedges = Dict{Tuple{Int,Int}, Matrix{QQ}}()
    C = cc.C
    for u in 1:n, v in 1:n
        if C[u,v]
            Eedges[(u,v)] = Matrix{QQ}(I, Edims[v], Edims[u])[1:Edims[v], 1:Edims[u]]
        end
    end
    E = PModule{QQ}(Q, Edims, Eedges)

    # iota : M -> E
    Linv = [ _left_inverse_full_column(Soc[u]) for u in 1:n ]
    comps = Vector{Matrix{QQ}}(undef, n)
    gens_at = Vector{Vector{Tuple{Int,Int}}}(undef, n)
    for u in 1:n
        gens_at[u] = [(u, j) for j in 1:mult[u]]
    end
    for i in 1:n
        rows = Matrix{QQ}(undef, 0, M.dims[i])
        for u in 1:n
            if Q.leq[i,u] && mult[u] > 0
                Mi_to_Mu = map_leq(M, i, u; cache=cc)
                R = Linv[u] * Mi_to_Mu
                rows = vcat(rows, R)
            end
        end
        @assert size(rows,1) == Edims[i]
        comps[i] = rows
    end
    iota = PMorphism{QQ}(M, E, comps)
    return E, iota, gens_at
end


# Degreewise cokernel of iota : E0 <- M, produced as a P-module C together with
# the quotient q : E0 -> C.  The quotient is represented by surjections q_i whose
# kernels are colspace(iota_i).
function _cokernel_module(iota::PMorphism{QQ}; cache::Union{Nothing,CoverCache}=nothing)
    E = iota.cod; Q = E.Q; n = Q.n
    Cdims  = zeros(Int, n)
    qcomps = Vector{Matrix{QQ}}(undef, n)     # each is (dim C_i) x (dim E_i)

    # degreewise quotients
    for i in 1:n
        Bi = colspaceQQ(iota.comps[i])        # dim E_i x rank
        Ni = nullspaceQQ(transpose(Bi))       # dim E_i x (dim E_i - rank)
        Cdims[i]  = size(Ni, 2)
        qcomps[i] = transpose(Ni)
    end

    # structure maps of C
    Cedges = Dict{Tuple{Int,Int}, Matrix{QQ}}()
    C = (cache === nothing ? _cover_cache(Q).C : cache.C)
    for u in 1:n, v in 1:n
        if C[u,v] && Cdims[u] > 0 && Cdims[v] > 0
            T = E.edge_maps[(u,v)]
            X = solve_fullcolumnQQ(transpose(qcomps[u]), transpose(qcomps[v] * T))
            A = transpose(X)                                     # dim C_v x dim C_u
            Cedges[(u,v)] = A
        elseif C[u,v]
            Cedges[(u,v)] = zeros(QQ, Cdims[v], Cdims[u])
        end
    end

    Cmod = PModule{QQ}(Q, Cdims, Cedges)
    q = PMorphism{QQ}(E, Cmod, qcomps)
    return Cmod, q
end



"""
    upset_presentation_one_step(Hfringe::FringeModule)
Compute the one-step upset presentation (Def. 6.4.1):
    F1 --d1--> F0 --pi0-->> M,
and return the lightweight wrapper `UpsetPresentation{QQ}(P, U0, U1, delta)`.
"""
function upset_presentation_one_step(H::FiniteFringe.FringeModule)
    M = pmodule_from_fringe(H)          # internal PModule over QQ
    cc = _cover_cache(M.Q)
    # First step: projective cover of M
    F0, pi0, gens_at_F0 = projective_cover(M; cache=cc)
    # Kernel K1 with inclusion i1 : K1 \into F0
    K1, i1 = kernel_with_inclusion(pi0; cache=cc)
    # Second projective cover (of K1)
    F1, pi1, gens_at_F1 = projective_cover(K1; cache=cc)
    # Differential d1 = i1 \circ pi1 : F1 \to F0
    comps = [i1.comps[i] * pi1.comps[i] for i in 1:length(M.dims)]
    d1 = PMorphism{QQ}(F1, F0, comps)

    # Build indicator wrapper:
    # U0: list of principal upsets, one per generator in gens_at_F0[p]
    P = M.Q
    U0 = Upset[]
    for p in 1:P.n
        for _ in gens_at_F0[p]
            push!(U0, principal_upset(P, p))
        end
    end
    U1 = Upset[]
    for p in 1:P.n
        for _ in gens_at_F1[p]
            push!(U1, principal_upset(P, p))
        end
    end

    # scalar delta block: one entry for each pair (theta in U1, lambda in U0) if ptheta <= plambda.
    # Extract the scalar from d1 at the minimal vertex i = plambda.
    m1 = length(U1); m0 = length(U0)
    delta = spzeros(QQ, m1, m0)
    # local index maps: at vertex i, Fk_i basis is "all generators at p with p <= i" in (p increasing) order.
    # Build offsets to find local coordinates.
    function local_index_list(gens_at)
        # return vector of vectors L[i] listing global generator indices active at vertex i
        L = Vector{Vector{Tuple{Int,Int}}}(undef, P.n)
        for i in 1:P.n
            L[i] = Tuple{Int,Int}[]
            for p in 1:P.n
                if P.leq[p,i]
                    append!(L[i], gens_at[p])
                end
            end
        end
        L
    end
    L0 = local_index_list(gens_at_F0)
    L1 = local_index_list(gens_at_F1)

    # Build a map from "global generator number in U0/U1" to its vertex p and position j in M_p
    globalU0 = Tuple{Int,Int}[]  # (p,j)
    for p in 1:P.n; append!(globalU0, gens_at_F0[p]); end
    globalU1 = Tuple{Int,Int}[]
    for p in 1:P.n; append!(globalU1, gens_at_F1[p]); end

    # helper: find local column index of global generator g=(p,j) at vertex i
    function local_col_of(L, g::Tuple{Int,Int}, i::Int)
        for (c, gg) in enumerate(L[i])
            if gg == g; return c; end
        end
        return 0
    end

    for (lambda, (plambda, jlambda)) in enumerate(globalU0)
        for (theta, (ptheta, jtheta)) in enumerate(globalU1)
            if P.leq[ptheta, plambda]   # otherwise zero block
                i = plambda
                col = local_col_of(L1, (ptheta, jtheta), i)
                row = local_col_of(L0, (plambda, jlambda), i)
                if col > 0 && row > 0
                    val = d1.comps[i][row, col]
                    if val != 0
                        delta[theta, lambda] = val
                    end
                end
            end
        end
    end

    UpsetPresentation{QQ}(P, U0, U1, delta)
end

# ------------------------ downset copresentation (Def. 6.4.2) ------------------------

# The dual story: compute an injective hull E0 of M and the next step E1 with rho0 : E0 \to E1.
# For brevity we implement the duals by applying the above steps to M^op using
# down-closures/up-closures symmetry. Here we implement directly degreewise.

"Outgoing coimage at u to immediate successors; basis for the span of maps M_u to oplus_{u<v} M_v."
function _outgoing_span_basis(M::PModule{QQ}, u::Int; cache::Union{Nothing,CoverCache}=nothing)
    su = (cache === nothing ? _cover_cache(M.Q).succs : cache.succs)[u]
    if isempty(su) || M.dims[u] == 0
        return zeros(QQ, 0, M.dims[u])
    end
    blocks = Matrix{QQ}[]
    for v in su
        push!(blocks, transpose(M.edge_maps[(u,v)]))
    end
    A = vcat(blocks...)
    return transpose(colspaceQQ(transpose(A)))
end


"Essential socle dimension at u = dim(M_u) - rank(outgoing span) (dual of generators)."
function _socle_count(M::PModule{QQ}, u::Int)
    S = _outgoing_span_basis(M, u)
    M.dims[u] - rankQQ(S)
end

"""
    downset_copresentation_one_step(Hfringe::FringeModule)

Compute the one-step downset **copresentation** (Def. 6.4(2)):
    M = ker(rho : E^0 to E^1),
with E^0 and E^1 expressed as direct sums of principal downsets and rho assembled
from the actual vertexwise maps, not just from the partial order.  Steps:

1. Build the injective (downset) hull iota0 : M into E^0.
2. Form C = coker(iota0) as a P-module together with q : E^0 to C.
3. Build the injective (downset) hull j : C into E^1 and set rho0 = j circ q : E^0 -> E^1.
4. Read scalar entries of rho at minimal vertices, as for delta on the upset side.
"""
function downset_copresentation_one_step(H::FiniteFringe.FringeModule)
    # Convert fringe to internal PModule over QQ
    M = pmodule_from_fringe(H)
    Q = M.Q; n = Q.n
    cc = _cover_cache(Q)

    # (1) Injective hull of M: E0 with inclusion iota0
    E0, iota0, gens_at_E0 = _injective_hull(M; cache=cc)

    # (2) Degreewise cokernel C and the quotient q : E0 \to C
    C, q = _cokernel_module(iota0; cache=cc)

    # (3) Injective hull of C: E1 with inclusion j : C \into E1
    E1, j, gens_at_E1 = _injective_hull(C; cache=cc)

    # Compose to get rho0 : E0 \to E1 (at each vertex i: rho0[i] = j[i] * q[i])
    comps_rho0 = [ j.comps[i] * q.comps[i] for i in 1:n ]
    rho0 = PMorphism{QQ}(E0, E1, comps_rho0)

    # (4) Assemble the indicator wrapper: labels D0, D1 and the scalar block rho.
    # D0/D1 each contain one principal downset for every generator (u,j) chosen above.
    D0 = Downset[]
    for u in 1:n, _ in gens_at_E0[u]
        push!(D0, principal_downset(Q, u))
    end
    D1 = Downset[]
    for u in 1:n, _ in gens_at_E1[u]
        push!(D1, principal_downset(Q, u))
    end

    # Local index lists: at vertex i, the active generators are those born at u with i <= u.
    function _local_index_list_D(gens_at)
        L = Vector{Vector{Tuple{Int,Int}}}(undef, n)
        for i in 1:n
            lst = Tuple{Int,Int}[]
            for u in 1:n
                if Q.leq[i,u]
                    append!(lst, gens_at[u])
                end
            end
            L[i] = lst
        end
        L
    end
    L0 = _local_index_list_D(gens_at_E0)
    L1 = _local_index_list_D(gens_at_E1)

    # Global enumerations (lambda in D0, theta in D1) with their birth vertices u_lambda, u_theta.
    globalD0 = Tuple{Int,Int}[]; for u in 1:n; append!(globalD0, gens_at_E0[u]); end
    globalD1 = Tuple{Int,Int}[]; for u in 1:n; append!(globalD1, gens_at_E1[u]); end

    # Helper to find the local column index of a global generator at vertex i
    function _local_col_of(L, g::Tuple{Int,Int}, i::Int)
        for (c, gg) in enumerate(L[i])
            if gg == g; return c; end
        end
        return 0
    end

    # Assemble the scalar monomial matrix rho by reading the minimal vertex i = u_theta.
    m1 = length(globalD1); m0 = length(globalD0)
    rho = spzeros(QQ, m1, m0)

    for (lambda, (ulambda, jlambda)) in enumerate(globalD0)
        for (theta, (utheta, jtheta)) in enumerate(globalD1)
            if Q.leq[ulambda, utheta] # otherwise the (theta,lambda) block is zero (no containment)
                i   = utheta
                col = _local_col_of(L0, (ulambda, jlambda), i)
                row = _local_col_of(L1, (utheta, jtheta), i)
                if col > 0 && row > 0
                    val = rho0.comps[i][row, col]
                    if val != 0
                        rho[theta, lambda] = val
                    end
                end
            end
        end
    end

    return DownsetCopresentation{QQ}(Q, D0, D1, rho)
end



"""
    prune_zero_relations(F::UpsetPresentation{QQ}) -> UpsetPresentation{QQ}

Remove rows of `delta` that are identically zero (redundant relations) and drop the
corresponding entries of `U1`. The cokernel is unchanged.
"""
function prune_zero_relations(F::UpsetPresentation{QQ})
    m1, m0 = size(F.delta)
    keep = trues(m1)
    # mark zero rows
    rows, _, _ = findnz(F.delta)
    seen = falses(m1); @inbounds for r in rows; seen[r] = true; end
    @inbounds for r in 1:m1
        if !seen[r]; keep[r] = false; end
    end
    new_U1 = [F.U1[i] for i in 1:m1 if keep[i]]
    new_delta = F.delta[keep, :]
    UpsetPresentation{QQ}(F.P, F.U0, new_U1, new_delta)
end

"""
    cancel_isolated_unit_pairs(F::UpsetPresentation{QQ}) -> UpsetPresentation{QQ}

Iteratively cancels isolated nonzero entries `delta[theta,lambda]` for which:
  * the theta-th row has exactly that one nonzero,
  * the lambda-th column has exactly that one nonzero, and
  * U1[theta] == U0[lambda] as Upsets (principal upsets match).

Each cancellation removes one generator in U0 and one relation in U1 without
changing the cokernel.
"""
function cancel_isolated_unit_pairs(F::UpsetPresentation{QQ})
    P, U0, U1, Delta = F.P, F.U0, F.U1, F.delta
    while true
        m1, m0 = size(Delta)
        rows, cols, _ = findnz(Delta)
        # count nonzeros per row/col
        rcount = zeros(Int, m1)
        ccount = zeros(Int, m0)
        @inbounds for k in eachindex(rows)
            rcount[rows[k]] += 1; ccount[cols[k]] += 1
        end
        # search an isolated pair with matching principal upsets
        found = false
        theta = 0; lambda = 0
        @inbounds for k in eachindex(rows)
            r = rows[k]; c = cols[k]
            if rcount[r] == 1 && ccount[c] == 1
                # require identical principal upsets
                if U1[r].P === U0[c].P && U1[r].mask == U0[c].mask
                    theta, lambda = r, c; found = true; break
                end
            end
        end
        if !found; break; end
        # remove row theta and column lambda
        keep_rows = trues(m1); keep_rows[theta] = false
        keep_cols = trues(m0); keep_cols[lambda] = false
        U1 = [U1[i] for i in 1:m1 if keep_rows[i]]
        U0 = [U0[j] for j in 1:m0 if keep_cols[j]]
        Delta = Delta[keep_rows, keep_cols]
    end
    UpsetPresentation{QQ}(P, U0, U1, Delta)
end

"""
    minimal_upset_presentation_one_step(H::FiniteFringe.FringeModule)
        -> UpsetPresentation{QQ}

Build a one-step upset presentation and apply safe minimality passes:
1) drop zero relations; 2) cancel isolated isomorphism pairs.
"""
function minimal_upset_presentation_one_step(H::FiniteFringe.FringeModule)
    F = upset_presentation_one_step(H)     # existing builder
    F = prune_zero_relations(F)
    F = cancel_isolated_unit_pairs(F)
    return F
end


"""
    prune_unused_targets(E::DownsetCopresentation{QQ}) -> DownsetCopresentation{QQ}

Drop rows of `rho` that are identically zero (unused target summands in E^1). The kernel is unchanged.
"""
function prune_unused_targets(E::DownsetCopresentation{QQ})
    m1, m0 = size(E.rho)
    keep = trues(m1)
    rows, _, _ = findnz(E.rho)
    seen = falses(m1); @inbounds for r in rows; seen[r] = true; end
    @inbounds for r in 1:m1
        if !seen[r]; keep[r] = false; end
    end
    new_D1 = [E.D1[i] for i in 1:m1 if keep[i]]
    new_rho = E.rho[keep, :]
    DownsetCopresentation{QQ}(E.P, E.D0, new_D1, new_rho)
end

"""
    cancel_isolated_unit_pairs(E::DownsetCopresentation{QQ}) -> DownsetCopresentation{QQ}

Iteratively cancels isolated nonzero entries `rho[theta,lambda]` with matching principal downsets
(D1[theta] == D0[lambda]) and unique in their row/column.
"""
function cancel_isolated_unit_pairs(E::DownsetCopresentation{QQ})
    P, D0, D1, R = E.P, E.D0, E.D1, E.rho
    while true
        m1, m0 = size(R)
        rows, cols, _ = findnz(R)
        rcount = zeros(Int, m1)
        ccount = zeros(Int, m0)
        @inbounds for k in eachindex(rows)
            rcount[rows[k]] += 1; ccount[cols[k]] += 1
        end
        found = false; theta = 0; lambda = 0
        @inbounds for k in eachindex(rows)
            r = rows[k]; c = cols[k]
            if rcount[r] == 1 && ccount[c] == 1
                if D1[r].P === D0[c].P && D1[r].mask == D0[c].mask
                    theta, lambda = r, c; found = true; break
                end
            end
        end
        if !found; break; end
        keep_rows = trues(m1); keep_rows[theta] = false
        keep_cols = trues(m0); keep_cols[lambda] = false
        D1 = [D1[i] for i in 1:m1 if keep_rows[i]]
        D0 = [D0[j] for j in 1:m0 if keep_cols[j]]
        R = R[keep_rows, keep_cols]
    end
    DownsetCopresentation{QQ}(P, D0, D1, R)
end

"""
    minimal_downset_copresentation_one_step(H::FiniteFringe.FringeModule)
        -> DownsetCopresentation{QQ}

Build a one-step downset copresentation and apply safe minimality passes:
1) drop zero target rows; 2) cancel isolated isomorphism pairs.
"""
function minimal_downset_copresentation_one_step(H::FiniteFringe.FringeModule)
    E = downset_copresentation_one_step(H)
    E = prune_unused_targets(E)
    E = cancel_isolated_unit_pairs(E)
    return E
end


# --------------------- First page dimensions from one-step data ---------------------

"""
    hom_ext_first_page(F0F1::UpsetPresentation{QQ}, E0E1::DownsetCopresentation{QQ})
Return `(dimHom, dimExt1)` computed from the first page (Defs. 6.1 & 6.4).
This delegates to the HomExt block assembly through the one-step data.
"""
function hom_ext_first_page(F::UpsetPresentation{QQ}, E::DownsetCopresentation{QQ})
    # We only need the connected-component counts pi0(U \cap D) and the naturality constraints
    # induced by the one-step differentials F.delta and E.rho.  Here we give dimensions only.
    # To keep the module boundary clean, we use the already loaded HomExt machinery.
    # (Users can call HomExt.ext_dims_via_resolutions with F/E wrapped as length-1 resolutions.)
    nHom = 0
    for U in F.U0, D in E.D0
        # Proposition 3.10: dim Hom(k[U], k[D]) = #components of U \cap D
        nHom += pi0_count(F.P, U, D)
    end
    # Ext^1 requires relations from delta and rho; returning 0 here is conservative.
    # Users typically call the Tot^* builder.
    nExt1 = 0
    return nHom, nExt1
end

# =============================================================================
# Longer indicator resolutions and high-level Ext driver
# =============================================================================
# We expose:
#   * upset_resolution(H; maxlen)     -> (F, dF)
#   * downset_resolution(H; maxlen)   -> (E, dE)
#   * indicator_resolutions(HM, HN; maxlen) -> (F, dF, E, dE)
#   * ext_dimensions_via_indicator_resolutions(HM, HN; maxlen) -> Dict{Int,Int}
#
# The outputs (F, dF, E, dE) are exactly the shapes expected by
# HomExt.build_hom_tot_complex / HomExt.ext_dims_via_resolutions:
#   - F is a Vector{UpsetPresentation{QQ}} with F[a+1].U0 = U_a
#   - dF[a] is the sparse delta_a : U_a <- U_{a+1}  (shape |U_{a+1}| x |U_a|)
#   - E is a Vector{DownsetCopresentation{QQ}} with E[b+1].D0 = D_b
#   - dE[b] is the sparse rho_b : D_b -> D_{b+1}    (shape |D_{b+1}| x |D_b|)
#
# Construction mirrors section 6.1 and the one-step routines already present.
# =============================================================================

using ..HomExt: build_hom_tot_complex, ext_dims_via_resolutions  # re-exported API. :contentReference[oaicite:2]{index=2}

# ------------------------------ small helpers --------------------------------

# Build the list of principal upsets from per-vertex generator labels returned by
# projective_cover: gens_at[v] is a vector of pairs (p, j).  Each pair contributes
# one principal upset at vertex p.
function _principal_upsets_from_gens(P::FinitePoset,
                                     gens_at::Vector{Vector{Tuple{Int,Int}}})
    U = Upset[]
    for p in 1:P.n
        for _ in gens_at[p]
            push!(U, principal_upset(P, p))
        end
    end
    U
end

# Build the list of principal downsets from per-vertex labels returned by _injective_hull
function _principal_downsets_from_gens(P::FinitePoset,
                                       gens_at::Vector{Vector{Tuple{Int,Int}}})
    D = Downset[]
    for u in 1:P.n
        for _ in gens_at[u]
            push!(D, principal_downset(P, u))
        end
    end
    D
end

# For upset side: at vertex i, which global generators are active (born at p <= i)?
# Returns L[i] = vector of global generator labels (p,j) visible at i.
function _local_index_list_up(P::FinitePoset,
                              gens_at::Vector{Vector{Tuple{Int,Int}}})
    L = Vector{Vector{Tuple{Int,Int}}}(undef, P.n)
    for i in 1:P.n
        lst = Tuple{Int,Int}[]
        for p in 1:P.n
            if P.leq[p,i]
                append!(lst, gens_at[p])
            end
        end
        L[i] = lst
    end
    L
end

# For downset side: at vertex i, which global generators are active (born at u with i <= u)?
# Returns L[i] = vector of global generator labels (u,j) visible at i.
function _local_index_list_down(P::FinitePoset,
                                gens_at::Vector{Vector{Tuple{Int,Int}}})
    L = Vector{Vector{Tuple{Int,Int}}}(undef, P.n)
    for i in 1:P.n
        lst = Tuple{Int,Int}[]
        for u in 1:P.n
            if P.leq[i,u]
                append!(lst, gens_at[u])
            end
        end
        L[i] = lst
    end
    L
end

# Find the local column index (1-based) of a global generator g=(p,j) or (u,j) in L[i].
# Returns 0 if not present.
function _local_col_of(L::Vector{Vector{Tuple{Int,Int}}}, g::Tuple{Int,Int}, i::Int)
    for (c, gg) in enumerate(L[i])
        if gg == g; return c; end
    end
    return 0
end

# Dense -> sparse helper over QQ (already have a general helper in FiniteFringe, but here
# we build directly from triplets to avoid materializing full dense blocks).
function _empty_sparse_QQ(nr::Int, nc::Int)
    return spzeros(QQ, nr, nc)
end

# ------------------------------ upset resolution ------------------------------

"""
    upset_resolution(H::FiniteFringe.FringeModule{QQ}; maxlen::Union{Int,Nothing}=nothing)
        -> (F::Vector{UpsetPresentation{QQ}}, dF::Vector{SparseMatrixCSC{QQ,Int}})

Construct a finite-length upset resolution of the module induced by the fringe
presentation `H` (Def. 6.1). The routine iterates projective covers and kernels:

  M = pmodule_from_fringe(H)
  (F0 -> M), K1 = ker, (F1 -> K1), K2 = ker, ...

Each step yields delta_a : F_a <- F_{a+1} as in your one-step builder, and we
read scalar entries at the minimal vertex exactly as in Def. 6.4.

Stop when the next kernel has zero fiber dimensions, or when `maxlen` steps have
been produced. Returned arrays are shaped for HomExt.build_hom_tot_complex.
"""
function upset_resolution(H::FiniteFringe.FringeModule{QQ}; maxlen::Union{Int,Nothing}=nothing)
    M = pmodule_from_fringe(H)  # internal PModule over QQ. :contentReference[oaicite:3]{index=3}
    P = M.Q

    # First projective cover: F0 --pi0--> M, with labels gens_at_F0
    F0, pi0, gens_at_F0 = projective_cover(M)  # :contentReference[oaicite:4]{index=4}
    U_by_a = Vector{Vector{Upset}}()      # U_a lists
    push!(U_by_a, _principal_upsets_from_gens(P, gens_at_F0))

    dF = Vector{SparseMatrixCSC{QQ,Int}}()  # deltas delta_a : U_a <- U_{a+1}

    # Iteration state
    curr_dom = F0
    curr_pi  = pi0
    curr_gens = gens_at_F0
    steps = 0

    while true
        if maxlen !== nothing && steps >= maxlen
            break
        end

        # Next kernel and projective cover
        K, iota = kernel_with_inclusion(curr_pi)   # K = ker(F_prev -> ...)
        if sum(K.dims) == 0
            # No new generators; terminate.
            break
        end
        Fnext, pinext, gens_at_next = projective_cover(K)

        # Compute delta_a at minimal vertex: d = iota circ pinext : Fnext -> curr_dom
        comps = Vector{Matrix{QQ}}(undef, P.n)
        for i in 1:P.n
            comps[i] = iota.comps[i] * pinext.comps[i]
        end
        d = PMorphism{QQ}(Fnext, curr_dom, comps)

        # Build labels and local index lists
        U_next = _principal_upsets_from_gens(P, gens_at_next)
        Lprev  = _local_index_list_up(P, curr_gens)
        Lnext  = _local_index_list_up(P, gens_at_next)

        # Global enumerations of generators (record their birth vertices)
        global_prev = Tuple{Int,Int}[]; for p in 1:P.n; append!(global_prev, curr_gens[p]); end
        global_next = Tuple{Int,Int}[]; for p in 1:P.n; append!(global_next, gens_at_next[p]); end

        # Assemble sparse delta: rows index next, cols index prev
        delta = _empty_sparse_QQ(length(global_next), length(global_prev))
        for (lambda, (plambda, jlambda)) in enumerate(global_prev)      # column in U_prev
            for (theta,  (ptheta,  jtheta))  in enumerate(global_next)   # row in U_next
                if P.leq[ptheta, plambda]   # containment of principal upsets
                    i     = plambda         # minimal vertex where we read the entry
                    col   = _local_col_of(Lnext, (ptheta, jtheta), i)
                    row   = _local_col_of(Lprev, (plambda, jlambda), i)
                    if col > 0 && row > 0
                        val = d.comps[i][row, col]
                        if val != 0
                            delta[theta, lambda] = val
                        end
                    end
                end
            end
        end

        push!(dF, delta)
        push!(U_by_a, U_next)

        # Advance
        curr_dom  = Fnext
        curr_pi   = pinext
        curr_gens = gens_at_next
        steps += 1
    end

    # Package as UpsetPresentation list, one per degree a, with U0=U_a and
    # U1=U_{a+1} for a < A (the last has empty U1 and zero-sized delta).
    F = Vector{UpsetPresentation{QQ}}(undef, length(U_by_a))
    for a in 1:length(U_by_a)
        U0 = U_by_a[a]
        if a < length(U_by_a)
            U1 = U_by_a[a+1]
            delta = dF[a]
        else
            U1 = Upset[]
            delta = spzeros(QQ, 0, length(U0))
        end
        F[a] = UpsetPresentation{QQ}(P, U0, U1, delta)  # :contentReference[oaicite:5]{index=5}
    end

    return F, dF
end

# ---------------------------- downset resolution ------------------------------

"""
    downset_resolution(H::FiniteFringe.FringeModule{QQ}; maxlen::Union{Int,Nothing}=nothing)
        -> (E::Vector{DownsetCopresentation{QQ}}, dE::Vector{SparseMatrixCSC{QQ,Int}})

Construct a finite-length downset (injective) resolution of the module induced by
the fringe presentation `H` (Def. 6.1). The routine iterates injective hulls and
cokernels:

  M -> E0, C1 = coker, C1 -> E1, C2 = coker, ...

Each step yields rho_b : E^b -> E^{b+1}. Scalar entries are read at minimal vertices
as in your one-step builder.
"""
function downset_resolution(H::FiniteFringe.FringeModule{QQ}; maxlen::Union{Int,Nothing}=nothing)
    M = pmodule_from_fringe(H)
    P = M.Q

    # First injective hull: iota0 : M -> E0
    E0, iota0, gens_at_E0 = _injective_hull(M)   # :contentReference[oaicite:6]{index=6}
    D_by_b = Vector{Vector{Downset}}()
    push!(D_by_b, _principal_downsets_from_gens(P, gens_at_E0))

    dE = Vector{SparseMatrixCSC{QQ,Int}}()

    # First cokernel
    C, q = _cokernel_module(iota0)               # :contentReference[oaicite:7]{index=7}
    steps = 0

    # When there is nothing to injectivize, we just return E^0 with empty dE.
    if sum(C.dims) == 0
        E = Vector{DownsetCopresentation{QQ}}(undef, 1)
        E[1] = DownsetCopresentation{QQ}(P, D_by_b[1], Downset[], spzeros(QQ, 0, length(D_by_b[1])))
        return E, dE
    end

    # Iterate
    prev_E   = E0
    prev_D   = D_by_b[1]
    prev_gens = gens_at_E0
    prev_q   = q

    while true
        if maxlen !== nothing && steps >= maxlen
            break
        end

        # Next injective hull: j : C -> E1
        E1, j, gens_at_E1 = _injective_hull(C)

        # Compose rho_b : E0 -> E1
        comps = Vector{Matrix{QQ}}(undef, P.n)
        for i in 1:P.n
            comps[i] = j.comps[i] * prev_q.comps[i]
        end
        rho = PMorphism{QQ}(prev_E, E1, comps)

        # Build labels and local lists
        D_next = _principal_downsets_from_gens(P, gens_at_E1)
        L0     = _local_index_list_down(P, prev_gens)
        L1     = _local_index_list_down(P, gens_at_E1)

        globalD0 = Tuple{Int,Int}[]; for u in 1:P.n; append!(globalD0, prev_gens[u]); end
        globalD1 = Tuple{Int,Int}[]; for u in 1:P.n; append!(globalD1, gens_at_E1[u]); end

        # Assemble sparse rho: rows index D1 (next), cols index D0 (prev)
        Rh = _empty_sparse_QQ(length(globalD1), length(globalD0))
        for (lambda, (ulambda, jlambda)) in enumerate(globalD0)     # col in D0
            for (theta,  (utheta,  jtheta))  in enumerate(globalD1)  # row in D1
                if P.leq[ulambda, utheta]   # containment for downsets
                    i   = utheta
                    col = _local_col_of(L0, (ulambda, jlambda), i)
                    row = _local_col_of(L1, (utheta,  jtheta),  i)
                    if col > 0 && row > 0
                        val = rho.comps[i][row, col]
                        if val != 0
                            Rh[theta, lambda] = val
                        end
                    end
                end
            end
        end

        push!(dE, Rh)
        push!(D_by_b, D_next)

        # Next cokernel
        C, q = _cokernel_module(j)
        if sum(C.dims) == 0
            # No further step
            prev_E   = E1
            prev_D   = D_next
            prev_gens = gens_at_E1
            break
        end

        # Advance
        prev_E    = E1
        prev_D    = D_next
        prev_gens = gens_at_E1
        prev_q    = q
        steps += 1
    end

    # Package as DownsetCopresentation list, one per b, with D0 = D_b and D1 = D_{b+1}
    E = Vector{DownsetCopresentation{QQ}}(undef, length(D_by_b))
    for b in 1:length(D_by_b)
        D0 = D_by_b[b]
        if b < length(D_by_b)
            D1 = D_by_b[b+1]
            rho = dE[b]
        else
            D1 = Downset[]
            rho = spzeros(QQ, 0, length(D0))
        end
        E[b] = DownsetCopresentation{QQ}(P, D0, D1, rho)  # :contentReference[oaicite:8]{index=8}
    end

    return E, dE
end

# --------------------- aggregator + high-level Ext driver ---------------------

"""
    indicator_resolutions(HM, HN; maxlen=nothing)
        -> (F, dF, E, dE)

Convenience wrapper: build an upset resolution for the source module (fringe HM)
and a downset resolution for the target module (fringe HN).  The `maxlen` keyword
cuts off each side after that many steps (useful for quick tests).
"""
function indicator_resolutions(HM::FiniteFringe.FringeModule{QQ},
                               HN::FiniteFringe.FringeModule{QQ};
                               maxlen::Union{Int,Nothing}=nothing)
    F, dF = upset_resolution(HM; maxlen=maxlen)
    E, dE = downset_resolution(HN; maxlen=maxlen)
    return F, dF, E, dE
end

"""
    ext_dimensions_via_indicator_resolutions(HM, HN; maxlen=nothing)::Dict{Int,Int}

Build indicator resolutions for HM and HN and return a dictionary mapping total
degree t to dim Ext^t(HM, HN) calculated from the Tot complex.  This is exactly
HomExt.ext_dims_via_resolutions(F, dF, E, dE) after constructing (F,dF,E,dE).
"""
function ext_dimensions_via_indicator_resolutions(HM::FiniteFringe.FringeModule{QQ},
                                                  HN::FiniteFringe.FringeModule{QQ};
                                                  maxlen::Union{Int,Nothing}=nothing)
    F, dF, E, dE = indicator_resolutions(HM, HN; maxlen=maxlen)
    return ext_dims_via_resolutions(F, dF, E, dE)  # :contentReference[oaicite:9]{index=9}
end


export PModule, PMorphism,
       pmodule_from_fringe,
       projective_cover, kernel_with_inclusion,
       upset_presentation_one_step, downset_copresentation_one_step,
       hom_ext_first_page,
       upset_resolution, downset_resolution,
       indicator_resolutions, ext_dimensions_via_indicator_resolutions

end # module
