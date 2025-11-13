module IndicatorResolutions
# Depends on your Phase-0/1 module `PosetModules.jl`
using ..PosetModules:
    Poset, PModule, PMorphism, leq, map_leq,
    colspace_basis, nullspace_QQ, rref_with_pivots, solve_fullcolumn


# Utilities already introduced previously (projective cover, upset resolution)
# kept intact. We extend with: injective hull, downset copresentation; and 
# Hom/Ext via the first page of the Hom double complex.
# References in comments use:
# - Indicator (co)presentations and resolutions: Def. 6.1-6.4.
# - Syzygy theorem guarantees existence over finite encodings: Thm. 6.12.

# ======== (A) Principal indicator modules (from earlier section) ==============
# (Repeat here the two short constructors so this file is standalone.)

"Direct sum of 'mult' copies of the principal **upset** k[\uparrow p] over finite poset `P`."
function principal_upset_module(P::Poset{ET}, p; mult::Int=1) where {ET}
    n = length(P.elements)
    dims = [leq(P, P.elements[p], P.elements[i]) ? mult : 0 for i in 1:n]
    maps = Dict{Tuple{Int,Int}, Matrix{Rational{BigInt}}}()
    for (u,v) in P.hasse_edges
        du, dv = dims[u], dims[v]
        M = zeros(Rational{BigInt}, dv, du)
        if du > 0 && dv > 0
            M[1:mult, 1:mult] .= 1//1
        end
        maps[(u,v)] = M
    end
    return PModule(P, dims, maps)
end

"Direct sum of 'mult' copies of the principal **downset** k[\downarrow p] (injective)."
function principal_downset_module(P::Poset{ET}, p; mult::Int=1) where {ET}
    n = length(P.elements)
    dims = [leq(P, P.elements[i], P.elements[p]) ? mult : 0 for i in 1:n]
    maps = Dict{Tuple{Int,Int}, Matrix{Rational{BigInt}}}()
    for (u,v) in P.hasse_edges
        du, dv = dims[u], dims[v]
        M = zeros(Rational{BigInt}, dv, du)
        if du > 0 && dv > 0
            M[1:mult, 1:mult] .= 1//1
        end
        maps[(u,v)] = M
    end
    return PModule(P, dims, maps)
end

# ======== (B) Projective cover and upset presentation (recall from Phase 2) ===

# Basis of \sum_{u<p} im(H_{u <= p}) inside H_p
function _incoming_image_basis(H::PModule{ET,S}, pidx::Int) where {ET,S<:Number}
    Q = H.Q
    hp = H.dims[pidx]
    B = zeros(Rational{BigInt}, hp, 0)
    for u in 1:length(Q.elements)
        if u != pidx && Q.leq_closure[u, pidx]          # u <= p
            A = map(Rational{BigInt}, map_leq(H, Q.elements[u], Q.elements[pidx]))
            if H.dims[u] > 0
                Im = A * Matrix{Rational{BigInt}}(I, H.dims[u], H.dims[u])
                B = colspace_basis(hcat(B, Im))
            end
        end
    end
    return B
end

# Columns of identity extending a given column space (complement basis)
function _complement_basis(hp::Int, B::Matrix{Rational{BigInt}})
    if size(B,2) >= hp
        return zeros(Rational{BigInt}, hp, 0)
    end
    M = hcat(B, Matrix{Rational{BigInt}}(I, hp, hp))
    _, piv = rref_with_pivots(M)
    chosen = Int[]
    for k in 1:hp
        if (size(B,2)+k) in piv
            push!(chosen, k)
        end
    end
    return Matrix{Rational{BigInt}}(I, hp, hp)[:, chosen]
end

"Projective cover F0 \to H by principal upsets; also returns bookkeeping 'gens_at'."
function projective_cover(H::PModule{ET,S}) where {ET,S<:Number}
    P = H.Q; n = length(P.elements)
    beta = zeros(Int, n); comps_at_p = Vector{Matrix{Rational{BigInt}}}(undef, n)
    for i in 1:n
        Bimg = _incoming_image_basis(H, i)
        C    = _complement_basis(H.dims[i], Bimg)
        comps_at_p[i] = C
        beta[i] = size(C,2)
    end

    # Direct sum \oplus_p k[\uparrow p]^{beta_p}
    dimsF0 = [sum(beta[u] for u in 1:n if P.leq_closure[u,i]) for i in 1:n]
    gens_at = [Tuple{Int,Int}[] for _ in 1:n]  # at i: list of (p,copy)
    for i in 1:n, p in 1:n
        if P.leq_closure[p,i] && beta[p] > 0
            for j in 1:beta[p]; push!(gens_at[i], (p,j)); end
        end
    end
    edgeF0 = Dict{Tuple{Int,Int}, Matrix{Rational{BigInt}}}()
    for (u,v) in P.hasse_edges
        Mu, Mv = length(gens_at[u]), length(gens_at[v])
        M = zeros(Rational{BigInt}, Mv, Mu)
        posv = Dict{Tuple{Int,Int},Int}(); for (idx,g) in enumerate(gens_at[v]); posv[g]=idx; end
        for (idx,g) in enumerate(gens_at[u]); M[posv[g], idx] = 1//1; end
        edgeF0[(u,v)] = M
    end
    F0 = PModule(P, dimsF0, edgeF0)

    # pi0 : F0 -> H
    comps = Vector{Matrix{Rational{BigInt}}}(undef, n)
    for i in 1:n
        Mi = zeros(Rational{BigInt}, H.dims[i], length(gens_at[i]))
        for (col,(p,j)) in enumerate(gens_at[i])
            c = comps_at_p[p][:,j]
            A = map(Rational{BigInt}, map_leq(H, P.elements[p], P.elements[i]))
            Mi[:,col] = A * c
        end
        comps[i] = Mi
    end
    pi0 = PMorphism(F0, H, comps)
    return F0, pi0, gens_at
end

"Kernel object together with its inclusion iota : ker(f) \to dom(f)."
function kernel_with_inclusion(f::PMorphism{ET,S}) where {ET,S<:Number}
    H, G = f.dom, f.cod; P = H.Q; n = length(P.elements)
    kerbasis = Vector{Matrix{Rational{BigInt}}}(undef, n); newdims = Int[]
    for i in 1:n
        Fi = map(Rational{BigInt}, f.comps[i])  # G_i \times H_i
        N  = nullspace_QQ(Fi)                   # H_i \times kerdim
        kerbasis[i] = N
        push!(newdims, size(N,2))
    end
    maps = Dict{Tuple{Int,Int}, Matrix{Rational{BigInt}}}()
    for (u,v) in P.hasse_edges
        Huv = map(Rational{BigInt}, H.edge_maps[(u,v)])
        Bu, Bv = kerbasis[u], kerbasis[v]
        images = Huv * Bu
        M = zeros(Rational{BigInt}, size(Bv,2), size(Bu,2))
        for j in 1:size(Bu,2)
            M[:,j] = solve_fullcolumn(Bv, images[:,j])
        end
        maps[(u,v)] = M
    end
    K = PModule(P, newdims, maps)
    iota = PMorphism(K, H, kerbasis)
    return K, iota
end

"One-step upset presentation F1 \to F0 \to H (Def. 6.4.1)."
function upset_presentation(H::PModule)
    F0, pi0, _ = projective_cover(H)
    K1, iota1    = kernel_with_inclusion(pi0)
    F1, pi1, _ = projective_cover(K1)
    d1 = PMorphism(F1, F0, [iota1.comps[i]*pi1.comps[i] for i in 1:length(H.Q.elements)])
    return F1, F0, d1, pi0
end

"Iterated upset (projective) resolution to length `maxlen` >= 1."
function upset_resolution(H::PModule; maxlen::Int=2)
    F = PModule[]; d = PMorphism[]
    F0, pi0, _ = projective_cover(H)
    push!(F, F0)
    K, iota = kernel_with_inclusion(pi0)
    i = 1
    while i <= maxlen && sum(K.dims) > 0
        Fi, pii, _ = projective_cover(K)
        di = PMorphism(Fi, F[i], [iota.comps[v]*pii.comps[v] for v in 1:length(H.Q.elements)])
        push!(d, di); push!(F, Fi)
        K, iota = kernel_with_inclusion(di)
        i += 1
    end
    return F, d, pi0
end

# ======== (C) Injective hull and downset copresentation =================
# Dual to projective cover. We compute the *socle* at each p:
#   Soc_p = \cap_{v>p} ker(H_{p <= v}); for maximal p this is H_p.
# Multiplicity alpha_p := dim Soc_p. The injective hull embeds H \into E0 = \oplus k[\downarrow p]^{alpha_p}.
# Then a **downset copresentation** H = ker(delta0: E0 \to E1) is produced by taking the
# cokernel Q of the hull, building an injective hull of Q, and composing.
# This realizes Def. 6.4.2 directly.  :contentReference[oaicite:3]{index=3}

# Left-inverse rows Lambda for a tall full-column-rank matrix S: Lambda*S = I
function _left_inverse_rows(S::Matrix{Rational{BigInt}})
    m, r = size(S)
    Lambda = zeros(Rational{BigInt}, r, m)
    for j in 1:r
        # Solve S^T * y = e_j  (S^T has full row rank r)
        A = transpose(S)                         # r \times m
        e = zeros(Rational{BigInt}, r); e[j]=1//1
        # Build augmented [A | e] and do RREF to solve exactly
        Aug = hcat(A, e)
        _, piv = rref_with_pivots(Aug[:, 1:m])
        # Solve by choosing pivot columns; we can reuse the helper `solve_fullcolumn`
        # by transposing the system: (A^T) x = e_j  with A^T full column rank.
        x = solve_fullcolumn(transpose(A), e)
        Lambda[j, :] = transpose(x)
    end
    Lambda
end

# Socle basis at p: intersection of kernels of maps to strictly higher degrees.
function _socle_basis(H::PModule{ET,S}, pidx::Int) where {ET,S<:Number}
    P = H.Q
    hp = H.dims[pidx]
    # Collect all maps H_{p <= v} for v > p (strict)
    maps = Matrix{Rational{BigInt}}[]
    for v in 1:length(P.elements)
        if v != pidx && P.leq_closure[pidx, v]
            A = map(Rational{BigInt}, map_leq(H, P.elements[pidx], P.elements[v]))
            push!(maps, A)
        end
    end
    if isempty(maps)
        # Maximal element: whole H_p is socle
        return Matrix{Rational{BigInt}}(I, hp, hp)
    end
    # Stack vertically and take nullspace to get the intersection of kernels
    A = vcat(maps...)                # (sum h_v) \times h_p
    N = nullspace_QQ(A)              # h_p \times alpha_p
    return N
end

"Injective hull eta0 : H \to E0 = \oplus_p k[\downarrow p]^{alpha_p}, with multiplicities alpha_p = dim soc_p."
function injective_hull(H::PModule{ET,S}) where {ET,S<:Number}
    P = H.Q; n = length(P.elements)
    alpha = zeros(Int, n)
    soc_at_p = Vector{Matrix{Rational{BigInt}}}(undef, n)
    for p in 1:n
        N = _socle_basis(H, p)      # H_p \times alpha_p
        soc_at_p[p] = N
        alpha[p] = size(N, 2)
    end

    # Build E0 = \olus k[\downarrow p]^{alpha_p}
    dimsE0 = [sum(alpha[v] for v in 1:n if P.leq_closure[i,v]) for i in 1:n]  # i <= p \implies p includes i
    gens_at = [Tuple{Int,Int}[] for _ in 1:n]  # at i: (p,copy) with i <= p
    for i in 1:n, p in 1:n
        if P.leq_closure[i,p] && alpha[p] > 0
            for j in 1:alpha[p]; push!(gens_at[i], (p,j)); end
        end
    end
    edges = Dict{Tuple{Int,Int}, Matrix{Rational{BigInt}}}()
    for (u,v) in P.hasse_edges
        Mu, Mv = length(gens_at[u]), length(gens_at[v])
        M = zeros(Rational{BigInt}, Mv, Mu)
        posv = Dict{Tuple{Int,Int},Int}(); for (idx,g) in enumerate(gens_at[v]); posv[g]=idx; end
        for (idx,g) in enumerate(gens_at[u])
            # If g=(p,j) appears at v, it appears at u (since u <= v <= p)
            if haskey(posv, g); M[posv[g], idx] = 1//1; end
        end
        edges[(u,v)] = M
    end
    E0 = PModule(P, dimsE0, edges)

    # Build eta0 : H -> E0 using functionals Lambda_p whose rows pick coordinates on Soc_p
    comps = Vector{Matrix{Rational{BigInt}}}(undef, n)
    Lambda = Vector{Matrix{Rational{BigInt}}}(undef, n)  # alpha_p \times h_p with Lambda_p * soc_at_p[p] = I
    for p in 1:n
        N = soc_at_p[p]
        Lambda[p] = size(N,2) == 0 ? zeros(Rational{BigInt}, 0, H.dims[p]) : _left_inverse_rows(N)
    end
    for i in 1:n
        Mi = zeros(Rational{BigInt}, length(gens_at[i]), H.dims[i])
        for (row,(p,j)) in enumerate(gens_at[i])
            # component: functionals at p composed with H_{i <= p}
            A = map(Rational{BigInt}, map_leq(H, P.elements[i], P.elements[p]))  # H_{i <= p}: H_i \to H_p
            Mi[row, :] = (Lambda[p][j, :] * A)
        end
        comps[i] = Mi
    end
    eta0 = PMorphism(H, E0, comps)
    return E0, eta0, gens_at
end

# Cokernel with projection pi : cod(f) \to coker(f)
function cokernel_with_projection(f::PMorphism{ET,S}) where {ET,S<:Number}
    H, G = f.dom, f.cod; P = H.Q; n = length(P.elements)
    proj_cols = Vector{Vector{Int}}(undef, n)           # columns of I forming a complement
    dimsC = Int[]; pi = Vector{Matrix{Rational{BigInt}}}(undef, n)
    for i in 1:n
        ImB = colspace_basis(map(Rational{BigInt}, f.comps[i]))  # G_i \times r
        # choose complement basis in G_i as columns of identity
        gi = G.dims[i]
        C = _complement_basis(gi, ImB)   # gi \times ci (unit columns)
        # projection pi_i: pick the coordinates in those columns
        pi[i] = transpose(C)               # ci \times gi
        push!(dimsC, size(C,2))
        # record which positions were chosen (not needed further here)
    end
    # edge maps C_uv = pi_v * G_uv * C_u  where C_u embeds Q-coords into G_u
    maps = Dict{Tuple{Int,Int}, Matrix{Rational{BigInt}}}()
    for (u,v) in P.hasse_edges
        Guv = map(Rational{BigInt}, G.edge_maps[(u,v)])
        Cu  = transpose(pi[u])     # G_u \times c_u
        CvT = pi[v]                # c_v \times G_v
        maps[(u,v)] = CvT * Guv * Cu
    end
    C = PModule(P, dimsC, maps)
    pi = PMorphism(G, C, pi)
    return C, pi
end

"""
    downset_copresentation(H::PModule) -> (E0, E1, delta0, eta0)

Construct a **downset copresentation** `H = ker(delta0 : E0 \to E1)` (Def. 6.4.2):
- `eta0 : H \to E0` is the injective hull by principal downsets (socle-based),
- `delta0` is obtained by pushing `E0` to its cokernel and then taking an injective hull of that cokernel.
"""
function downset_copresentation(H::PModule)
    E0, eta0, _ = injective_hull(H)            # H \into E0
    Q, pi  = cokernel_with_projection(eta0)     # E0 \onto Q
    E1, eta1, _ = injective_hull(Q)            # Q \into E1
    delta0 = PMorphism(E0, E1, [eta1.comps[i] * pi.comps[i] for i in 1:length(H.Q.elements)])
    return E0, E1, delta0, eta0
end


"""
    downset_resolution(H::FringeModule{K}; maxlen::Int=3)
        -> (E::Vector{DownsetCopresentation{K}}, rho::Vector{SparseMatrixCSC{K,Int}}, eta0::PMorphism)

Build a *downset resolution*  E^0 \to E^1 \to ... \to E^B  with B <= maxlen by iterating:
injective hull (E^b, eta_b : Q^{b} \into E^b), then cokernel (Q^{b+1}), then hull again, etc.
Terminates when the new cokernel is zero.  (Def. 6.1, dual to upset resolution.)  :contentReference[oaicite:11]{index=11}
"""
function downset_resolution(H::FiniteFringe.FringeModule{K}; maxlen::Int=3) where {K}
    E = DownsetCopresentation{K}[]
    rho = SparseMatrixCSC{K,Int}[]
    # Start from H
    Hcur = H
    # We also carry the inclusion eta_b : H^{(b)} \into E^b, but callers usually only need eta0.
    E0, eta0, _ = injective_hull(Hcur)   # re-use your injective hull
    push!(E, DownsetCopresentations.DownsetCopresentation{K}(H.P, E0.D0, E0.D1, E0.rho))
    push!(rho, E0.rho)
    # Iterate
    b = 0
    while b < maxlen
        Qb, pib = cokernel_with_projection(eta0) # Q^b
        # stop if zero
        if sum(Qb.dims) == 0; break; end
        Eb1, etab1, _ = injective_hull(Qb)
        push!(E, DownsetCopresentations.DownsetCopresentation{K}(H.P, Eb1.D0, Eb1.D1, Eb1.rho))
        push!(rho, Eb1.rho)
        # next
        Hcur = Qb
        eta0 = etab1
        b += 1
    end
    return E, rho, eta0
end

# ======== (D) Hom/Ext via the first page of the Hom double complex ============

# Linear space of natural transformations Phi: A \to B:
# unknowns are block-matrices Phi_i; constraints: B_uv Phi_u = Phi_v A_uv for all Hasse edges.
function _dim_hom_space(A::PModule{ET,S}, B::PModule{ET,S}) where {ET,S<:Number}
    P = A.Q; n = length(P.elements)
    # Build one big homogeneous linear system M * vec(Phi) = 0
    # We do it vertex-by-vertex; to keep things simple (and exact) we stack constraints
    # in blocks and compute the nullspace dimension by sequential elimination.
    # Implementation detail: we do not materialize M explicitly sparse -- it suffices
    # to accumulate equations and then call nullspace_QQ on the concatenated matrix.
    # For moderate sizes this is fine; move to Nemo for large instances.
    # Variables: concatenation of vec(Phi_i) (column-stacked) for i=1..n.
    # We'll form an "evaluation matrix" E so that solutions are nullspace(E).
    # Construct row-blocks per edge:
    rows_total = 0
    for (u,v) in P.hasse_edges
        rows_total += B.dims[v]*A.dims[u]  # number of scalar equations
    end
    # We can't preallocate sparsity comfortably; build dense and rely on exact arithmetic.
    tot_vars = sum(B.dims[i]*A.dims[i] for i in 1:n)
    E = zeros(Rational{BigInt}, rows_total, tot_vars)
    rowptr = 1
    col_offsets = cumsum([0; (B.dims[i]*A.dims[i] for i in 1:n)...])
    for (u,v) in P.hasse_edges
        Bu = B.dims[u]; Bv = B.dims[v]; Au = A.dims[u]; Av = A.dims[v]
        Guv = map(Rational{BigInt}, B.edge_maps[(u,v)])   # B_v \times B_u
        Huv = map(Rational{BigInt}, A.edge_maps[(u,v)])   # A_v \times A_u
        # Equation: (I \otimes Guv) vec(Phi_u)  -  (Huv^T \otimes I) vec(Phi_v)  = 0
        # Fill the corresponding columns
        # Block for u:
        blk_u = kron(Matrix{Rational{BigInt}}(I, Au, Au), Guv)   # (Bv*Au) \times (Bu*Au)
        E[rowptr:rowptr+Bv*Au-1, col_offsets[u]+1 : col_offsets[u]+Bu*Au] += blk_u
        # Block for v:
        blk_v = kron(transpose(Huv), Matrix{Rational{BigInt}}(I, Bv, Bv))  # (Bv*Au) \times (Bv*Av)
        E[rowptr:rowptr+Bv*Au-1, col_offsets[v]+1 : col_offsets[v]+Bv*Av] -= blk_v
        rowptr += Bv*Au
    end
    N = nullspace_QQ(E)     # each column of N is a basis vector of Hom(A,B)
    return size(N, 2)
end

# Dimension of {Phi : A \to B natural | Phi \circ L = 0  and  R \circ Phi = 0 }.
function _dim_hom_with_composition_constraints(A::PModule{ET,S}, B::PModule{ET,S};
                                               L::Union{Nothing,PMorphism}=nothing,
                                               R::Union{Nothing,PMorphism}=nothing) where {ET,S<:Number}
    P = A.Q; n = length(P.elements)
    # Unknowns are Phi_i (B_i \times A_i). Constraints:
    # (i) Naturality on each edge (as above).
    # (ii) For each vertex i: Phi_i * L_i = 0 if L given (map A'->A)  -> Phi \circ L = 0 in Hom(A',B)
    # (iii) For each vertex i: R_i * Phi_i = 0 if R given (map B->B')  -> R \circ Phi = 0 in Hom(A,B')
    # Build one block matrix of all constraints.
    # Count equations
    eqs = 0
    for (u,v) in P.hasse_edges
        eqs += B.dims[v]*A.dims[u]
    end
    if L !== nothing
        A2 = L.dom; for i in 1:n; eqs += B.dims[i]*A2.dims[i]; end
    end
    if R !== nothing
        B2 = R.cod; for i in 1:n; eqs += B2.dims[i]*A.dims[i]; end
    end
    tot_vars = sum(B.dims[i]*A.dims[i] for i in 1:n)
    E = zeros(Rational{BigInt}, eqs, tot_vars)
    rowptr = 1
    col_offsets = cumsum([0; (B.dims[i]*A.dims[i] for i in 1:n)...])

    # Naturality
    for (u,v) in P.hasse_edges
        Guv = map(Rational{BigInt}, B.edge_maps[(u,v)])
        Huv = map(Rational{BigInt}, A.edge_maps[(u,v)])
        Au = A.dims[u]; Bv = B.dims[v]; Bu = B.dims[u]; Av = A.dims[v]
        blk_u = kron(Matrix{Rational{BigInt}}(I, Au, Au), Guv)
        blk_v = kron(transpose(Huv), Matrix{Rational{BigInt}}(I, Bv, Bv))
        E[rowptr:rowptr+Bv*Au-1, col_offsets[u]+1 : col_offsets[u]+Bu*Au] += blk_u
        E[rowptr:rowptr+Bv*Au-1, col_offsets[v]+1 : col_offsets[v]+Bv*Av] -= blk_v
        rowptr += Bv*Au
    end
    # Phi \circ L = 0
    if L !== nothing
        A2 = L.dom
        for i in 1:n
            Li = map(Rational{BigInt}, L.comps[i])  # A_i \times A2_i
            blk = kron(transpose(Li), Matrix{Rational{BigInt}}(I, B.dims[i], B.dims[i]))  # (B_i*A2_i) \times (B_i*A_i)
            E[rowptr:rowptr + B.dims[i]*A2.dims[i]-1, col_offsets[i]+1 : col_offsets[i]+B.dims[i]*A.dims[i]] += blk
            rowptr += B.dims[i]*A2.dims[i]
        end
    end
    # R \circ Phi = 0
    if R !== nothing
        B2 = R.cod
        for i in 1:n
            Ri = map(Rational{BigInt}, R.comps[i])  # B2_i \times B_i
            blk = kron(Matrix{Rational{BigInt}}(I, A.dims[i], A.dims[i]), Ri)  # (B2_i*A_i) \times (B_i*A_i)
            E[rowptr:rowptr + B2.dims[i]*A.dims[i]-1, col_offsets[i]+1 : col_offsets[i]+B.dims[i]*A.dims[i]] += blk
            rowptr += B2.dims[i]*A.dims[i]
        end
    end
    N = nullspace_QQ(E)
    return size(N, 2)
end

"""
    hom_ext_first_page(F1,F0,d1, E0,E1,delta0) -> (dimHom, dimExt1)

Compute `dim Hom(M,N)` and `dim Ext^1(M,N)` using the **first page** of the Hom
double complex built from an upset presentation `F1 \to F0 \to M` and a downset copresentation
`H=N \to E0 \to E1` (Def. 6.1, 6.4). No bases are constructed-only dimensions via exact
linear constraints (naturality + composition), all over the rationals.
"""
function hom_ext_first_page(F1::PModule, F0::PModule, d1::PMorphism,
                            E0::PModule, E1::PModule, delta0::PMorphism)
    # H^0 = Hom(M,N) = { Phi : F0 \to E0 nat. | Phi \circ d1=0 and delta0 \circ Phi=0 }
    dimHom = _dim_hom_with_composition_constraints(F0, E0; L=d1, R=delta0)

    # Z^1 = {(A,B) | A:F1 \to E0 nat., B:F0 \to E1 nat., delta0 \circ A + B \circ d1 = 0}
    # Build a single system for (A,B) by stacking variables and constraints.
    # Unknowns: vec(A) followed by vec(B).
    P = F0.Q; n = length(P.elements)
    # Dimensions
    dimA = sum(E0.dims[i]*F1.dims[i] for i in 1:n)
    dimB = sum(E1.dims[i]*F0.dims[i] for i in 1:n)
    # Build block constraints matrix for Z^1
    # We re-use helpers by constructing a big "direct sum" module A \oplus B with tailored constraints.
    # Implement ad hoc: stack three families of equations.
    # (1) Naturality for A
    eqsA = sum(E0.dims[v]*F1.dims[u] for (u,v) in P.hasse_edges)
    # (2) Naturality for B
    eqsB = sum(E1.dims[v]*F0.dims[u] for (u,v) in P.hasse_edges)
    # (3) Cross equation delta0 \circ A + B \circ d1 = 0, vertexwise
    eqsC = sum(E1.dims[i]*F1.dims[i] for i in 1:n)

    tot_vars = dimA + dimB
    E = zeros(Rational{BigInt}, eqsA+eqsB+eqsC, tot_vars)
    rowptr = 1
    # Offsets in variable vector
    # For A: offsetsA[i]; for B: offsetsB[i]
    offsetsA = cumsum([0; (E0.dims[i]*F1.dims[i] for i in 1:n)...])
    offsetsB = dimA .+ cumsum([0; (E1.dims[i]*F0.dims[i] for i in 1:n)...])

    # (1) Naturality for A
    for (u,v) in P.hasse_edges
        Guv = map(Rational{BigInt}, E0.edge_maps[(u,v)])
        Huv = map(Rational{BigInt}, F1.edge_maps[(u,v)])
        Au = F1.dims[u]; Bv = E0.dims[v]; Bu = E0.dims[u]; Av = F1.dims[v]
        blk_u = kron(Matrix{Rational{BigInt}}(I, Au, Au), Guv)
        blk_v = kron(transpose(Huv), Matrix{Rational{BigInt}}(I, Bv, Bv))
        E[rowptr:rowptr+Bv*Au-1, offsetsA[u]+1 : offsetsA[u]+Bu*Au] += blk_u
        E[rowptr:rowptr+Bv*Au-1, offsetsA[v]+1 : offsetsA[v]+Bv*Av] -= blk_v
        rowptr += Bv*Au
    end
    # (2) Naturality for B
    for (u,v) in P.hasse_edges
        Guv = map(Rational{BigInt}, E1.edge_maps[(u,v)])
        Huv = map(Rational{BigInt}, F0.edge_maps[(u,v)])
        Au = F0.dims[u]; Bv = E1.dims[v]; Bu = E1.dims[u]; Av = F0.dims[v]
        blk_u = kron(Matrix{Rational{BigInt}}(I, Au, Au), Guv)
        blk_v = kron(transpose(Huv), Matrix{Rational{BigInt}}(I, Bv, Bv))
        E[rowptr:rowptr+Bv*Au-1, offsetsB[u]+1 : offsetsB[u]+Bu*Au] += blk_u
        E[rowptr:rowptr+Bv*Au-1, offsetsB[v]+1 : offsetsB[v]+Bv*Av] -= blk_v
        rowptr += Bv*Au
    end
    # (3) delta0 \circ A + B \circ d1 = 0   \implies  (I \oplus delta0_i) vec(A_i) + (d1_i^T \oplus I) vec(B_i) = 0
    for i in 1:n
        delta = map(Rational{BigInt}, delta0.comps[i])   # E1_i \times E0_i
        d = map(Rational{BigInt}, d1.comps[i])   # F0_i \times F1_i
        blkA = kron(Matrix{Rational{BigInt}}(I, F1.dims[i], F1.dims[i]), delta)
        blkB = kron(transpose(d), Matrix{Rational{BigInt}}(I, E1.dims[i], E1.dims[i]))
        r = E1.dims[i]*F1.dims[i]
        E[rowptr:rowptr+r-1, offsetsA[i]+1 : offsetsA[i]+E0.dims[i]*F1.dims[i]] += blkA
        E[rowptr:rowptr+r-1, offsetsB[i]+1 : offsetsB[i]+E1.dims[i]*F0.dims[i]] += blkB
        rowptr += r
    end
    Z1dim = size(nullspace_QQ(E), 2)

    # Boundaries B^1 = {(Phi \circ d1, -delta0 \circ Phi) | Phi:F0 \to E0 nat.}
    dimDom = _dim_hom_space(F0, E0)
    # ker(boundary-map) = { Phi nat. | Phi \circ d1 = 0 and delta0 \circ Phi = 0 }
    kerDim = _dim_hom_with_composition_constraints(F0, E0; L=d1, R=delta0)
    rankIm = dimDom - kerDim
    dimExt1 = Z1dim - rankIm

    return dimHom, dimExt1
end

export projective_cover, upset_presentation, upset_resolution,
       injective_hull, downset_copresentation,
       hom_ext_first_page

end # module
