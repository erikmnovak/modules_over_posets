module FiniteFringe

using SparseArrays, LinearAlgebra

# =========================================
# Finite poset and indicator sets
# =========================================

struct FinitePoset
    n::Int
    leq::BitMatrix           # leq[i,j] = true  iff i <= j
    function FinitePoset(leq::BitMatrix)
        n1, n2 = size(leq)
        @assert n1 == n2 "leq must be square"
        @assert all(diag(leq)) "leq must be reflexive"
        new(n1, leq)
    end
end
leq(P::FinitePoset, i::Int, j::Int) = P.leq[i,j]

# Hasse cover relation (needed for resolutions):
# edge i -> j iff i<j and no k with i<k<j
function cover_edges(P::FinitePoset)
    C = falses(P.n, P.n)
    for i in 1:P.n, j in 1:P.n
        if i != j && leq(P, i, j) && !leq(P, j, i) # i<j
            # check minimality
            is_cover = true
            for k in 1:P.n
                if k != i && k != j && leq(P,i,k) && leq(P,k,j) && (leq(P,k,i)==false) && (leq(P,j,k)==false)
                    is_cover = false
                    break
                end
            end
            C[i,j] = is_cover
        end
    end
    C
end

struct Upset
    P::FinitePoset
    mask::BitVector
end
struct Downset
    P::FinitePoset
    mask::BitVector
end
Base.length(U::Upset) = length(U.mask)
Base.length(D::Downset) = length(D.mask)

is_subset(U1::Upset, U2::Upset) = all(.!U1.mask .| U2.mask)
is_subset(D1::Downset, D2::Downset) = all(.!D1.mask .| D2.mask)
intersects(U::Upset, D::Downset) = any(U.mask .& D.mask)

# Upset/downset closures and principal sets (used in resolutions)
function upset_closure(P::FinitePoset, S::BitVector)
    U = copy(S)
    for i in 1:P.n, j in 1:P.n
        if U[i] && leq(P,i,j); U[j] = true; end
    end
    Upset(P, U)
end
function downset_closure(P::FinitePoset, S::BitVector)
    D = copy(S)
    for j in 1:P.n, i in 1:P.n
        if D[j] && leq(P,i,j); D[i] = true; end
    end
    Downset(P, D)
end

upset_from_generators(P::FinitePoset, gens::Vector{Int}) =
    upset_closure(P, BitVector([i in gens for i in 1:P.n]))
downset_from_generators(P::FinitePoset, gens::Vector{Int}) =
    downset_closure(P, BitVector([i in gens for i in 1:P.n]))

# Principal upset \uparrow p and principal downset \downarrow p (representables/corepresentables).
principal_upset(P::FinitePoset, p::Int) = Upset(P, BitVector([leq(P, p, q) for q in 1:P.n]))
principal_downset(P::FinitePoset, p::Int) = Downset(P, BitVector([leq(P, q, p) for q in 1:P.n]))

# =========================================
# Fringe presentations (Defs. 3.16 - 3.17)
# =========================================

struct FringeModule{K}
    P::FinitePoset
    U::Vector{Upset}                # births (columns)
    D::Vector{Downset}              # deaths (rows)
    phi::SparseMatrixCSC{K,Int}       # size |D| \times |U|
end

# Prop. 3.18: Nonzero entry only if U_i \cap D_j \neq \emptyset.
function _check_monomial_condition(M::FringeModule)
    m, n = size(M.phi)
    @assert m == length(M.D) && n == length(M.U) "Dimension mismatch"
    for j in 1:m, i in 1:n
        v = M.phi[j,i]
        if v != zero(v)
            @assert intersects(M.U[i], M.D[j]) "Nonzero phi[j,i] requires U[i] \cap D[j] \neq \emptyset (Prop. 3.18)"
        end
    end
end

FringeModule{K}(P::FinitePoset, U::Vector{Upset}, D::Vector{Downset}, phi::SparseMatrixCSC{K,Int}) where {K} = begin
    M = FringeModule{K}(P, U, D, phi)
    _check_monomial_condition(M)
    M
end

# ------------------ evaluation (degreewise image; after Def. 3.17) ------------------

# Simple exact RREF rank
function _rank_rref!(A::AbstractMatrix)
    M = copy(Matrix(A)); m, n = size(M)
    r = 0; c = 1
    while r < m && c <= n
        pivot = 0
        for i in (r+1):m
            if !iszero(M[i,c]); pivot = i; break; end
        end
        if pivot == 0; c += 1; continue; end
        r += 1
        if pivot != r; M[r,:], M[pivot,:] = M[pivot,:], M[r,:]; end
        p = M[r,c]; M[r,:] ./= p
        for i in 1:m
            if i != r && !iszero(M[i,c]); M[i,:] .-= M[i,c] .* M[r,:]; end
        end
        c += 1
    end
    r
end

"""
    fiber_dimension(M::FringeModule{K}, q::Int) -> Int

Compute dim_k M_q as rank of phi_q : F_q \to E_q (degreewise image).
"""
function fiber_dimension(M::FringeModule{K}, q::Int) where {K}
    cols = findall(U -> U.mask[q], M.U)
    rows = findall(D -> D.mask[q], M.D)
    if isempty(cols) || isempty(rows); return 0; end
    phi_q = Matrix(M.phi[rows, cols])
    _rank_rref!(phi_q)
end

# ------------------ Hom for fringe modules via commuting squares ------------------

# Bases of allowed indicator Hom blocks (connected homomorphisms; Prop. 3.10 + Cor. 3.11)

struct HomBasis
    pairs::Vector{Tuple{Int,Int}}
    index::Dict{Tuple{Int,Int},Int}
    dim::Int
end

function _V1_basis(M::FringeModule, N::FringeModule)
    pairs = Tuple{Int,Int}[]
    for (iM, UM) in enumerate(M.U), (jN, UN) in enumerate(N.U)
        if is_subset(UM, UN)  # Hom(k[UM],k[UN]) is 1-dim iff UM \subseteq UN
            push!(pairs, (iM, jN))
        end
    end
    HomBasis(pairs, Dict((p,k) for (k,p) in enumerate(pairs)), length(pairs))
end

function _V2_basis(M::FringeModule, N::FringeModule)
    pairs = Tuple{Int,Int}[]
    for (sM, DM) in enumerate(M.D), (tN, DN) in enumerate(N.D)
        if is_subset(DN, DM) # Hom(k[DM],k[DN]) 1-dim iff DN \subseteq DM
            push!(pairs, (sM, tN))
        end
    end
    HomBasis(pairs, Dict((p,k) for (k,p) in enumerate(pairs)), length(pairs))
end

function _W_basis(M::FringeModule, N::FringeModule)
    pairs = Tuple{Int,Int}[]
    for (iM, UM) in enumerate(M.U), (tN, DN) in enumerate(N.D)
        if intersects(UM, DN) # Hom(k[UM],k[DN]) 1-dim iff UM \cap DN \neq \emptyset (connected)
            push!(pairs, (iM, tN))
        end
    end
    HomBasis(pairs, Dict((p,k) for (k,p) in enumerate(pairs)), length(pairs))
end

"""
    hom_dimension(M, N) -> Int

dim_k Hom_Q(M,N), computed by solving phi_N \circ bega = alpha \circ phi_M and modding out null pairs.
(See discussion under Defs. 3.16 - 3.17 and Prop. 3.18; this is the two-term Hom complex in degree 0.)
"""
function hom_dimension(M::FringeModule{K}, N::FringeModule{K}) where {K}
    V1 = _V1_basis(M,N)   # beta : F_M \to F_N
    V2 = _V2_basis(M,N)   # alpha : E_M \to E_N
    W  = _W_basis(M,N)    # F_M \to E_N

    T = spzeros(K, W.dim, V1.dim)  # T(beta) = phi_N \circ beta
    for (col, (iM, jN)) in enumerate(V1.pairs)
        for tN in 1:length(N.D)
            val = N.phi[tN, jN]
            if val != zero(K)
                if haskey(W.index, (iM, tN))
                    row = W.index[(iM, tN)]
                    T[row, col] += val
                end
            end
        end
    end

    S = spzeros(K, W.dim, V2.dim)  # S(alpha) = alpha \circ phi_M
    for (col, (sM, tN)) in enumerate(V2.pairs)
        for iM in 1:length(M.U)
            val = M.phi[sM, iM]
            if val != zero(K)
                if haskey(W.index, (iM, tN))
                    row = W.index[(iM, tN)]
                    S[row, col] += val
                end
            end
        end
    end

    Tdense = Matrix(T); Sdense = Matrix(S); big = hcat(Tdense, -Sdense)

    rT   = _rank_rref!(Tdense)
    rS   = _rank_rref!(Sdense)
    rbig = _rank_rref!(big)

    dimKer_T   = size(Tdense,2) - rT
    dimKer_S   = size(Sdense,2) - rS
    dimKer_big = size(big,2)    - rbig

    dimKer_big - (dimKer_T + dimKer_S)
end

# ---------- utility: dense\tosparse over K ----------
function dense_to_sparse_K(A::AbstractMatrix{T}, ::Type{K}) where {T,K}
    m,n = size(A)
    S = spzeros(K, m, n)
    for j in 1:n, i in 1:m
        v = K(A[i,j])
        if v != zero(K); S[i,j] = v; end
    end
    S
end

export FinitePoset, Upset, Downset, principal_upset, principal_downset,
       upset_from_generators, downset_from_generators, cover_edges,
       FringeModule, fiber_dimension, hom_dimension, dense_to_sparse_K

end # module
