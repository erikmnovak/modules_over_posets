module FlangeZn
# ------------------------------------------------------------------------------
# Flange presentations over Z^n (Miller section 5).
#
# A flange Phi : (\oplus flats) \to (\oplus injectives) is stored as a scalar matrix Phi with
# rows = indecomposable injectives, cols = indecomposable flats.  At a degree g,
# take the submatrix with the active rows/cols; dim M_g = rank(Phi_g).
# (Remark 5.5, 5.11; Def. 5.14; Prop. 5.17.)  See also section 1.4 for the narrative.
# ------------------------------------------------------------------------------

using LinearAlgebra
using ..CoreModules: QQ

# ---------------- faces, flats, injectives (ASCII) --------------------------------
"Face tau of N^n: a bitmask of 'free' coordinates (Z-directions)."
struct Face
    n::Int
    coords::BitVector
    function Face(n::Int, coords::AbstractVector{Bool})
        length(coords) == n || error("Face: coords must have length n")
        new(n, BitVector(coords))
    end
end

"Indecomposable flat F = k[b + N^n + Z^tau]."
struct IndFlat{K}
    b::Vector{Int}
    tau::Face
    id::Symbol
end

"Indecomposable injective E = k[b + Z^tau - N^n]."
struct IndInj{K}
    b::Vector{Int}
    tau::Face
    id::Symbol
end

# Membership tests at degree g in Z^n
@inline function in_flat(F::IndFlat, g::AbstractVector{<:Integer})
    @inbounds for i in 1:length(g)
        F.tau.coords[i] && continue
        if g[i] < F.b[i]; return false; end
    end
    true
end

@inline function in_inj(E::IndInj, g::AbstractVector{<:Integer})
    @inbounds for i in 1:length(g)
        E.tau.coords[i] && continue
        if g[i] > E.b[i]; return false; end
    end
    true
end

"Non-emptiness of the intersection F cap E (Prop. 5.17)."
function intersects(F::IndFlat, E::IndInj)
    n = length(F.b); n == length(E.b) || error("dimension mismatch")
    @inbounds for i in 1:n
        if !F.tau.coords[i] && !E.tau.coords[i] && (F.b[i] > E.b[i])
            return false
        end
    end
    true
end

# ---------------- flange container -------------------------------------------------
"Flange Phi : (oplus flats) to (oplus injectives) with scalar matrix Phi."
struct Flange{K}
    n::Int
    flats::Vector{IndFlat{K}}
    injectives::Vector{IndInj{K}}
    Phi::Matrix{K}
    function Flange{K}(n::Int, flats::Vector{IndFlat{K}},
                       injectives::Vector{IndInj{K}}, Phi::AbstractMatrix{K}) where {K}
        size(Phi,1) == length(injectives) || error("Phi rows must equal #injectives")
        size(Phi,2) == length(flats)      || error("Phi cols must equal #flats")
        M = Matrix{K}(Phi)
        for q in 1:length(injectives), p in 1:length(flats)
            if !intersects(flats[p], injectives[q]); M[q,p] = zero(K); end
        end
        new{K}(n, flats, injectives, M)
    end
end

"Canonical scalar matrix (1 when F_p cap E_q neq emptyset, else 0)."
function canonical_matrix(flats::Vector{IndFlat{K}},
                          injectives::Vector{IndInj{K}}) where {K}
    Phi = zeros(K, length(injectives), length(flats))
    for q in 1:length(injectives), p in 1:length(flats)
        if intersects(flats[p], injectives[q]); Phi[q,p] = one(K); end
    end
    Phi
end

# ---------------- degree-wise evaluation ------------------------------------------
"Active column indices (flats) at degree g."
active_flats(FG::Flange, g::AbstractVector{<:Integer}) =
    [p for p in 1:length(FG.flats) if in_flat(FG.flats[p], g)]

"Active row indices (injectives) at degree g."
active_injectives(FG::Flange, g::AbstractVector{<:Integer}) =
    [q for q in 1:length(FG.injectives) if in_inj(FG.injectives[q], g)]

"Return Phi_g = Phi[rows,cols] together with the row/col indices."
function degree_matrix(FG::Flange{K}, g::AbstractVector{<:Integer}) where {K}
    cols = active_flats(FG, g)
    rows = active_injectives(FG, g)
    return FG.Phi[rows, cols], rows, cols
end

"Dimension dim M_g = rank(Phi_g)."
function dim_at(FG::Flange, g::AbstractVector{<:Integer}; rankfun=rank)
    A, _, _ = degree_matrix(FG, g)
    isempty(A) ? 0 : rankfun(Matrix{eltype(A)}(A))
end

# ---------------- conservative bounding box for finite encoding --------------------
"Crude bounding box [a,b] that contains all non-trivial behaviour."
function bounding_box(FG::Flange; margin::Int=1)
    n = FG.n
    a = fill(div(typemin(Int),4), n)
    b = fill(div(typemax(Int),4), n)
    # Coordinates constrained by flats set lower bounds; by injectives set upper bounds
    for F in FG.flats, i in 1:n
        if !F.tau.coords[i]; a[i] = max(a[i], F.b[i] - margin); end
    end
    for E in FG.injectives, i in 1:n
        if !E.tau.coords[i]; b[i] = min(b[i], E.b[i] + margin); end
    end
    a, b
end


# --------------------- flange minimization ----------------------------------------
"""
    minimize(FG::Flange{K}) -> Flange{K}

Return a flange with:
  * zero columns (flat summands that never contribute) removed,
  * zero rows   (injective summands unused) removed,
  * duplicate flat columns that are proportional merged to one,
  * duplicate injective rows that are proportional merged to one.

The represented image submodule does not change; only the presentation shrinks.
"""
function minimize(FG::Flange{K}) where {K}
    Phi = FG.Phi
    m, n = size(Phi)

    # 1) drop zero columns and rows
    keep_cols = [any(x -> !iszero(x), Phi[:, j]) for j in 1:n]
    keep_rows = [any(x -> !iszero(x), Phi[i, :]) for i in 1:m]
    flats1 = [FG.flats[j] for j in 1:n if keep_cols[j]]
    inject1 = [FG.injectives[i] for i in 1:m if keep_rows[i]]
    Phi1 = Phi[keep_rows, keep_cols]

    # helper: detect proportional duplicates
    function proportional_groups_cols(A::AbstractMatrix{K})
        n1 = size(A, 2)
        groups = Dict{Int, Vector{Int}}()
        used = falses(n1)
        for j in 1:n1
            if used[j]; continue; end
            v = A[:, j]
            if all(iszero, v)
                groups[j] = [j]; used[j] = true; continue
            end
            groups[j] = [j]; used[j] = true
            for k in (j+1):n1
                if used[k]; continue; end
                w = A[:, k]
                if all(iszero, w); continue; end
                # test proportionality v ~ w
                t = 0
                for i in 1:length(v)
                    if !iszero(v[i])
                        t = i; break
                    end
                end
                if t == 0; continue; end
                alpha = w[t] / v[t]
                ok = true
                for i in 1:length(v)
                    if w[i] != alpha * v[i]; ok = false; break; end
                end
                if ok
                    push!(groups[j], k); used[k] = true
                end
            end
        end
        groups
    end

    function proportional_groups_rows(A::AbstractMatrix{K})
        proportional_groups_cols(transpose(A))
    end

    # 2) merge proportional duplicate columns (keep one per group)
    groupsC = proportional_groups_cols(Phi1)
    keepC = falses(size(Phi1,2)); @inbounds for j in keys(groupsC); keepC[j] = true; end
    flats2 = [flats1[j] for j in 1:length(flats1) if keepC[j]]
    Phi2 = Phi1[:, keepC]

    # 3) merge proportional duplicate rows (keep one per group)
    groupsR = proportional_groups_rows(Phi2)
    keepR = falses(size(Phi2,1)); @inbounds for i in keys(groupsR); keepR[i] = true; end
    inject2 = [inject1[i] for i in 1:length(inject1) if keepR[i]]
    Phi3 = Phi2[keepR, :]

    # Rebuild flange (constructor will re-zero forbidden entries)
    return Flange{K}(FG.n, flats2, inject2, Phi3)
end



# -------------- compatibility aliases for existing client code ---------------------
const ZnFace       = Face
const ZnFlat       = IndFlat
const ZnInjective  = IndInj
const FlangePresentation = Flange

export Face, IndFlat, IndInj, Flange, canonical_matrix,
       active_flats, active_injectives, degree_matrix, dim_at, bounding_box,
       ZnFace, ZnFlat, ZnInjective, FlangePresentation
end # module
