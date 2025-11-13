module FlangeZn
# ------------------------------------------------------------------------------
# Flange presentations over Z^n (Section 5 in the paper).
#
# A flange presentation phi : F \to E has:
#   - F = \oplus flats   F_p  with F_p = k[b_p + N^n + Z*tau_p] (indecomp. FLAT)   (Rem. 5.11),
#   - E = \oplus inject. E_q  with E_q = k[b_q + Z*tau_q - N^n] (indecomp. INJECT.) (Rem. 5.5),
#   - scalar "monomial matrix" Phi = (phi_{q,p}); entry zero unless F_p \cap E_q \neq \emptyset
#     (Def. 5.14; Prop. 5.17).
# At degree g \in Z^n, the linear map phi_g is the submatrix of Phi with active
# rows/cols; the image equals M_g, so dim M_g = rank(phi_g).                    : see paper
# ------------------------------------------------------------------------------

using LinearAlgebra, SparseArrays

# --- Faces of the positive cone -------------------------------------------------

"""
    Face(n, coords)

A face `tau` of `N^n` encoded by a BitVector of length `n`: `coords[i] == true`
iff `e_i \in tau`.

- For indecomposable flat F = k[b + N^n + Z*tau], coordinates in `tau` are free (Z),
  others are *lower* bounded by `b`.
- For indecomp. injective E = k[b + Z*tau - N^n], coordinates in `tau` are free (Z),
  others are *upper* bounded by `b`.
"""
struct Face
    n::Int
    coords::BitVector  # length n
    function Face(n::Int, coords::AbstractVector{Bool})
        length(coords) == n || error("Face: length(coords) must be n")
        new(n, BitVector(coords))
    end
end

Base.show(io::IO, tau::Face) = print(io, "Face(", tau.n, ", ", findall(tau.coords), ")")

# --- Indecomposable flats/injectives -------------------------------------------

"""
    IndFlat{K}(b, tau; id=:F_p)

Indecomposable flat over Z^n:  `k[b + N^n + Z*tau]`.
- `b::Vector{Int}`: multidegree shift (length n).
- `tau::Face`: a face indicating which coordinates are free (Z-directions).
- `id::Symbol`: optional label for debugging/pretty printing.
"""
struct IndFlat{K}
    b::Vector{Int}
    tau::Face
    id::Symbol
end

"""
    IndInj{K}(b, tau; id=:E_q)

Indecomposable injective over Z^n: `k[b + Z*tau - N^n]`.
"""
struct IndInj{K}
    b::Vector{Int}
    tau::Face
    id::Symbol
end

# Membership at lattice degree g \in Z^n.
# For flats:  g \in b + N^n + Z*tau   \iff  g_i \geq b_i  for all i \not \in tau.
# For inj.:   g \in b + Z*tau - N^n   \iff  g_i \leq b_i  for all i \not \in tau.

@inline function in_flat(F::IndFlat, g::AbstractVector{<:Integer})
    @inbounds for i in 1:length(g)
        F.tau.coords[i] && continue
        if g[i] < F.b[i]; return false; end
    end
    return true
end

@inline function in_inj(E::IndInj, g::AbstractVector{<:Integer})
    @inbounds for i in 1:length(g)
        E.tau.coords[i] && continue
        if g[i] > E.b[i]; return false; end
    end
    return true
end

# Nonemptiness test for intersection F_p \cap E_q (Prop. 5.17):
# For each coordinate i with i \not \in tau_F and i \not \in tau_E, require b_F[i] \leq b_E[i].
function intersects(F::IndFlat, E::IndInj)
    n = length(F.b)
    n == length(E.b) || error("dimension mismatch")
    @inbounds for i in 1:n
        if !F.tau.coords[i] && !E.tau.coords[i] && (F.b[i] > E.b[i])
            return false
        end
    end
    return true
end

# --- Flange presentation container ---------------------------------------------

"""
    Flange{K}(n, flats, injectives, Phi)

Flange presentation `phi : (\oplus flats) \to (\oplus injectives)` over field `K`.

- `n`: ambient dimension.
- `flats::Vector{IndFlat{K}}`, `injectives::Vector{IndInj{K}}`.
- `Phi::Matrix{K}` has size (#injectives) \times (#flats), with `Phi[q,p] = 0` unless
  `intersects(flats[p], injectives[q])`.
"""
struct Flange{K}
    n::Int
    flats::Vector{IndFlat{K}}
    injectives::Vector{IndInj{K}}
    Phi::Matrix{K}
    function Flange{K}(n::Int, flats::Vector{IndFlat{K}},
                       injectives::Vector{IndInj{K}}, Phi::AbstractMatrix{K}) where {K}
        size(Phi,1) == length(injectives) || error("Phi rows must equal #injectives")
        size(Phi,2) == length(flats)      || error("Phi cols must equal #flats")
        # Zero-out forbidden entries for safety.
        Phi_c = Matrix{K}(Phi)
        for q in 1:length(injectives), p in 1:length(flats)
            if !intersects(flats[p], injectives[q]); Phi_c[q,p] = zero(K); end
        end
        new{K}(n, flats, injectives, Phi_c)
    end
end

"Canonical scalar matrix with 1's exactly when indecomposables intersect."
function canonical_matrix(flats::Vector{IndFlat{K}},
                          injectives::Vector{IndInj{K}}) where {K}
    Phi = zeros(K, length(injectives), length(flats))
    for q in 1:length(injectives), p in 1:length(flats)
        if intersects(flats[p], injectives[q]); Phi[q,p] = one(K); end
    end
    Phi
end

# --- Degree-wise evaluation -----------------------------------------------------

"Active flat column indices at degree g."
active_flats(FG::Flange, g::AbstractVector{<:Integer}) =
    [p for p in 1:length(FG.flats) if in_flat(FG.flats[p], g)]

"Active injective row indices at degree g."
active_injectives(FG::Flange, g::AbstractVector{<:Integer}) =
    [q for q in 1:length(FG.injectives) if in_inj(FG.injectives[q], g)]

"""
    degree_matrix(FG, g) -> (A, rows, cols)

Submatrix `A = phi_g` of the flange at degree `g`, obtained by restricting `Phi`
to active rows (injectives) and columns (flats).  `M_g = im(A)` so `dim M_g = rank(A)`.
"""
function degree_matrix(FG::Flange{K}, g::AbstractVector{<:Integer}) where {K}
    cols = active_flats(FG, g)
    rows = active_injectives(FG, g)
    return FG.Phi[rows, cols], rows, cols
end

"Dimension dim M_g = rank(phi_g). Supply an exact `rankfun` if desired."
function dim_at(FG::Flange, g::AbstractVector{<:Integer}; rankfun=rank)
    A, _, _ = degree_matrix(FG, g)
    isempty(A) ? 0 : rankfun(Matrix{eltype(A)}(A))
end

# --- Bounding box for finite encoding (Example 4.5) -----------------------------
# Heuristic [a,b] capturing nontrivial behavior: use lower bounds from flats (coords not in tau),
# and upper bounds from injectives (coords not in tau); inflate by `margin`.  Cf. paper Ex. 4.5.

"""
    bounding_box(FG; margin=1) -> (a::Vector{Int}, b::Vector{Int})

Return a conservative Z^n box `[a,b]` that contains the interesting part of the module.
"""
function bounding_box(FG::Flange; margin::Int=1)
    n = FG.n
    lo = fill( typemax(Int), n)
    hi = fill( typemin(Int), n)
    # From flats: lower bounds in coords not in tau.
    for F in FG.flats
        for i in 1:n
            if !F.tau.coords[i]
                lo[i] = min(lo[i], F.b[i])
            end
        end
    end
    # From injectives: upper bounds in coords not in tau.
    for E in FG.injectives
        for i in 1:n
            if !E.tau.coords[i]
                hi[i] = max(hi[i], E.b[i])
            end
        end
    end
    # Fill unconstrained coordinates reasonably and inflate margins.
    for i in 1:n
        if lo[i] == typemax(Int); lo[i] = -5; end
        if hi[i] == typemin(Int); hi[i] =  5; end
        lo[i] -= margin; hi[i] += margin
    end
    return lo, hi
end

export Face, IndFlat, IndInj, Flange, canonical_matrix,
       in_flat, in_inj, intersects,
       active_flats, active_injectives, degree_matrix, dim_at, bounding_box

end # module
