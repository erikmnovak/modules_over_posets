module DerivedFunctors

using LinearAlgebra
using SparseArrays

using ..CoreModules: QQ
using ..ExactQQ: rankQQ, nullspaceQQ, colspaceQQ, solve_fullcolumnQQ
using ..FiniteFringe: FinitePoset, FringeModule, cover_edges
using ..IndicatorResolutions: PModule, PMorphism, pmodule_from_fringe, map_leq, projective_cover, kernel_with_inclusion, _injective_hull, _cokernel_module
using ..ChainComplexes
using ..ChainComplexes: CochainComplex, cohomology_data, cohomology_coordinates, induced_map_on_cohomology
using ..ZnEncoding

# ----------------------------
# Basic utilities: morphism composition (local, explicit, reliable)
# ----------------------------

function compose(g::PMorphism{QQ}, f::PMorphism{QQ})
    @assert f.cod === g.dom
    Q = f.dom.Q
    comps = Vector{Matrix{QQ}}(undef, Q.n)
    for i in 1:Q.n
        comps[i] = g.comps[i] * f.comps[i]
    end
    return PMorphism{QQ}(f.dom, g.cod, comps)
end

# ----------------------------
# Hom space with explicit basis
# ----------------------------

struct HomSpace{K}
    dom::PModule{K}
    cod::PModule{K}
    basis::Vector{PMorphism{K}}
    basis_matrix::Matrix{K}  # columns are vectorizations of basis morphisms
    offsets::Vector{Int}     # per vertex block offsets in the vectorization
end

function _hom_offsets(M::PModule{QQ}, N::PModule{QQ})
    Q = M.Q
    @assert N.Q === Q
    offs = zeros(Int, Q.n + 1)
    for i in 1:Q.n
        offs[i+1] = offs[i] + N.dims[i] * M.dims[i]
    end
    return offs
end

function _morphism_to_vector(f::PMorphism{QQ}, offs::Vector{Int})
    Q = f.dom.Q
    v = zeros(QQ, offs[end], 1)
    for i in 1:Q.n
        di = f.cod.dims[i]
        ei = f.dom.dims[i]
        if di == 0 || ei == 0
            continue
        end
        block = vec(f.comps[i]) # column-major
        s = offs[i] + 1
        t = offs[i+1]
        v[s:t, 1] = block
    end
    return v
end

function _vector_to_morphism(dom::PModule{QQ}, cod::PModule{QQ}, offs::Vector{Int}, x::Vector{QQ})
    Q = dom.Q
    comps = Vector{Matrix{QQ}}(undef, Q.n)
    for i in 1:Q.n
        di = cod.dims[i]
        ei = dom.dims[i]
        if di == 0 || ei == 0
            comps[i] = zeros(QQ, di, ei)
            continue
        end
        s = offs[i] + 1
        t = offs[i+1]
        comps[i] = reshape(x[s:t], di, ei)
    end
    return PMorphism{QQ}(dom, cod, comps)
end

# Compute a full explicit Hom basis by solving naturality constraints along cover edges.
function Hom(M::PModule{QQ}, N::PModule{QQ})
    Q = M.Q
    @assert N.Q === Q
    C = cover_edges(Q)

    offs = _hom_offsets(M, N)
    nvars = offs[end]

    # count equations
    neqs = 0
    for u in 1:Q.n
        for v in 1:Q.n
            if C[u, v]
                neqs += N.dims[v] * M.dims[u]
            end
        end
    end

    A = zeros(QQ, neqs, nvars)
    row = 0

    # variable index for entry (i, r, c) in matrix X_i : N_i x M_i
    # column-major: index = offs[i] + (c-1)*N.dims[i] + r
    function var_index(i::Int, r::Int, c::Int)
        return offs[i] + (c - 1) * N.dims[i] + r
    end

    for u in 1:Q.n
        for v in 1:Q.n
            if !C[u, v]
                continue
            end
            Nu = N.edge_maps[(u, v)]  # N_u -> N_v
            Mu = M.edge_maps[(u, v)]  # M_u -> M_v

            for rr in 1:N.dims[v]
                for cc in 1:M.dims[u]
                    row += 1
                    # left side: (Nu * X_u)[rr, cc]
                    for kk in 1:N.dims[u]
                        A[row, var_index(u, kk, cc)] += Nu[rr, kk]
                    end
                    # right side: (X_v * Mu)[rr, cc]
                    for kk in 1:M.dims[v]
                        A[row, var_index(v, rr, kk)] -= Mu[kk, cc]
                    end
                end
            end
        end
    end

    NS = nullspaceQQ(A)          # nvars x dimHom
    dimHom = size(NS, 2)

    basis = Vector{PMorphism{QQ}}(undef, dimHom)
    for j in 1:dimHom
        basis[j] = _vector_to_morphism(M, N, offs, NS[:, j])
    end

    return HomSpace{QQ}(M, N, basis, NS, offs)
end

# Convenience overload for fringe modules
function Hom(M::FringeModule{QQ}, N::FringeModule{QQ})
    Mp = pmodule_from_fringe(M)
    Np = pmodule_from_fringe(N)
    return Hom(Mp, Np)
end

dimension(H::HomSpace) = length(H.basis)
basis(H::HomSpace) = H.basis

# ----------------------------
# Projective resolution (explicit summands + coefficient matrices)
# ----------------------------

struct ProjectiveResolution{K}
    M::PModule{K}
    Pmods::Vector{PModule{K}}                    # P_0 .. P_L
    gens::Vector{Vector{Int}}                    # base vertex per summand (same order as summands)
    d_mor::Vector{PMorphism{K}}                  # d_a : P_a -> P_{a-1}, a=1..L
    d_mat::Vector{SparseMatrixCSC{K, Int}}       # coefficient matrices (rows cod summands, cols dom summands)
    aug::PMorphism{K}                            # P_0 -> M
end

function _flatten_gens_at(gens_at)
    out = Int[]
    for u in 1:length(gens_at)
        for tup in gens_at[u]
            push!(out, tup[1])
        end
    end
    return out
end

function _active_upset_indices(P::FinitePoset, base_vertices::Vector{Int})
    active = Vector{Vector{Int}}(undef, P.n)
    for u in 1:P.n
        idxs = Int[]
        for i in 1:length(base_vertices)
            if P.leq[base_vertices[i], u]
                push!(idxs, i)
            end
        end
        active[u] = idxs
    end
    return active
end


"""
    _active_downset_indices(P::FinitePoset, base_vertices::Vector{Int}) -> Vector{Vector{Int}}

For a direct sum of principal downsets

    ⊕_i k[Dn(base_vertices[i])],

return `active[u]` = the list of summand indices that are active at vertex `u`.

Convention:
- A principal downset Dn(v) contains u iff u <= v.
- Summand i is active at u iff P.leq[u, base_vertices[i]].

The returned lists are in increasing summand index order. This matches the fiber
basis ordering used in `_injective_hull` and makes coefficient extraction stable.
"""
function _active_downset_indices(P::FinitePoset, base_vertices::Vector{Int})
    active = [Int[] for _ in 1:P.n]
    for i in 1:length(base_vertices)
        v = base_vertices[i]
        for u in 1:P.n
            if P.leq[u, v]
                push!(active[u], i)
            end
        end
    end
    return active
end


"""
    _coeff_matrix_downsets(P, dom_bases, cod_bases, f) -> SparseMatrixCSC{QQ,Int}

Extract the scalar coefficient matrix of a morphism between direct sums of
principal downsets.

Inputs:
- `dom_bases`: base vertices of the domain summands (each summand is k[Dn(v)]).
- `cod_bases`: base vertices of the codomain summands (each summand is k[Dn(w)]).
- `f`: a `PMorphism{QQ}` whose domain/codomain are those direct sums.

Output:
- A sparse matrix C of size (length(cod_bases) x length(dom_bases)) such that
  C[row, col] is the scalar multiplying the unique (up to scalar) map

      k[Dn(dom_bases[col])] -> k[Dn(cod_bases[row])].

Implementation detail:
- For downsets, the distinguished generator of k[Dn(w)] lives at vertex w.
  To read the scalar for a map into k[Dn(w)], evaluate at u = w (the codomain
  base vertex), where the codomain generator is visible.
"""
function _coeff_matrix_downsets(P::FinitePoset,
                                dom_bases::Vector{Int},
                                cod_bases::Vector{Int},
                                f::PMorphism{QQ})
    n_dom = length(dom_bases)
    n_cod = length(cod_bases)

    active_dom = _active_downset_indices(P, dom_bases)
    active_cod = _active_downset_indices(P, cod_bases)

    C = spzeros(QQ, n_cod, n_dom)

    for row in 1:n_cod
        u = cod_bases[row]   # read at the codomain base vertex
        act_dom_u = active_dom[u]
        act_cod_u = active_cod[u]

        # Locate this codomain summand in the fiber basis at u.
        pos_row = findfirst(x -> x == row, act_cod_u)
        if pos_row === nothing
            error("_coeff_matrix_downsets: could not locate cod summand in fiber basis")
        end

        Fu = f.comps[u]
        # Read the row restricted to active domain summands at u.
        for pos_col in 1:length(act_dom_u)
            col = act_dom_u[pos_col]
            val = Fu[pos_row, pos_col]
            if !iszero(val)
                C[row, col] = val
            end
        end
    end

    return C
end



# Extract coefficient matrix for a morphism between sums of principal upsets.
# Domain and codomain are direct sums of principal upsets indexed by base vertex lists.
function _coeff_matrix_upsets(P::FinitePoset,
                              dom_bases::Vector{Int}, cod_bases::Vector{Int},
                              f::PMorphism{QQ})
    n_dom = length(dom_bases)
    n_cod = length(cod_bases)

    active_dom = _active_upset_indices(P, dom_bases)
    active_cod = _active_upset_indices(P, cod_bases)

    M = zeros(QQ, n_cod, n_dom)

    for i in 1:n_dom
        u = dom_bases[i]
        act_dom_u = active_dom[u]
        pos_dom = searchsortedfirst(act_dom_u, i)
        if pos_dom > length(act_dom_u) || act_dom_u[pos_dom] != i
            error("_coeff_matrix_upsets: could not locate dom summand in fiber basis")
        end

        act_cod_u = active_cod[u]
        Cu = f.comps[u]
        for (rowpos, j) in enumerate(act_cod_u)
            M[j, i] = Cu[rowpos, pos_dom]
        end
    end

    return sparse(M)
end

function projective_resolution(M::PModule{QQ}; maxlen::Int=3)
    # Step 0
    P0, pi0, gens0 = projective_cover(M)
    bases0 = _flatten_gens_at(gens0)

    Pmods = PModule{QQ}[P0]
    gens = Vector{Int}[bases0]
    d_mor = PMorphism{QQ}[]
    d_mat = SparseMatrixCSC{QQ, Int}[]

    # kernel K1 -> P0
    K, iota = kernel_with_inclusion(pi0)

    prevP = P0
    prevBases = bases0
    prevK = K
    prevIota = iota

    for step in 1:maxlen
        # stop if kernel is zero
        if sum(prevK.dims) == 0
            break
        end

        Pn, pin, gensn = projective_cover(prevK)
        basesn = _flatten_gens_at(gensn)

        # differential d_step = prevIota circ pin : Pn -> prevP
        d = compose(prevIota, pin)
        push!(Pmods, Pn)
        push!(gens, basesn)
        push!(d_mor, d)
        push!(d_mat, _coeff_matrix_upsets(M.Q, basesn, prevBases, d))

        # next kernel
        Kn, iotan = kernel_with_inclusion(pin)

        prevP = Pn
        prevBases = basesn
        prevK = Kn
        prevIota = iotan
    end

    return ProjectiveResolution{QQ}(M, Pmods, gens, d_mor, d_mat, pi0)
end

# Convenience overload for fringe modules
function projective_resolution(M::FringeModule{QQ}; maxlen::Int=3)
    return projective_resolution(pmodule_from_fringe(M); maxlen=maxlen)
end

# =============================================================================
# Betti and Bass numbers (multiplicities of indecomposable summands)
# =============================================================================

"""
    betti(res::ProjectiveResolution{QQ}) -> Dict{Tuple{Int,Int},Int}

Return the Betti numbers of a projective resolution.

Interpretation:
- `res.Pmods[a+1]` is the projective in homological degree `a`.
- Each term splits as a direct sum of indecomposable projectives k[Up(v)].
- `res.gens[a+1]` stores the base vertex `v` for each summand, with repetition.

Output convention:
- The dictionary key `(a, v)` means "homological degree a, vertex v".
- The value is the multiplicity of k[Up(v)] in P_a.
- Only positive multiplicities appear as keys.

This is the poset-module analogue of multigraded Betti numbers in commutative algebra.
It is *not* a polynomial-ring Betti table unless you have explicitly modeled a
polynomial-ring module as a poset module and computed its resolution in that category.
"""
function betti(res::ProjectiveResolution{QQ})
    out = Dict{Tuple{Int,Int},Int}()
    L = length(res.Pmods) - 1
    for a in 0:L
        for v in res.gens[a+1]
            key = (a, v)
            out[key] = get(out, key, 0) + 1
        end
    end
    return out
end


"""
    betti_table(res::ProjectiveResolution{QQ}) -> Matrix{Int}

Return a dense Betti table B.

- Rows are homological degrees a = 0,1,2,...
- Columns are vertices v = 1,...,Q.n
- Entry B[a+1, v] is the multiplicity of k[Up(v)] in P_a.

This is purely a formatting/convenience layer over `betti(res)`.
"""
function betti_table(res::ProjectiveResolution{QQ})
    Q = res.M.Q
    L = length(res.Pmods) - 1
    B = zeros(Int, L+1, Q.n)
    for a in 0:L
        for v in res.gens[a+1]
            B[a+1, v] += 1
        end
    end
    return B
end


"""
    betti(M::PModule{QQ}; maxlen::Int=3) -> Dict{Tuple{Int,Int},Int}

Convenience wrapper:
- build `projective_resolution(M; maxlen=maxlen)`,
- return its Betti numbers.

If you need full control over the chosen resolution object, call
`projective_resolution` yourself and then call `betti(res)`.
"""
function betti(M::PModule{QQ}; maxlen::Int=3)
    return betti(projective_resolution(M; maxlen=maxlen))
end

# ----------------------------
# Minimality diagnostics for projective resolutions
# ----------------------------

"""
    _vertex_counts(bases::Vector{Int}, nverts::Int) -> Vector{Int}

Return the multiplicity vector of vertices in `bases`.

If `bases` is the list of base vertices for a direct sum of principal upsets,
then the output c satisfies:

    c[v] = number of copies of k[Up(v)].

We use this for minimality certification, since multiplicities are the canonical
data of a minimal resolution (up to isomorphism).
"""
function _vertex_counts(bases::Vector{Int}, nverts::Int)
    c = zeros(Int, nverts)
    for v in bases
        c[v] += 1
    end
    return c
end


"""
    ProjectiveMinimalityReport

Returned by `minimality_report(res::ProjectiveResolution{QQ})`.

Fields:
- `minimal`:
    True iff all requested checks passed.

- `cover_ok`:
    True iff the augmentation P0 -> M is a projective cover, checked by comparing
    vertex multiplicities with a freshly computed cover of M.

- `cover_expected`, `cover_actual`:
    Multiplicity vectors of principal upsets in a projective cover of M and in
    the resolution's P0.

- `diagonal_violations`:
    A list of tuples (a, v, row, col, val) witnessing non-minimality in higher
    degrees. Interpretation:
      - a is the homological degree of the differential d_a : P_a -> P_{a-1},
      - v is the vertex,
      - (row, col) is an entry in the coefficient matrix of d_a,
      - val is the nonzero scalar coefficient,
      - and the entry corresponds to a map k[Up(v)] -> k[Up(v)].
    Any such nonzero scalar is an isomorphism on that indecomposable summand,
    hence it splits off a contractible subcomplex. Minimal projective resolutions
    forbid this.
"""
struct ProjectiveMinimalityReport
    minimal::Bool
    cover_ok::Bool
    cover_expected::Vector{Int}
    cover_actual::Vector{Int}
    diagonal_violations::Vector{Tuple{Int,Int,Int,Int,QQ}}
end


"""
    minimality_report(res::ProjectiveResolution{QQ}; check_cover=true) -> ProjectiveMinimalityReport

Certify minimality of a projective resolution in the standard finite-dimensional
algebra sense (incidence algebra / poset representation sense).

Checks performed:

1. (Optional) `check_cover`:
   Verify that P0 -> M is a projective cover by comparing multiplicities of
   principal upsets against a freshly computed `projective_cover(M)`.

2. Verify the "no units on diagonal" condition:
   For every differential d_a : P_a -> P_{a-1}, there is no nonzero coefficient
   from a k[Up(v)] summand in P_a to a k[Up(v)] summand in P_{a-1}.
"""
function minimality_report(res::ProjectiveResolution{QQ}; check_cover::Bool=true)
    Q = res.M.Q
    n = Q.n

    cover_actual = _vertex_counts(res.gens[1], n)
    cover_expected = copy(cover_actual)
    cover_ok = true

    if check_cover
        _, _, gens_at = projective_cover(res.M)
        cover_expected = _vertex_counts(_flatten_gens_at(gens_at), n)
        cover_ok = (cover_expected == cover_actual)
    end

    # Detect forbidden k[Up(v)] -> k[Up(v)] coefficients in the differentials.
    violations = Tuple{Int,Int,Int,Int,QQ}[]
    for a in 1:length(res.d_mat)
        D = res.d_mat[a]
        dom = res.gens[a+1]   # bases in P_a
        cod = res.gens[a]     # bases in P_{a-1}

        for col in 1:size(D, 2)
            for ptr in D.colptr[col]:(D.colptr[col+1] - 1)
                row = D.rowval[ptr]
                val = D.nzval[ptr]
                if !iszero(val) && (cod[row] == dom[col])
                    v = dom[col]
                    push!(violations, (a, v, row, col, val))
                end
            end
        end
    end

    minimal = cover_ok && isempty(violations)
    return ProjectiveMinimalityReport(minimal, cover_ok, cover_expected, cover_actual, violations)
end


"""
    is_minimal(res::ProjectiveResolution{QQ}; check_cover=true) -> Bool

Return `true` iff `minimality_report(res; check_cover=check_cover).minimal` is true.
"""
function is_minimal(res::ProjectiveResolution{QQ}; check_cover::Bool=true)
    return minimality_report(res; check_cover=check_cover).minimal
end


"""
    assert_minimal(res::ProjectiveResolution{QQ}; check_cover=true) -> true

Throw a descriptive error if the resolution fails minimality checks.
Return `true` otherwise.

This is intended for:
- test suites,
- defensively checking assumptions before extracting "minimal Betti invariants".
"""
function assert_minimal(res::ProjectiveResolution{QQ}; check_cover::Bool=true)
    R = minimality_report(res; check_cover=check_cover)
    if R.minimal
        return true
    end

    if !R.cover_ok
        error("Projective resolution is not minimal: P0 -> M is not a projective cover. " *
              "Expected cover multiplicities = $(R.cover_expected), got = $(R.cover_actual).")
    end

    if !isempty(R.diagonal_violations)
        (a, v, row, col, val) = R.diagonal_violations[1]
        error("Projective resolution is not minimal: differential d_$a has a nonzero coefficient " *
              "val = $val from k[Up($v)] in P_$a (column $col) to k[Up($v)] in P_$(a-1) (row $row).")
    end

    error("Projective resolution failed minimality checks for an unknown reason.")
end


"""
    minimal_projective_resolution(M::PModule{QQ}; maxlen=3, check=true) -> ProjectiveResolution{QQ}

Build a projective resolution via successive projective covers (the standard minimal
construction in the finite-dimensional algebra setting).

If `check=true`, call `assert_minimal` before returning.
"""
function minimal_projective_resolution(M::PModule{QQ}; maxlen::Int=3, check::Bool=true)
    res = projective_resolution(M; maxlen=maxlen)
    if check
        assert_minimal(res; check_cover=true)
    end
    return res
end


"""
    minimal_betti(M::PModule{QQ}; maxlen=3, check=true) -> Dict{Tuple{Int,Int},Int}

Compute Betti multiplicities from a certified minimal projective resolution.

This is a user-facing "invariant" function:
- it constructs the minimal resolution,
- optionally certifies minimality,
- and returns the Betti table.

If you already have a resolution in hand, use `betti(res)` instead.
"""
function minimal_betti(M::PModule{QQ}; maxlen::Int=3, check::Bool=true)
    res = minimal_projective_resolution(M; maxlen=maxlen, check=check)
    return betti(res)
end


# ----------------------------
# Ext via projective resolution: explicit cochains, cycles, boundaries, basis
# ----------------------------

struct ExtSpaceProjective{K}
    res::ProjectiveResolution{K}
    N::PModule{K}
    complex::ChainComplexes.CochainComplex{K}
    offsets::Vector{Vector{Int}}                       # per degree, block offsets
    cohom::Vector{ChainComplexes.CohomologyData{K}}     # per degree
end

# -----------------------------------------------------------------------------
# Convenience properties for Ext spaces
#
# We store the degree range on the underlying cochain complex, but much of the code
# is written in the natural mathematical style `E.tmax` instead of `E.complex.tmax`.
# These forwarders make that syntax correct and explicit.
# -----------------------------------------------------------------------------

function Base.getproperty(E::ExtSpaceProjective{K}, s::Symbol) where {K}
    if s === :tmin
        return getfield(getfield(E, :complex), :tmin)
    elseif s === :tmax
        return getfield(getfield(E, :complex), :tmax)
    else
        return getfield(E, s)
    end
end

Base.propertynames(E::ExtSpaceProjective{K}; private::Bool=false) where {K} =
    (fieldnames(typeof(E))..., :tmin, :tmax)

function _block_offsets_for_gens(N::PModule{QQ}, gens::Vector{Int})
    offs = zeros(Int, length(gens) + 1)
    for i in 1:length(gens)
        offs[i+1] = offs[i] + N.dims[gens[i]]
    end
    return offs
end

function _build_hom_differential(res::ProjectiveResolution{QQ}, N::PModule{QQ}, a::Int,
                                 offs_cod::Vector{Int}, offs_dom::Vector{Int})
    # a is the chain degree of the projective differential d_a: P_a -> P_{a-1}
    # On cochains: d^{a-1} : Hom(P_{a-1}, N) -> Hom(P_a, N)
    dom_gens = res.gens[a+1]      # summands of P_a
    cod_gens = res.gens[a]        # summands of P_{a-1}
    delta = res.d_mat[a]          # rows cod, cols dom

    out_dim = offs_dom[end]
    in_dim = offs_cod[end]
    D = zeros(QQ, out_dim, in_dim)

    I, J, V = findnz(delta)
    for k in 1:length(V)
        j = I[k]   # cod summand index (in degree a-1)
        i = J[k]   # dom summand index (in degree a)
        c = V[k]

        vj = cod_gens[j]
        ui = dom_gens[i]

        A = map_leq(N, vj, ui)  # N_vj -> N_ui
        rows = (offs_dom[i] + 1):(offs_dom[i+1])
        cols = (offs_cod[j] + 1):(offs_cod[j+1])
        D[rows, cols] .+= c .* A
    end

    return sparse(D)
end

function Ext(M::PModule{QQ}, N::PModule{QQ}; maxdeg::Int=3)
    res = projective_resolution(M; maxlen=maxdeg)
    _pad_projective_resolution!(res, maxdeg)
    return Ext(res, N)
end


function Ext(res::ProjectiveResolution{QQ}, N::PModule{QQ})
    L = length(res.Pmods) - 1
    # build cochain dims and offsets
    dimsC = Int[]
    offs = Vector{Vector{Int}}()
    for a in 0:L
        oa = _block_offsets_for_gens(N, res.gens[a+1])
        push!(offs, oa)
        push!(dimsC, oa[end])
    end

    # differentials d^a : C^a -> C^{a+1} for a=0..L-1
    dC = Vector{SparseMatrixCSC{QQ, Int}}()
    for a in 1:L
        push!(dC, _build_hom_differential(res, N, a, offs[a], offs[a+1]))
    end

    C = ChainComplexes.CochainComplex{QQ}(0, L, dimsC, dC, [Any[] for _ in 0:L])
    cohom = [ChainComplexes.cohomology_data(C, t) for t in 0:L]
    return ExtSpaceProjective{QQ}(res, N, C, offs, cohom)
end

# Convenience: fringe modules
function Ext(M::FringeModule{QQ}, N::FringeModule{QQ}; maxdeg::Int=3)
    Mp = pmodule_from_fringe(M)
    Np = pmodule_from_fringe(N)
    return Ext(Mp, Np; maxdeg=maxdeg)
end

function dim(E::ExtSpaceProjective, t::Int)
    if t < E.tmin || t > E.tmax
        return 0
    end
    return E.cohom[t+1].dimH
end

function cycles(E::ExtSpaceProjective, t::Int)
    return E.cohom[t+1].K
end

function boundaries(E::ExtSpaceProjective, t::Int)
    return E.cohom[t+1].B
end

function representative(E::ExtSpaceProjective, t::Int, i::Int)
    Hrep = E.cohom[t+1].Hrep
    return Hrep[:, i]
end

"""
    representative(E::ExtSpaceProjective, t::Int, coords::AbstractVector{QQ}) -> Vector{QQ}

Return an explicit cocycle representative in the cochain space C^t of the Ext class
whose coordinates (in the basis chosen internally by `E`) are given by `coords`.

Mathematically:
- `E` stores a basis of H^t(C) by choosing cocycle representatives (columns of `Hrep`).
- This function returns the linear combination of those cocycles.

This is useful when you want:
- explicit chain-level representatives of arbitrary Ext elements,
- Yoneda products on representatives,
- custom linear combinations without manually forming them.
"""
function representative(E::ExtSpaceProjective{QQ}, t::Int, coords::AbstractVector{QQ})
    if t < 0 || t > E.tmax
        error("representative: degree t must satisfy 0 <= t <= tmax.")
    end
    Hrep = E.cohom[t+1].Hrep
    if length(coords) != size(Hrep, 2)
        error("representative: coordinate vector has length $(length(coords)), expected $(size(Hrep,2)).")
    end
    v = Hrep * reshape(coords, :, 1)
    return vec(v)
end


function basis(E::ExtSpaceProjective, t::Int)
    Hrep = E.cohom[t+1].Hrep
    out = Vector{Vector{QQ}}(undef, size(Hrep, 2))
    for i in 1:size(Hrep, 2)
        out[i] = Vector{QQ}(Hrep[:, i])
    end
    return out
end

# Split a cochain vector into a list of fiber-vectors, one per projective summand generator.
function split_cochain(E::ExtSpaceProjective, t::Int, v::AbstractVector{QQ})
    offs = E.offsets[t+1]
    gens = E.res.gens[t+1]
    parts = Vector{Vector{QQ}}(undef, length(gens))
    for i in 1:length(gens)
        parts[i] = Vector{QQ}(v[(offs[i]+1):offs[i+1]])
    end
    return gens, parts
end

# Reduce a cocycle in C^t to Ext-coordinates in the chosen basis.
function coordinates(E::ExtSpaceProjective, t::Int, cocycle::AbstractVector{QQ})
    data = E.cohom[t+1]
    return ChainComplexes.cohomology_coordinates(data, cocycle)[:, 1]
end

# ----------------------------
# Functoriality in second argument: g : N -> N2 induces Ext(M,N) -> Ext(M,N2)
# ----------------------------

function _blockdiag_on_hom_cochains(g::PMorphism{QQ}, gens::Vector{Int}, offs_src::Vector{Int}, offs_tgt::Vector{Int})
    F = zeros(QQ, offs_tgt[end], offs_src[end])
    for i in 1:length(gens)
        u = gens[i]
        rows = (offs_tgt[i] + 1):(offs_tgt[i+1])
        cols = (offs_src[i] + 1):(offs_src[i+1])
        F[rows, cols] = g.comps[u]
    end
    return F
end

function ext_map_second(E1::ExtSpaceProjective{QQ}, E2::ExtSpaceProjective{QQ}, g::PMorphism{QQ}; t::Int)
    # Both E1 and E2 must be built from the same projective resolution (same M).
    @assert length(E1.res.gens) == length(E2.res.gens)
    gens_t = E1.res.gens[t+1]
    F = _blockdiag_on_hom_cochains(g, gens_t, E1.offsets[t+1], E2.offsets[t+1])
    return ChainComplexes.induced_map_on_cohomology(E1.cohom[t+1], E2.cohom[t+1], F)
end

# ----------------------------
# Connecting homomorphism for 0 -> A --i--> B --p--> C -> 0 in the second argument
# Uses the projective model, so Hom(P_a,-) is exact.
# ----------------------------

function connecting_hom(EA::ExtSpaceProjective{QQ}, EB::ExtSpaceProjective{QQ}, EC::ExtSpaceProjective{QQ},
                        i::PMorphism{QQ}, p::PMorphism{QQ}; t::Int)
    # delta : Ext^t(M,C) -> Ext^{t+1}(M,A)

    # cochain degree t maps:
    gens_t = EA.res.gens[t+1]
    It = _blockdiag_on_hom_cochains(i, gens_t, EA.offsets[t+1], EB.offsets[t+1])   # CA^t -> CB^t
    Pt = _blockdiag_on_hom_cochains(p, gens_t, EB.offsets[t+1], EC.offsets[t+1])   # CB^t -> CC^t

    gens_tp1 = EA.res.gens[t+2]
    Itp1 = _blockdiag_on_hom_cochains(i, gens_tp1, EA.offsets[t+2], EB.offsets[t+2])
    Ptp1 = _blockdiag_on_hom_cochains(p, gens_tp1, EB.offsets[t+2], EC.offsets[t+2])

    dBt = Matrix{QQ}(EB.complex.d[t+1])     # CB^t -> CB^{t+1}
    # Basis of Ext^t(M,C) as cocycles (columns)
    HrepC = EC.cohom[t+1].Hrep
    out = zeros(QQ, EA.cohom[t+2].dimH, EC.cohom[t+1].dimH)

    for j in 1:size(HrepC, 2)
        z = HrepC[:, j]  # cocycle in CC^t

        # lift to y in CB^t: Pt * y = z
        y = ChainComplexes.solve_particularQQ(Pt, Matrix{QQ}(reshape(z, :, 1)))[:, 1]

        # dy in CB^{t+1}
        dy = dBt * y

        # dy is in ker(Ptp1) = im(Itp1), solve Itp1 * x = dy
        x = ChainComplexes.solve_particularQQ(Itp1, Matrix{QQ}(reshape(dy, :, 1)))[:, 1]

        # reduce x to Ext^{t+1}(M,A) coordinates
        coords = ChainComplexes.cohomology_coordinates(EA.cohom[t+2], x)
        out[:, j] = coords[:, 1]
    end

    return out
end

# =============================================================================
# Yoneda product on Ext (projective-resolution model)
# =============================================================================

# -----------------------------------------------------------------------------
# Poset/module compatibility checks (do not assume FinitePoset has a `covers` field)
# -----------------------------------------------------------------------------

function _same_poset(Q1::FinitePoset, Q2::FinitePoset)::Bool
    # The leq-matrix is the canonical source of truth.
    if Q1.n != Q2.n
        return false
    end
    if Q1.leq != Q2.leq
        return false
    end

    # Some layers of the codebase historically stored cover edges as a cached field
    # `Q.covers`.  The current finite-poset layer computes cover edges from `leq`
    # via `cover_edges(Q)`.  We avoid relying on a non-existent or stale cache.
    C1 = cover_edges(Q1)
    C2 = cover_edges(Q2)
    return convert(BitMatrix, C1) == convert(BitMatrix, C2)
end


function _assert_same_pmodule_structure(A::PModule{QQ}, B::PModule{QQ}, ctx::String)
    if !_same_poset(A.Q, B.Q)
        error("$ctx: modules live on different posets.")
    end
    if A.dims != B.dims
        error("$ctx: modules have different fiber dimensions.")
    end

    # Compare cover-edge structure maps.  Iterate over cover edges computed from leq,
    # rather than assuming a cached list exists on the FinitePoset object.
    for (u, v) in cover_edges(A.Q)
        Auv = get(A.edge_maps, (u, v), zeros(QQ, A.dims[v], A.dims[u]))
        Buv = get(B.edge_maps, (u, v), zeros(QQ, B.dims[v], B.dims[u]))
        if Auv != Buv
            error("$ctx: modules have different structure maps on cover edge ($u,$v).")
        end
    end
    return nothing
end



# -----------------------------------------------------------------------------
# Internal: lift a cocycle in Hom(P_q(L), M) to a degree-q chain map P(L) -> P(M)
# (enough components to compose with a degree-p cocycle from Ext^p(M,N)).
# -----------------------------------------------------------------------------

"""
    _lift_cocycle_to_chainmap_coeff(resL, resM, E_LM, q, alpha_cocycle; upto)

Given:
- `resL`: a projective resolution of L,
- `resM`: a projective resolution of M,
- a cocycle `alpha_cocycle` in C^q = Hom(P_q(L), M),

construct (non-canonically, but deterministically) coefficient matrices describing
a degree-q chain map
    F : P(L) -> P(M),
i.e. maps
    F_k : P_{q+k}(L) -> P_k(M)
for k = 0,1,...,upto

Return value:
- a vector `F` where `F[k+1]` is the coefficient matrix of F_k.

This is the standard "comparison map" construction used to implement the Yoneda
product via projective resolutions.
"""
function _lift_cocycle_to_chainmap_coeff(resL::ProjectiveResolution{QQ},
                                         resM::ProjectiveResolution{QQ},
                                         E_LM::ExtSpaceProjective{QQ},
                                         q::Int,
                                         alpha_cocycle::AbstractVector{QQ};
                                         upto::Int)

    if q < 0
        error("_lift_cocycle_to_chainmap_coeff: q must be >= 0.")
    end
    if upto < 0
        error("_lift_cocycle_to_chainmap_coeff: upto must be >= 0.")
    end
    if q + upto > (length(resL.Pmods) - 1)
        error("_lift_cocycle_to_chainmap_coeff: resolution of L is too short for q+upto = $(q+upto).")
    end
    if upto > (length(resM.Pmods) - 1)
        error("_lift_cocycle_to_chainmap_coeff: resolution of M is too short for upto = $upto.")
    end

    Q = resM.M.Q

    # Split alpha as images of the generators of each principal upset summand in P_q(L).
    dom_gens_q, alpha_parts = split_cochain(E_LM, q, alpha_cocycle)  # dom_gens_q == resL.gens[q+1]

    # Precompute active summand indices for each projective term P_k(M).
    active_M = Vector{Vector{Vector{Int}}}(undef, upto + 1)
    for k in 0:upto
        active_M[k+1] = _active_upset_indices(Q, resM.gens[k+1])
    end

    F = Vector{Matrix{QQ}}(undef, upto + 1)

    # Step k = 0: lift alpha through the augmentation P_0(M) -> M.
    cod_bases0 = resM.gens[1]
    F0 = zeros(QQ, length(cod_bases0), length(dom_gens_q))

    for i in 1:length(dom_gens_q)
        u = dom_gens_q[i]
        act = active_M[1][u]  # indices of summands in P_0(M) active at vertex u

        # If M_u is 0-dimensional, the equation is vacuous.
        if resM.M.dims[u] == 0
            continue
        end

        Aug_u = resM.aug.comps[u]  # M_u x (P0_u)
        rhs   = alpha_parts[i]

        x = ChainComplexes.solve_particularQQ(Aug_u, reshape(rhs, :, 1))  # (P0_u) x 1

        # Place the solution into the global coefficient matrix column i.
        # The order of columns of Aug_u matches `act` by construction.
        for (pos, j) in enumerate(act)
            F0[j, i] = x[pos, 1]
        end
    end

    F[1] = F0

    # Steps k >= 1: solve the chain-map equations
    #   d_k^M * F_k = F_{k-1} * d_{q+k}^L
    for k in 1:upto
        DkM = Matrix(resM.d_mat[k])          # P_k(M) -> P_{k-1}(M) coefficients
        DqkL = Matrix(resL.d_mat[q + k])     # P_{q+k}(L) -> P_{q+k-1}(L) coefficients
        RHS = F[k] * DqkL

        cod_bases_k = resM.gens[k+1]         # bases of summands in P_k(M)
        dom_bases_qk = resL.gens[q+k+1]      # bases of summands in P_{q+k}(L)

        Fk = zeros(QQ, length(cod_bases_k), length(dom_bases_qk))

        for col in 1:length(dom_bases_qk)
            u = dom_bases_qk[col]
            allowed = active_M[k+1][u]       # indices in P_k(M) with base <= u

            # If nothing can map into u, then RHS column must be 0.
            if isempty(allowed)
                if any(!iszero, RHS[:, col])
                    error("_lift_cocycle_to_chainmap_coeff: inconsistent constraints at (k=$k, dom_summand=$col).")
                end
                continue
            end

            A = DkM[:, allowed]
            b = RHS[:, col]
            x = ChainComplexes.solve_particularQQ(A, reshape(b, :, 1))  # length(allowed) x 1

            for (pos, j) in enumerate(allowed)
                Fk[j, col] = x[pos, 1]
            end
        end

        F[k+1] = Fk
    end

    return F
end


# -----------------------------------------------------------------------------
# Internal: compose a chain-map component into a cocycle for Hom(P_{p+q}(L), N).
# -----------------------------------------------------------------------------

function _compose_into_module_cocycle(resL::ProjectiveResolution{QQ},
                                      resM::ProjectiveResolution{QQ},
                                      N::PModule{QQ},
                                      p::Int,
                                      q::Int,
                                      Fp::Matrix{QQ},
                                      beta_cocycle::AbstractVector{QQ},
                                      E_MN::ExtSpaceProjective{QQ},
                                      E_LN::ExtSpaceProjective{QQ})

    deg = p + q
    dom_bases = resL.gens[deg+1]   # bases of summands in P_{p+q}(L)
    mid_bases = resM.gens[p+1]     # bases of summands in P_p(M)

    # Split beta: images of generators of summands in P_p(M), stored as vectors in N at their base vertices.
    _, beta_parts = split_cochain(E_MN, p, beta_cocycle)

    offs = E_LN.offsets[deg+1]
    out = zeros(QQ, offs[end])

    for i in 1:length(dom_bases)
        u = dom_bases[i]
        block = zeros(QQ, N.dims[u])

        for j in 1:length(mid_bases)
            c = Fp[j, i]
            if iszero(c)
                continue
            end
            v = mid_bases[j]
            if !resL.M.Q.leq[v, u]
                # In a valid morphism, this should not happen, but keep it defensive.
                continue
            end

            A = map_leq(N, v, u)           # N_v -> N_u
            block .+= c .* (A * beta_parts[j])
        end

        out[(offs[i]+1):offs[i+1]] = block
    end

    return out
end


# -----------------------------------------------------------------------------
# Public API: Yoneda product
# -----------------------------------------------------------------------------

"""
    yoneda_product(E_MN, p, beta_coords, E_LM, q, alpha_coords; ELN=nothing, return_cocycle=false)

Compute the Yoneda product

    Ext^p(M, N) x Ext^q(L, M) -> Ext^{p+q}(L, N).

Inputs:
- `E_MN` is an `ExtSpaceProjective` for (M,N).
- `E_LM` is an `ExtSpaceProjective` for (L,M).
- `beta_coords` are coordinates of a class in Ext^p(M,N) in the basis used by `E_MN`.
- `alpha_coords` are coordinates of a class in Ext^q(L,M) in the basis used by `E_LM`.

Output:
- `(E_LN, coords)` where `coords` are coordinates of the product class in Ext^{p+q}(L,N)
  in the basis used by `E_LN`.
- If `return_cocycle=true`, returns `(E_LN, coords, cocycle)` where `cocycle` is an explicit
  representative in the cochain space Hom(P_{p+q}(L), N).

Notes for mathematicians:
- This implements the classical Yoneda product by constructing a comparison map
  between projective resolutions and composing at the chain level.
- The result is well-defined in cohomology; chain-level representatives depend on
  deterministic but non-canonical lift choices (as always).

Technical requirements:
- `E_MN` must have `tmax >= p`.
- `E_LM` must have `tmax >= p+q` (because we need P_{p+q}(L)).
- The "middle" module M used by `E_MN.res` and the second argument of `E_LM` must agree
  as poset-modules (same fibers and structure maps).
"""
function yoneda_product(E_MN::ExtSpaceProjective{QQ},
                        p::Int,
                        beta_coords::AbstractVector{QQ},
                        E_LM::ExtSpaceProjective{QQ},
                        q::Int,
                        alpha_coords::AbstractVector{QQ};
                        ELN::Union{Nothing,ExtSpaceProjective{QQ}}=nothing,
                        return_cocycle::Bool=false)

    if p < 0 || q < 0
        error("yoneda_product: degrees p and q must be >= 0.")
    end
    if p > E_MN.tmax
        error("yoneda_product: E_MN.tmax is too small for p = $p.")
    end
    if (p + q) > E_LM.tmax
        error("yoneda_product: E_LM.tmax is too small for p+q = $(p+q).")
    end

    resM = E_MN.res
    resL = E_LM.res
    N = E_MN.N

    # Compatibility: the middle module in Ext^q(L,M) must match the resolved module M.
    _assert_same_pmodule_structure(E_LM.N, resM.M, "yoneda_product (middle module check)")

    # Build (or validate) the target Ext space Ext(L,N).
    if ELN === nothing
        ELN_use = Ext(resL, N)
    else
        ELN_use = ELN
        # Very conservative checks: same resolved L and same N.
        _assert_same_pmodule_structure(ELN_use.N, N, "yoneda_product (target N check)")
        _assert_same_pmodule_structure(ELN_use.res.M, resL.M, "yoneda_product (target L check)")
        if ELN_use.tmax < (p + q)
            error("yoneda_product: provided ELN has tmax < p+q.")
        end
    end

    # Convert coordinates to explicit cocycles.
    beta_cocycle  = representative(E_MN, p, beta_coords)
    alpha_cocycle = representative(E_LM, q, alpha_coords)

    # Lift alpha to a degree-q chain map into the projective resolution of M, up to component p.
    F = _lift_cocycle_to_chainmap_coeff(resL, resM, E_LM, q, alpha_cocycle; upto=p)
    Fp = F[p+1]  # P_{p+q}(L) -> P_p(M)

    # Compose at chain level to get a cocycle in Hom(P_{p+q}(L), N).
    cocycle = _compose_into_module_cocycle(resL, resM, N, p, q, Fp, beta_cocycle, E_MN, ELN_use)

    coords = coordinates(ELN_use, p+q, cocycle)

    if return_cocycle
        return (ELN_use, coords, cocycle)
    else
        return (ELN_use, coords)
    end
end

# =============================================================================
# Ext algebra: Ext^*(M,M) with cached Yoneda multiplication
# =============================================================================

"""
    ExtAlgebra(M::PModule{QQ}; maxdeg::Int=3) -> ExtAlgebra{QQ}
    ExtAlgebra(M::FringeModule{QQ}; maxdeg::Int=3) -> ExtAlgebra{QQ}

Construct the truncated graded Ext algebra Ext^*(M,M) up to degree `maxdeg`,
with multiplication given by the Yoneda product.

This wrapper is intentionally "mathematician-facing":

- It chooses (once) a projective resolution and Ext bases via `Ext(M,M; maxdeg=...)`.
- It exposes homogeneous elements as `ExtElement` objects.
- It supports `*` for Ext multiplication and caches the structure constants.

Caching model (key point):
For each bidegree (p,q) with p+q <= tmax, we cache a matrix

    MU[p,q] : Ext^p(M,M) x Ext^q(M,M) -> Ext^{p+q}(M,M)

in coordinate bases as a linear map on the Kronecker product coordinates.

Column convention:
If dim_p = dim Ext^p and dim_q = dim Ext^q, we index basis pairs (i,j) by

    col = (i-1)*dim_q + j

and we use Julia's `kron(x, y)` to build the vector of coefficients x_i*y_j
in the same ordering.  This makes multiplication a single matrix-vector product:

    coords(x * y) = MU[p,q] * kron(coords(x), coords(y)).

Truncation:
The product is only defined when deg(x) + deg(y) <= A.tmax.
"""
mutable struct ExtAlgebra{K}
    E::ExtSpaceProjective{K}
    mult_cache::Dict{Tuple{Int,Int}, Matrix{K}}
    unit_coords::Union{Nothing, Vector{K}}
end

"""
    ExtElement(A::ExtAlgebra{QQ}, deg::Int, coords::Vector{QQ})

A homogeneous element of Ext^deg(M,M), expressed in the basis chosen by `A.E`.

This is deliberately lightweight: it is just (algebra handle, degree, coordinate vector).
Use:
- `element(A, deg, coords)` to construct,
- `basis(A, deg)` or `A[deg, i]` for basis elements,
- multiplication via `*`.
"""
struct ExtElement{K}
    A::ExtAlgebra{K}
    deg::Int
    coords::Vector{K}
end


# ----------------------------
# Construction
# ----------------------------

function ExtAlgebra(M::PModule{QQ}; maxdeg::Int=3)
    E = Ext(M, M; maxdeg=maxdeg)
    return ExtAlgebra{QQ}(E, Dict{Tuple{Int,Int}, Matrix{QQ}}(), nothing)
end

function ExtAlgebra(M::FringeModule{QQ}; maxdeg::Int=3)
    return ExtAlgebra(pmodule_from_fringe(M); maxdeg=maxdeg)
end


# Optional ergonomics: let users write A.tmin / A.tmax
function Base.getproperty(A::ExtAlgebra{K}, s::Symbol) where {K}
    if s === :tmin
        return A.E.tmin
    elseif s === :tmax
        return A.E.tmax
    else
        return getfield(A, s)
    end
end

Base.propertynames(A::ExtAlgebra{K}; private::Bool=false) where {K} =
    (fieldnames(typeof(A))..., :tmin, :tmax)


# ----------------------------
# Basic queries and constructors for elements
# ----------------------------

"Dimension of Ext^deg(M,M) in the basis chosen by the underlying Ext space."
dim(A::ExtAlgebra{QQ}, deg::Int) = dim(A.E, deg)

"""
    element(A::ExtAlgebra{QQ}, deg::Int, coords::AbstractVector{QQ}) -> ExtElement{QQ}

Construct a homogeneous Ext element in degree `deg` with the given coordinate vector.
"""
function element(A::ExtAlgebra{QQ}, deg::Int, coords::AbstractVector{QQ})
    if deg < 0 || deg > A.tmax
        error("element: degree must satisfy 0 <= deg <= tmax.")
    end
    d = dim(A, deg)
    if length(coords) != d
        error("element: expected coordinate vector of length $d in degree $deg, got length $(length(coords)).")
    end
    return ExtElement{QQ}(A, deg, Vector{QQ}(coords))
end


"""
    basis(A::ExtAlgebra{QQ}, deg::Int) -> Vector{ExtElement{QQ}}

Return the standard coordinate basis of Ext^deg(M,M) as ExtElement objects.
"""
function basis(A::ExtAlgebra{QQ}, deg::Int)
    d = dim(A, deg)
    out = Vector{ExtElement{QQ}}(undef, d)
    for i in 1:d
        c = zeros(QQ, d)
        c[i] = one(QQ)
        out[i] = ExtElement{QQ}(A, deg, c)
    end
    return out
end


"""
    A[deg, i] -> ExtElement

Indexing convenience: the i-th basis element in degree `deg`.
"""
function Base.getindex(A::ExtAlgebra{QQ}, deg::Int, i::Int)
    d = dim(A, deg)
    if i < 1 || i > d
        error("ExtAlgebra getindex: basis index i=$i out of range 1:$d in degree $deg.")
    end
    c = zeros(QQ, d)
    c[i] = one(QQ)
    return ExtElement{QQ}(A, deg, c)
end


"Return the coordinate vector of an ExtElement."
coordinates(x::ExtElement{QQ}) = x.coords

"""
    representative(x::ExtElement{QQ}) -> Vector{QQ}

Return a cocycle representative (in the internal cochain model) for the Ext class.
This is useful for debugging and for users who want explicit representatives.
"""
representative(x::ExtElement{QQ}) = representative(x.A.E, x.deg, x.coords)


# ----------------------------
# Linear structure on ExtElement
# ----------------------------

function _assert_same_algebra(x::ExtElement{QQ}, y::ExtElement{QQ}, ctx::String)
    if x.A !== y.A
        error("$ctx: elements live in different ExtAlgebra objects.")
    end
    if x.deg != y.deg
        error("$ctx: degrees differ (deg(x)=$(x.deg), deg(y)=$(y.deg)).")
    end
    return nothing
end

Base.:+(x::ExtElement{QQ}, y::ExtElement{QQ}) = (_assert_same_algebra(x, y, "ExtElement +");
                                                ExtElement{QQ}(x.A, x.deg, x.coords + y.coords))

Base.:-(x::ExtElement{QQ}, y::ExtElement{QQ}) = (_assert_same_algebra(x, y, "ExtElement -");
                                                ExtElement{QQ}(x.A, x.deg, x.coords - y.coords))

Base.:-(x::ExtElement{QQ}) = ExtElement{QQ}(x.A, x.deg, -x.coords)

Base.:*(c::QQ, x::ExtElement{QQ}) = ExtElement{QQ}(x.A, x.deg, c .* x.coords)
Base.:*(x::ExtElement{QQ}, c::QQ) = c * x

Base.:*(c::Integer, x::ExtElement{QQ}) = QQ(c) * x
Base.:*(x::ExtElement{QQ}, c::Integer) = QQ(c) * x

Base.iszero(x::ExtElement{QQ}) = all(x.coords .== 0)


# ----------------------------
# Unit element in Ext^0(M,M)
# ----------------------------

# Internal: encode a morphism P_t -> N as a cochain vector in C^t = Hom(P_t, N)
# using the generator ordering stored in the projective resolution.
function _cochain_vector_from_morphism(E::ExtSpaceProjective{QQ}, t::Int, f::PMorphism{QQ})
    if t < 0 || t > E.tmax
        error("_cochain_vector_from_morphism: degree out of range.")
    end
    if f.dom !== E.res.Pmods[t+1] || f.cod !== E.N
        error("_cochain_vector_from_morphism: expected a morphism P_t -> N for the given Ext space.")
    end

    bases = E.res.gens[t+1]
    offs  = E.offsets[t+1]
    out = zeros(QQ, offs[end])

    # Which summands of P_t are active at each vertex?
    active = _active_upset_indices(E.res.M.Q, bases)

    # For each summand i with base vertex u = bases[i],
    # locate the column position of i inside the fiber (P_t)_u,
    # then read off the image of that generator under f.
    for i in 1:length(bases)
        u = bases[i]
        du = E.N.dims[u]
        if du == 0
            continue
        end

        act_u = active[u]
        pos = searchsortedfirst(act_u, i)
        if pos > length(act_u) || act_u[pos] != i
            error("_cochain_vector_from_morphism: internal mismatch locating summand $i at vertex $u.")
        end

        out[(offs[i]+1):offs[i+1]] = f.comps[u][:, pos]
    end

    return out
end


"""
    unit(A::ExtAlgebra{QQ}) -> ExtElement{QQ}

Return the multiplicative identity in Ext^0(M,M).

Mathematically, Ext^0(M,M) = Hom(M,M), and the unit is id_M.
In the projective-resolution model
    ... -> P_1 -> P_0 -> M -> 0,
the inclusion Hom(M,M) -> Hom(P_0,M) sends id_M to the augmentation map P_0 -> M.
That augmentation is a cocycle in C^0 and represents the unit class in H^0.
"""
function unit(A::ExtAlgebra{QQ})
    if A.unit_coords === nothing
        if dim(A, 0) == 0
            # Zero module edge case: Ext^0(0,0) is 0 as a vector space.
            A.unit_coords = zeros(QQ, 0)
        else
            cocycle = _cochain_vector_from_morphism(A.E, 0, A.E.res.aug)
            A.unit_coords = coordinates(A.E, 0, cocycle)
        end
    end
    return ExtElement{QQ}(A, 0, copy(A.unit_coords))
end

Base.one(A::ExtAlgebra{QQ}) = unit(A)


# ----------------------------
# Cached multiplication: ExtElement * ExtElement
# ----------------------------

# Ensure the multiplication matrix MU[p,q] is present in the cache.
function _ensure_mult_cache!(A::ExtAlgebra{QQ}, p::Int, q::Int)
    key = (p, q)
    if haskey(A.mult_cache, key)
        return A.mult_cache[key]
    end

    if p < 0 || q < 0
        error("_ensure_mult_cache!: degrees must be nonnegative.")
    end
    if p + q > A.tmax
        error("_ensure_mult_cache!: requested product degree p+q=$(p+q) exceeds truncation tmax=$(A.tmax).")
    end

    dp = dim(A, p)
    dq = dim(A, q)
    dr = dim(A, p + q)

    MU = zeros(QQ, dr, dp * dq)

    # Cache even the trivial cases so repeated calls are O(1).
    if dp == 0 || dq == 0 || dr == 0
        A.mult_cache[key] = MU
        return MU
    end

    # Precompute all products of basis elements e_i in Ext^p and e_j in Ext^q.
    # Each product is computed by the trusted "mathematical core" `yoneda_product`,
    # then stored as a column of MU in the kron(x,y) ordering.
    ei = zeros(QQ, dp)
    ej = zeros(QQ, dq)

    for i in 1:dp
        fill!(ei, zero(QQ))
        ei[i] = one(QQ)
        for j in 1:dq
            fill!(ej, zero(QQ))
            ej[j] = one(QQ)

            # Multiply e_i (degree p) by e_j (degree q) in Ext(M,M).
            _, coords = yoneda_product(A.E, p, ei, A.E, q, ej; ELN=A.E)

            MU[:, (i - 1) * dq + j] = coords
        end
    end

    A.mult_cache[key] = MU
    return MU
end


"""
    multiply(A::ExtAlgebra{QQ}, p::Int, x::AbstractVector{QQ}, q::Int, y::AbstractVector{QQ}) -> Vector{QQ}

Multiply two homogeneous elements given by coordinate vectors x in Ext^p and y in Ext^q.
Returns the coordinate vector in Ext^{p+q}.
"""
function multiply(A::ExtAlgebra{QQ}, p::Int, x::AbstractVector{QQ}, q::Int, y::AbstractVector{QQ})
    dp = dim(A, p)
    dq = dim(A, q)
    if length(x) != dp
        error("multiply: left coordinate vector has length $(length(x)) but dim Ext^$p = $dp.")
    end
    if length(y) != dq
        error("multiply: right coordinate vector has length $(length(y)) but dim Ext^$q = $dq.")
    end

    MU = _ensure_mult_cache!(A, p, q)

    # kron(x,y) uses exactly the ordering we used for MU columns.
    v = kron(Vector{QQ}(x), Vector{QQ}(y))
    out = MU * v
    return Vector{QQ}(out)
end


"""
    precompute!(A::ExtAlgebra{QQ}) -> ExtAlgebra{QQ}

Eagerly compute and cache all multiplication matrices MU[p,q] with p+q <= A.tmax.
This is optional. Most users will rely on lazy caching via `*`.
"""
function precompute!(A::ExtAlgebra{QQ})
    for p in 0:A.tmax
        for q in 0:(A.tmax - p)
            _ensure_mult_cache!(A, p, q)
        end
    end
    return A
end


# The user-facing multiplication on homogeneous Ext elements.
function Base.:*(x::ExtElement{QQ}, y::ExtElement{QQ})
    if x.A !== y.A
        error("ExtElement *: elements live in different ExtAlgebra objects.")
    end
    A = x.A
    p = x.deg
    q = y.deg
    if p + q > A.tmax
        error("ExtElement *: degree p+q=$(p+q) exceeds truncation tmax=$(A.tmax).")
    end
    coords = multiply(A, p, x.coords, q, y.coords)
    return ExtElement{QQ}(A, p + q, coords)
end


# =============================================================================
# Injective resolutions
# =============================================================================

"""
Injective resolution of a module N:
    0 -> N -> E^0 -> E^1 -> ...

The field `gens[b+1]` stores the base vertices of the indecomposable injective
summands k[Dn(v)] appearing in E^b, with repetition.

This makes it possible to extract Bass-type multiplicity data (injective summands
by vertex and cohomological degree) in a canonical, user-facing way.
"""
struct InjectiveResolution{K}
    N::PModule{K}
    Emods::Vector{PModule{K}}         # E^0, E^1, ...
    gens::Vector{Vector{Int}}         # base vertices per injective summand in each E^b
    d_mor::Vector{PMorphism{K}}       # E^b -> E^{b+1}
    iota0::PMorphism{K}               # N -> E^0
end


"""
    injective_resolution(N::PModule{QQ}; maxlen::Int=3) -> InjectiveResolution{QQ}

Build an injective resolution
    0 -> N -> E^0 -> E^1 -> ... -> E^maxlen

Implementation notes:
- `injective_hull` is computed degreewise via `_injective_hull`.
- The differentials are obtained by extending the map N^b -> E^b to E^b -> E^{b+1}.

The resulting resolution is intended to be suitable for Ext computations and
Bass-number extraction (multiplicity of injective indecomposables).
"""
function injective_resolution(N::PModule{QQ}; maxlen::Int=3)
    E0, iota0, gens0 = _injective_hull(N)
    Emods = [E0]
    gens  = [_flatten_gens_at(gens0)]
    d_mor = PMorphism{QQ}[]

    C0, pi0 = _cokernel_module(iota0)
    prevC  = C0
    prevPi = pi0

    for step in 1:maxlen
        En, iotan, gensn = _injective_hull(prevC)
        push!(Emods, En)
        push!(gens, _flatten_gens_at(gensn))

        dn = compose(iotan, prevPi)   # E^{step-1} -> E^{step}
        push!(d_mor, dn)

        Cn, pin = _cokernel_module(iotan)
        prevC  = Cn
        prevPi = pin
    end

    return InjectiveResolution{QQ}(N, Emods, gens, d_mor, iota0)
end


"""
    bass(res::InjectiveResolution{QQ}) -> Dict{Tuple{Int,Int},Int}

Bass numbers for an injective resolution.

Interpretation:
- `res.Emods[b+1]` is the injective in cohomological degree b (i.e. E^b).
- Each E^b splits as a direct sum of indecomposable injectives k[Dn(v)].
- `res.gens[b+1]` stores the base vertex v for each summand, with repetition.

Output convention:
- Key `(b, v)` means "cohomological degree b, vertex v".
- Value is multiplicity of k[Dn(v)] in E^b.
"""
function bass(res::InjectiveResolution{QQ})
    out = Dict{Tuple{Int,Int},Int}()
    L = length(res.Emods) - 1
    for b in 0:L
        for v in res.gens[b+1]
            key = (b, v)
            out[key] = get(out, key, 0) + 1
        end
    end
    return out
end


"""
    bass_table(res::InjectiveResolution{QQ}) -> Matrix{Int}

Dense Bass table, analogous to `betti_table`:

- Rows are cohomological degrees b = 0,1,2,...
- Columns are vertices v = 1,...,Q.n
- Entry B[b+1, v] is the multiplicity of k[Dn(v)] in E^b.
"""
function bass_table(res::InjectiveResolution{QQ})
    Q = res.N.Q
    L = length(res.Emods) - 1
    B = zeros(Int, L+1, Q.n)
    for b in 0:L
        for v in res.gens[b+1]
            B[b+1, v] += 1
        end
    end
    return B
end


"""
    bass(N::PModule{QQ}; maxlen::Int=3) -> Dict{Tuple{Int,Int},Int}

Convenience wrapper:
- build `injective_resolution(N; maxlen=maxlen)`,
- return its Bass numbers.
"""
function bass(N::PModule{QQ}; maxlen::Int=3)
    return bass(injective_resolution(N; maxlen=maxlen))
end


# ----------------------------
# Minimality diagnostics for injective resolutions
# ----------------------------

"""
    InjectiveMinimalityReport

Returned by `minimality_report(res::InjectiveResolution{QQ})`.

Fields mirror `ProjectiveMinimalityReport`, but for injective resolutions:

- `hull_ok` compares multiplicities in E^0 against a freshly computed injective hull.
- `diagonal_violations` records nonzero coefficients k[Dn(v)] -> k[Dn(v)] in the
  differentials E^{b-1} -> E^b. Such coefficients split off contractible summands
  and are forbidden in a minimal injective resolution.
"""
struct InjectiveMinimalityReport
    minimal::Bool
    hull_ok::Bool
    hull_expected::Vector{Int}
    hull_actual::Vector{Int}
    diagonal_violations::Vector{Tuple{Int,Int,Int,Int,QQ}}
end


"""
    minimality_report(res::InjectiveResolution{QQ}; check_hull=true) -> InjectiveMinimalityReport

Certify minimality of an injective resolution in the standard finite-dimensional
algebra sense.

Checks performed:

1. (Optional) `check_hull`:
   Verify that N -> E^0 is an injective hull by comparing multiplicities of principal
   downsets against a freshly computed `_injective_hull(N)`.

2. Verify the "no units on diagonal" condition:
   For every differential d^b : E^{b-1} -> E^b, there is no nonzero coefficient
   k[Dn(v)] -> k[Dn(v)].
"""
function minimality_report(res::InjectiveResolution{QQ}; check_hull::Bool=true)
    Q = res.N.Q
    n = Q.n

    hull_actual = _vertex_counts(res.gens[1], n)
    hull_expected = copy(hull_actual)
    hull_ok = true

    if check_hull
        _, _, gens_at = _injective_hull(res.N)
        hull_expected = _vertex_counts(_flatten_gens_at(gens_at), n)
        hull_ok = (hull_expected == hull_actual)
    end

    violations = Tuple{Int,Int,Int,Int,QQ}[]
    for b in 1:length(res.d_mor)
        dom = res.gens[b]     # bases in E^{b-1}
        cod = res.gens[b+1]   # bases in E^b

        D = _coeff_matrix_downsets(Q, dom, cod, res.d_mor[b])

        for col in 1:size(D, 2)
            for ptr in D.colptr[col]:(D.colptr[col+1] - 1)
                row = D.rowval[ptr]
                val = D.nzval[ptr]
                if !iszero(val) && (cod[row] == dom[col])
                    v = dom[col]
                    push!(violations, (b, v, row, col, val))
                end
            end
        end
    end

    minimal = hull_ok && isempty(violations)
    return InjectiveMinimalityReport(minimal, hull_ok, hull_expected, hull_actual, violations)
end


"""
    is_minimal(res::InjectiveResolution{QQ}; check_hull=true) -> Bool
"""
function is_minimal(res::InjectiveResolution{QQ}; check_hull::Bool=true)
    return minimality_report(res; check_hull=check_hull).minimal
end


"""
    assert_minimal(res::InjectiveResolution{QQ}; check_hull=true) -> true

Throw a descriptive error if the injective resolution fails minimality checks.
"""
function assert_minimal(res::InjectiveResolution{QQ}; check_hull::Bool=true)
    R = minimality_report(res; check_hull=check_hull)
    if R.minimal
        return true
    end

    if !R.hull_ok
        error("Injective resolution is not minimal: N -> E0 is not an injective hull. " *
              "Expected hull multiplicities = $(R.hull_expected), got = $(R.hull_actual).")
    end

    if !isempty(R.diagonal_violations)
        (b, v, row, col, val) = R.diagonal_violations[1]
        error("Injective resolution is not minimal: differential d^$b has a nonzero coefficient " *
              "val = $val from k[Dn($v)] in E_$(b-1) (column $col) to k[Dn($v)] in E_$b (row $row).")
    end

    error("Injective resolution failed minimality checks for an unknown reason.")
end


"""
    minimal_injective_resolution(N::PModule{QQ}; maxlen=3, check=true) -> InjectiveResolution{QQ}

Build an injective resolution via successive injective hulls (the standard minimal
construction in the finite-dimensional algebra setting).

If `check=true`, call `assert_minimal` before returning.
"""
function minimal_injective_resolution(N::PModule{QQ}; maxlen::Int=3, check::Bool=true)
    res = injective_resolution(N; maxlen=maxlen)
    if check
        assert_minimal(res; check_hull=true)
    end
    return res
end


"""
    minimal_bass(N::PModule{QQ}; maxlen=3, check=true) -> Dict{Tuple{Int,Int},Int}

Compute Bass multiplicities from a certified minimal injective resolution.
"""
function minimal_bass(N::PModule{QQ}; maxlen::Int=3, check::Bool=true)
    res = minimal_injective_resolution(N; maxlen=maxlen, check=check)
    return bass(res)
end



struct ExtSpaceInjective{K}
    M::PModule{K}
    res::InjectiveResolution{K}
    homs::Vector{HomSpace{K}}                     # Hom(M, E^b)
    complex::ChainComplexes.CochainComplex{K}
    cohom::Vector{ChainComplexes.CohomologyData{K}}
end

# Derived property: the maximum and minimum cohomological degree stored in this Ext space.
function Base.getproperty(E::ExtSpaceInjective{K}, s::Symbol) where {K}
    if s === :tmin
        return getfield(getfield(E, :complex), :tmin)
    elseif s === :tmax
        return getfield(getfield(E, :complex), :tmax)
    else
        return getfield(E, s)
    end
end

Base.propertynames(E::ExtSpaceInjective{K}; private::Bool=false) where {K} =
    (fieldnames(typeof(E))..., :tmin, :tmax)

"""
    representative(E::ExtSpaceInjective, t::Int, coords::AbstractVector{QQ}) -> Vector{QQ}

Same as the projective-model method, but for an Ext space computed via an injective
resolution of the second argument.

Returns a cocycle in the cochain space Hom(M, E^t) (assembled over all degrees).
"""
function representative(E::ExtSpaceInjective{QQ}, t::Int, coords::AbstractVector{QQ})
    if t < 0 || t > E.tmax
        error("representative: degree t must satisfy 0 <= t <= tmax.")
    end
    Hrep = E.cohom[t+1].Hrep
    if length(coords) != size(Hrep, 2)
        error("representative: coordinate vector has length $(length(coords)), expected $(size(Hrep,2)).")
    end
    v = Hrep * reshape(coords, :, 1)
    return vec(v)
end

function ExtInjective(M::PModule{QQ}, N::PModule{QQ}; maxdeg::Int=3)
    resN = injective_resolution(N; maxlen=maxdeg)
    return ExtInjective(M, resN)
end

function ExtInjective(M::PModule{QQ}, resN::InjectiveResolution{QQ})
    L = length(resN.Emods) - 1
    homs = HomSpace{QQ}[]
    dims = Int[]
    for b in 0:L
        Hb = Hom(M, resN.Emods[b+1])
        push!(homs, Hb)
        push!(dims, size(Hb.basis_matrix, 2))
    end

    # differential on Hom(M, E^b) is postcomposition with d^b : E^b -> E^{b+1}
    dC = Vector{SparseMatrixCSC{QQ, Int}}()
    for b in 0:(L-1)
        Hb = homs[b+1]
        Hb1 = homs[b+2]
        db = resN.d_mor[b+1]

        # build matrix dim(Hb1) x dim(Hb)
        D = zeros(QQ, size(Hb1.basis_matrix, 2), size(Hb.basis_matrix, 2))
        for j in 1:size(Hb.basis_matrix, 2)
            fj = Hb.basis[j]
            img = compose(db, fj)                       # M -> E^{b+1}
            vimg = _morphism_to_vector(img, Hb1.offsets)
            coeffs = solve_fullcolumnQQ(Hb1.basis_matrix, vimg)
            D[:, j] = coeffs[:, 1]
        end
        push!(dC, sparse(D))
    end

    C = ChainComplexes.CochainComplex{QQ}(0, L, dims, dC, [Any[] for _ in 0:L])
    cohom = [ChainComplexes.cohomology_data(C, t) for t in 0:L]
    return ExtSpaceInjective{QQ}(M, resN, homs, C, cohom)
end

# Contravariant map in first argument: f : M -> Mp induces Ext^t(Mp,N) -> Ext^t(M,N)
function ext_map_first(EMN::ExtSpaceInjective{QQ}, EMPN::ExtSpaceInjective{QQ}, f::PMorphism{QQ}; t::Int)
    # map on cochains at degree t: Hom(Mp, E^t) -> Hom(M, E^t), g |-> g circ f
    Hsrc = EMPN.homs[t+1]
    Htgt = EMN.homs[t+1]

    F = zeros(QQ, size(Htgt.basis_matrix, 2), size(Hsrc.basis_matrix, 2))
    for j in 1:size(Hsrc.basis_matrix, 2)
        gj = Hsrc.basis[j]
        img = compose(gj, f)
        vimg = _morphism_to_vector(img, Htgt.offsets)
        coeffs = solve_fullcolumnQQ(Htgt.basis_matrix, vimg)
        F[:, j] = coeffs[:, 1]
    end

    return ChainComplexes.induced_map_on_cohomology(EMPN.cohom[t+1], EMN.cohom[t+1], F)
end

# -----------------------------------------------------------------------------
# Internal helper: precomposition matrices on Hom spaces
# -----------------------------------------------------------------------------

"""
    _precompose_matrix(Hdom, Hcod, f) -> Matrix{QQ}

Given a morphism f: A -> B and Hom spaces

- Hcod = Hom(B, E)
- Hdom = Hom(A, E)

return the matrix of the linear map
    f^* : Hom(B, E) -> Hom(A, E),   g |-> g circ f
in the bases stored inside the HomSpace objects.

This is the cochain-level map used by Ext functoriality in the first argument
when Ext is computed via an injective resolution.
"""
function _precompose_matrix(Hdom::HomSpace{QQ}, Hcod::HomSpace{QQ}, f::PMorphism{QQ})
    F = zeros(QQ, dim(Hdom), dim(Hcod))
    for j in 1:dim(Hcod)
        gj = Hcod.basis[j]                 # B -> E
        comp = compose(gj, f)              # A -> E
        x = ChainComplexes.solve_fullcolumnQQ(Hdom.basis_matrix,
                                              reshape(_morphism_vector(Hdom, comp), :, 1))
        F[:, j] = x[:, 1]
    end
    return F
end


"""
    connecting_hom_first(EA, EB, EC, i, p; t) -> Matrix{QQ}

Connecting homomorphism for a short exact sequence in the first (contravariant) argument:

    0 -> A --i--> B --p--> C -> 0.

Fix an injective resolution of N and compute Ext via the cochain complexes Hom(-, E^*).
The associated long exact sequence contains the connecting map

    delta^t : Ext^t(A, N) -> Ext^{t+1}(C, N).

This function returns the matrix of delta^t in the chosen Ext bases.

Requirements:
- EA, EB, EC must be `ExtSpaceInjective` objects built from the *same*
  `InjectiveResolution` of N.
- i: A -> B and p: B -> C should define a short exact sequence in the first argument.
"""
function connecting_hom_first(EA::ExtSpaceInjective{QQ},
                              EB::ExtSpaceInjective{QQ},
                              EC::ExtSpaceInjective{QQ},
                              i::PMorphism{QQ},
                              p::PMorphism{QQ}; t::Int)

    if EA.res !== EB.res || EA.res !== EC.res
        error("connecting_hom_first: EA, EB, EC must share the same InjectiveResolution.")
    end
    if t < 0 || t >= EA.tmax
        error("connecting_hom_first: t must satisfy 0 <= t <= tmax-1.")
    end

    # Cochain-level maps at the relevant degrees:
    #   i^* : Hom(B, E^t) -> Hom(A, E^t)
    #   p^* : Hom(C, E^{t+1}) -> Hom(B, E^{t+1})
    It   = _precompose_matrix(EA.homs[t+1], EB.homs[t+1], i)
    Ptp1 = _precompose_matrix(EB.homs[t+2], EC.homs[t+2], p)

    # Coboundary on Hom(B, E^*):
    dBt  = EB.complex.d[t+1]   # degree t: Hom(B,E^t) -> Hom(B,E^{t+1})

    # Domain and codomain cohomology data:
    HdA  = EA.cohom[t+1]       # Ext^t(A,N)
    HdC  = EC.cohom[t+2]       # Ext^{t+1}(C,N)

    delta = zeros(QQ, size(HdC.Hrep, 2), size(HdA.Hrep, 2))

    # For each basis class [z] in Ext^t(A,N), pick a cocycle rep z in Hom(A,E^t),
    # lift it to y in Hom(B,E^t), take dy, then lift dy back through p^* to x in Hom(C,E^{t+1}).
    for j in 1:size(HdA.Hrep, 2)
        z = HdA.Hrep[:, j]

        y = ChainComplexes.solve_particularQQ(It, reshape(z, :, 1))
        dy = dBt * y

        x = ChainComplexes.solve_particularQQ(Ptp1, dy)

        delta[:, j] = cohomology_coordinates(HdC, vec(x))
    end

    return delta
end


# ----------------------------
# Tor for right module (as left module over opposite poset) vs left module
# ----------------------------

struct TorSpace{K}
    resRop::ProjectiveResolution{K}    # projective resolution computed on P^op
    L::PModule{K}                      # left module on P
    bd::Vector{SparseMatrixCSC{K, Int}}  # boundaries bd_s : C_s -> C_{s-1}, s=1..S
    dims::Vector{Int}                    # dim C_s for s=0..S
    offsets::Vector{Vector{Int}}         # offsets per degree
    homol::Vector{ChainComplexes.HomologyData{K}}  # homology data per degree
end

function _op_poset(P::FinitePoset)
    leq = transpose(P.leq)
    return FinitePoset(leq)
end

# Tor(Rop, L) where Rop is a left module over P^op (a right module over P), L is a left module over P.
function Tor(Rop::PModule{QQ}, L::PModule{QQ}; maxdeg::Int=3)
    Pop = Rop.Q
    P = _op_poset(Pop)
    @assert L.Q.leq == P.leq  # sanity

    res = projective_resolution(Rop; maxlen=maxdeg)
    S = length(res.Pmods) - 1

    dims = Int[]
    offs = Vector{Vector{Int}}()
    for s in 0:S
        gens_s = res.gens[s+1]  # vertices in P^op indexing, same as P indexing
        os = zeros(Int, length(gens_s) + 1)
        for i in 1:length(gens_s)
            os[i+1] = os[i] + L.dims[gens_s[i]]
        end
        push!(offs, os)
        push!(dims, os[end])
    end

    # boundary bd_s : C_s -> C_{s-1} for s=1..S
    bd = Vector{SparseMatrixCSC{QQ, Int}}()
    for s in 1:S
        dom_gens = res.gens[s+1]
        cod_gens = res.gens[s]
        delta = res.d_mat[s]  # rows cod, cols dom, computed in P^op

        out_dim = offs[s][end]
        in_dim = offs[s+1][end]
        B = zeros(QQ, out_dim, in_dim)

        I, J, V = findnz(delta)
        for k in 1:length(V)
            j = I[k]  # cod summand index (degree s-1)
            i = J[k]  # dom summand index (degree s)
            c = V[k]

            u = dom_gens[i]
            v = cod_gens[j]
            # In P^op, v <=op u means u <= v in P, so L has map L_u -> L_v
            A = map_leq(L, u, v)
            rows = (offs[s][j] + 1):(offs[s][j+1])
            cols = (offs[s+1][i] + 1):(offs[s+1][i+1])
            B[rows, cols] .+= c .* A
        end

        push!(bd, sparse(B))
    end

    # homology data H_s = ker(bd_s) / im(bd_{s+1})
    homol = Vector{ChainComplexes.HomologyData{QQ}}(undef, S+1)
    for s in 0:S
        bd_curr = (s == 0) ? zeros(QQ, 0, dims[1]) : Matrix{QQ}(bd[s])
        bd_next = (s == S) ? zeros(QQ, dims[S+1], 0) : Matrix{QQ}(bd[s+1])
        homol[s+1] = ChainComplexes.homology_data(bd_next, bd_curr, s)
    end

    return TorSpace{QQ}(res, L, bd, dims, offs, homol)
end

dim(T::TorSpace, s::Int) = T.homol[s+1].dimH
cycles(T::TorSpace, s::Int) = T.homol[s+1].Z
boundaries(T::TorSpace, s::Int) = T.homol[s+1].B

function basis(T::TorSpace, s::Int)
    Hrep = T.homol[s+1].Hrep
    out = Vector{Vector{QQ}}(undef, size(Hrep, 2))
    for i in 1:size(Hrep, 2)
        out[i] = Vector{QQ}(Hrep[:, i])
    end
    return out
end

# ----------------------------
# Zn wrappers: compute on a box by first building a finite poset module
# ----------------------------

function pmodule_on_box(FG; a::Vector{Int}, b::Vector{Int})
    return ZnEncoding.pmodule_on_box(FG; a=a, b=b)
end

function ExtZn(FG1, FG2; a::Vector{Int}, b::Vector{Int}, maxdeg::Int=3)
    M = pmodule_on_box(FG1; a=a, b=b)
    N = pmodule_on_box(FG2; a=a, b=b)
    return Ext(M, N; maxdeg=maxdeg)
end

# Pad a projective resolution with zeros so Ext(M,N; maxdeg=d) always has tmax=d.

function _zero_pmodule(Q::FinitePoset)
    edge = Dict{Tuple{Int,Int},Matrix{QQ}}()
    for (u,v) in cover_edges(Q)
        edge[(u,v)] = zeros(QQ, 0, 0)
    end
    return PModule{QQ}(Q, zeros(Int, Q.n), edge)
end

function _pad_projective_resolution!(res::ProjectiveResolution{QQ}, maxdeg::Int)
    L = length(res.Pmods) - 1
    if L >= maxdeg
        return
    end
    Q = res.M.Q
    for a in (L+1):maxdeg
        push!(res.Pmods, _zero_pmodule(Q))
        push!(res.gens, Int[])
        push!(res.d, Dict{Tuple{Int,Int},Matrix{QQ}}())
        cod_summands = length(res.gens[a])     # P_{a-1}
        dom_summands = length(res.gens[a+1])   # P_a
        push!(res.d_mat, spzeros(QQ, cod_summands, dom_summands))
    end
end


end
