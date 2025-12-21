module ChainComplexes

using LinearAlgebra
using SparseArrays

using ..CoreModules: QQ
using ..ExactQQ: rrefQQ, rankQQ, nullspaceQQ, colspaceQQ, solve_fullcolumnQQ

# ----------------------------
# Small, reliable linear solvers
# ----------------------------

# Solve A * X = B over QQ, returning one particular solution with all free vars set to 0.
# Throws if inconsistent.
function solve_particularQQ(A::AbstractMatrix{QQ}, B::AbstractMatrix{QQ})
    A0 = Matrix{QQ}(A)
    B0 = Matrix{QQ}(B)
    m, n = size(A0)
    @assert size(B0, 1) == m

    Aug = hcat(A0, B0)
    R, pivs_all = rrefQQ(Aug)

    rhs = size(B0, 2)

    # consistency check: a zero row in A-part with a nonzero in RHS-part
    for i in 1:m
        if all(R[i, 1:n] .== 0)
            if any(R[i, n+1:n+rhs] .!= 0)
                error("solve_particularQQ: inconsistent system")
            end
        end
    end

    pivs = Int[]
    for p in pivs_all
        if p <= n
            push!(pivs, p)
        end
    end

    X = zeros(QQ, n, rhs)
    # In RREF, pivot rows occur first; set free vars to 0, pivot vars read from RHS.
    for (row, pcol) in enumerate(pivs)
        X[pcol, :] = R[row, n+1:n+rhs]
    end
    return X
end

# Extend columns of C (k x r) to an invertible (k x k) matrix by adding standard basis vectors.
function extend_to_basisQQ(C::Matrix{QQ})
    k = size(C, 1)
    r = size(C, 2)
    if k == 0
        return zeros(QQ, 0, 0)
    end
    if r == 0
        B = zeros(QQ, k, k)
        for i in 1:k
            B[i, i] = one(QQ)
        end
        return B
    end

    B = Matrix{QQ}(C)
    for i in 1:k
        e = zeros(QQ, k, 1)
        e[i, 1] = one(QQ)
        if rankQQ(hcat(B, e)) > size(B, 2)
            B = hcat(B, e)
        end
        if size(B, 2) == k
            break
        end
    end

    if size(B, 2) != k
        error("extend_to_basisQQ: could not extend to a full basis")
    end
    return B
end

# ----------------------------
# Cochain complexes
# ----------------------------

struct CochainComplex{K}
    tmin::Int
    tmax::Int
    dims::Vector{Int}                         # dims[t - tmin + 1] = dim C^t
    d::Vector{SparseMatrixCSC{K, Int}}        # d[idx] : C^t -> C^{t+1}
    labels::Vector{Vector{Any}}               # optional, per degree
end

function degree_index(C::CochainComplex, t::Int)
    if t < C.tmin || t > C.tmax
        error("degree out of range")
    end
    return t - C.tmin + 1
end

# Holds cycle, boundary, and quotient data in a concrete basis.
struct CohomologyData{K}
    t::Int
    dimC::Int
    dimZ::Int
    dimB::Int
    dimH::Int
    K::Matrix{K}          # cycle basis in C^t
    B::Matrix{K}          # boundary basis in C^t
    Cx::Matrix{K}         # boundary subspace basis in K-coordinates
    Q::Matrix{K}          # complement basis in K-coordinates
    Bfull::Matrix{K}      # [Cx Q], square dimZ x dimZ, invertible
    Hrep::Matrix{K}       # cocycle representatives: K * Q
end

# Compute cohomology data at degree t:
# Z^t = ker(d^t), B^t = im(d^{t-1}), H^t = Z^t / B^t
function cohomology_data(C::CochainComplex{QQ}, t::Int)
    idx = degree_index(C, t)
    dimCt = C.dims[idx]

    # d_prev: C^{t-1} -> C^t, d_curr: C^t -> C^{t+1}
    d_prev = (idx == 1) ? zeros(QQ, dimCt, 0) : Matrix{QQ}(C.d[idx-1])
    d_curr = (idx > length(C.d)) ? zeros(QQ, 0, dimCt) : Matrix{QQ}(C.d[idx])

    # cycles
    K = if dimCt == 0
        zeros(QQ, 0, 0)
    elseif size(d_curr, 1) == 0
        I = zeros(QQ, dimCt, dimCt)
        for i in 1:dimCt
            I[i, i] = one(QQ)
        end
        I
    else
        nullspaceQQ(d_curr)
    end

    # boundaries
    B = if dimCt == 0
        zeros(QQ, 0, 0)
    elseif size(d_prev, 2) == 0
        zeros(QQ, dimCt, 0)
    else
        colspaceQQ(d_prev)
    end

    dimZ = size(K, 2)
    dimB = size(B, 2)

    if dimCt == 0
        return CohomologyData{QQ}(t, 0, 0, 0, 0, K, B, zeros(QQ, 0, 0), zeros(QQ, 0, 0), zeros(QQ, 0, 0), zeros(QQ, 0, 0))
    end

    # coordinates of boundaries in the cycle basis (guaranteed by d^t d^{t-1} = 0)
    X = if dimB == 0
        zeros(QQ, dimZ, 0)
    else
        solve_fullcolumnQQ(K, B)
    end

    Cx = if size(X, 2) == 0
        zeros(QQ, dimZ, 0)
    else
        colspaceQQ(X)
    end

    rB = size(Cx, 2)
    Bfull = extend_to_basisQQ(Cx)
    Q = Bfull[:, rB+1:end]
    Hrep = K * Q
    dimH = size(Q, 2)

    return CohomologyData{QQ}(t, dimCt, dimZ, rB, dimH, K, B, Cx, Q, Bfull, Hrep)
end

# Reduce a cycle z in C^t to coordinates in H^t, using precomputed cohomology data.
function cohomology_coordinates(data::CohomologyData{QQ}, z::AbstractVector{QQ})
    if data.dimH == 0
        return zeros(QQ, 0, 1)
    end
    z0 = Matrix{QQ}(reshape(Vector{QQ}(z), :, 1))
    alpha = solve_fullcolumnQQ(data.K, z0)           # K-coordinates
    gamma = solve_fullcolumnQQ(data.Bfull, alpha)    # (boundary + H) coordinates
    rB = data.dimB
    return gamma[rB+1:end, :]
end

# Given a linear map f: C^t -> D^t and cohomology data for both sides, compute induced map on H^t.
function induced_map_on_cohomology(src::CohomologyData{QQ}, tgt::CohomologyData{QQ}, f::AbstractMatrix{QQ})
    if src.dimH == 0 || tgt.dimH == 0
        return zeros(QQ, tgt.dimH, src.dimH)
    end
    F = Matrix{QQ}(f)
    M = zeros(QQ, tgt.dimH, src.dimH)
    for j in 1:src.dimH
        y = F * src.Hrep[:, j]
        coords = cohomology_coordinates(tgt, y)
        M[:, j] = coords[:, 1]
    end
    return M
end

# ----------------------------
# Homology (chain complexes) for Tor, etc.
# ----------------------------

struct HomologyData{K}
    s::Int
    dimC::Int
    dimZ::Int
    dimB::Int
    dimH::Int
    Z::Matrix{K}          # cycles in C_s
    B::Matrix{K}          # boundaries in C_s
    Cx::Matrix{K}         # boundaries in Z-coordinates
    Q::Matrix{K}          # complement in Z-coordinates
    Bfull::Matrix{K}
    Hrep::Matrix{K}       # cycle representatives in C_s
end

# Homology at degree s uses:
# cycles = ker(bd_s : C_s -> C_{s-1})
# boundaries = im(bd_{s+1} : C_{s+1} -> C_s)
function homology_data(bd_next::AbstractMatrix{QQ}, bd_curr::AbstractMatrix{QQ}, s::Int)
    bdN = Matrix{QQ}(bd_next)
    bdC = Matrix{QQ}(bd_curr)

    dimCs = size(bdC, 2)

    Z = if dimCs == 0
        zeros(QQ, 0, 0)
    elseif size(bdC, 1) == 0
        I = zeros(QQ, dimCs, dimCs)
        for i in 1:dimCs
            I[i, i] = one(QQ)
        end
        I
    else
        nullspaceQQ(bdC)
    end

    B = if dimCs == 0
        zeros(QQ, 0, 0)
    elseif size(bdN, 2) == 0
        zeros(QQ, dimCs, 0)
    else
        colspaceQQ(bdN)
    end

    dimZ = size(Z, 2)
    dimB = size(B, 2)

    X = if dimB == 0
        zeros(QQ, dimZ, 0)
    else
        solve_fullcolumnQQ(Z, B)
    end

    Cx = if size(X, 2) == 0
        zeros(QQ, dimZ, 0)
    else
        colspaceQQ(X)
    end

    rB = size(Cx, 2)
    Bfull = extend_to_basisQQ(Cx)
    Q = Bfull[:, rB+1:end]
    Hrep = Z * Q
    dimH = size(Q, 2)

    return HomologyData{QQ}(s, dimCs, dimZ, rB, dimH, Z, B, Cx, Q, Bfull, Hrep)
end

function homology_coordinates(data::HomologyData{QQ}, z::AbstractVector{QQ})
    if data.dimH == 0
        return zeros(QQ, 0, 1)
    end
    z0 = Matrix{QQ}(reshape(Vector{QQ}(z), :, 1))
    alpha = solve_fullcolumnQQ(data.Z, z0)
    gamma = solve_fullcolumnQQ(data.Bfull, alpha)
    rB = data.dimB
    return gamma[rB+1:end, :]
end

# =====================================================================================
# Additional chain-level and derived-category infrastructure
# =====================================================================================

# -------------------------------------------------------------------------------------
# Induced maps on homology
# -------------------------------------------------------------------------------------

"""
    induced_map_on_homology(src, tgt, f) -> Matrix{QQ}

Given homology data
  * `src = homology_data(dimsC, dC, s)` for a chain complex C, and
  * `tgt = homology_data(dimsD, dD, s)` for a chain complex D,

and a linear map `f : C_s -> D_s`, compute the induced map on homology

    H_s(C) -> H_s(D)

in the *specific bases* stored inside `src` and `tgt` (namely the columns of `src.Hrep`
and `tgt.Hrep`).

The output matrix has size `tgt.dimH times src.dimH`.

Mathematician-friendly interpretation:
- The j-th basis vector of `H_s(C)` is represented by the cycle `src.Hrep[:,j]`.
- We apply `f` to that cycle and reduce mod boundaries in `D_s` to obtain coordinates
  in the target basis.
"""
function induced_map_on_homology(src::HomologyData{QQ}, tgt::HomologyData{QQ}, f::AbstractMatrix{QQ})
    if src.dimH == 0 || tgt.dimH == 0
        return zeros(QQ, tgt.dimH, src.dimH)
    end

    F = zeros(QQ, tgt.dimH, src.dimH)
    for j in 1:src.dimH
        z = src.Hrep[:, j:j]           # representative cycle in C_s (as a column matrix)
        img = f * z                    # element of D_s
        coords = homology_coordinates(tgt, img)
        F[:, j] = coords[:, 1]
    end
    return F
end


# -------------------------------------------------------------------------------------
# Derived-category style constructions for cochain complexes
# -------------------------------------------------------------------------------------

# These helpers are intentionally small and explicit. They are designed to make it easy
# to build mapping cones, distinguished triangles, and their long exact sequences in
# a way that keeps bases coherent and inspectable.

# Internal: dimension of C^t, returning 0 outside the defined range.
_dim_at(C::CochainComplex{QQ}, t::Int) =
    (t < C.tmin || t > C.tmax) ? 0 : C.dims[t - C.tmin + 1]

# Internal: differential d^t : C^t -> C^{t+1}, returning the appropriate zero matrix
# outside the defined range.
function _diff_at(C::CochainComplex{QQ}, t::Int)
    if t < C.tmin || t >= C.tmax
        return spzeros(QQ, _dim_at(C, t+1), _dim_at(C, t))
    end
    return C.d[t - C.tmin + 1]
end

"""
    shift(C, k) -> CochainComplex{QQ}

Return the shifted cochain complex `C[k]` in the standard triangulated-category
convention:

- `(C[k])^t = C^{t+k}`
- `d_{C[k]}^t = (-1)^k cdot d_C^{t+k}`

In particular, `C[1]` has differential `-d` (after reindexing).
"""
function shift(C::CochainComplex{QQ}, k::Int)
    tmin = C.tmin - k
    tmax = C.tmax - k

    dims = Int[]
    for t in tmin:tmax
        push!(dims, _dim_at(C, t+k))
    end

    d = SparseMatrixCSC{QQ,Int}[]
    for t in tmin:(tmax-1)
        dt = _diff_at(C, t+k)
        if isodd(k)
            dt = -dt
        end
        push!(d, dt)
    end

    return CochainComplex{QQ}(tmin, tmax, dims, d)
end


"""
    CochainMap(C, D, maps; check=true)

A degreewise linear map of cochain complexes `f : C -> D`.

- `maps[i]` is the matrix `C^t -> D^t` where `t = C.tmin + (i-1)`.
- For now, we require `C` and `D` to have the same degree range.

If `check=true`, we verify the cochain map identity
`d_D circ f = f circ d_C` in each degree.
"""
struct CochainMap{K}
    C::CochainComplex{K}
    D::CochainComplex{K}
    maps::Vector{SparseMatrixCSC{K,Int}}
end

function CochainMap(C::CochainComplex{QQ},
                    D::CochainComplex{QQ},
                    maps::Vector{<:AbstractMatrix{QQ}}; check::Bool=true)
    if C.tmin != D.tmin || C.tmax != D.tmax
        error("CochainMap: currently requires C and D to have the same degree range.")
    end
    if length(maps) != length(C.dims)
        error("CochainMap: expected one map per degree t = tmin..tmax.")
    end

    spmaps = SparseMatrixCSC{QQ,Int}[]
    for t in C.tmin:C.tmax
        ft = sparse(maps[t - C.tmin + 1])
        if size(ft, 2) != _dim_at(C, t) || size(ft, 1) != _dim_at(D, t)
            error("CochainMap: size mismatch at degree $(t).")
        end
        push!(spmaps, ft)
    end

    m = CochainMap{QQ}(C, D, spmaps)

    if check
        for t in C.tmin:(C.tmax-1)
            f_t   = m.maps[t - C.tmin + 1]
            f_tp1 = m.maps[t - C.tmin + 2]
            lhs = _diff_at(D, t) * f_t
            rhs = f_tp1 * _diff_at(C, t)
            if lhs != rhs
                error("CochainMap: check failed at degree $(t).")
            end
        end
    end

    return m
end


"""
    mapping_cone(f) -> CochainComplex{QQ}

Return the mapping cone `Cone(f)` of a cochain map `f : C -> D`.

Convention (standard for cochain complexes):
- `Cone(f)^t = D^t oplus C^{t+1}`
- `d_Cone^t(d, c) = ( d_D(d) + f^{t+1}(c),  - d_C(c) )`

This fits into the standard distinguished triangle

    C --f--> D --> Cone(f) --> C[1].
"""
function mapping_cone(f::CochainMap{QQ})
    C = f.C
    D = f.D

    # We keep the cone indexed over the same t-range as D, which is the same as C here.
    tmin = D.tmin
    tmax = D.tmax

    dims_cone = Int[]
    for t in tmin:tmax
        dimD_t  = _dim_at(D, t)
        dimC_t1 = _dim_at(C, t+1)
        push!(dims_cone, dimD_t + dimC_t1)
    end

    d_cone = SparseMatrixCSC{QQ,Int}[]
    for t in tmin:(tmax-1)
        dimD_t    = _dim_at(D, t)
        dimD_tp1  = _dim_at(D, t+1)
        dimC_t1   = _dim_at(C, t+1)
        dimC_t2   = _dim_at(C, t+2)

        dD = _diff_at(D, t)         # D^t -> D^{t+1}
        dC = _diff_at(C, t+1)       # C^{t+1} -> C^{t+2} (possibly zero at the top)
        f_tp1 = f.maps[t - tmin + 2]  # f^{t+1} : C^{t+1} -> D^{t+1}

        # Block form:
        # [ dD    f^{t+1} ]
        # [  0    -dC    ]
        Z = spzeros(QQ, dimC_t2, dimD_t)
        d_cone_t = [dD f_tp1; Z -dC]
        push!(d_cone, d_cone_t)
    end

    return CochainComplex{QQ}(tmin, tmax, dims_cone, d_cone)
end


"""
    DistinguishedTriangle

A concrete distinguished triangle of cochain complexes

    C --f--> D --i--> Cone(f) --p--> C[1]

constructed via the mapping cone.
"""
struct DistinguishedTriangle{K}
    C::CochainComplex{K}
    D::CochainComplex{K}
    f::CochainMap{K}
    cone::CochainComplex{K}
    i::CochainMap{K}          # D -> Cone(f)
    p::CochainMap{K}          # Cone(f) -> C[1]
    Cshift::CochainComplex{K} # C[1]
end


"""
    mapping_cone_triangle(f) -> DistinguishedTriangle{QQ}

Construct the standard distinguished triangle

    C --f--> D --> Cone(f) --> C[1]

together with explicit cochain maps for the inclusion `D -> Cone(f)` and projection
`Cone(f) -> C[1]` in degreewise matrix form.
"""
function mapping_cone_triangle(f::CochainMap{QQ})
    C = f.C
    D = f.D
    cone = mapping_cone(f)
    Cshift = shift(C, 1)

    if C.tmin != D.tmin || C.tmax != D.tmax
        error("mapping_cone_triangle: requires C and D to have the same degree range.")
    end

    tmin = C.tmin
    tmax = C.tmax

    # Inclusion i^t : D^t -> D^t oplus C^{t+1}
    imaps = SparseMatrixCSC{QQ,Int}[]
    # Projection p^t : D^t oplus C^{t+1} -> C^{t+1}  (this is (C[1])^t)
    pmaps = SparseMatrixCSC{QQ,Int}[]

    for t in tmin:tmax
        dimD_t  = _dim_at(D, t)
        dimC_t1 = _dim_at(C, t+1)

        # i = [I; 0]
        I_D = spdiagm(0 => fill(one(QQ), dimD_t))
        Z   = spzeros(QQ, dimC_t1, dimD_t)
        push!(imaps, [I_D; Z])

        # p = [0  I]
        Z2  = spzeros(QQ, dimC_t1, dimD_t)
        I_C = spdiagm(0 => fill(one(QQ), dimC_t1))
        push!(pmaps, [Z2 I_C])
    end

    i = CochainMap(D, cone, imaps; check=true)
    p = CochainMap(cone, Cshift, pmaps; check=true)

    return DistinguishedTriangle{QQ}(C, D, f, cone, i, p, Cshift)
end


"""
    LongExactSequence

A packaged long exact sequence in cohomology coming from a distinguished triangle.
We store the cohomology data objects and the induced maps in the corresponding bases.

For the mapping-cone triangle C -> D -> Cone(f) -> C[1], the long exact sequence is

    ... -> H^t(C) -> H^t(D) -> H^t(Cone(f)) -> H^t(C[1]) -> H^{t+1}(D) -> ...

Remember that `H^t(C[1])` is canonically isomorphic to `H^{t+1}(C)`; here we keep it
as `H^t(C[1])` to avoid silently changing bases.
"""
struct LongExactSequence{K}
    triangle::DistinguishedTriangle{K}
    HC::Vector{CohomologyData{K}}
    HD::Vector{CohomologyData{K}}
    Hcone::Vector{CohomologyData{K}}
    HCshift::Vector{CohomologyData{K}}
    fH::Vector{Matrix{K}}     # H^t(C) -> H^t(D)
    iH::Vector{Matrix{K}}     # H^t(D) -> H^t(Cone)
    pH::Vector{Matrix{K}}     # H^t(Cone) -> H^t(C[1])
end

"""
    cohomology_data(C::CochainComplex{QQ}) -> Vector{CohomologyData{QQ}}

Compute `CohomologyData` in every degree of the complex, using the built-in
linear algebra over QQ. The output vector is indexed by `t = C.tmin..C.tmax`.
"""
function cohomology_data(C::CochainComplex{QQ})
    out = CohomologyData{QQ}[]
    for t in C.tmin:C.tmax
        push!(out, cohomology_data(C.dims, C.d, t))
    end
    return out
end

"""
    long_exact_sequence(tri) -> LongExactSequence{QQ}

Package the long exact sequence in cohomology induced by a distinguished triangle,
together with explicit matrices for the induced maps in the stored bases.
"""
function long_exact_sequence(tri::DistinguishedTriangle{QQ})
    C = tri.C
    D = tri.D
    cone = tri.cone
    Cshift = tri.Cshift

    HC = cohomology_data(C)
    HD = cohomology_data(D)
    Hcone = cohomology_data(cone)
    HCshift = cohomology_data(Cshift)

    fH = Matrix{QQ}[]
    iH = Matrix{QQ}[]
    pH = Matrix{QQ}[]

    for t in C.tmin:C.tmax
        idx = t - C.tmin + 1
        push!(fH, induced_map_on_cohomology(HC[idx], HD[idx], tri.f.maps[idx]))
        push!(iH, induced_map_on_cohomology(HD[idx], Hcone[idx], tri.i.maps[idx]))
        push!(pH, induced_map_on_cohomology(Hcone[idx], HCshift[idx], tri.p.maps[idx]))
    end

    return LongExactSequence{QQ}(tri, HC, HD, Hcone, HCshift, fH, iH, pH)
end


# -------------------------------------------------------------------------------------
# Spectral sequences from (finite) double complexes
# -------------------------------------------------------------------------------------

"""
    DoubleComplex{K}

A finite first-quadrant style cochain bicomplex with bidegrees (a,b):

- Objects:  C^{a,b}  for amin <= a <= amax and bmin <= b <= bmax
- Vertical differential:   dv^{a,b} : C^{a,b} -> C^{a,b+1}
- Horizontal differential: dh^{a,b} : C^{a,b} -> C^{a+1,b}

We assume:
- dv circ dv = 0 and dh circ dh = 0 (within range),
- dv circ dh + dh circ dv = 0 (anti-commutation), as is standard for a bicomplex.

Storage convention:
- `dims[aidx,bidx] = dim(C^{a,b})` where aidx = a-amin+1 and bidx = b-bmin+1.
- `dv[aidx,bidx]` is the matrix for dv^{a,b} (or a correctly-sized zero matrix at the boundary).
- `dh[aidx,bidx]` is the matrix for dh^{a,b} (or a correctly-sized zero matrix at the boundary).
"""
struct DoubleComplex{K}
    amin::Int
    amax::Int
    bmin::Int
    bmax::Int
    dims::Matrix{Int}
    dv::Array{SparseMatrixCSC{K,Int},2}
    dh::Array{SparseMatrixCSC{K,Int},2}
end

"""
    total_complex(DC) -> CochainComplex{QQ}

Form the total cochain complex Tot(DC) with grading t = a+b and differential d = dv + dh.
(We assume dh already carries whatever sign convention was used to enforce anti-commutation.)

This produces a *concrete* cochain complex suitable for cohomology computations.
"""
function total_complex(DC::DoubleComplex{QQ})
    amin, amax = DC.amin, DC.amax
    bmin, bmax = DC.bmin, DC.bmax

    tmin = amin + bmin
    tmax = amax + bmax

    # dims of Tot^t
    dims_tot = Int[]
    for t in tmin:tmax
        s = 0
        for a in amin:amax
            b = t - a
            if bmin <= b <= bmax
                s += DC.dims[a - amin + 1, b - bmin + 1]
            end
        end
        push!(dims_tot, s)
    end

    # offsets: for each t, store the starting index of each (a,b) block inside Tot^t
    # We store as a Dict keyed by (a,b).
    offsets = Vector{Dict{Tuple{Int,Int},Int}}(undef, length(dims_tot))
    for (ti, t) in enumerate(tmin:tmax)
        d = Dict{Tuple{Int,Int},Int}()
        off = 1
        for a in amin:amax
            b = t - a
            if bmin <= b <= bmax
                d[(a,b)] = off
                off += DC.dims[a - amin + 1, b - bmin + 1]
            end
        end
        offsets[ti] = d
    end

    # build differentials Tot^t -> Tot^{t+1}
    d_tot = SparseMatrixCSC{QQ,Int}[]
    for t in tmin:(tmax-1)
        dom_dim = dims_tot[t - tmin + 1]
        cod_dim = dims_tot[t - tmin + 2]

        I = Int[]
        J = Int[]
        V = QQ[]

        # contributions from each block (a,b) with a+b = t
        for a in amin:amax
            b = t - a
            if !(bmin <= b <= bmax)
                continue
            end
            aidx = a - amin + 1
            bidx = b - bmin + 1

            dom0 = offsets[t - tmin + 1][(a,b)]
            # dv goes to (a,b+1)
            if b < bmax
                cod0 = offsets[t - tmin + 2][(a,b+1)]
                B = DC.dv[aidx, bidx]
                rows, cols, vals = findnz(B)
                for k in eachindex(vals)
                    push!(I, cod0 + rows[k] - 1)
                    push!(J, dom0 + cols[k] - 1)
                    push!(V, vals[k])
                end
            end
            # dh goes to (a+1,b)
            if a < amax
                cod0 = offsets[t - tmin + 2][(a+1,b)]
                B = DC.dh[aidx, bidx]
                rows, cols, vals = findnz(B)
                for k in eachindex(vals)
                    push!(I, cod0 + rows[k] - 1)
                    push!(J, dom0 + cols[k] - 1)
                    push!(V, vals[k])
                end
            end
        end

        push!(d_tot, sparse(I, J, V, cod_dim, dom_dim))
    end

    return CochainComplex{QQ}(tmin, tmax, dims_tot, d_tot)
end


"""
    SpectralSequence{K}

A minimal, explicit spectral sequence object attached to a double complex.
We currently compute only:
- E1 page (dimensions and d1),
- E2 page (dimensions),

for either the "vertical-first" or "horizontal-first" filtration.

This is deliberately limited but mathematically coherent: E1/E2 are the pages that are
most often inspected in concrete computations, and they are directly computable by
linear algebra on columns/rows.
"""
struct SpectralSequence{K}
    DC::DoubleComplex{K}
    first::Symbol            # :vertical or :horizontal

    E1_dims::Matrix{Int}     # dims of E1^{a,b}
    d1::Array{SparseMatrixCSC{K,Int},2}  # d1 maps (direction depends on `first`)
    E2_dims::Matrix{Int}     # dims of E2^{a,b}
end

"""
    spectral_sequence(DC; first=:vertical) -> SpectralSequence{QQ}

Compute the (E1, d1, E2) data for the spectral sequence of a double complex.

- If `first=:vertical`, we take cohomology of (C^{a,*}, dv) first, so
  d1 is induced by dh and goes (a,b) -> (a+1,b).
- If `first=:horizontal`, we take cohomology of (C^{*,b}, dh) first, so
  d1 is induced by dv and goes (a,b) -> (a,b+1).

We only compute through the E2 page.
"""
function spectral_sequence(DC::DoubleComplex{QQ}; first::Symbol = :vertical)
    amin, amax = DC.amin, DC.amax
    bmin, bmax = DC.bmin, DC.bmax
    Alen = amax - amin + 1
    Blen = bmax - bmin + 1

    E1 = zeros(Int, Alen, Blen)
    E2 = zeros(Int, Alen, Blen)
    d1 = Array{SparseMatrixCSC{QQ,Int},2}(undef, Alen, Blen)

    if first == :vertical
        # Column cohomology first.
        col_coh = Array{CohomologyData{QQ},2}(undef, Alen, Blen)

        for aidx in 1:Alen
            dims_col = [DC.dims[aidx, bidx] for bidx in 1:Blen]
            d_col = SparseMatrixCSC{QQ,Int}[]
            for bidx in 1:(Blen-1)
                push!(d_col, DC.dv[aidx, bidx])
            end
            Ccol = CochainComplex{QQ}(bmin, bmax, dims_col, d_col)
            Hcol = cohomology_data(Ccol)
            for bidx in 1:Blen
                col_coh[aidx, bidx] = Hcol[bidx]
                E1[aidx, bidx] = Hcol[bidx].dimH
            end
        end

        # d1 induced by dh: E1^{a,b} -> E1^{a+1,b}
        for aidx in 1:Alen
            for bidx in 1:Blen
                if aidx < Alen
                    f = DC.dh[aidx, bidx]  # C^{a,b} -> C^{a+1,b}
                    d1[aidx, bidx] = sparse(induced_map_on_cohomology(col_coh[aidx, bidx],
                                                                      col_coh[aidx+1, bidx],
                                                                      f))
                else
                    d1[aidx, bidx] = spzeros(QQ, 0, E1[aidx, bidx])
                end
            end
        end

        # E2 is cohomology of the rows in the a-direction on the E1 page, separately for each b.
        for bidx in 1:Blen
            dims_row = [E1[aidx, bidx] for aidx in 1:Alen]
            d_row = SparseMatrixCSC{QQ,Int}[]
            for aidx in 1:(Alen-1)
                push!(d_row, d1[aidx, bidx])
            end
            Crow = CochainComplex{QQ}(amin, amax, dims_row, d_row)
            Hrow = cohomology_data(Crow)
            for aidx in 1:Alen
                E2[aidx, bidx] = Hrow[aidx].dimH
            end
        end

        return SpectralSequence{QQ}(DC, :vertical, E1, d1, E2)
    elseif first == :horizontal
        # Row cohomology first.
        row_coh = Array{CohomologyData{QQ},2}(undef, Alen, Blen)

        for bidx in 1:Blen
            dims_row = [DC.dims[aidx, bidx] for aidx in 1:Alen]
            d_row = SparseMatrixCSC{QQ,Int}[]
            for aidx in 1:(Alen-1)
                push!(d_row, DC.dh[aidx, bidx])
            end
            Crow = CochainComplex{QQ}(amin, amax, dims_row, d_row)
            Hrow = cohomology_data(Crow)
            for aidx in 1:Alen
                row_coh[aidx, bidx] = Hrow[aidx]
                E1[aidx, bidx] = Hrow[aidx].dimH
            end
        end

        # d1 induced by dv: E1^{a,b} -> E1^{a,b+1}
        for aidx in 1:Alen
            for bidx in 1:Blen
                if bidx < Blen
                    f = DC.dv[aidx, bidx]  # C^{a,b} -> C^{a,b+1}
                    d1[aidx, bidx] = sparse(induced_map_on_cohomology(row_coh[aidx, bidx],
                                                                      row_coh[aidx, bidx+1],
                                                                      f))
                else
                    d1[aidx, bidx] = spzeros(QQ, 0, E1[aidx, bidx])
                end
            end
        end

        # E2 is cohomology in the b-direction on the E1 page, separately for each a.
        for aidx in 1:Alen
            dims_col = [E1[aidx, bidx] for bidx in 1:Blen]
            d_col = SparseMatrixCSC{QQ,Int}[]
            for bidx in 1:(Blen-1)
                push!(d_col, d1[aidx, bidx])
            end
            Ccol = CochainComplex{QQ}(bmin, bmax, dims_col, d_col)
            Hcol = cohomology_data(Ccol)
            for bidx in 1:Blen
                E2[aidx, bidx] = Hcol[bidx].dimH
            end
        end

        return SpectralSequence{QQ}(DC, :horizontal, E1, d1, E2)
    else
        error("spectral_sequence: `first` must be :vertical or :horizontal.")
    end
end


end
