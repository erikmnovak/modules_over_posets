module PLPolyhedra
# =============================================================================
# Piecewise-linear (PL) backend using Polyhedra + CDD exact arithmetic.
#
# NOTE ON OPTIONAL DEPENDENCIES
# - Polyhedra/CDDLib are optional.
# - This module still loads without them.
# - Membership tests in HPoly / PolyUnion are implemented directly from stored
#   H-representations A*x <= b (exact QQ) and DO NOT require Polyhedra.
# - Feasibility enumeration (encode_from_PL_fringe) DOES require Polyhedra/CDDLib.
#
# Key robustness change:
#   We store the H-rep matrix (A,b) inside HPoly and avoid Polyhedra API helpers
#   like hrep_matrix (which may not exist in the installed Polyhedra version).
# =============================================================================

using ..FiniteFringe
using ..CoreModules: QQ

# ------------------------------ Optional deps ---------------------------------
const HAVE_POLY = try
    @eval begin
        import Polyhedra
        import CDDLib
    end
    true
catch
    false
end

const _CDD = HAVE_POLY ? CDDLib.Library(:exact) : nothing

# Small positive slack for strict facet violation (kept exact)
const STRICT_EPS_QQ = 1//(big(1) << 40)

# ------------------------------ QQ conversion ---------------------------------
# Robust conversion to QQ = Rational{BigInt}.
_toQQ(u)::QQ = u isa QQ       ? u :
               u isa Integer  ? (BigInt(u) // BigInt(1)) :
               u isa Rational ? (BigInt(numerator(u)) // BigInt(denominator(u))) :
               (u isa AbstractFloat ? rationalize(QQ, u) : rationalize(QQ, float(u)))

function _toQQ_vec(v::AbstractVector)
    out = Vector{QQ}(undef, length(v))
    for i in eachindex(v)
        out[i] = _toQQ(v[i])
    end
    return out
end

function _toQQ_mat(A::AbstractMatrix)
    m, n = size(A)
    out = Matrix{QQ}(undef, m, n)
    for i in 1:m, j in 1:n
        out[i,j] = _toQQ(A[i,j])
    end
    return out
end

# ------------------------------ Basic types -----------------------------------

"""
HPoly

A single convex polyhedron in H-representation:
    { x in R^n : A*x <= b }.

We store A,b explicitly (QQ) to avoid relying on Polyhedra internals for
membership tests and facet extraction.

Field `poly` is an optional Polyhedra object (Any). It is present only when
HAVE_POLY is true and the polyhedron was constructed with Polyhedra.
"""
struct HPoly
    n::Int
    A::Matrix{QQ}
    b::Vector{QQ}
    poly::Any
end

"""
PolyUnion

Finite union of convex HPolys. Membership is disjunction.
"""
struct PolyUnion
    n::Int
    parts::Vector{HPoly}
end

"""
PLUpset / PLDownset

Birth and death shapes as unions of convex parts.
"""
struct PLUpset
    U::PolyUnion
end
struct PLDownset
    D::PolyUnion
end

# ----------------------------- Geometry classes --------------------------------
const GeometryClass = Symbol
geometry_class(::HPoly)      = :PL::GeometryClass
geometry_class(::PolyUnion)  = :PL::GeometryClass
geometry_class(::PLUpset)    = :PL::GeometryClass
geometry_class(::PLDownset)  = :PL::GeometryClass

function _assert_PL_inputs(Ups::Vector{PLUpset}, Downs::Vector{PLDownset})
    for u in Ups
        @assert geometry_class(u) === :PL "Non-PL upset encountered"
    end
    for d in Downs
        @assert geometry_class(d) === :PL "Non-PL downset encountered"
    end
    nothing
end

# --------------------------- Construction helpers -----------------------------

"""
    make_hpoly(A, b) -> HPoly

Build a convex polyhedron { x : A*x <= b }.

- Stores A,b exactly in QQ inside the returned HPoly.
- If Polyhedra/CDDLib are available, also builds an exact Polyhedra object
  for feasibility testing and witness extraction used by encode_from_PL_fringe.
"""
function make_hpoly(A::AbstractMatrix, b::AbstractVector)
    m, n = size(A)
    length(b) == m || error("make_hpoly: size mismatch A (m,n) vs b (m).")

    Aqq = _toQQ_mat(A)
    bqq = _toQQ_vec(b)

    poly = nothing
    if HAVE_POLY
        hrep = Polyhedra.hrep(Aqq, bqq)
        poly = Polyhedra.polyhedron(hrep, _CDD)
    end

    return HPoly(n, Aqq, bqq, poly)
end

"Convenience: single-part PolyUnion."
poly_union(h::HPoly) = PolyUnion(h.n, [h])

# ---------------------------- Membership checks -------------------------------

"Exact membership using stored A*x <= b in QQ."
function _in_hpoly(h::HPoly, x::AbstractVector)
    length(x) == h.n || error("dimension mismatch in _in_hpoly")
    xqq = _toQQ_vec(x)
    m = size(h.A, 1)
    for i in 1:m
        s = zero(QQ)
        @inbounds for j in 1:h.n
            s += h.A[i,j] * xqq[j]
        end
        if s > h.b[i]
            return false
        end
    end
    return true
end

"Point membership in a union of polytopes."
function _in_union(U::PolyUnion, x::AbstractVector)
    for p in U.parts
        if _in_hpoly(p, x)
            return true
        end
    end
    return false
end

contains(U::PLUpset, x::AbstractVector)   = _in_union(U.U, x)
contains(D::PLDownset, x::AbstractVector) = _in_union(D.D, x)

# ---------------------- Feasibility helper for enumeration --------------------

# Collect facet inequalities (a, b) representing a'*x <= b from stored A,b.
function _facets_of(hp::HPoly)::Vector{Tuple{Vector{QQ},QQ}}
    m, n = size(hp.A)
    out = Vector{Tuple{Vector{QQ},QQ}}(undef, m)
    @inbounds for i in 1:m
        ai = Vector{QQ}(undef, n)
        for j in 1:n
            ai[j] = hp.A[i,j]
        end
        out[i] = (ai, hp.b[i])
    end
    return out
end

# Build intersection of in_parts plus "outside" constraints represented as:
#   a'*x >= b0  (to emulate strict violation we use b0 + STRICT_EPS_QQ)
function _internal_build_poly(in_parts::Vector{HPoly},
                              out_halfspaces::Vector{Tuple{Vector{T},T}};
                              strict_eps::QQ=STRICT_EPS_QQ) where {T<:Real}
    HAVE_POLY || error("Polyhedra/CDDLib not available; install Polyhedra.jl and CDDLib.jl.")

    # Determine ambient dimension.
    n = if !isempty(in_parts)
        in_parts[1].n
    elseif !isempty(out_halfspaces)
        length(out_halfspaces[1][1])
    else
        0
    end
    n == 0 && return (nothing, nothing, true)

    # Count total constraints.
    m_in = 0
    for p in in_parts
        p.n == n || error("_internal_build_poly: dimension mismatch among in_parts")
        m_in += size(p.A, 1)
    end
    m_out = length(out_halfspaces)
    m_tot = m_in + m_out

    A = Matrix{QQ}(undef, m_tot, n)
    b = Vector{QQ}(undef, m_tot)

    # Fill with in-parts inequalities.
    row = 1
    for p in in_parts
        mp = size(p.A, 1)
        if mp > 0
            A[row:row+mp-1, :] .= p.A
            b[row:row+mp-1]    .= p.b
            row += mp
        end
    end

    # Add outside constraints: a'*x >= b0 + eps  <=>  (-a)'*x <= -(b0 + eps)
    for (a0, b0) in out_halfspaces
        length(a0) == n || error("_internal_build_poly: outside halfspace has wrong dimension")
        for j in 1:n
            A[row, j] = -_toQQ(a0[j])
        end
        b[row] = -(_toQQ(b0) + strict_eps)
        row += 1
    end

    hrep = Polyhedra.hrep(A, b)
    P = Polyhedra.polyhedron(hrep, _CDD)

    if Polyhedra.isempty(P)
        return (nothing, nothing, true)
    end

    # Best-effort witness (not required for correctness; safe if Polyhedra API differs).
    witness = nothing
    try
        V = Polyhedra.vrep(P)
        pts = Polyhedra.points(V)
        if length(pts) > 0
            witness = Vector{Float64}(pts[1])
        end
    catch
        witness = nothing
    end

    return (HPoly(n, A, b, P), witness, false)
end

# ------------------------- Region enumeration (Y-signatures) ------------------

function enumerate_feasible_regions(Ups::Vector{PLUpset}, Downs::Vector{PLDownset};
                                    max_regions::Int=10_000,
                                    strict_eps::QQ=STRICT_EPS_QQ)
    HAVE_POLY || error("Polyhedra/CDDLib not available; install Polyhedra.jl and CDDLib.jl.")

    m = length(Ups)
    r = length(Downs)
    n = (m > 0 ? Ups[1].U.n : (r > 0 ? Downs[1].D.n : 0))

    results = Vector{Tuple{BitVector,BitVector,HPoly,Vector{Float64}}}()

    # Helper: all ways to force OUTSIDE a union of HPolys:
    # pick one facet inequality to violate for each part.
    function outside_choices(union::PolyUnion)
        if isempty(union.parts)
            return [Tuple{Vector{QQ},QQ}[]]  # outside(empty) = whole space, no constraints
        end
        facet_lists = Vector{Vector{Tuple{Vector{QQ},QQ}}}(undef, length(union.parts))
        for i in 1:length(union.parts)
            facet_lists[i] = _facets_of(union.parts[i])
            if isempty(facet_lists[i])
                return Vector{Vector{Tuple{Vector{QQ},QQ}}}() # cannot be outside full space part
            end
        end
        out = Vector{Vector{Tuple{Vector{QQ},QQ}}}()
        function rec(i::Int, acc::Vector{Tuple{Vector{QQ},QQ}})
            if i > length(facet_lists)
                push!(out, copy(acc))
                return
            end
            for f in facet_lists[i]
                push!(acc, f)
                rec(i+1, acc)
                pop!(acc)
            end
        end
        rec(1, Tuple{Vector{QQ},QQ}[])
        return out
    end

    # Iterate over all signatures y in {0,1}^m and z in {0,1}^r.
    total = 1 << (m + r)
    for mask in 0:(total-1)
        y = falses(m)
        z = falses(r)
        for i in 1:m
            y[i] = ((mask >> (i-1)) & 1) == 1
        end
        for j in 1:r
            z[j] = ((mask >> (m + j - 1)) & 1) == 1
        end

        # Build "inside" disjunction choices (pick one part for each inside union constraint).
        in_choices = Vector{Vector{HPoly}}()
        push!(in_choices, HPoly[])

        # For Upsets: y[i] == 1 means inside U_i; y[i] == 0 means outside U_i.
        out_choices = Vector{Vector{Tuple{Vector{QQ},QQ}}}()
        push!(out_choices, Tuple{Vector{QQ},QQ}[])

        # Upset constraints
        feasible = true
        for i in 1:m
            if y[i]
                # inside union: choose one part
                parts = Ups[i].U.parts
                if isempty(parts)
                    feasible = false
                    break
                end
                new_in = Vector{Vector{HPoly}}()
                for base in in_choices, part in parts
                    push!(new_in, vcat(base, [part]))
                end
                in_choices = new_in
            else
                # outside union
                choices = outside_choices(Ups[i].U)
                if isempty(choices)
                    feasible = false
                    break
                end
                out_choices = [vcat(base, ch) for base in out_choices, ch in choices]
            end
        end
        feasible || continue

        # Downset constraints: z[j] == 0 means inside D_j; z[j] == 1 means outside D_j.
        for j in 1:r
            if !z[j]
                parts = Downs[j].D.parts
                if isempty(parts)
                    feasible = false
                    break
                end
                new_in = Vector{Vector{HPoly}}()
                for base in in_choices, part in parts
                    push!(new_in, vcat(base, [part]))
                end
                in_choices = new_in
            else
                choices = outside_choices(Downs[j].D)
                if isempty(choices)
                    feasible = false
                    break
                end
                out_choices = [vcat(base, ch) for base in out_choices, ch in choices]
            end
        end
        feasible || continue

        # Test feasibility for each branch.
        for iparts in in_choices
            for out_hs in out_choices
                hp, wit, isemp = _internal_build_poly(iparts, out_hs; strict_eps=strict_eps)
                if !isemp
                    push!(results, (BitVector(y), BitVector(z), hp, wit === nothing ? Float64[] : wit))
                    if length(results) >= max_regions
                        break
                    end
                end
            end
            if length(results) >= max_regions
                break
            end
        end

        if length(results) >= max_regions
            @warn "enumerate_feasible_regions: reached max_regions cap; stopping early."
            break
        end
    end

    # Collapse equivalent (y,z) signatures. Use tuple-of-bools keys (content-hashable).
    seen = Dict{Tuple{Tuple,Tuple},Int}()
    collapsed = Vector{Tuple{BitVector,BitVector,HPoly,Vector{Float64}}}()
    for rec in results
        key = (Tuple(rec[1]), Tuple(rec[2]))
        if !haskey(seen, key)
            seen[key] = 1
            push!(collapsed, rec)
        end
    end
    return collapsed
end

# --------------------------- Encoding (P, H_hat, pi) --------------------------

struct PLEncodingMap
    n::Int
    sig_y::Vector{BitVector}
    sig_z::Vector{BitVector}
    regions::Vector{HPoly}
    witnesses::Vector{Vector{Float64}}
end

# Locate by scanning stored region reps (works as long as each signature has a nonempty rep).
function locate(pi::PLEncodingMap, x::AbstractVector)
    length(x) == pi.n || error("locate: dimension mismatch")
    for (idx, hp) in enumerate(pi.regions)
        if _in_hpoly(hp, x)
            return idx
        end
    end
    return 0
end

function _uptight_from_signatures(sig_y::Vector{BitVector}, sig_z::Vector{BitVector})
    rN = length(sig_y)
    leq = falses(rN, rN)
    for i in 1:rN
        leq[i,i] = true
    end
    for i in 1:rN, j in 1:rN
        yi, zi = sig_y[i], sig_z[i]
        yj, zj = sig_y[j], sig_z[j]
        leq[i,j] = all(yi .<= yj) && all(zi .<= zj)
    end
    for k in 1:rN, i in 1:rN, j in 1:rN
        leq[i,j] = leq[i,j] || (leq[i,k] && leq[k,j])
    end
    return FiniteFringe.FinitePoset(leq)
end

function _images_on_P(P::FiniteFringe.FinitePoset,
                      sig_y::Vector{BitVector}, sig_z::Vector{BitVector},
                      m::Int, r::Int)
    Uhat = Vector{FiniteFringe.Upset}(undef, m)
    Dhat = Vector{FiniteFringe.Downset}(undef, r)
    for i in 1:m
        mask = BitVector([sig_y[t][i] == 1 for t in 1:P.n])
        Uhat[i] = FiniteFringe.upset_closure(P, mask)
    end
    for j in 1:r
        mask = BitVector([sig_z[t][j] == 0 for t in 1:P.n])
        Dhat[j] = FiniteFringe.downset_closure(P, mask)
    end
    return Uhat, Dhat
end

function _monomialize_phi(phi::AbstractMatrix{QQ}, Uhat, Dhat)
    m = length(Dhat)
    n = length(Uhat)
    Phi = copy(phi)
    for j in 1:m, i in 1:n
        if !FiniteFringe.intersects(Uhat[i], Dhat[j])
            Phi[j,i] = zero(QQ)
        end
    end
    return Phi
end

function encode_from_PL_fringe(Ups::Vector{PLUpset},
                               Downs::Vector{PLDownset},
                               Phi_in::AbstractMatrix{QQ};
                               max_regions::Int=10_000,
                               strict_eps::QQ=STRICT_EPS_QQ)
    _assert_PL_inputs(Ups, Downs)
    HAVE_POLY || error("Polyhedra/CDDLib not available; install Polyhedra.jl and CDDLib.jl.")

    feasible = enumerate_feasible_regions(Ups, Downs; max_regions=max_regions, strict_eps=strict_eps)

    if isempty(feasible)
        P = FiniteFringe.FinitePoset(reshape(Bool[true], 1, 1))
        Uhat = FiniteFringe.Upset[FiniteFringe.upset_closure(P, BitVector([false])) for _ in 1:length(Ups)]
        Dhat = FiniteFringe.Downset[FiniteFringe.downset_closure(P, BitVector([false])) for _ in 1:length(Downs)]
        Phi0 = zeros(QQ, length(Downs), length(Ups))
        H = FiniteFringe.FringeModule{QQ}(P, Uhat, Dhat, Phi0)
        pi = PLEncodingMap( (length(Ups)>0 ? Ups[1].U.n : (length(Downs)>0 ? Downs[1].D.n : 0)),
                            BitVector[], BitVector[], HPoly[], Vector{Vector{Float64}}() )
        return P, H, pi
    end

    sigy = Vector{BitVector}(undef, length(feasible))
    sigz = Vector{BitVector}(undef, length(feasible))
    regs = Vector{HPoly}(undef, length(feasible))
    wits = Vector{Vector{Float64}}(undef, length(feasible))
    for (k, rec) in enumerate(feasible)
        sigy[k] = rec[1]
        sigz[k] = rec[2]
        regs[k] = rec[3]
        wits[k] = rec[4]
    end

    P = _uptight_from_signatures(sigy, sigz)
    m = length(Ups)
    r = length(Downs)
    Uhat, Dhat = _images_on_P(P, sigy, sigz, m, r)

    Phi = _monomialize_phi(Matrix{QQ}(Phi_in), Uhat, Dhat)
    H = FiniteFringe.FringeModule{QQ}(P, Uhat, Dhat, Phi)

    pi = PLEncodingMap(regs[1].n, sigy, sigz, regs, wits)
    return P, H, pi
end

function encode_from_PL_fringe_with_tag(Ups, Downs, Phi_in; kwargs...)
    P, H, pi = encode_from_PL_fringe(Ups, Downs, Phi_in; kwargs...)
    return P, H, pi, :PL
end

export HPoly, PolyUnion, PLUpset, PLDownset, PLEncodingMap, locate, make_hpoly,
       encode_from_PL_fringe

end # module
