module PLPolyhedra
# =============================================================================
# Piecewise-linear (PL) backend using Polyhedra + CDD exact arithmetic.
#
# What this module provides:
#   * HPoly:    a single convex polyhedron stored in H-representation.
#   * PolyUnion:finite union of HPoly (for convenience; membership is "any").
#   * PLUpset / PLDownset: birth/death shapes as unions of convex parts.
#   * PLEncodingMap: classifier pi : R^n -> P (finite) with region H-reps
#     and interior witnesses for fast lookups.
#   * encode_from_PL_fringe: build an uptight encoding (P, H_hat, pi) from
#     PL upsets and downsets and a monomial matrix Phi, verifying feasibility
#     of Y-patterns under Y = Ups U {complement(Downs)}.
#
# Design choices:
#   - Polyhedra is optional but, when present, we do real feasibility with
#     CDD exact mode to line up with QQ scalars used elsewhere.
#   - We enumerate feasible Y-signatures by arrangement-style branching over
#     complements (each "outside D" is implemented by violating >= 1 facet).
#     Every feasible branch yields an H-representation and an interior point.
#   - Equivalent region signatures are collapsed: each distinct Y-membership
#     vector defines one region node, exactly Defs. 4.12-4.17 (uptight).
#   - The induced P-fringe obeys the monomial condition: phi_hat[j,i] = 0
#     unless im(U_i) intersects im(D_j) on P (Prop. 3.18).
#   - We return an actual PLEncodingMap pi along with (P, H_hat).
#
# References inside the repo:
#   - Finite posets, upset/downset masks, FringeModule, dense_to_sparse_K:
#       FiniteFringe.jl  (constructs masks and posets) 
#   - Encoding / uptight construction over *finite* posets:
#       Encoding.jl      (we mimic that strategy geometrically)
# =============================================================================

using ..FiniteFringe
using ..CoreModules: QQ
using ..ExactQQ: rankQQ   # used only in comments/testing; not needed below

# ------------------------------ Optional deps ---------------------------------
# We guard imports; the code runs with graceful error messages if Polyhedra
# is missing. When present, we force exact arithmetic via CDDLib(:exact).
const HAVE_POLY = try
    @eval begin
        import Polyhedra
        import CDDLib
    end
    true
catch
    false
end

# CDD "exact" backend object (constructed only if libs present)
const _CDD = HAVE_POLY ? CDDLib.Library(:exact) : nothing

# -----------tiny positive slack to emulate strict facet violation ---------------
# Prefer a rational eps so the intent is exact, then convert to Float64 when needed.
const STRICT_EPS_QQ  = 1//(big(1) << 40)        # approx 9.09e-13 as a Rational{BigInt}
const STRICT_EPS_F64 = Float64(STRICT_EPS_QQ)   # fallback for Float64 H-reps

# ------------------------------ Basic types -----------------------------------

"""
HPoly

A single convex polyhedron in H-representation (intersection of finitely
many closed half-spaces). We store the Polyhedra object directly and also
remember the ambient dimension n.
"""
struct HPoly
    n::Int
    poly::Any   # Polyhedra.Polyhedron, but kept as Any to avoid hard dep in signatures
end

"""
PolyUnion

A finite union of convex polyhedra. Membership is disjunction ("in any").
"""
struct PolyUnion
    n::Int
    parts::Vector{HPoly}
end

"""
PLUpset / PLDownset

Birth and death shapes as unions of convex pieces. This module does not
attempt to *prove* these are monotone sets under the ambient poset order;
it simply treats them as the geometric carriers for Y-membership tests.
"""
struct PLUpset
    U::PolyUnion
end
struct PLDownset
    D::PolyUnion
end

# ----------------------------- Geometry classes --------------------------------
# Bookkeeping tags that mirror Def. 2.15 and Thm. 4.22(3) in Miller (2008.00063v2).
# For this backend, all sets we construct are PL by design (finite unions of convex H-polytopes).

"Symbolic tag for geometry class of a set: :PL, :semialgebraic, or :subanalytic."
const GeometryClass = Symbol

"Return the geometry class tag for our objects; currently always :PL."
geometry_class(::HPoly)      = :PL::GeometryClass
geometry_class(::PolyUnion)  = :PL::GeometryClass
geometry_class(::PLUpset)    = :PL::GeometryClass
geometry_class(::PLDownset)  = :PL::GeometryClass

"Assert that inputs are PL; used to document and enforce the hypothesis of Thm. 4.22(3)."
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

Build a convex polyhedron { x in R^n : A*x <= b } in exact arithmetic.
A must be size (m,n), b length m; entries can be Int, QQ, or Float64.
"""
function make_hpoly(A::AbstractMatrix, b::AbstractVector)
    HAVE_POLY || error("Polyhedra/CDDLib not available; install Polyhedra.jl and CDDLib.jl.")
    m, n = size(A)
    length(b) == m || error("make_hpoly: size mismatch A (m,n) vs b (m).")
    
    
    # Convert A,b to QQ exactly
    # Robust conversion to QQ
    toQQ(u) = u isa QQ       ? u :
              u isa Rational ? (BigInt(numerator(u)) // BigInt(denominator(u))) :
              u isa Integer  ? (BigInt(u) // BigInt(1)) :
                               rationalize(QQ, float(u))

    Ap = Matrix{QQ}(undef, m, n)
    bp = Vector{QQ}(undef, m)
    
    for i in 1:m, j in 1:n
        Ap[i,j] = toQQ(A[i,j])
    end
    for i in 1:m
        bp[i] = toQQ(b[i])
    end

    hrep = Polyhedra.hrep(Ap, bp)
    P = Polyhedra.polyhedron(hrep, _CDD)
    return HPoly(n, P)
end

"Convenience: single-part PolyUnion."
poly_union(h::HPoly) = PolyUnion(h.n, [h])

# ---------------------------- Membership checks -------------------------------

"Exact membership: test A*x <= b with QQ arithmetic."
function _in_hpoly(h::HPoly, x::AbstractVector)
    HAVE_POLY || error("Polyhedra/CDDLib not available.")
    length(x) == h.n || error("dimension mismatch in _in_hpoly")

    # Fetch exact H-rep and compare in QQ.
    H  = Polyhedra.hrep(h.poly)
    Ab = Polyhedra.hrep_matrix(H)

    # Robust QQ conversion (shared with _facets_of).
    toQQ(u) = u isa QQ       ? u :
              u isa Rational ? (BigInt(numerator(u)) // BigInt(denominator(u))) :
              u isa Integer  ? (BigInt(u) // BigInt(1)) :
                               rationalize(QQ, float(u))

    Aqq = Matrix{QQ}(undef, size(Ab.A,1), size(Ab.A,2))
    bqq = Vector{QQ}(undef, length(Ab.b))
    for i in 1:size(Ab.A,1), j in 1:size(Ab.A,2)
        Aqq[i,j] = toQQ(Ab.A[i,j])
    end
    for i in 1:length(Ab.b)
        bqq[i] = toQQ(Ab.b[i])
    end
    xqq = [toQQ(xi) for xi in x]
    for i in 1:size(Aqq,1)
        s = zero(QQ)
        @inbounds for j in 1:h.n
            s += Aqq[i,j] * xqq[j]
        end
        if s > bqq[i]
            return false
        end
    end
    return true
end


"Point membership in a union of polytopes."
function _in_union(U::PolyUnion, x::AbstractVector)
    for p in U.parts
        if _in_hpoly(p, x); return true; end
    end
    return false
end

"in(Upset, x) and in(Downset, x) predicates."
contains(U::PLUpset, x::AbstractVector)  = _in_union(U.U, x)
contains(D::PLDownset, x::AbstractVector) = _in_union(D.D, x)

# ---------------------- Feasibility of Y-membership patterns -------------------
# Y := [U_1,...,U_m] U { complement(D_1),..., complement(D_r) }.
# A Y-signature is a bit-vector (y,z) with:
#   y[i] = 1  means  x in U_i
#   z[j] = 1  means  x outside D_j
#
# We realize feasibility as follows:
#   - Start with a list of "in" constraints from all U_i with y[i]=1; each adds
#     an HPoly intersection.
#   - For each "outside D_j" (z[j]=1), we *branch by facets*: pick at least one
#     facet of at least one HPoly in the union D_j to violate. This yields a
#     set of halfspaces "a*x >= b + eps" in spirit. We emulate by flipping the
#     inequality to a*x >= b and add a tiny positive slack in Float64 space,
#     which is enough for separation inside an exact-feasible intersection,
#     because we confirm non-emptiness with Polyhedra.isempty() on the final
#     H-rep.
#   - For each "inside D_j" (z[j]=0), we must include x in union(D_j), so we
#     branch over the union parts as disjunction. Each branch adds the part
#     as "in" constraints (intersection with that part).
#
# The overall search is a tree of conjunctions with disjunction branching.
# We prune infeasible nodes early using Polyhedra.isempty. We also record one
# interior point (witness) from Polyhedra for each feasible leaf. We cap the
# number of feasible leaves by 'max_regions' for complexity safeguards.

"""
_internal_build_poly(in_parts, out_parts) -> (P::Polyhedra.Polyhedron, witness::Vector{Float64} or nothing, empty::Bool)

Intersect all "in_parts" HPolys and the (weak) complements requested in out_parts,
where each out-part is represented by a pair (a, b) recording the half-space
      a' * x >= b
(used to approximate strict violation).  The element type
of the coefficients can be any <:Real (e.g. Float64 or QQ).
"""
function _internal_build_poly(in_parts::Vector{HPoly},
                              out_halfspaces::Vector{Tuple{Vector{T},T}};   # a' * x >= b
                              strict_eps::QQ=STRICT_EPS_QQ) where {T<:Real}
    HAVE_POLY || error("Polyhedra/CDDLib not available.")

    # Build base as intersection of all "in" polytopes
    # We accumulate H-rep: A*x <= b, then append flipped out-constraints as -a*x <= -b.
    # Accumulate an H-rep in exact rationals (QQ)
    Aqq = QQ[]    # flattened row-major A; we reshape later
    bqq = QQ[]

   # helper to append a poly's hrep, converting its (possibly Float64) data to QQ
    function add_poly_QQ!(Hp::HPoly)
        H = Polyhedra.hrep(Hp.poly)
        Ab = Polyhedra.hrep_matrix(H)
        A = Array(Ab.A)  # may be Float64 or QQ, so convert entrywise
        b = Array(Ab.b)
        for i in 1:size(A,1)
            for j in 1:size(A,2)
                push!(Aqq, QQ(A[i,j]))
            end
            push!(bqq, QQ(b[i]))
        end
    end

    for P in in_parts
        add_poly_QQ!(P)
    end

    # Strict complement facets: -a'*x <= -(b + eps) in QQ
    for (a,b0) in out_halfspaces
        for j in 1:length(a)
            push!(Aqq, -QQ(a[j]))
        end
        push!(bqq, -(QQ(b0) + strict_eps))   # exact slack here
    end

    m = length(bqq)
    n = (m == 0) ? (isempty(in_parts) ? 0 : in_parts[1].n) : (length(Aqq) / m)
    n == 0 && return (nothing, nothing, true)

    Am = reshape(Vector{QQ}(Aqq), m, n)
    bm = Vector{QQ}(bqq)

    H = Polyhedra.hrep(Am, bm)
    P = Polyhedra.polyhedron(H, _CDD)

    if Polyhedra.isempty(P)
        return (P, nothing, true)
    else
        # Get a witness interior point. We use a simple vertex or any point.
        # Polyhedra offers "chebyshev center" in some backends; pick first vertex as fallback.
        try
            V = Polyhedra.vrep(P)
            pts = Polyhedra.points(V)
            x = length(pts) > 0 ? Vector{Float64}(pts[1]) : nothing
            return (P, x, false)
        catch
            return (P, nothing, false)
        end
    end
end

"Collect outward-pointing facet inequalities a cdot x <= b for an HPoly, with QQ coefficients."
function _facets_of(hp::HPoly)::Vector{Tuple{Vector{QQ},QQ}}
    H  = Polyhedra.hrep(hp.poly)
    Ab = Polyhedra.hrep_matrix(H)
    A  = Array(Ab.A);  b = Array(Ab.b)
    # robust conversion to QQ regardless of underlying element types
    toQQ(x) = x isa QQ        ? x :
              x isa Rational  ? (BigInt(numerator(x)) // BigInt(denominator(x))) :
              x isa Integer   ? (BigInt(x) // BigInt(1)) :
                                 rationalize(QQ, float(x))
    m, n = size(A)
    out = Vector{Tuple{Vector{QQ},QQ}}(undef, m)
    @inbounds for i in 1:m
        ai = Vector{QQ}(undef, n)
        for j in 1:n
            ai[j] = toQQ(A[i,j])
        end
        out[i] = (ai, toQQ(b[i]))
    end
    return out
end

"""
enumerate_feasible_regions(Ups, Downs; max_regions)

Return a vector of records:
  (signature_y::BitVector, signature_z::BitVector, poly::HPoly, witness::Vector{Float64})
Each record represents a feasible Y-region with its H-rep and one interior point.

We branch:
  - for each z[j]==0 (inside D_j union), choose one union part to include
  - for each z[j]==1 (outside D_j union), choose at least one violating facet
We deduplicate identical (y,z) signatures at the end (region signatures).
"""
function enumerate_feasible_regions(Ups::Vector{PLUpset}, Downs::Vector{PLDownset};
                                    max_regions::Int=10_000,
                                    strict_eps::QQ=STRICT_EPS_QQ)
    HAVE_POLY || error("Polyhedra/CDDLib not available.")

    m = length(Ups)
    r = length(Downs)
    n = (m>0 ? Ups[1].U.n : (r>0 ? Downs[1].D.n : 0))

    results = Vector{Tuple{BitVector,BitVector,HPoly,Vector{Float64}}}()

    # Iterate over all y in {0,1}^m and z in {0,1}^r.
    # In practice this grows fast; rely on feasibility pruning and max_regions cap.
    function process_signature(y::BitVector, z::BitVector)
        # seed choice lists
        in_choices = Vector{Vector{HPoly}}()
        push!(in_choices, HPoly[])
        inside_choices = Vector{Vector{HPoly}}()
        push!(inside_choices, HPoly[])
        # collect required Upsets (y[i]==1) as conjunctions
        for i in 1:m
            if y[i] == 1
                new_choices = Vector{Vector{HPoly}}()
                for base in in_choices, part in Ups[i].U.parts
                    push!(new_choices, vcat(base, [part]))
                end
                in_choices = new_choices
            end
        end

        # For "inside D_j" (z[j]==0): union disjunction -> branch over parts
        inside_choices = Vector{Vector{HPoly}}()  # each element is a vector of parts chosen to include
        push!(inside_choices, HPoly[])  # seed with empty selection

        for j in 1:r
            if z[j] == 0
                # expand branches by picking exactly one part from D_j union
                new_choices = Vector{Vector{HPoly}}()
                if isempty(Downs[j].D.parts)
                    # degenerate; no part -> impossible to be inside; skip
                    return
                end
                for base in inside_choices
                    for part in Downs[j].D.parts
                        push!(new_choices, vcat(base, [part]))
                    end
                end
                inside_choices = new_choices
            end
        end

        # For "outside D_j" (z[j]==1): for each D_j we must violate at least one facet
        # of at least one union part. We collect all candidate facets and branch by
        # selecting at least one facet per D_j.  Coefficients are exact QQ.
        outside_facets_choices = Vector{Vector{Tuple{Vector{QQ},QQ}}}()
        push!(outside_facets_choices, Tuple{Vector{QQ},QQ}[])
        for j in 1:r
            if z[j] == 1
                # collect facets by part
                part_facets = [ _facets_of(part) for part in Downs[j].D.parts ]
                new_out = Vector{Vector{Tuple{Vector{QQ},QQ}}}()
                push!(outside_facets_choices, Tuple{Vector{QQ},QQ}[])
                # Cartesian product over "choose one facet per part"
                function product(lo::Int, acc)
                    if lo > length(part_facets)
                        push!(new_out, copy(acc)); return
                    end
                    for f in part_facets[lo]
                        push!(acc, f); product(lo+1, acc); pop!(acc)
                    end
                end
                product(1, Tuple{Vector{QQ},QQ}[])
                # combine with previous choices
                outside_facets_choices = [ vcat(base, choice) for base in outside_facets_choices, choice in new_out ]
            end
        end

        # Now for each inside-choice and outside-facets-choice, test feasibility.
        for chosen_up in in_choices, chosen_inside in inside_choices
            iparts = vcat(chosen_up, chosen_inside)
            for chosen_out in outside_facets_choices
                P, witness, isemp = _internal_build_poly(iparts, chosen_out; strict_eps=strict_eps)
                if !isemp
                    hp = HPoly(n, P)
                    push!(results, (copy(y), copy(z), hp, witness === nothing ? Float64[] : witness))
                    if length(results) >= max_regions
                        return
                    end
                end
            end
            if length(results) >= max_regions
                return
            end
        end
    end

    # Full enumeration with early quit on cap
    total = 1 << (m + r)
    for mask in 0:(total-1)
        # decode mask into y and z
        y = falses(m); z = falses(r)
        for i in 1:m
            y[i] = (mask >> (i-1)) & 1 == 1
        end
        for j in 1:r
            z[j] = (mask >> (m + j - 1)) & 1 == 1
        end
        process_signature(y,z)
        if length(results) >= max_regions
            @warn "enumerate_feasible_regions: reached max_regions cap; stopping early."
            break
        end
    end

    # Collapse equivalent region signatures (Def. 4.12-4.17) by (y,z) equality.
    # Keep the first feasible geometry as representative.
    seen = Dict{Tuple{BitVector,BitVector},Int}()
    collapsed = Vector{Tuple{BitVector,BitVector,HPoly,Vector{Float64}}}()
    for rec in results
        key = (rec[1], rec[2])
        if !haskey(seen, key)
            seen[key] = 1
            push!(collapsed, rec)
        end
    end
    return collapsed
end

# --------------------------- Encoding (P, H_hat, pi) --------------------------

"""
PLEncodingMap

A classifier implementing pi : R^n -> {1,...,P.n} by testing Y-membership
signatures. Each region stores:
  - its signature (y,z)
  - an H-representation (A*x <= b) via Polyhedra object
  - one interior representative point witness for fast approximate checks

The 'locate' method returns the index of the unique region whose H-rep
contains x (first match), or 0 if no region matches.
"""
struct PLEncodingMap
    n::Int
    # region data
    sig_y::Vector{BitVector}         # length = number of regions
    sig_z::Vector{BitVector}
    regions::Vector{HPoly}
    witnesses::Vector{Vector{Float64}}
end

"Locate region index of x by geometric membership."
function locate(pi::PLEncodingMap, x::AbstractVector)
    HAVE_POLY || error("Polyhedra/CDDLib not available.")
    length(x) == pi.n || error("locate: dimension mismatch")
    for (idx, hp) in enumerate(pi.regions)
        if _in_hpoly(hp, x); return idx; end
    end
    return 0
end

# Build P as in Defs. 4.12-4.17: order by inclusion of Y-membership sets.
function _uptight_from_signatures(sig_y::Vector{BitVector}, sig_z::Vector{BitVector})
    rN = length(sig_y)
    leq = falses(rN, rN)
    for i in 1:rN
        leq[i,i] = true
    end
    for i in 1:rN, j in 1:rN
        yi, zi = sig_y[i], sig_z[i]
        yj, zj = sig_y[j], sig_z[j]
        # inclusion test componentwise: yi <= yj and zi <= zj
        leq[i,j] = all(yi .<= yj) && all(zi .<= zj)
    end
    # transitive closure
    for k in 1:rN, i in 1:rN, j in 1:rN
        leq[i,j] = leq[i,j] || (leq[i,k] && leq[k,j])
    end
    return FiniteFringe.FinitePoset(leq)
end

"Image upsets/downsets on P from Y-signatures."
function _images_on_P(P::FiniteFringe.FinitePoset,
                      sig_y::Vector{BitVector}, sig_z::Vector{BitVector},
                      m::Int, r::Int)
    Uhat = Vector{FiniteFringe.Upset}(undef, m)
    Dhat = Vector{FiniteFringe.Downset}(undef, r)
    # Upsets: region t is in image(U_i) iff y_i(t) == 1
    for i in 1:m
        mask = BitVector([ sig_y[t][i] == 1 for t in 1:P.n ])
        Uhat[i] = FiniteFringe.upset_closure(P, mask)
    end
    # Downsets: region t is in image(D_j) iff z_j(t) == 0 (i.e., not outside)
    for j in 1:r
        mask = BitVector([ sig_z[t][j] == 0 for t in 1:P.n ])
        Dhat[j] = FiniteFringe.downset_closure(P, mask)
    end
    return Uhat, Dhat
end

"Enforce monomial condition phi_hat[j,i] = 0 if image(U_i) and image(D_j) do not intersect."
function _monomialize_phi(phi::AbstractMatrix{QQ}, Uhat, Dhat)
    m = length(Dhat); n = length(Uhat)
    Phi = copy(phi)
    for j in 1:m, i in 1:n
        # empty intersection on P means zero entry
        if !FiniteFringe.intersects(Uhat[i], Dhat[j])
            Phi[j,i] = zero(QQ)
        end
    end
    return Phi
end

"""
    encode_from_PL_fringe(Ups, Downs, Phi; max_regions=10000)

Build a finite uptight encoding from PL upsets/downsets and a scalar
monomial matrix Phi (rows = Downs, cols = Ups).

Returns (P::FinitePoset, H_hat::FringeModule{QQ}, pi::PLEncodingMap).
"""
# Miller linkage (Def. 2.15 and Thm. 4.22(3)):
# With PL upsets/downsets (finite unions of convex H-polytopes) and the polyhedral cone R^n_+,
# the uptight encoding constructed here is PL in the sense of Def. 2.15(2) and satisfies
# Thm. 4.22(3): PL constant regions -> PL encoding -> monomialized phi on P.
function encode_from_PL_fringe(Ups::Vector{PLUpset},
                               Downs::Vector{PLDownset},
                               Phi_in::AbstractMatrix{QQ};
                               max_regions::Int=10_000,
                               strict_eps::QQ=STRICT_EPS_QQ)

    # Sanity: inputs are PL sets; the construction then yields a PL (uptight) encoding
    # in the sense of Miller Def. 2.15(2) and Thm. 4.22(3).
    _assert_PL_inputs(Ups, Downs)

    HAVE_POLY || error("Polyhedra/CDDLib not available.")

    # 1) enumerate feasible region signatures with geometry
    feasible = enumerate_feasible_regions(Ups, Downs; max_regions=max_regions, strict_eps=strict_eps)
    if isempty(feasible)
        # degenerate: no feasible regions; return the 1-point poset with zero module
        P = FiniteFringe.FinitePoset(reshape(Bool[true], 1, 1))
        Uhat = FiniteFringe.Upset[FiniteFringe.upset_closure(P, BitVector([false])) for _ in 1:length(Ups)]
        Dhat = FiniteFringe.Downset[FiniteFringe.downset_closure(P, BitVector([false])) for _ in 1:length(Downs)]
        Phi0 = zeros(QQ, length(Downs), length(Ups))
        H = FiniteFringe.FringeModule{QQ}(P, Uhat, Dhat, sparse(Phi0))
        pi = PLEncodingMap( (length(Ups)>0 ? Ups[1].U.n : (length(Downs)>0 ? Downs[1].D.n : 0)),
                             BitVector[], BitVector[], HPoly[], Vector{Vector{Float64}}() )
        return P, H, pi
    end

    # split records
    sigy = Vector{BitVector}(undef, length(feasible))
    sigz = Vector{BitVector}(undef, length(feasible))
    regs = Vector{HPoly}(undef, length(feasible))
    wits = Vector{Vector{Float64}}(undef, length(feasible))
    for (k, rec) in enumerate(feasible)
        sigy[k] = rec[1]; sigz[k] = rec[2]; regs[k] = rec[3]; wits[k] = rec[4]
    end

    # 2) build uptight poset on region signatures
    P = _uptight_from_signatures(sigy, sigz)

    # 3) build images of Ups/Downs on P
    m = length(Ups)
    r = length(Downs)
    Uhat, Dhat = _images_on_P(P, sigy, sigz, m, r)

    # 4) enforce monomial condition and build FringeModule over P
    Phi = _monomialize_phi(Matrix{QQ}(Phi_in), Uhat, Dhat)
    H = FiniteFringe.FringeModule{QQ}(P, Uhat, Dhat, FiniteFringe.dense_to_sparse_K(Phi, QQ))

    # 5) return the classifier pi with stored region H-reps and witnesses
    pi = PLEncodingMap(regs[1].n, sigy, sigz, regs, wits)

    return P, H, pi
end

"Return (P, H, pi, :PL). Non-breaking convenience."
function encode_from_PL_fringe_with_tag(Ups, Downs, Phi_in; kwargs...)
    P, H, pi = encode_from_PL_fringe(Ups, Downs, Phi_in; kwargs...)
    return P, H, pi, :PL
end

# ------------------------------- Exports --------------------------------------
export HPoly, PolyUnion, PLUpset, PLDownset, PLEncodingMap, locate, make_hpoly,
       encode_from_PL_fringe

end # module
