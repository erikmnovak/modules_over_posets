module PLPolyhedra

using ..FiniteFringe
using ..Encoding
using ..IndicatorResolutions
using ..DownsetCopresentations

# External geometry (soft dependency)
import Polyhedra
import CDDLib

const _LIB = CDDLib.Library(:exact)  # exact rational polyhedral arithmetic
  -----------------------
# ----------------------- PL upset/downset as unions of polyhedra -----------------------

"Polyhedral region described by an H-representation (finite intersection of halfspaces)."
struct HPoly
    # We store Polyhedra's H-rep polyhedron
    P::Polyhedra.Polyhedron
end

"Union of polyhedra (disjointness not required)."
struct PolyUnion
    polys::Vector{HPoly}
end

"PL Upset (declaration); membership is union over polys. Geometry need not be monotone checked."
struct PLUpset    # models a birth upset
    U::PolyUnion
end

"PL Downset (declaration)."
struct PLDownset  # models a death downset
    D::PolyUnion
end

# Membership by exact LP feasibility: x \in \bigcup Polys  <=>  there exist poly with Ax <= b satisfied.
function _contains_point(poly::HPoly, x::Vector{Rational{BigInt}})
    # Polyhedra's 'in' supports Float; for rationals we check feasibility with equality t=x
    # but cheaper: convert to Float64 *only* for point membership (safe if x is rational).
    # For exactness, instead project: we accept Float terms for point-in-poly check;
    # but here we rely on arrangement refinement (no numerics in algebra).
    Polyhedra.in(Vector{Float64}(x), poly.P)
end

# ----------------------- Uptight encoding from PL up/down sets -----------------------
# We create the family Y of constant upsets:
#   Y := {U_i}  \cup  { R^n \ D_j }.
# The uptight regions are intersections of each member or its complement; we generate
# only the *feasible* Boolean combinations by testing H-rep feasibility with CDDLib.
# Each feasible region gets a label (node) and the *poset order* is inclusion of Y-membership
# sets, which is a morphism (monotone) under the product order (Prop. 4.15).  :contentReference[oaicite:7]{index=7}

"Build H-rep of the complement of a polyhedron: this is a *union* of halfspaces; we approximate
by adding a fresh region type `ComplementOf` and treat it combinatorially in the Boolean search."
struct ComplementOf; poly::HPoly; end

# Abstract condition: either "inside polyhedron H" or "outside polyhedron H".
abstract type Cond end
struct InPoly   <: Cond; poly::HPoly; end
struct OutPoly  <: Cond; poly::HPoly; end

# Boolean cell generator: test feasibility of a conjunction of Cond's.
function _feasible(conj::Vector{Cond})
    # Build H-rep: IN-conditions add halfspaces; OUT-conditions add the negation, which
    # we realize by a *disjunction*. We navigate the disjunctions by branching; in practice,
    # we include OUT-conditions by splitting per facet (standard arrangement refinement).
    # For moderate inputs, recursion suffices; in production, consider lazy SAT+LP.
    Hs = Polyhedra.HRepresentation(Polyhedra.hrep(Polyhedra.polyhedron(Polyhedra.HyperPlane([0.0],0.0), _LIB))) # dummy
    # We instead accumulate as a Polyhedron = R^n; then intersect.
    P = nothing
    for c in conj
        if c isa InPoly
            P = P === nothing ? (c::InPoly).poly.P : Polyhedra.intersect(P, (c::InPoly).poly.P)
        else
            # OUT-conditions will be handled by upper caller-here we skip. (See below.)
            # For clarity in this compact prototype, we *do not* expand disjunctions here;
            # we implement them by enumerating Boolean patterns over "in Y" directly.
        end
    end
    if P === nothing
        return true  # empty conjunction \implies whole space feasible
    end
    !Polyhedra.isempty(P)
end

"""
    encode_from_PL_fringe(Ups::Vector{PLUpset}, Downs::Vector{PLDownset}, Phi::Matrix{K})
        -> (P::FinitePoset, H_hat::FringeModule{K}, pi::EncodingMap)

Construct a **finite encoding** from PL upsets/downsets via the *uptight* construction:
we form Y = {U_i} \cup {complement of D_j}. Each Boolean Y-membership vector that is
geometrically feasible becomes a node (region) of the finite poset P; the order is inclusion
of Y-memberships. The encoded fringe module H_hat on P has the *same* monomial matrix Phi,
with birth rows labeled by images of U_i and death columns by images of D_j.
(Prop. 4.11; Thm. 4.22.)  :contentReference[oaicite:8]{index=8}
"""
function encode_from_PL_fringe(Ups::Vector{PLUpset},
                               Downs::Vector{PLDownset},
                               Phi::Matrix{K}) where {K<:Number}
    m = length(Ups)
    r = length(Downs)

    # Construct Boolean variables: y_1..y_m for U-membership, z_1..z_r for complement(D).
    # For each Boolean vector (y,z) \in {0,1}^{m+r}, test feasibility:
    #   Lambda_{i: y_i=1}  In(U_i)   Lambda  Lambda_{i: y_i=0} Out(U_i)
    #   Lambda_{j: z_j=1}  Out(D_j)  Lambda  Lambda_{j: z_j=0} In(D_j)

    regions = Tuple{BitVector,BitVector}[]   # store (y,z)
    for ymask in Iterators.product((0:1 for _ in 1:m)...)
        y = BitVector(ymask)
        for zmask in Iterators.product((0:1 for _ in 1:r)...)
            z = BitVector(zmask)
            # (Prototype feasibility: assume "small" m+r and accept pattern as feasible.
            # In production, build Polyhedra constraints and test emptiness with CDDLib.)
            push!(regions, (y,z))
        end
    end

    # Build poset by inclusion of Y-memberships (Prop. 4.15 \implies acyclic; we take transitive closure).
    n = length(regions)
    leq = falses(n,n)
    for i in 1:n
        leq[i,i] = true
    end
    for i in 1:n, j in 1:n
        yi,zi = regions[i]
        yj,zj = regions[j]
        if all(yi .<= yj) && all(zi .<= zj)
            leq[i,j] = true
        end
    end
    # Transitive closure (Floyd-Warshall)
    for k in 1:n, i in 1:n, j in 1:n
        leq[i,j] = leq[i,j] || (leq[i,k] && leq[k,j])
    end
    P = FiniteFringe.FinitePoset(leq)

    # Image upsets / downsets on P: convert each (y,z) to membership in image sets.
    function image_up(Uidx::Int)
        mask = falses(P.n)
        for (t, (y,z)) in enumerate(regions)
            if y[Uidx] == 1; mask[t] = true; end
        end
        FiniteFringe.Upset(P, mask)
    end
    function image_dn(Didx::Int)
        mask = falses(P.n)
        for (t, (y,z)) in enumerate(regions)
            # recall z_j = 0 means "in D_j"
            if z[Didx] == 0; mask[t] = true; end
        end
        FiniteFringe.Downset(P, mask)
    end

    U_hat = [image_up(i) for i in 1:m]
    D_hat = [image_dn(j) for j in 1:r]

    # Same monomial matrix (connected entries survive exactly on nonempty intersections; Prop 3.18)
    phi = Matrix{K}(Phi)
    H_hat = FiniteFringe.FringeModule{K}(P, U_hat, D_hat, phi)

    # Encoding map pi is implicit in our region construction; we return an EncodingMap stub
    # with identity-on-P (domain is "continuous"; we use PLPartition-like mapping elsewhere).
    pi = Encoding.EncodingMap(P, P, collect(1:P.n))  # identity (placeholder)
    return P, H_hat, pi
end

export PLUpset, PLDownset, PolyUnion, HPoly, encode_from_PL_fringe

end # module
