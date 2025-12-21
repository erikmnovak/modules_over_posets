using Test

# Load the library either as an installed package or via local include.
# This makes "julia --project tests/runtests.jl" work even without Pkg setup.
try
    using PosetModules
catch
    include(joinpath(@__DIR__, "..", "src", "PosetModules.jl"))
    using .PosetModules
end

const PM  = PosetModules
const FF  = PM.FiniteFringe
const EN  = PM.Encoding
const HE  = PM.HomExt
const IR  = PM.IndicatorResolutions
const FZ  = PM.FlangeZn
const SER = PM.Serialization
const BR  = PM.M2SingularBridge
const CV  = PM.CrossValidateFlangePL
const PLP = PM.PLPolyhedra
const EX  = PM.ExactQQ
const QQ  = PM.CoreModules.QQ

using SparseArrays

# ---------------- Helpers used by multiple test files -------------------------

"Chain poset on {1,...,n} with i <= j iff i <= j as integers."
function chain_poset(n::Int)
    leq = falses(n, n)
    for i in 1:n
        for j in i:n
            leq[i, j] = true
        end
    end
    return FF.FinitePoset(leq)
end

"Disjoint union of two chains: 1 < 2 and 3 < 4, no relations across."
function disjoint_two_chains_poset()
    leq = falses(4, 4)
    for i in 1:4
        leq[i, i] = true
    end
    leq[1, 2] = true
    leq[3, 4] = true
    return FF.FinitePoset(leq)
end


"""
Diamond poset on {1,2,3,4} with relations

    1 < 2 < 4
    1 < 3 < 4

and with 2 incomparable to 3.

This is the smallest non-chain poset where "two different length-2 paths"
exist (1->2->4 and 1->3->4), which is exactly the situation where indicator
resolutions can have length > 1 and Ext^2 can be nonzero.
"""
function diamond_poset()
    leq = falses(4, 4)
    for i in 1:4
        leq[i, i] = true
    end
    leq[1, 2] = true
    leq[1, 3] = true
    leq[2, 4] = true
    leq[3, 4] = true
    leq[1, 4] = true  # transitive closure needed explicitly for FinitePoset
    return FF.FinitePoset(leq)
end



"Convenience: 1x1 fringe module with scalar on the unique entry."
function one_by_one_fringe(P::FF.FinitePoset, U::FF.Upset, D::FF.Downset; scalar=QQ(1))
    phi = spzeros(QQ, 1, 1)
    phi[1, 1] = scalar
    return FF.FringeModule{QQ}(P, [U], [D], phi)
end

"Simple modules on the chain 1 < 2: S1 supported at 1, S2 supported at 2."
function simple_modules_chain2()
    P = chain_poset(2)
    S1 = one_by_one_fringe(P, FF.principal_upset(P, 1), FF.principal_downset(P, 1))
    S2 = one_by_one_fringe(P, FF.principal_upset(P, 2), FF.principal_downset(P, 2))
    return P, S1, S2
end

# ---------------- Run test files ---------------------------------------------

include("test_ascii_only.jl")
include("test_exactqq.jl")
include("test_finite_fringe.jl")
include("test_homext_pi0.jl")
include("test_encoding.jl")
include("test_indicator_resolutions.jl")
include("test_indicator_resolutions_diamond.jl")
include("test_flange_zn.jl")
include("test_serialization.jl")
include("test_cross_validate.jl")
include("test_plpolyhedra_optional.jl")
include("test_derived_functors.jl")
include("test_theory_by_hand.jl")
include("test_plbackend_axis.jl")
include("test_tor_by_hand.jl")
include("test_chain_complexes_homology.jl")
include("test_random_stress.jl")
