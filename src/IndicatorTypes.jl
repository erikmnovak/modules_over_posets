module IndicatorTypes
# Small shared types for indicator presentations/copresentations.
# These mirror the data in Def. 6.4 (Miller) and are used by Hom/Ext assembly.

using SparseArrays
using ..FiniteFringe: FinitePoset, Upset, Downset  # P, labeling sets

"""
    UpsetPresentation{K}

A one-step **upset presentation** (Def. 6.4): F1 --delta--> F0 \to M with labels by upsets.
Fields:
- `P`: underlying finite poset
- `U0`: column labels for F0 (birth upsets)
- `U1`: row labels for F1
- 'delta': block monomial matrix delta: (#U1) \times (#U0), entries in `K`
"""
struct UpsetPresentation{K}
    P::FinitePoset
    U0::Vector{Upset}
    U1::Vector{Upset}
    delta::SparseMatrixCSC{K,Int}
end

"""
    DownsetCopresentation{K}

A one-step **downset copresentation** (Def. 6.4):  M = ker(\rho : E^0 \to E^1) with labels by downsets.
Fields:
- `P`: underlying finite poset
- `D0`: row labels for E^0 (death downsets)
- `D1`: column labels for E^1
- `rho`: block monomial matrix rho: (#D1) \times (#D0), entries in `K`
"""
struct DownsetCopresentation{K}
    P::FinitePoset
    D0::Vector{Downset}
    D1::Vector{Downset}
    rho::SparseMatrixCSC{K,Int}
end

export UpsetPresentation, DownsetCopresentation
end # module
