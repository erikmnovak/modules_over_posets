module CrossValidateFlangePL
# ------------------------------------------------------------------------------
# Cross-validate a Z^n flange against an axis-aligned PL evaluation on the same
# lattice points in a convex-projection box [a,b].  No external Axis* module.
# ------------------------------------------------------------------------------

using LinearAlgebra
import ..FlangeZn: Flange, bounding_box, dim_at

# Minimal axis-aligned shapes for cross-check (not exported elsewhere).
struct AxisUpset{T}
    a::Vector{T}  # lower thresholds; +\infty on free coords is emulated with a large negative sentinel
end
struct AxisDownset{T}
    b::Vector{T}  # upper thresholds; -\infty on free coords is emulated with a large positive sentinel
end
struct AxisFringe{TU,TD,TF}
    n::Int
    births::Vector{AxisUpset{TU}}
    deaths::Vector{AxisDownset{TD}}
    Phi::Matrix{TF}  # same scalar "monomial" matrix
end

"Translate Zn indecomposables to axis-aligned PL up/down sets for cross-checking."
function flange_to_axis(fr::Flange{K}) where {K}
    n = fr.n
    BIG = 10_000.0
    births = AxisUpset{Float64}[]
    for F in fr.flats
        a = Vector{Float64}(undef, n)
        for i in 1:n
            a[i] = F.tau.coords[i] ? -BIG : float(F.b[i])
        end
        push!(births, AxisUpset(a))
    end
    deaths = AxisDownset{Float64}[]
    for E in fr.injectives
        b = Vector{Float64}(undef, n)
        for i in 1:n
            b[i] = E.tau.coords[i] ? +BIG : float(E.b[i])
        end
        push!(deaths, AxisDownset(b))
    end
    Phi = Array{Float64}(fr.Phi)
    return AxisFringe{Float64,Float64,Float64}(n, births, deaths, Phi)
end

"Test membership of a lattice point x in an axis-aligned upset or downset."
_contains(U::AxisUpset, x::Vector{Float64})  = all(x[i] >= U.a[i] for i in 1:length(x))
_contains(D::AxisDownset, x::Vector{Float64})= all(x[i] <= D.b[i] for i in 1:length(x))

"""
    cross_validate(fr::Flange; margin=1, rankfun=rank)

1. Build the convex-projection box [a,b] (heuristic 'bounding_box').
2. Evaluate 'dim M_g' for all integer 'g \in [a,b]' via the flange.
3. Build an axis-aligned proxy and evaluate the same lattice points.
4. Compare; return '(all_equal?, report::Dict)'.
"""
function cross_validate(fr::Flange; margin=1, rankfun=rank)
    a, b = bounding_box(fr; margin)
    ranges = (a[i]:b[i] for i in 1:fr.n)
    pts = collect(Iterators.product(ranges...))

    # Flange evaluation (exact)
    dims_Z = Dict{NTuple{Int,Int},Int}()
    for t in pts
        g = collect(Int.(t))
        dims_Z[Tuple(g...)] = dim_at(fr, g; rankfun=rankfun)
    end

    # Axis-aligned proxy
    afr = flange_to_axis(fr)
    dims_PL = Dict{NTuple{Int,Int},Int}()
    for t in pts
        x = Float64.(collect(Int.(t)))
        rows = [ _contains(d, x) for d in afr.deaths ]
        cols = [ _contains(u, x) for u in afr.births ]
        idxr = findall(identity, rows)
        idxc = findall(identity, cols)
        if isempty(idxr) || isempty(idxc)
            dims_PL[Tuple(Int.(t)...)] = 0
        else
            Phi_x = afr.Phi[idxr, idxc]
            dims_PL[Tuple(Int.(t)...)] = rank(Matrix(Phi_x))
        end
    end

    # Compare
    mism = Dict{NTuple{Int,Int},Tuple{Int,Int}}()
    for (k,v) in dims_Z
        if v != dims_PL[k]; mism[k] = (v, dims_PL[k]); end
    end
    ok = isempty(mism)
    report = Dict("box" => (a,b), "mismatches" => mism,
                  "tested" => length(pts), "agree" => length(pts) - length(mism))
    return ok, report
end

export cross_validate, flange_to_axis
end # module
