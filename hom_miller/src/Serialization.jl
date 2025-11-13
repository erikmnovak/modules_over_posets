module Serialization
# ==============================================================================
# Serialization for:
#   - FlangeZn.Flange{QQ}          (Zn flange presentations)
#   - FiniteFringe.(FinitePoset, FringeModule{QQ})  (finite encodings)
#
# Design choices:
#   - ASCII variable names in this file (phi, tau, etc.)
#   - When touching FlangeZn types, we *read* Greek-named fields (tau, phi) but
#     keep local variables ASCII to maximize readability.
#   - All scalars are exact rationals QQ serialized as "num/den".
# ==============================================================================

using JSON3
import ..FlangeZn: Face, IndFlat, IndInj, Flange          # actual types in repo
import ..FiniteFringe: FinitePoset, Upset, Downset, FringeModule, dense_to_sparse_K
using ..CoreModules: QQ, rational_to_string, string_to_rational

export save_flange_json, load_flange_json,
       save_encoding_json, load_encoding_json

# ------------------------------------------------------------------------------
# (1) FlangeZn.Flange{QQ}  <->  JSON
# Schema:
# {
#   "kind": "FlangeZn",
#   "n": 2,
#   "flats":      [ {"b":[...], "tau":[i1,i2,...]}, ... ],
#   "injectives": [ {"b":[...], "tau":[...]} , ... ],
#   "phi": [[ "num/den", ...], ...]   # rows = #injectives, cols = #flats
# }
# ------------------------------------------------------------------------------

"""
    save_flange_json(path::AbstractString, fr::Flange{QQ}) -> String

Write a Z^n flange presentation to JSON.  See the schema above.
"""
function save_flange_json(path::AbstractString, fr::Flange{QQ})
    n = fr.n
    # Note: in FlangeZn, the face is stored as tau::Face with a BitVector of coords.
    flats = [Dict("b" => F.b,
                  "tau" => findall(identity, F.tau.coords)) for F in fr.flats]
    injectives = [Dict("b" => E.b,
                       "tau" => findall(identity, E.tau.coords)) for E in fr.injectives]
    # Phi is the scalar matrix (rows=#injectives, cols=#flats)
    phi_mat = [[rational_to_string(fr.Phi[i,j]) for j in 1:length(fr.flats)]
                                        for i in 1:length(fr.injectives)]
    obj = Dict(
        "kind" => "FlangeZn",
        "n" => n,
        "flats" => flats,
        "injectives" => injectives,
        "phi" => phi_mat
    )
    open(path, "w") do io
        JSON3.write(io, obj; allow_inf=true, indent=2)
    end
    return path
end

"""
    load_flange_json(path::AbstractString) -> Flange{QQ}

Inverse of `save_flange_json`.
"""
function load_flange_json(path::AbstractString)::Flange{QQ}
    obj = open(JSON3.read, path)
    @assert haskey(obj, "kind") && String(obj["kind"]) == "FlangeZn"
    n = Int(obj["n"])

    # helper: face from an index list
    function _mkface(n::Int, idxs)
        bits = falses(n)
        for t in idxs
            bits[Int(t)] = true
        end
        Face(n, bits)
    end

    flats = [IndFlat{QQ}(Vector{Int}(f["b"]),
                         _mkface(n, Vector{Int}(f["tau"])),
                         Symbol(get(f, "id", "F$(i)")))
             for (i, f) in enumerate(obj["flats"])]

    injectives = [IndInj{QQ}(Vector{Int}(e["b"]),
                             _mkface(n, Vector{Int}(e["tau"])),
                             Symbol(get(e, "id", "E$(i)")))
                  for (i, e) in enumerate(obj["injectives"])]

    m = length(injectives); k = length(flats)
    Phi = Matrix{QQ}(undef, m, k)
    for i in 1:m, j in 1:k
        Phi[i,j] = string_to_rational(String(obj["phi"][i][j]))
    end
    return Flange{QQ}(n, flats, injectives, Phi)
end

# ------------------------------------------------------------------------------
# (2) Finite encodings (FinitePoset, FringeModule{QQ})  <->  JSON
#
# Schema:
# {
#   "kind":"FiniteEncoding",
#   "poset":{
#      "n": N,
#      "leq": [ [true,false,...], [...], ... ]   # NxN boolean matrix
#   },
#   "fringe":{
#      "U":   [ [i1,i2,...], ... ],              # membership indices for each upset
#      "D":   [ [j1,j2,...], ... ],              # membership indices for each downset
#      "phi": [ ["num/den",...], ... ]           # rows=#D, cols=#U
#   }
# }
#
# This matches the finite-encoding viewpoint (Def. 4.1) with a monomial matrix
# (Defs. 3.16-3.17). See also Example 4.5 on convex projection to finite boxes.      [Miller]
# ------------------------------------------------------------------------------

"""
    save_encoding_json(path, P::FinitePoset, H::FringeModule{QQ}) -> String

Serialize a finite encoding (poset + fringe module) to JSON.
"""
function save_encoding_json(path::AbstractString,
                            P::FinitePoset,
                            H::FringeModule{QQ})
    # poset order as an NxN boolean matrix
    leq_rows = [ [ P.leq[i,j] for j in 1:P.n ] for i in 1:P.n ]

    # upsets/downsets as membership index lists (1-based)
    U_lists = [ findall(identity, U.mask) for U in H.U ]
    D_lists = [ findall(identity, D.mask) for D in H.D ]

    # scalar matrix phi is SparseCSC in H; serialize dense as strings
    m, n = size(H.phi)
    phi_str = [ [ rational_to_string(QQ(H.phi[i,j])) for j in 1:n ] for i in 1:m ]

    obj = Dict(
      "kind"  => "FiniteEncoding",
      "poset" => Dict("n" => P.n, "leq" => leq_rows),
      "fringe"=> Dict("U" => U_lists, "D" => D_lists, "phi" => phi_str)
    )
    open(path, "w") do io
        JSON3.write(io, obj; allow_inf=true, indent=2)
    end
    return path
end

"""
    load_encoding_json(path) -> (P::FinitePoset, H::FringeModule{QQ})

Inverse of `save_encoding_json`.
"""
function load_encoding_json(path::AbstractString)
    obj = open(JSON3.read, path)
    @assert haskey(obj, "kind") && String(obj["kind"]) == "FiniteEncoding"

    # Poset
    N = Int(obj["poset"]["n"])
    leq = falses(N, N)
    for i in 1:N, j in 1:N
        leq[i,j] = Bool(obj["poset"]["leq"][i][j])
    end
    P = FinitePoset(leq)

    # Upsets/Downsets (mask from index lists)
    function _mask_from_indices(idxs::Vector{Int})
        m = falses(P.n)
        for t in idxs; m[t] = true; end
        m
    end
    Ups = [Upset(P, _mask_from_indices(Vector{Int}(u))) for u in obj["fringe"]["U"]]
    Dns = [Downset(P, _mask_from_indices(Vector{Int}(d))) for d in obj["fringe"]["D"]]

    # phi matrix
    rows = length(obj["fringe"]["phi"])
    cols = length(obj["fringe"]["phi"][1])
    phi_dense = Matrix{QQ}(undef, rows, cols)
    for i in 1:rows, j in 1:cols
        phi_dense[i,j] = string_to_rational(String(obj["fringe"]["phi"][i][j]))
    end
    phi_sparse = dense_to_sparse_K(phi_dense, QQ)

    H = FringeModule{QQ}(P, Ups, Dns, phi_sparse)
    return P, H
end

end # module
