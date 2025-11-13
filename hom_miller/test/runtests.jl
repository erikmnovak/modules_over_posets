using Test
using .PosetModules
using .IndicatorResolutions
using .PLBackend


const K = Rational{BigInt}

# A tiny finite poset: two incomparable chains 1<3, 2<4
leq = falses(4,4); for i in 1:4; leq[i,i] = true; end
leq[1,3] = true; leq[2,4] = true
Q = FinitePoset(leq)

# Build a small fringe module M on Q
U1 = upset_from_generators(Q, [1])      # {1,3}
U2 = upset_from_generators(Q, [2])      # {2,4}
D1 = downset_from_generators(Q, [3])    # {1,3}
D2 = downset_from_generators(Q, [4])    # {2,4}
φM = FiniteFringe.dense_to_sparse_K([1 0; 0 1], K)
M  = FringeModule{K}(Q, [U1,U2], [D1,D2], φM)

# 1) Build finite encoding via uptight poset (Φ uses U_i and complements of D_j)
enc = build_uptight_encoding_from_fringe(M)
π = enc.π
P = π.P
@test P.n ≥ 2  # nontrivial encoding

# 2) Build H on P by pushing labels forward and copying φ (Prop. 4.11 reversed)
Û = [image_upset(π, U) for U in M.U]
D̂ = [image_downset(π, D) for D in M.D]
Ĥ = FringeModule{K}(P, Û, D̂, φM)

# 3) Pull back Ĥ fringe; compare fibers with M
M′ = pullback_fringe_along_encoding(Ĥ, π)
for q in 1:Q.n
    @test fiber_dimension(M, q) == fiber_dimension(M′, q)
end

# 4) Build an upset presentation over P; check degreewise cokernel dimension equals Ĥ fiber
preŝ = build_upset_presentation_over_encoding(Ĥ)
for p in 1:P.n
    # dim coker δ_p = (#U0 containing p) - rank(δ_p)
    dim_F0p = count(U -> U.mask[p], preŝ.U0)
    rp = rank_at(preŝ, p)
    dim_cok = dim_F0p - rp
    @test dim_cok == fiber_dimension(Ĥ, p)
end

# 5) Pull back the upset presentation along π; (sanity) coker dimensions at q match M
presQ = pullback_upset_presentation(preŝ, π)
for q in 1:Q.n
    dim_F0q = count(U -> U.mask[q], presQ.U0)
    rq = rank_at(presQ, q)
    dim_cok = dim_F0q - rq
    @test dim_cok == fiber_dimension(M, q)
end


# A small chain poset 1<2<3
Q = PosetModules.Poset([1,2,3], [(1,2),(2,3)])

# A tiny module H on Q
dims = [1,2,1]
maps = Dict{Tuple{Int,Int}, Matrix{Rational{BigInt}}}()
maps[(1,2)] = [1//1, 0//1]'
maps[(2,3)] = [1//1 0//1]
H = PosetModules.PModule(Q, dims, maps)

# 1) Upset presentation and downset copresentation
F1, F0, d1, π0 = IndicatorResolutions.upset_presentation(H)
E0, E1, δ0, η0 = IndicatorResolutions.downset_copresentation(H)

# 2) Hom/Ext¹ via first page (self-Hom for sanity)
dimHom, dimExt1 = IndicatorResolutions.hom_ext_first_page(F1, F0, d1, E0, E1, δ0)
# Compare Hom with direct computation on PModules (via commuting squares)
dimHom_direct = PosetModules.hom(H, H)[1]   # Phase-0 function that returns (dim, ...)
@test dimHom == dimHom_direct
@test dimExt1 ≥ 0   # should be nonnegative

# 3) PL example: 2D box-fringe encoding
Ups = [PLBackend.BoxUpset([0.0, 0.0]), PLBackend.BoxUpset([1.0, -1.0])]
Downs = [PLBackend.BoxDownset([2.0, 2.0])]
φ = Rational{BigInt}[1//1  0//1] # 1×2 matrix
P, Henc = PLBackend.encode_fringe_boxes(Ups, Downs, φ)
# Projective resolution on the finite encoding is now available:
F, d, aug = IndicatorResolutions.upset_resolution(Henc; maxlen=2)
@test length(F) ≥ 1


@testset "Higher Ext via longer resolutions (finite poset)" begin
    # A tiny diamond poset: 1<3, 2<3
    leq = falses(3,3); for i in 1:3; leq[i,i]=true; end
    leq[1,3]=true; leq[2,3]=true
    P = FinitePoset(leq)

    # Simple module H on P via a fringe presentation
    U = [upset_from_generators(P,[1]), upset_from_generators(P,[2])]
    D = [downset_from_generators(P,[3])]
    φ = dense_to_sparse_K([1 1], K)
    H = FringeModule{K}(P, U, D, φ)

    # Upset and downset resolutions (length up to 2)
    F, δF, _ = IndicatorResolutions.upset_resolution(H; maxlen=2)
    E, ρE, _ = DownsetCopresentations.downset_resolution(H; maxlen=2)

    extdims = HomExt.ext_dims_via_resolutions(F, δF, E, ρE)
    @test extdims[1] == hom_dimension(H,H)     # Hom via Tot^0 equals commuting-squares result
    @test extdims[2] ≥ 0                       # Ext¹ is nonnegative
end

@testset "PL Polyhedra backend (uptight encoding)" begin
    # 2D example with two PL upsets (upper half-planes) and one downset (left half-plane)
    using Polyhedra, CDDLib
    A1 = Polyhedra.polyhedron(Polyhedra.HalfSpace([0.0,1.0], 0.0), CDDLib.Library(:exact))  # y ≥ 0
    A2 = Polyhedra.polyhedron(Polyhedra.HalfSpace([1.0,0.0], -1.0), CDDLib.Library(:exact)) # x ≥ 1
    L  = Polyhedra.polyhedron(Polyhedra.HalfSpace([-1.0,0.0], 0.0), CDDLib.Library(:exact)) # x ≤ 0

    U1 = PLUpset(PolyUnion([HPoly(A1)]))
    U2 = PLUpset(PolyUnion([HPoly(A2)]))
    D1 = PLDownset(PolyUnion([HPoly(L)]))
    Φ  = K[1 0; 0 1]     # 1×2 or 2×1 — any small monomial matrix

    P̂, Ĥ, π = PLPolyhedra.encode_from_PL_fringe([U1,U2], [D1], Φ)
    @test P̂.n > 0
    F, δF, _ = IndicatorResolutions.upset_resolution(Ĥ; maxlen=1)
    E, ρE, _ = DownsetCopresentations.downset_resolution(Ĥ; maxlen=1)
    extdims = HomExt.ext_dims_via_resolutions(F, δF, E, ρE)
    @test extdims[1] ≥ 0
end

@testset "Serialization: flange Zn" begin
    using Serialization, ZnFlange, ExactQQ
    τ = ZnFlange.ZnFace(falses(2))
    U = ZnFlange.ZnFlat([0,0], τ)
    E = ZnFlange.ZnInjective([1,1], τ)
    FG = ZnFlange.FlangePresentation([U],[E], ExactQQ.QQ[1//1])
    path = tempname()*".json"
    save_flange_json(path, FG)
    FG2 = load_flange_json(path)
    @test length(FG2.flats)==1 && length(FG2.injectives)==1
    @test FG2.ϕ[1,1] == 1//1
end

@testset "Serialization: encoding (P,H)" begin
    using Serialization, PosetModules, ExactQQ
    P = PosetModules.FinitePoset(collect(1:3), [(1,2),(2,3)])
    dims = [1,2,1]
    maps = Dict{Tuple{Int,Int}, Matrix{ExactQQ.QQ}}()
    maps[(1,2)] = [1//1 0//1; 0//1 1//1]
    maps[(2,3)] = [1//1 1//1]
    H = PosetModules.PModule(P, dims, maps)
    path = tempname()*".json"
    save_encoding_json(path, P, H)
    P2, H2 = load_encoding_json(path)
    @test length(P2.elts) == 3
    @test H2.dims == dims
end