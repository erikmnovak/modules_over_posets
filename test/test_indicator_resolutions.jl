using Test

@testset "IndicatorResolutions internal invariants + Ext on A2" begin
    # Internal PModule should match fiber_dimension from the fringe.
    P = chain_poset(3)
    M = one_by_one_fringe(P, FF.principal_upset(P, 2), FF.principal_downset(P, 2))
    PMM = IR.pmodule_from_fringe(M)
    for q in 1:P.n
        @test PMM.dims[q] == FF.fiber_dimension(M, q)
    end

    # Projective cover should be surjective on each vertex (full row rank).
    F0, pi0, _ = IR.projective_cover(PMM)
    for q in 1:P.n
        @test EX.rankQQ(pi0.comps[q]) == PMM.dims[q]
    end

    # Kernel inclusion iota: K -> F0 should be injective and satisfy pi0 * iota = 0.
    K, iota = IR.kernel_with_inclusion(pi0)
    for q in 1:P.n
        @test EX.rankQQ(iota.comps[q]) == K.dims[q]
        Z = pi0.comps[q] * iota.comps[q]
        @test Z == zeros(QQ, size(Z,1), size(Z,2))
    end

    # Now test Ext dimensions on the A2 chain: 1 < 2
    _, S1, S2 = simple_modules_chain2()

    ext12 = IR.ext_dimensions_via_indicator_resolutions(S1, S2; maxlen=3)
    ext21 = IR.ext_dimensions_via_indicator_resolutions(S2, S1; maxlen=3)
    ext11 = IR.ext_dimensions_via_indicator_resolutions(S1, S1; maxlen=3)
    ext22 = IR.ext_dimensions_via_indicator_resolutions(S2, S2; maxlen=3)

    # Known quiver A2 facts:
    # Hom(S1,S2)=0, Ext^1(S1,S2)=1
    # Hom(S2,S1)=0, Ext^1(S2,S1)=0
    # Endomorphisms: Hom(Si,Si)=1, Ext^1(Si,Si)=0
    @test get(ext12, 0, 0) == 0
    @test get(ext12, 1, 0) == 1

    @test get(ext21, 0, 0) == 0
    @test get(ext21, 1, 0) == 0

    @test get(ext11, 0, 0) == 1
    @test get(ext11, 1, 0) == 0

    @test get(ext22, 0, 0) == 1
    @test get(ext22, 1, 0) == 0

    # Ext^0 should agree with FiniteFringe.hom_dimension on these simple cases.
    @test get(ext12, 0, 0) == FF.hom_dimension(S1, S2)
    @test get(ext21, 0, 0) == FF.hom_dimension(S2, S1)
    @test get(ext11, 0, 0) == FF.hom_dimension(S1, S1)
    @test get(ext22, 0, 0) == FF.hom_dimension(S2, S2)
end

@testset "Cover-edge maps are label-consistent on non-chain posets" begin
    # Poset with relations: 1<3<4 and 2<4 (2 incomparable with 3)
    leq = falses(4,4)
    for i in 1:4
        leq[i,i] = true
    end
    leq[1,3] = true
    leq[3,4] = true
    leq[1,4] = true
    leq[2,4] = true
    P = FF.FinitePoset(leq)

    # A tiny fringe module that typically forces generators at vertices 2 and 3.
    U = [FF.principal_upset(P, 2), FF.principal_upset(P, 3)]
    D = [FF.principal_downset(P, 4)]
    Phi = spzeros(QQ, 1, 2)
    Phi[1,1] = QQ(1)
    Phi[1,2] = QQ(1)
    H = FF.FringeModule{QQ}(P, U, D, Phi)

    M = IR.pmodule_from_fringe(H)

    # Projective cover must be a P-module morphism.
    F0, pi0, _ = IR.projective_cover(M)
    C = FF.cover_edges(P)
    for u in 1:P.n, v in 1:P.n
        if C[u,v]
            lhs = M.edge_maps[(u,v)] * pi0.comps[u]
            rhs = pi0.comps[v] * F0.edge_maps[(u,v)]
            @test lhs == rhs
        end
    end

    # Injective hull inclusion must be a P-module morphism.
    E, iota, _ = IR._injective_hull(M)
    for u in 1:P.n, v in 1:P.n
        if C[u,v]
            lhs = E.edge_maps[(u,v)] * iota.comps[u]
            rhs = iota.comps[v] * M.edge_maps[(u,v)]
            @test lhs == rhs
        end
    end
end

@testset "verify_upset_resolution / verify_downset_resolution catch illegal monomial support" begin
    P = diamond_poset()

    # Deliberately invalid "resolution step": for upset resolutions, nonzero in delta (row i, col j)
    # requires U_row subset U_col. We violate that on the diamond poset.
    U2 = FF.principal_upset(P, 2)
    U3 = FF.principal_upset(P, 3)

    F0 = IR.UpsetPresentation{QQ}(P, [U2, U3], FF.Upset[], spzeros(QQ, 0, 2))
    F1 = IR.UpsetPresentation{QQ}(P, [U2],     FF.Upset[], spzeros(QQ, 0, 1))

    # delta is |U1| x |U0| = 1 x 2. Put a nonzero at (U2 row, U3 col).
    # Since U2 is not a subset of U3, this must be rejected.
    delta_bad = spzeros(QQ, 1, 2)
    delta_bad[1, 2] = QQ(1)

    @test_throws ErrorException IR.verify_upset_resolution([F0, F1], [delta_bad];
        check_d2=false, check_exactness=false)

    # Dual check for downset copresentations: nonzero in rho (row i, col j) requires
    # D_row subset D_col, where rows come from the later stage.
    D2 = FF.principal_downset(P, 2)
    D3 = FF.principal_downset(P, 3)

    E0 = IR.DownsetCopresentation{QQ}(P, [D3], FF.Downset[], spzeros(QQ, 0, 1))
    E1 = IR.DownsetCopresentation{QQ}(P, [D2], FF.Downset[], spzeros(QQ, 0, 1))

    rho_bad = spzeros(QQ, 1, 1)
    rho_bad[1, 1] = QQ(1)  # D2 is not a subset of D3

    @test_throws ErrorException IR.verify_downset_resolution([E0, E1], [rho_bad];
        check_d2=false, check_exactness=false)
end
