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
