using Test

@testset "HomExt pi0_count (Prop 3.10 sanity)" begin
    # Disjoint union of two chains: intersection has two connected components.
    P = disjoint_two_chains_poset()

    U = FF.upset_from_generators(P, [2, 4])      # {2,4}
    D = FF.downset_from_generators(P, [2, 4])    # {1,2,3,4}

    @test HE.pi0_count(P, U, D) == 2

    # Connected case: chain
    Q = chain_poset(3)
    U2 = FF.principal_upset(Q, 2)
    D2 = FF.principal_downset(Q, 2)
    @test HE.pi0_count(Q, U2, D2) == 1
end
