using Test
using SparseArrays

@testset "FiniteFringe basics" begin
    P = chain_poset(3)

    # cover edges on a 3-chain should be 1->2 and 2->3 only
    C = FF.cover_edges(P)
    @test C[1,2] == true
    @test C[2,3] == true
    @test C[1,3] == false

    # principal sets
    U2 = FF.principal_upset(P, 2)      # {2,3}
    D2 = FF.principal_downset(P, 2)    # {1,2}
    @test U2.mask == BitVector([false, true,  true])
    @test D2.mask == BitVector([true,  true,  false])

    # 1x1 interval module supported at {2}
    phi = spzeros(QQ, 1, 1)
    phi[1,1] = QQ(1)
    M = FF.FringeModule{QQ}(P, [U2], [D2], phi)

    @test FF.fiber_dimension(M, 1) == 0
    @test FF.fiber_dimension(M, 2) == 1
    @test FF.fiber_dimension(M, 3) == 0

    # Endomorphisms of a connected indicator should be 1-dimensional
    @test FF.hom_dimension(M, M) == 1

    # Monomial condition should reject nonzero entry when U cap D is empty
    U3 = FF.principal_upset(P, 3)      # {3}
    D1 = FF.principal_downset(P, 1)    # {1}
    phi_bad = spzeros(QQ, 1, 1)
    phi_bad[1,1] = QQ(1)
    @test_throws AssertionError FF.FringeModule{QQ}(P, [U3], [D1], phi_bad)
end
