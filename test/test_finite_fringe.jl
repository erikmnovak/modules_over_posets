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


        @testset "FinitePoset validation errors" begin
        # Not reflexive.
        leq1 = trues(2, 2)
        leq1[1, 1] = false
        @test_throws ErrorException FF.FinitePoset(leq1)

        # Violates antisymmetry: 1<=2 and 2<=1.
        leq2 = trues(2, 2)
        @test_throws ErrorException FF.FinitePoset(leq2)

        # Violates transitivity: 1<=2, 2<=3, but not 1<=3.
        leq3 = falses(3, 3)
        for i in 1:3
            leq3[i, i] = true
        end
        leq3[1, 2] = true
        leq3[2, 3] = true
        @test_throws ErrorException FF.FinitePoset(leq3)
    end

    @testset "Upset/downset closure and generators" begin
        P4 = chain_poset(4)
        S = falses(P4.n)
        S[2] = true

        Ucl = FF.upset_closure(P4, S)
        Dcl = FF.downset_closure(P4, S)

        @test Ucl.mask == BitVector([false, true, true, true])
        @test Dcl.mask == BitVector([true, true, false, false])

        Pdia = diamond_poset()
        Ugen = FF.upset_from_generators(Pdia, [2, 3])
        Dgen = FF.downset_from_generators(Pdia, [2, 3])

        @test Ugen.mask == BitVector([false, true, true, true])  # {2,3,4}
        @test Dgen.mask == BitVector([true, true, true, false])  # {1,2,3}
    end

    @testset "cover_edges on a non-chain (diamond) poset" begin
        Pdia = diamond_poset()
        C = FF.cover_edges(Pdia)
        edges = Set([(I[1], I[2]) for I in findall(C)])
        @test edges == Set([(1, 2), (1, 3), (2, 4), (3, 4)])
    end


    @testset "Dense phi branch and dense_to_sparse_K" begin
        P3 = chain_poset(3)
        U2 = FF.principal_upset(P3, 2)
        D2 = FF.principal_downset(P3, 2)

        # Dense 1x1 phi exercises the non-sparse path in _check_monomial_condition.
        phi_dense = reshape(QQ[1], 1, 1)
        Mdense = FF.FringeModule{QQ}(P3, [U2], [D2], phi_dense)
        @test FF.fiber_dimension(Mdense, 2) == 1

        # Converting dense -> sparse should preserve values.
        phi_sparse = FF.dense_to_sparse_K(phi_dense)
        @test phi_sparse isa SparseMatrixCSC{QQ, Int}
        @test Matrix(phi_sparse) == phi_dense

        # Sparse phi should behave the same.
        Msparse = FF.FringeModule{QQ}(P3, [U2], [D2], phi_sparse)
        @test FF.fiber_dimension(Msparse, 2) == 1

        # Dense constructor should reject nonzero phi when U cap D is empty.
        U3 = FF.principal_upset(P3, 3)
        D1 = FF.principal_downset(P3, 1)
        @test_throws AssertionError FF.FringeModule{QQ}(P3, [U3], [D1], reshape(QQ[1], 1, 1))
    end


end
