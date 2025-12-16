using Test
using SparseArrays

@testset "Finite encoding from fringe (Defs 4.12-4.18)" begin
    P = chain_poset(3)
    U2 = FF.principal_upset(P, 2)
    D2 = FF.principal_downset(P, 2)

    M = one_by_one_fringe(P, U2, D2)

    enc = EN.build_uptight_encoding_from_fringe(M)
    pi = enc.pi

    # pi should be order-preserving: i <= j in Q => pi(i) <= pi(j) in P_Y
    for i in 1:pi.Q.n, j in 1:pi.Q.n
        if pi.Q.leq[i,j]
            @test pi.P.leq[pi.pi_of_q[i], pi.pi_of_q[j]]
        end
    end

    # Each generator upset U_i and each death downset D_j should be unions of fibers.
    # Hence preimage(image(U)) = U and preimage(image(D)) = D.
    for U in M.U
        Uhat = EN.image_upset(pi, U)
        Uback = EN.preimage_upset(pi, Uhat)
        @test Uback.mask == U.mask
    end
    for D in M.D
        Dhat = EN.image_downset(pi, D)
        Dback = EN.preimage_downset(pi, Dhat)
        @test Dback.mask == D.mask
    end

    # Build the induced module on the encoded poset by pushing U,D forward.
    Uhat = [EN.image_upset(pi, U) for U in M.U]
    Dhat = [EN.image_downset(pi, D) for D in M.D]
    Hhat = FF.FringeModule{QQ}(pi.P, Uhat, Dhat, M.phi)

    # Pull back and recover the original U,D exactly.
    M2 = EN.pullback_fringe_along_encoding(Hhat, pi)
    @test M2.U[1].mask == M.U[1].mask
    @test M2.D[1].mask == M.D[1].mask
    for q in 1:P.n
        @test FF.fiber_dimension(M2, q) == FF.fiber_dimension(M, q)
    end
end
