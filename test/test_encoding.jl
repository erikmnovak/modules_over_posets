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

@testset "Uptight poset uses transitive closure (Example 4.16)" begin
    # Miller Example 4.16 exhibits non-transitivity of the raw "exists a<=b" relation
    # between uptight regions; Definition 4.17 then defines the uptight poset as its
    # transitive closure. We build a finite truncation of the N^2 example to test that
    # Encoding._uptight_poset really closes transitively.

    # Q = {0..3} x {0..2} with product order.
    function grid_poset(ax::Int, by::Int)
        coords = Tuple{Int,Int}[]
        for a in 0:ax, b in 0:by
            push!(coords, (a, b))
        end
        nQ = length(coords)
        leq = falses(nQ, nQ)
        for i in 1:nQ, j in 1:nQ
            ai, bi = coords[i]
            aj, bj = coords[j]
            leq[i, j] = (ai <= aj) && (bi <= bj)
        end
        Q = FF.FinitePoset(leq)
        idx = Dict{Tuple{Int,Int}, Int}()
        for (i, c) in enumerate(coords)
            idx[c] = i
        end
        return Q, coords, idx
    end

    Q, coords, idx = grid_poset(3, 2)

    # Upsets corresponding to the monomial ideals in Example 4.16:
    #   U1 = <x^2, y>    => (a>=2) or (b>=1)
    #   U2 = <x^3, y>    => (a>=3) or (b>=1)
    #   U3 = <x*y>       => (a>=1) and (b>=1)
    #   U4 = <x^2*y>     => (a>=2) and (b>=1)
    U1 = FF.upset_closure(Q, BitVector([(a >= 2) || (b >= 1) for (a, b) in coords]))
    U2 = FF.upset_closure(Q, BitVector([(a >= 3) || (b >= 1) for (a, b) in coords]))
    U3 = FF.upset_closure(Q, BitVector([(a >= 1) && (b >= 1) for (a, b) in coords]))
    U4 = FF.upset_closure(Q, BitVector([(a >= 2) && (b >= 1) for (a, b) in coords]))

    # No deaths: we only need the upsets to form Y for uptight signatures.
    phi0 = spzeros(QQ, 0, 4)
    M = FF.FringeModule{QQ}(Q, [U1, U2, U3, U4], FF.Downset[], phi0)

    enc = EN.build_uptight_encoding_from_fringe(M)
    pi = enc.pi
    P = pi.P

    # Identify the regions by representative lattice points (degrees):
    #   A: x^2  = (2,0) has signature {U1}
    #   B: x^3  = (3,0) and y = (0,1) share signature {U1,U2}
    #   C: x*y  = (1,1) has signature {U1,U2,U3}
    q_x2 = idx[(2, 0)]
    q_x3 = idx[(3, 0)]
    q_y  = idx[(0, 1)]
    q_xy = idx[(1, 1)]

    A = pi.pi_of_q[q_x2]
    B = pi.pi_of_q[q_x3]
    B2 = pi.pi_of_q[q_y]
    C = pi.pi_of_q[q_xy]

    @test B == B2
    @test length(Set([A, B, C])) == 3

    # In the transitive closure P, we must have A <= B <= C, hence A <= C.
    @test P.leq[A, B]
    @test P.leq[B, C]
    @test P.leq[A, C]

    # But in the underlying "exists a<=c" relation on regions, there is no witness for A <= C:
    # the only point in A is (2,0), and every point in C has x=1, so (2,0) is not <= any c in C.
    has_witness = false
    for a in 1:Q.n, c in 1:Q.n
        if pi.pi_of_q[a] == A && pi.pi_of_q[c] == C && Q.leq[a, c]
            has_witness = true
            break
        end
    end
    @test !has_witness
end

