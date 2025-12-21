using Test

# This file assumes the helper constructors defined in runtests.jl:
# - chain_poset(n)
# - diamond_poset()
# - one_by_one_fringe(P, U, D)

# -----------------------------------------------------------------------------
# Small helper: direct sum of poset-modules and the split short exact sequence
# -----------------------------------------------------------------------------

function direct_sum_with_split_sequence(A::IR.PModule{QQ}, C::IR.PModule{QQ})
    Q = A.Q
    @test Q.leq == C.Q.leq

    dimsB = [A.dims[v] + C.dims[v] for v in 1:Q.n]

    # Block-diagonal structure maps on cover edges.
    edge_mapsB = Dict{Tuple{Int,Int}, Matrix{QQ}}()
    for (i, j) in FF.cover_edges(Q)
        Aij = get(A.edge_maps, (i,j), zeros(QQ, A.dims[j], A.dims[i]))
        Cij = get(C.edge_maps, (i,j), zeros(QQ, C.dims[j], C.dims[i]))

        top    = hcat(Aij, zeros(QQ, size(Aij,1), size(Cij,2)))
        bottom = hcat(zeros(QQ, size(Cij,1), size(Aij,2)), Cij)
        edge_mapsB[(i,j)] = vcat(top, bottom)
    end

    B = IR.PModule{QQ}(Q, dimsB, edge_mapsB)

    # Inclusion i: A -> A oplus C (first summand)
    comps_i = Vector{Matrix{QQ}}(undef, Q.n)
    for v in 1:Q.n
        comps_i[v] = vcat(Matrix{QQ}(I, A.dims[v], A.dims[v]),
                          zeros(QQ, C.dims[v], A.dims[v]))
    end
    i = IR.PMorphism{QQ}(A, B, comps_i)

    # Projection p: A oplus C -> C (second summand)
    comps_p = Vector{Matrix{QQ}}(undef, Q.n)
    for v in 1:Q.n
        comps_p[v] = hcat(zeros(QQ, C.dims[v], A.dims[v]),
                          Matrix{QQ}(I, C.dims[v], C.dims[v]))
    end
    p = IR.PMorphism{QQ}(B, C, comps_p)

    return B, i, p
end


@testset "Betti extraction from projective resolutions (diamond)" begin
    P = diamond_poset()

    # Simple modules S1..S4 as fringe modules, then as poset-modules.
    Sm = Vector{IR.PModule{QQ}}(undef, P.n)
    for v in 1:P.n
        Hv = one_by_one_fringe(P, FF.principal_upset(P, v), FF.principal_downset(P, v))
        Sm[v] = IR.pmodule_from_fringe(Hv)
    end
    S1, S2, S3, S4 = Sm

    # Minimal projective resolution of S1 on the diamond should have:
    # P0 = P1, P1 = P2 oplus P3, P2 = P4.
    res = PM.projective_resolution(S1; maxlen=2)
    b = PM.betti(res)

    @test length(b) == 4
    @test b[(0, 1)] == 1
    @test b[(1, 2)] == 1
    @test b[(1, 3)] == 1
    @test b[(2, 4)] == 1

    Btbl = PM.betti_table(res)
    @test Btbl[1,1] == 1
    @test Btbl[2,2] == 1
    @test Btbl[2,3] == 1
    @test Btbl[3,4] == 1
end


@testset "Yoneda product (diamond: Ext^1 x Ext^1 -> Ext^2)" begin
    P = diamond_poset()

    Sm = Vector{IR.PModule{QQ}}(undef, P.n)
    for v in 1:P.n
        Hv = one_by_one_fringe(P, FF.principal_upset(P, v), FF.principal_downset(P, v))
        Sm[v] = IR.pmodule_from_fringe(Hv)
    end
    S1, S2, S3, S4 = Sm

    # Compute the target Ext space once so coordinates are comparable.
    E14 = PM.Ext(S1, S4; maxdeg=2)
    @test PM.dim(E14, 2) == 0

    # Via the chain 1 -> 2 -> 4
    E24 = PM.Ext(S2, S4; maxdeg=1)
    E12 = PM.Ext(S1, S2; maxdeg=2)  # needs tmax >= 2 because p+q = 2
    @test PM.dim(E24, 1) == 0
    @test PM.dim(E12, 1) == 1

    beta = [QQ(1)]
    alpha = [QQ(1)]
    _, coords_2 = PM.yoneda_product(E24, 1, beta, E12, 1, alpha; ELN=E14)
    @test coords_2[1] != 0

    # Via the chain 1 -> 3 -> 4
    E34 = PM.Ext(S3, S4; maxdeg=1)
    E13 = PM.Ext(S1, S3; maxdeg=2)
    _, coords_3 = PM.yoneda_product(E34, 1, [QQ(1)], E13, 1, [QQ(1)]; ELN=E14)
    @test coords_3[1] != 0

    # In a 1-dimensional target, the two products must be proportional.
    # With our deterministic lifts/basis choices, they should agree up to sign.
    @test coords_2[1] == coords_3[1] || coords_2[1] == -coords_3[1]
end


@testset "Yoneda associativity sanity check (chain of length 4)" begin
    Q = chain_poset(4)

    Sm = Vector{IR.PModule{QQ}}(undef, Q.n)
    for v in 1:Q.n
        Hv = one_by_one_fringe(Q, FF.principal_upset(Q, v), FF.principal_downset(Q, v))
        Sm[v] = IR.pmodule_from_fringe(Hv)
    end
    S1, S2, S3, S4 = Sm

    # Target space for comparisons.
    E14 = PM.Ext(S1, S4; maxdeg=3)
    @test PM.dim(E14, 3) == 1

    # Degree-1 generators on consecutive covers.
    E34 = PM.Ext(S3, S4; maxdeg=1)
    E23 = PM.Ext(S2, S3; maxdeg=2)   # needs tmax >= 2 for intermediate products
    E12 = PM.Ext(S1, S2; maxdeg=3)   # needs tmax >= 3 for final product

    @test PM.dim(E34, 1) == 1
    @test PM.dim(E23, 1) == 1
    @test PM.dim(E12, 1) == 1

    # First bracketing: (e34 * e23) * e12
    E24 = PM.Ext(S2, S4; maxdeg=2)
    _, x = PM.yoneda_product(E34, 1, [QQ(1)], E23, 1, [QQ(1)]; ELN=E24)   # x in Ext^2(S2,S4)
    _, left = PM.yoneda_product(E24, 2, x, E12, 1, [QQ(1)]; ELN=E14)

    # Second bracketing: e34 * (e23 * e12)
    E13 = PM.Ext(S1, S3; maxdeg=3)
    _, y = PM.yoneda_product(E23, 1, [QQ(1)], E12, 1, [QQ(1)]; ELN=E13)   # y in Ext^2(S1,S3)
    _, right = PM.yoneda_product(E34, 1, [QQ(1)], E13, 2, y; ELN=E14)

    @test left[1] != 0
    @test right[1] != 0
    @test left[1] == right[1] || left[1] == -right[1]
end


@testset "Connecting homomorphisms: split exact sequences give zero maps" begin
    Q = chain_poset(4)

    Sm = Vector{IR.PModule{QQ}}(undef, Q.n)
    for v in 1:Q.n
        Hv = one_by_one_fringe(Q, FF.principal_upset(Q, v), FF.principal_downset(Q, v))
        Sm[v] = IR.pmodule_from_fringe(Hv)
    end
    S1, S2, S3, S4 = Sm

    # -------------------------------------------------------------------------
    # Second argument LES: 0 -> A -> A oplus C -> C -> 0
    # delta : Ext^t(M,C) -> Ext^{t+1}(M,A)
    # Choose M=S1, C=S3 (Ext^2), A=S4 (Ext^3), t=2.
    # -------------------------------------------------------------------------
    A = S4
    C = S3
    B, i, p = direct_sum_with_split_sequence(A, C)

    resM = PM.projective_resolution(S1; maxlen=3)
    EMA = PM.Ext(resM, A)
    EMB = PM.Ext(resM, B)
    EMC = PM.Ext(resM, C)

    delta2 = PM.connecting_hom(EMA, EMB, EMC, i, p; t=2)
    @test all(delta2 .== 0)

    # -------------------------------------------------------------------------
    # First argument LES: 0 -> A -> A oplus C -> C -> 0
    # delta : Ext^t(A,N) -> Ext^{t+1}(C,N)
    # Choose N=S4, A=S3 (Ext^1), C=S2 (Ext^2), t=1.
    # -------------------------------------------------------------------------
    A1 = S3
    C1 = S2
    B1, i1, p1 = direct_sum_with_split_sequence(A1, C1)

    resN = PM.injective_resolution(S4; maxlen=2)
    EA = PM.ExtInjective(A1, resN)
    EB = PM.ExtInjective(B1, resN)
    EC = PM.ExtInjective(C1, resN)

    delta1 = PM.connecting_hom_first(EA, EB, EC, i1, p1; t=1)
    @test all(delta1 .== 0)
end

@testset "ExtAlgebra: cached multiplication agrees with yoneda_product" begin
    P = diamond_poset()

    # Build simples S1..S4 as 1x1 fringe modules, then as poset-modules.
    Sm = Vector{IR.PModule{QQ}}(undef, P.n)
    for v in 1:P.n
        Hv = one_by_one_fringe(P, FF.principal_upset(P, v), FF.principal_downset(P, v))
        Sm[v] = IR.pmodule_from_fringe(Hv)
    end
    S1, S2, S3, S4 = Sm

    # We take a direct sum with enough structure to produce nontrivial Ext in degree 1
    # and allow products in degree 2 on the diamond.
    #
    # M = S1 oplus S2 oplus S4 is small but already contains:
    # - degree-1 extension classes among summands, and
    # - degree-2 composites (in the full Ext(M,M)).
    M12, _, _ = direct_sum_with_split_sequence(S1, S2)
    M, _, _ = direct_sum_with_split_sequence(M12, S4)

    A = PM.ExtAlgebra(M; maxdeg=2)
    E = A.E

    # Sanity: dimensions agree with the underlying Ext space.
    for t in 0:E.tmax
        @test PM.dim(A, t) == PM.dim(E, t)
    end

    # The unit should act as both-sided identity on every homogeneous degree <= tmax.
    oneA = one(A)
    for t in 0:E.tmax
        dt = PM.dim(A, t)
        if dt == 0
            continue
        end

        # Deterministic "generic" element: (1,2,3,...,dt).
        x = PM.element(A, t, [QQ(i) for i in 1:dt])

        @test (oneA * x).coords == x.coords
        @test (x * oneA).coords == x.coords
    end

    # Cache behavior: after one multiplication in (p,q), the multiplication matrix should exist,
    # and repeated multiplication should not grow the cache.
    if E.tmax >= 2 && PM.dim(A, 1) > 0
        d1 = PM.dim(A, 1)
        x = PM.element(A, 1, [QQ(i) for i in 1:d1])
        y = PM.element(A, 1, [QQ(d1 - i + 1) for i in 1:d1])

        prod1 = x * y
        @test haskey(A.mult_cache, (1, 1))
        nkeys = length(A.mult_cache)

        prod2 = x * y
        @test length(A.mult_cache) == nkeys

        # Cached multiplication must match a direct call to the mathematical core (Yoneda product)
        # in the same Ext space and bases.
        _, coords_direct = PM.yoneda_product(A.E, 1, x.coords, A.E, 1, y.coords; ELN=A.E)
        @test prod1.coords == coords_direct
        @test prod2.coords == coords_direct
    end

    # Associativity in the cached algebra (within truncation).
    #
    # On the diamond, Ext^3 is expected to vanish for many modules; we therefore test an
    # associativity instance that stays inside degrees <= 2 by including a degree-0 factor.
    if E.tmax >= 2 && PM.dim(A, 1) > 0
        d1 = PM.dim(A, 1)

        # First basis direction e_1
        a = PM.element(A, 1, [one(QQ); zeros(QQ, d1 - 1)])

        # Last basis direction e_{d1}
        b = PM.element(A, 1, [zeros(QQ, d1 - 1); one(QQ)])
        
        c = oneA

        @test ((a * b) * c).coords == (a * (b * c)).coords
        @test ((c * a) * b).coords == (c * (a * b)).coords
    end
end
