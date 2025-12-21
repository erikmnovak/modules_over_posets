using Test

@testset "Indicator resolutions on the diamond poset (by-hand checks)" begin
    # -------------------------------------------------------------------------
    # The diamond poset (a.k.a. the rank-2 Boolean lattice) is the first place
    # where projective/injective resolutions can have length 2 (not just 1),
    # because there are two different length-2 chains from bottom to top:
    #   1 -> 2 -> 4
    #   1 -> 3 -> 4
    #
    # In the category algebra kP (representations of the poset), this produces
    # a genuine relation between the two composites, and it manifests as:
    #   Ext^2(S1, S4) = 1
    # while Ext^1(S1, S4) = 0 (since 1<4 is not a cover).
    #
    # These tests check:
    #  * the shape of the computed resolutions for S1 and S4 (length 2),
    #  * the "by hand" pattern of the differentials at the coefficient-matrix
    #    level (equal-entry and sum-to-zero patterns),
    #  * Ext^0/1/2 between ALL simples on the diamond against an interval-based
    #    computation (reduced cohomology in degrees -1 and 0).
    # -------------------------------------------------------------------------

    P = diamond_poset()

    # Sanity: cover edges should be exactly 1->2, 1->3, 2->4, 3->4.
    C = FF.cover_edges(P)
    @test C[1, 2] == true
    @test C[1, 3] == true
    @test C[2, 4] == true
    @test C[3, 4] == true
    @test C[1, 4] == false
    @test C[2, 3] == false
    @test C[3, 2] == false

    # Sanity: principal upsets/downsets on a non-chain poset.
    @test FF.principal_upset(P, 1).mask == BitVector([true,  true,  true,  true])
    @test FF.principal_upset(P, 2).mask == BitVector([false, true,  false, true])
    @test FF.principal_upset(P, 3).mask == BitVector([false, false, true,  true])
    @test FF.principal_upset(P, 4).mask == BitVector([false, false, false, true])

    @test FF.principal_downset(P, 1).mask == BitVector([true,  false, false, false])
    @test FF.principal_downset(P, 2).mask == BitVector([true,  true,  false, false])
    @test FF.principal_downset(P, 3).mask == BitVector([true,  false, true,  false])
    @test FF.principal_downset(P, 4).mask == BitVector([true,  true,  true,  true])

    # Simple module S_p at vertex p: support only at p, with all structure maps zero.
    simple_at(p::Int) = one_by_one_fringe(P, FF.principal_upset(P, p), FF.principal_downset(P, p))
    S = [simple_at(p) for p in 1:P.n]
    S1, S2, S3, S4 = S

    # -------------------------------------------------------------------------
    # (A) Resolution shape checks for the "interesting" case S1 and S4.
    # -------------------------------------------------------------------------

    # Projective (upset) resolution of S1 should have length 2:
    #   F2 -> F1 -> F0 -> S1 -> 0
    # with:
    #   U0 = [Up(1)]
    #   U1 = [Up(2), Up(3)]
    #   U2 = [Up(4)]
    F, dF = IR.upset_resolution(S1; maxlen=10)

    @test length(F) == 3
    @test length(dF) == 2

    # Verify structural correctness of the upset resolution (d^2=0 + exactness).
    @test IR.verify_upset_resolution(F, dF)

    U_by_a = [f.U0 for f in F]
    @test length(U_by_a[1]) == 1
    @test U_by_a[1][1].mask == FF.principal_upset(P, 1).mask

    @test length(U_by_a[2]) == 2
    @test U_by_a[2][1].mask == FF.principal_upset(P, 2).mask
    @test U_by_a[2][2].mask == FF.principal_upset(P, 3).mask

    @test length(U_by_a[3]) == 1
    @test U_by_a[3][1].mask == FF.principal_upset(P, 4).mask

    # Differential patterns, robust to a global nonzero scalar choice:
    #
    # delta0 : U1 -> U0 should send both branch generators (at 2 and 3)
    # to the unique generator at 1 with the SAME coefficient.
    #
    # delta1 : U2 -> U1 encodes the relation between the two branches at the top;
    # the two coefficients must sum to zero (so they are negatives of each other).
    delta0 = dF[1]
    delta1 = dF[2]

    @test size(delta0) == (2, 1)
    @test size(delta1) == (1, 2)

    a = delta0[1, 1]
    b = delta0[2, 1]
    @test a != 0 && b != 0
    @test a == b

    c = delta1[1, 1]
    d = delta1[1, 2]
    @test c != 0 && d != 0
    @test c + d == 0

    # Composition must be 0 (also checked by verify_upset_resolution, but this is "by hand").
    @test Matrix(delta1 * delta0) == zeros(QQ, 1, 1)

    # Injective (downset) resolution of S4 should have length 2:
    #   0 -> S4 -> E0 -> E1 -> E2
    # with:
    #   D0 = [Down(4)]
    #   D1 = [Down(2), Down(3)]
    #   D2 = [Down(1)]
    E, dE = IR.downset_resolution(S4; maxlen=10)

    @test length(E) == 3
    @test length(dE) == 2

    # Verify structural correctness of the downset resolution.
    @test IR.verify_downset_resolution(E, dE)

    D_by_b = [e.D0 for e in E]
    @test length(D_by_b[1]) == 1
    @test D_by_b[1][1].mask == FF.principal_downset(P, 4).mask

    @test length(D_by_b[2]) == 2
    @test D_by_b[2][1].mask == FF.principal_downset(P, 2).mask
    @test D_by_b[2][2].mask == FF.principal_downset(P, 3).mask

    @test length(D_by_b[3]) == 1
    @test D_by_b[3][1].mask == FF.principal_downset(P, 1).mask

    rho0 = dE[1]
    rho1 = dE[2]

    @test size(rho0) == (2, 1)
    @test size(rho1) == (1, 2)

    r1 = rho0[1, 1]
    r2 = rho0[2, 1]
    @test r1 != 0 && r2 != 0
    @test r1 == r2

    s1 = rho1[1, 1]
    s2 = rho1[1, 2]
    @test s1 != 0 && s2 != 0
    @test s1 + s2 == 0

    @test Matrix(rho1 * rho0) == zeros(QQ, 1, 1)

    # -------------------------------------------------------------------------
    # (B) Ext^0/1/2 between ALL simple modules on the diamond, computed "by hand"
    #     from the open interval (x,y).
    #
    # For degree 1 and 2 on simples, we only need:
    #   - Ext^1 corresponds to reduced H^{-1} of the order complex of (x,y),
    #     which is 1 iff (x,y) is empty (i.e. x<y is a cover), else 0.
    #   - Ext^2 corresponds to reduced H^0 of that order complex, i.e.
    #       (#connected components of the open interval) - 1,
    #     and is 0 for empty intervals.
    #
    # On the diamond: (1,4) = {2,3} has 2 components, so Ext^2(S1,S4)=1.
    # -------------------------------------------------------------------------

    function strict_interval(P::FF.FinitePoset, x::Int, y::Int)
        if x == y || !P.leq[x, y]
            return Int[]
        end
        return [z for z in 1:P.n if z != x && z != y && P.leq[x, z] && P.leq[z, y]]
    end

    # Count connected components in the induced Hasse graph on 'verts' (undirected).
    function interval_components(P::FF.FinitePoset, verts::Vector{Int})
        isempty(verts) && return 0

        inV = falses(P.n)
        for v in verts
            inV[v] = true
        end

        C = FF.cover_edges(P)
        adj = [Int[] for _ in 1:P.n]
        for u in 1:P.n, v in 1:P.n
            if C[u, v] && inV[u] && inV[v]
                push!(adj[u], v)
                push!(adj[v], u)
            end
        end

        seen = falses(P.n)
        comps = 0
        for v in verts
            if seen[v]
                continue
            end
            comps += 1
            stack = [v]
            seen[v] = true
            while !isempty(stack)
                a = pop!(stack)
                for b in adj[a]
                    if !seen[b]
                        seen[b] = true
                        push!(stack, b)
                    end
                end
            end
        end
        return comps
    end

    function expected_ext_dims_simple(P::FF.FinitePoset, x::Int, y::Int)
        # Expected (Ext^0, Ext^1, Ext^2) for simples at x and y.
        if x == y
            return (ext0 = 1, ext1 = 0, ext2 = 0)
        end
        if !P.leq[x, y]
            return (ext0 = 0, ext1 = 0, ext2 = 0)
        end

        verts = strict_interval(P, x, y)
        if isempty(verts)
            # Cover relation => reduced H^{-1}(empty) = k => Ext^1 = 1.
            return (ext0 = 0, ext1 = 1, ext2 = 0)
        end

        # Nonempty interval => Ext^1 = 0 and Ext^2 = reduced H^0 = components-1.
        c = interval_components(P, verts)
        return (ext0 = 0, ext1 = 0, ext2 = max(c - 1, 0))
    end

    for x in 1:P.n, y in 1:P.n
        ext_xy = IR.ext_dimensions_via_indicator_resolutions(S[x], S[y]; maxlen=6)
        exp = expected_ext_dims_simple(P, x, y)

        @test get(ext_xy, 0, 0) == exp.ext0
        @test get(ext_xy, 1, 0) == exp.ext1
        @test get(ext_xy, 2, 0) == exp.ext2

        # On this poset, the only possible nonzero higher group for simples would
        # come from higher reduced cohomology of intervals, which does not occur here.
        @test get(ext_xy, 3, 0) == 0
        @test get(ext_xy, 4, 0) == 0
    end

    # -------------------------------------------------------------------------
    # (C) First-page vs full Ext: S1 -> S4 is the motivating case.
    # The "one-step" data cannot see Ext^2, but the full resolution must.
    # -------------------------------------------------------------------------
    F1 = IR.upset_presentation_one_step(S1)
    E1 = IR.downset_copresentation_one_step(S4)
    hom0, ext1 = IR.hom_ext_first_page(F1, E1)

    @test hom0 == 0
    @test ext1 == 0

    ext_full = IR.ext_dimensions_via_indicator_resolutions(S1, S4; maxlen=6)
    @test get(ext_full, 2, 0) == 1
end
