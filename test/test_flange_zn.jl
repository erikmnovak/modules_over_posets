using Test

@testset "FlangeZn dim_at + minimize invariance" begin
    # n = 1 interval [b,c] via flat (>= b) and injective (<= c)
    n = 1
    tau0 = FZ.Face(n, [false])
    b = 1
    c = 3
    F1 = FZ.IndFlat{QQ}([b], tau0, :F1)
    E1 = FZ.IndInj{QQ}([c], tau0, :E1)
    Phi = reshape(QQ[QQ(1)], 1, 1)
    FG = FZ.Flange{QQ}(n, [F1], [E1], Phi)

    # dim is 1 on [b,c], 0 otherwise
    for g in (b-2):(c+2)
        d = FZ.dim_at(FG, [g]; rankfun=EX.rankQQ)
        expected = (b <= g <= c) ? 1 : 0
        @test d == expected
    end

    # intersects should detect empty intersection when b > c
    F_bad = FZ.IndFlat{QQ}([5], tau0, :Fbad)
    E_bad = FZ.IndInj{QQ}([2], tau0, :Ebad)
    @test FZ.intersects(F_bad, E_bad) == false

    # Minimize should merge proportional duplicate columns without changing dim_at
    F2 = FZ.IndFlat{QQ}([b], tau0, :F2)
    Phi2 = reshape(QQ[QQ(1), QQ(2)], 1, 2)    # second column is 2x the first
    FG2 = FZ.Flange{QQ}(n, [F1, F2], [E1], Phi2)
    FG2m = FZ.minimize(FG2)

    for g in (b-1):(c+1)
        d1 = FZ.dim_at(FG2, [g];  rankfun=EX.rankQQ)
        d2 = FZ.dim_at(FG2m, [g]; rankfun=EX.rankQQ)
        @test d1 == d2
    end

    @testset "canonical_matrix / degree_matrix / bounding_box" begin
        # 1D: flat is x >= 1, inj is x <= 3.
        F = [FZ.Flat(:F, [1], [false])]
        E = [FZ.Injective(:E, [3], [false])]
        Phi = FZ.canonical_matrix(F, E)
        @test Phi == reshape(QQ[1], 1, 1)

        # Non-intersecting pair: x >= 5 and x <= 3.
        Fbad = [FZ.Flat(:Fbad, [5], [false])]
        Ebad = [FZ.Injective(:Ebad, [3], [false])]
        Phibad = FZ.canonical_matrix(Fbad, Ebad)
        @test Phibad == reshape(QQ[0], 1, 1)

        # degree_matrix should pick out exactly the active row/col at a given degree.
        Phi2 = reshape(QQ[2], 1, 1)
        H = FZ.Flange{QQ}(1, F, E, Phi2)
        Phi_sub, rows, cols = FZ.degree_matrix(H, [2])
        @test rows == [1]
        @test cols == [1]
        @test Phi_sub == reshape(QQ[2], 1, 1)

        # Outside the intersection, there should be no active flats or injectives.
        Phi_sub2, rows2, cols2 = FZ.degree_matrix(H, [10])
        @test rows2 == Int[]
        @test cols2 == Int[]
        @test size(Phi_sub2) == (0, 0)

        # bounding_box in 1D with margin 1:
        #   flats force a >= (b_flat - margin) = 0
        #   injectives force b <= (b_inj + margin) = 4
        a, b = FZ.bounding_box(H; margin=1)
        @test a == [0]
        @test b == [4]
    end

    @testset "minimize: do not merge different labels, but merge proportional duplicates" begin
        # Two proportional columns with *different* underlying flats must not be merged.
        F1 = FZ.Flat(:F1, [0], [false])
        F2 = FZ.Flat(:F2, [1], [false])  # different threshold => different upset
        E1 = FZ.Injective(:E1, [2], [false])

        # Columns are proportional but flats differ.
        Phi = QQ[1 2]
        H = FZ.Flange{QQ}(1, [F1, F2], [E1], Phi)
        Hmin = FZ.minimize(H)
        @test length(Hmin.F) == 2

        # Two proportional rows with identical injectives should be merged.
        Fin = [FZ.Flat(:F, [0], [false])]
        Einj1 = FZ.Injective(:E, [0], [false])
        Einj2 = FZ.Injective(:Edup, [0], [false]) # same underlying downset as Einj1
        Phi_rows = reshape(QQ[1, 2], 2, 1)        # 2x1, proportional rows
        H2 = FZ.Flange{QQ}(1, Fin, [Einj1, Einj2], Phi_rows)
        H2min = FZ.minimize(H2)
        @test length(H2min.E) == 1

        # Rank at degree 0 should be unchanged by minimization.
        @test FZ.dim_at(H2, [0]) == FZ.dim_at(H2min, [0])
    end



end
