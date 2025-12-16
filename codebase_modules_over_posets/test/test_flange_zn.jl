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
end
