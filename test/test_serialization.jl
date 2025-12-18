using Test

@testset "JSON serialization round-trips" begin
    # Flange round-trip
    n = 1
    tau0 = FZ.Face(n, [false])
    F1 = FZ.IndFlat{QQ}([1], tau0, :F1)
    E1 = FZ.IndInj{QQ}([3], tau0, :E1)
    Phi = reshape(QQ[QQ(1)], 1, 1)
    FG = FZ.Flange{QQ}(n, [F1], [E1], Phi)

    mktempdir() do dir
        path = joinpath(dir, "flange.json")
        SER.save_flange_json(path, FG)
        FG2 = SER.load_flange_json(path)

        @test FG2.n == FG.n
        @test length(FG2.flats) == length(FG.flats)
        @test length(FG2.injectives) == length(FG.injectives)
        @test FG2.Phi == FG.Phi
        @test FG2.flats[1].b == FG.flats[1].b
        @test FG2.injectives[1].b == FG.injectives[1].b
        @test FG2.flats[1].tau.coords == FG.flats[1].tau.coords
        @test FG2.injectives[1].tau.coords == FG.injectives[1].tau.coords
    end

    # Finite encoding fringe round-trip
    P = chain_poset(3)
    M = one_by_one_fringe(P, FF.principal_upset(P, 2), FF.principal_downset(P, 2))

    mktempdir() do dir
        path = joinpath(dir, "encoding.json")
        SER.save_encoding_json(path, M)
        M2 = SER.load_encoding_json(path)

        @test M2.P.n == M.P.n
        @test M2.P.leq == M.P.leq
        @test M2.U[1].mask == M.U[1].mask
        @test M2.D[1].mask == M.D[1].mask
        @test Matrix(M2.phi) == Matrix(M.phi)
        for q in 1:M.P.n
            @test FF.fiber_dimension(M2, q) == FF.fiber_dimension(M, q)
        end
    end

    # M2/Singular bridge parser (pure JSON input)
    json = """
    {
      "n": 1,
      "field": "QQ",
      "flats":      [ {"b":[1], "tau":[false], "id":"F1"} ],
      "injectives": [ {"b":[3], "tau":[false], "id":"E1"} ],
      "phi": [ ["1/1"] ]
    }
    """
    FG3 = BR.parse_flange_json(json)
    @test FG3.n == 1
    @test length(FG3.flats) == 1
    @test length(FG3.injectives) == 1
    @test FG3.Phi[1,1] == QQ(1)
end
