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
        M_loaded = SER.load_encoding_json(path)

        @test M_loaded.P.n == M.P.n
        @test M_loaded.P.leq == M.P.leq
        @test M_loaded.U[1].mask == M.U[1].mask
        @test M_loaded.D[1].mask == M.D[1].mask
        @test Matrix(M_loaded.phi) == Matrix(M.phi)
        for q in 1:M.P.n
            @test FF.fiber_dimension(M_loaded, q) == FF.fiber_dimension(M, q)
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

@testset "M2SingularBridge.parse_flange_json edge cases" begin
    # tau given as index list (1-based) rather than Bool vector; phi omitted -> canonical_matrix
    json1 = """
    {
      "n": 1,
      "flats": [ { "id": "F", "b": [0], "tau": [1] } ],
      "injectives": [ { "id": "E", "b": [0], "tau": [1] } ]
    }
    """
    H1 = BR.parse_flange_json(json1)
    @test H1.n == 1
    @test length(H1.flats) == 1
    @test length(H1.injectives) == 1
    @test Matrix(H1.Phi) == reshape(QQ[1], 1, 1)

    # Explicit phi entries can be rationals in string form.
    json2 = """
    {
      "n": 1,
      "flats": [ { "id": "F", "b": [0], "tau": [false] } ],
      "injectives": [ { "id": "E", "b": [0], "tau": [false] } ],
      "phi": [ [ "-2/3" ] ]
    }
    """
    H2 = BR.parse_flange_json(json2)
    @test Matrix(H2.Phi)[1, 1] == (-QQ(2) / QQ(3))

    # Non-intersecting flat/injective pairs must force Phi entries to 0 (monomial condition).
    json3 = """
    {
      "n": 1,
      "flats": [ { "id": "F", "b": [5], "tau": [false] } ],
      "injectives": [ { "id": "E", "b": [3], "tau": [false] } ],
      "phi": [ [ 1 ] ]
    }
    """
    H3 = BR.parse_flange_json(json3)
    @test Matrix(H3.Phi) == reshape(QQ[0], 1, 1)
    @test FZ.dim_at(H3, [0]) == 0
end

@testset "Serialization.load_encoding_json accepts flat leq format" begin
    # load_encoding_json supports:
    #  (1) leq as an n-by-n Boolean matrix (vector-of-vectors)
    #  (2) leq as a flat n*n Boolean vector in Julia column-major order.
    #
    # This test exercises (2) with a 3-chain.
    json = """
    {
      "P": { "n": 3,
             "leq": [ true, false, false,
                      true, true,  false,
                      true, true,  true ] },
      "U": [ [2, 3] ],
      "D": [ [1, 2] ],
      "phi": [ [ "-2/3" ] ]
    }
    """
    mktempdir() do dir
        path = joinpath(dir, "encoding_flat.json")
        write(path, json)
        M = SER.load_encoding_json(path)

        P = chain_poset(3)
        @test M.P.leq == P.leq

        @test length(M.U) == 1
        @test length(M.D) == 1
        @test M.U[1].mask == BitVector([false, true, true])
        @test M.D[1].mask == BitVector([true, true, false])
        @test Matrix(M.phi)[1, 1] == (-QQ(2) / QQ(3))
    end
end
