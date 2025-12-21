using Test

@testset "CrossValidateFlangePL smoke test" begin
    n = 1
    tau0 = FZ.Face(n, [false])
    F1 = FZ.IndFlat{QQ}([1], tau0, :F1)
    E1 = FZ.IndInj{QQ}([3], tau0, :E1)
    Phi = reshape(QQ[QQ(1)], 1, 1)
    FG = FZ.Flange{QQ}(n, [F1], [E1], Phi)

    ok, report = CV.cross_validate(FG; margin=1, rankfun=EX.rankQQ)
    @test ok == true
    @test haskey(report, "mismatches")
    @test isempty(report["mismatches"])
end
