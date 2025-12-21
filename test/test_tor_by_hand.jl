using Test

const PM = PosetModules
const QQ = PM.QQ
const FF = PM.FiniteFringe
const IR = PM.IndicatorResolutions

@testset "Tor by hand on chain of length 2" begin
    P = chain_poset(2)
    Pop = FF.FinitePoset(transpose(P.leq))

    # L = simple at 1 on P
    Lfr = one_by_one_fringe(P, FF.principal_upset(P, 1), FF.principal_downset(P, 1))
    L = IR.pmodule_from_fringe(Lfr)

    # Rop = simple at 2 on Pop (= P^op)
    Rfr = one_by_one_fringe(Pop, FF.principal_upset(Pop, 2), FF.principal_downset(Pop, 2))
    Rop = IR.pmodule_from_fringe(Rfr)

    T = PM.Tor(Rop, L; maxdeg=3)

    @test PM.dim(T, 0) == 0
    @test PM.dim(T, 1) == 1
    @test PM.dim(T, 2) == 0
    @test PM.dim(T, 3) == 0
end
