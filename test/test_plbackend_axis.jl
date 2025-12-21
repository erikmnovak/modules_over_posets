using Test

@testset "PLBackend axis encoding (axis-aligned boxes)" begin
    # PLBackend is optional at package-load time. For test coverage, load it
    # into the PosetModules namespace if it is not already present.
    if !isdefined(PosetModules, :PLBackend)
        @eval PosetModules include(joinpath(@__DIR__, "..", "src", "PLBackend.jl"))
    end
    PLB = PosetModules.PLBackend

    # One-dimensional example:
    # Birth upset:  x >= 0
    # Death downset: x <= 2
    # phi = [1], so the represented module should be supported exactly in the middle region.
    Ups = [PLB.BoxUpset([0.0])]
    Downs = [PLB.BoxDownset([2.0])]
    Phi = reshape(QQ[1], 1, 1)

    # Error guard: if max_regions is too small, the backend should refuse.
    @test_throws ErrorException PLB.encode_fringe_boxes(Ups, Downs, Phi; max_regions=2)

    P, H, pi = PLB.encode_fringe_boxes(Ups, Downs, Phi)

    @test P.n == 3
    @test pi.coords[1] == [0.0, 2.0]

    # The encoded poset should be a 3-element chain.
    @test Set(FF.cover_edges(P)) == Set([(1, 2), (2, 3)])

    # Locate regions by sampling points in each cell and check fiber dimensions.
    @test FF.fiber_dimension(H, PLB.locate(pi, [-1.0])) == 0   # left of 0: outside upset
    @test FF.fiber_dimension(H, PLB.locate(pi, [ 1.0])) == 1   # between: in upset and downset
    @test FF.fiber_dimension(H, PLB.locate(pi, [ 3.0])) == 0   # right of 2: outside downset
end
