using Test
using SparseArrays

const PM = PosetModules
const QQ = PM.QQ
const CC = PM.ChainComplexes

@testset "ChainComplexes homology_data and homology_coordinates by hand" begin
    # ----------------
    # Circle: C1=QQ, C0=QQ, d1=0
    # ----------------
    d1 = zeros(QQ, 1, 1)      # C1 -> C0
    d0 = zeros(QQ, 0, 1)      # C0 -> 0
    d2 = zeros(QQ, 1, 0)      # 0  -> C1

    H1 = CC.homology_data(d2, d1, 1)
    @test H1.dimH == 1

    H0 = CC.homology_data(d1, d0, 0)
    @test H0.dimH == 1

    # coordinate sanity on the chosen basis representative
    c = CC.homology_coordinates(H1, H1.Hrep[:, 1])
    @test c == reshape(QQ[1], 1, 1)

    # ----------------
    # Interval: C1=QQ, C0=QQ^2, d1 = [ -1; 1 ]
    # ----------------
    d1_int = reshape(QQ[-1, 1], 2, 1)
    d0_int = zeros(QQ, 0, 2)
    d2_int = zeros(QQ, 1, 0)

    H1_int = CC.homology_data(d2_int, d1_int, 1)
    @test H1_int.dimH == 0

    H0_int = CC.homology_data(d1_int, d0_int, 0)
    @test H0_int.dimH == 1

    # any nonzero vector in C1 is not a cycle since d1_int is injective
    @test_throws ErrorException CC.homology_coordinates(H1_int, reshape(QQ[1], 1, 1))

    # boundary class should map to 0 in H0
    b = H0_int.B[:, 1]
    c0 = CC.homology_coordinates(H0_int, b)
    @test c0 == zeros(QQ, H0_int.dimH, 1)

    # ----------------
    # Filled triangle: C2=QQ, C1=QQ^3, C0=QQ^3
    # edge basis: e01,e02,e12; vertex basis: v0,v1,v2; face basis: f012
    # d2(f) = e12 - e02 + e01 => column [1,-1,1]
    # d1 columns are boundary of edges:
    #   e01 -> v1 - v0  => [-1, 1, 0]
    #   e02 -> v2 - v0  => [-1, 0, 1]
    #   e12 -> v2 - v1  => [ 0,-1, 1]
    # ----------------
    d2_tri = reshape(QQ[1, -1, 1], 3, 1)
    d1_tri = QQ[
        -1  -1   0;
         1   0  -1;
         0   1   1
    ]
    d0_tri = zeros(QQ, 0, 3)

    H2_tri = CC.homology_data(zeros(QQ, 1, 0), d2_tri, 2)
    @test H2_tri.dimH == 0

    H1_tri = CC.homology_data(d2_tri, d1_tri, 1)
    @test H1_tri.dimH == 0

    H0_tri = CC.homology_data(d1_tri, d0_tri, 0)
    @test H0_tri.dimH == 1
end
