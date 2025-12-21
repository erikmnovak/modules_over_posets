using Test

@testset "PLPolyhedra optional backend" begin
    if PLP.HAVE_POLY
        # Unit square: 0 <= x <= 1, 0 <= y <= 1
        A = QQ[QQ(1)  QQ(0);
               QQ(0)  QQ(1);
               QQ(-1) QQ(0);
               QQ(0)  QQ(-1)]
        b = QQ[QQ(1), QQ(1), QQ(0), QQ(0)]
        h = PLP.make_hpoly(A, b)

        @test PLP._in_hpoly(h, [0, 0]) == true
        @test PLP._in_hpoly(h, [1, 1]) == true
        @test PLP._in_hpoly(h, [2, 0]) == false
        @test PLP._in_hpoly(h, [-1, 0]) == false
    else
        # If Polyhedra/CDDLib is unavailable, make_hpoly must error deterministically.
        A = reshape(QQ[QQ(1)], 1, 1)
        b = QQ[QQ(1)]
        @test_throws ErrorException PLP.make_hpoly(A, b)
    end
end
