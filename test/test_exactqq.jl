using Test

@testset "ExactQQ linear algebra" begin
    # Rank test
    A = QQ[QQ(1) QQ(2);
           QQ(2) QQ(4)]
    @test EX.rankQQ(A) == 1

    # Nullspace test: A * v = 0
    N = EX.nullspaceQQ(A)
    @test size(N, 1) == 2
    @test size(N, 2) == 1
    v = N[:, 1]
    @test A * v == zeros(QQ, 2)

    # Solve full column rank system B*x = y
    B = QQ[QQ(1) QQ(0);
           QQ(0) QQ(1);
           QQ(1) QQ(1)]
    x_true = QQ[QQ(1), QQ(2)]
    y = B * x_true
    x = EX.solve_fullcolumnQQ(B, y)
    @test B * x == y

    # Multiple right-hand sides
    Y = hcat(y, QQ(2) .* y)
    X = EX.solve_fullcolumnQQ(B, Y)
    @test B * X == Y
end
