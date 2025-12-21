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


       @testset "rrefQQ / colspaceQQ / solve_fullcolumnQQ edge cases" begin
              # A has rank 2: second row is 2*first, third row breaks full dependence.
              A = QQ[1 2 3;
                     2 4 6;
                     1 1 1]
              R, piv = EX.rrefQQ(A)

              @test piv == [1, 2]
              @test R == QQ[1 0 -1;
                            0 1  2;
                            0 0  0]

              # colspaceQQ should return exactly the pivot columns of A.
              C = EX.colspaceQQ(A)
              @test size(C) == (3, 2)
              @test C == A[:, piv]

              # solve_fullcolumnQQ should reject matrices that are not full column rank.
              B = QQ[1 2;
                     2 4;
                     3 6]
              b = QQ[1, 2, 3]
              @test_throws ErrorException EX.solve_fullcolumnQQ(B, b)
       end
end