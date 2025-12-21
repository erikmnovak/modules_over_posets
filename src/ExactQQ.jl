module ExactQQ
# Exact linear algebra over QQ. If Nemo.jl is present we use fmpq_mat; otherwise
# a pure-Julia exact fallback is provided here.  ASCII-only version.

using ..CoreModules: QQ

const _HAVE_NEMO = try
    @eval import Nemo
    true
catch
    false
end

if _HAVE_NEMO
    # Nemo-backed implementations
    _to_fmpq_mat(A::AbstractMatrix{<:QQ}) = Nemo.fmpq_mat(A)

    """
    rrefQQ(A) -> (R, pivs)
    Reduced row echelon form over QQ and the 1-based pivot-column indices.
    """
    function rrefQQ(A::AbstractMatrix{<:QQ})
        R_fmpq, pivs = Nemo.rref(_to_fmpq_mat(A))
        R = Matrix{QQ}(R_fmpq)
        return R, Vector{Int}(pivs)
    end

    """
    rankQQ(A) -> Integer
    Rank over QQ computed from the pivot columns returned by rrefQQ.
    """
    function rankQQ(A::AbstractMatrix{<:QQ})
        _, pivs = rrefQQ(A)
        return length(pivs)
    end

    """
    nullspaceQQ(A) -> Matrix{QQ}
    Columns form a QQ-basis of the nullspace of A.
    """
    function nullspaceQQ(A::AbstractMatrix{<:QQ})
        R, pivs = rrefQQ(A)
        m, n = size(R)
        pivset = Set(pivs)
        free = [j for j in 1:n if !(j in pivset)]
        if isempty(free)
            return zeros(QQ, n, 0)
        end
        N = zeros(QQ, n, length(free))
        for (col, j) in enumerate(free)
            N[j, col] = 1//1
            for (row, pj) in enumerate(pivs)
                N[pj, col] = -R[row, j]
            end
        end
        return N
    end

    """
    colspaceQQ(A) -> Matrix{QQ}
    Columns form a QQ-basis of the column space of A.
    """
    function colspaceQQ(A::AbstractMatrix{<:QQ})
        _, pivs = rrefQQ(A)
        if isempty(pivs)
            return zeros(QQ, size(A,1), 0)
        end
        return Matrix{QQ}(A[:, pivs])
    end

    """
    solve_fullcolumnQQ(B, y) -> x
    Solve B * x = y over QQ assuming B has full column rank.
    """
    function solve_fullcolumnQQ(B::AbstractMatrix{<:QQ},
                            y::AbstractVector{<:QQ})
        m, n = size(B)
        length(y) == m || error("solve_fullcolumnQQ: dimension mismatch")

        # One RREF pass on the augmented matrix.
        By = hcat(B, reshape(y, m, 1))
        R, pivs = rrefQQ(By)

        # Enforce: B has full column rank and y lies in col(B).
        pivB = [p for p in pivs if p <= n]
        (length(pivB) == n) || error("solve_fullcolumnQQ: B must have full column rank")
        any(p > n for p in pivs) && error("solve_fullcolumnQQ: right-hand side is not in the column space of B")

        x = zeros(QQ, n)
        for (row, pj) in enumerate(pivs)
            if pj <= n
                x[pj] = R[row, n + 1]
            end
        end
        return x
    end


    """
    solve_fullcolumnQQ(B, Y) -> X
    Solve B * X = Y with multiple right-hand sides assuming B has full column rank.
    """
    function solve_fullcolumnQQ(B::AbstractMatrix{<:QQ},
                            Y::AbstractMatrix{<:QQ})
        m, n = size(B)
        k = size(Y, 2)
        size(Y, 1) == m || error("solve_fullcolumnQQ: dimension mismatch")

        By = hcat(B, Y)
        R, pivs = rrefQQ(By)

        pivB = [p for p in pivs if p <= n]
        (length(pivB) == n) || error("solve_fullcolumnQQ: B must have full column rank")
        any(p > n for p in pivs) && error("solve_fullcolumnQQ: at least one RHS column is not in the column space of B")

        X = zeros(QQ, n, k)
        for (row, pj) in enumerate(pivs)
            if pj <= n
                for j in 1:k
                    X[pj, j] = R[row, n + j]
                end
            end
        end
        return X
    end


else
    # Pure Julia exact routines (no Nemo)
    "Reduced row echelon form with pivot columns."
    function rrefQQ(A::AbstractMatrix{<:QQ})
        M = Matrix{QQ}(A); m, n = size(M)
        pivs = Int[]; r = 0; c = 1
        while r < m && c <= n
            pivot = 0
            for i in (r+1):m
                if !iszero(M[i,c]); pivot = i; break; end
            end
            if pivot == 0; c += 1; continue; end
            r += 1
            if pivot != r; M[r,:], M[pivot,:] = M[pivot,:], M[r,:]; end
            p = M[r,c]; M[r,:] ./= p
            for i in 1:m
                if i != r && !iszero(M[i,c]); M[i,:] .-= M[i,c] .* M[r,:]; end
            end
            push!(pivs, c)
            c += 1
        end
        return M, pivs
    end

    "Nullspace as a matrix with columns = basis vectors."
    function nullspaceQQ(A::AbstractMatrix{<:QQ})
        R, pivs = rrefQQ(A)
        m, n = size(R); pivset = Set(pivs)
        free = [j for j in 1:n if !(j in pivset)]
        if isempty(free); return zeros(QQ, n, 0); end
        N = zeros(QQ, n, length(free))
        for (col, j) in enumerate(free)
            N[j, col] = 1//1
            for (row, pj) in enumerate(pivs)
                N[pj, col] = -R[row, j]
            end
        end
        N
    end

    "Rank via RREF."
    function rankQQ(A::AbstractMatrix{<:QQ})
        _, pivs = rrefQQ(A); length(pivs)
    end

    "Column space basis via pivot columns."
    function colspaceQQ(A::AbstractMatrix{<:QQ})
        R, pivs = rrefQQ(Matrix{QQ}(A))
        if isempty(pivs)
            return zeros(QQ, size(A, 1), 0)
        end
        return Matrix{QQ}(A[:, pivs])
    end

    "Solve B * X = Y with B full column rank. Y can be a vector or a matrix."
    "Solves B * X = Y with B full column rank. Y can be a vector or a matrix."
    function solve_fullcolumnQQ(B::AbstractMatrix{<:QQ}, Y::AbstractVecOrMat{<:QQ})
        m, n = size(B)

        # Convert RHS to a QQ matrix with correct shape.
        want_vec = false
        Ymat = if Y isa AbstractVector
            length(Y) == m || error("solve_fullcolumnQQ: dimension mismatch")
            want_vec = true
            reshape(Vector{QQ}(Y), m, 1)
        else
            size(Y, 1) == m || error("solve_fullcolumnQQ: dimension mismatch")
            Matrix{QQ}(Y)
        end

        k = size(Ymat, 2)
        By = hcat(Matrix{QQ}(B), Ymat)
        R, pivs = rrefQQ(By)

        pivB = [p for p in pivs if p <= n]
        (length(pivB) == n) || error("solve_fullcolumnQQ: B must have full column rank")
        any(p > n for p in pivs) && error("solve_fullcolumnQQ: at least one RHS column is not in the column space of B")



        X = zeros(QQ, n, k)
        for (row, pj) in enumerate(pivs)
            if pj <= n
                for j in 1:k
                    X[pj, j] = R[row, n + j]
                end
            end
        end

        return want_vec ? vec(X) : X
    end

end

export rankQQ, nullspaceQQ, colspaceQQ, rrefQQ, solve_fullcolumnQQ, _HAVE_NEMO
end # module
