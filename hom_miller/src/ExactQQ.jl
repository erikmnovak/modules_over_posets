module ExactQQ
# Exact linear algebra over QQ. If Nemo.jl is present we use fmpq_mat; otherwise
# a pure-Julia exact fallback is provided here.

const _HAVE_NEMO = try
    @eval import Nemo
    true
catch
    false
end

# ---------------- Nemo-backed implementations ----------------
if _HAVE_NEMO
    _to_fmpq_mat(A::AbstractMatrix{<:Rational{BigInt}}) = Nemo.fmpq_mat(A)

    function rankQQ(A::AbstractMatrix{<:Rational{BigInt}})
        Nemo.rank(_to_fmpq_mat(A))
    end

    function nullspaceQQ(A::AbstractMatrix{<:Rational{BigInt}})
        N = Nemo.nullspace_right(_to_fmpq_mat(A))
        Matrix{Rational{BigInt}}(N)           # columns = basis
    end

    function rrefQQ(A::AbstractMatrix{<:Rational{BigInt}})
        R, pivs = Nemo.rref(_to_fmpq_mat(A))
        Matrix{Rational{BigInt}}(R), Vector{Int}(pivs)
    end

    # Solve B * x = y with B full column rank
    function solve_fullcolumnQQ(B::AbstractMatrix{<:Rational{BigInt}},
                                y::AbstractVector{<:Rational{BigInt}})
        m, n = size(B)
        By = hcat(B, y)
        R, pivs = rrefQQ(By)
        x = zeros(Rational{BigInt}, n)
        for (k, pk) in enumerate(pivs)
            if pk <= n
                x[pk] = R[k, n+1]
            end
        end
        x
    end

else
# ---------------- Pure-Julia exact fallback -------------------

    # RREF with pivot list (no mutation of input)
    function rrefQQ(A::AbstractMatrix{<:Rational{BigInt}})
        M = Matrix{Rational{BigInt}}(A)
        m, n = size(M)
        r = 0
        pivots = Int[]
        c = 1
        while r < m && c <= n
            # find pivot
            pivot = 0
            for i in r+1:m
                if M[i,c] != 0
                    pivot = i; break
                end
            end
            if pivot == 0
                c += 1
                continue
            end
            r += 1
            if pivot != r
                M[r,:], M[pivot,:] = M[pivot,:], M[r,:]
            end
            # normalize row r
            p = M[r,c]
            M[r,:] ./= p
            # eliminate column c
            for i in 1:m
                if i != r && M[i,c] != 0
                    M[i,:] .-= M[i,c] .* M[r,:]
                end
            end
            push!(pivots, c)
            c += 1
        end
        return M, pivots
    end

    function rankQQ(A::AbstractMatrix{<:Rational{BigInt}})
        R, _ = rrefQQ(A); # rank = number of nonzero rows
        r = 0
        for i in 1:size(R,1)
            if any(!iszero, R[i,:]); r += 1; end
        end
        r
    end

    # Nullspace: columns form a basis
    function nullspaceQQ(A::AbstractMatrix{<:Rational{BigInt}})
        R, piv = rrefQQ(A)
        m, n = size(R)
        free = setdiff(collect(1:n), piv)
        if isempty(free)
            return zeros(Rational{BigInt}, n, 0)
        end
        N = zeros(Rational{BigInt}, n, length(free))
        for (j, f) in enumerate(free)
            N[f, j] = 1//1
            # back-substitute pivots
            for (row, c) in enumerate(piv)
                N[c, j] = -R[row, f]
            end
        end
        N
    end

    # Solve B * x = y with B full column rank via RREF([B | y])
    function solve_fullcolumnQQ(B::AbstractMatrix{<:Rational{BigInt}},
                                y::AbstractVector{<:Rational{BigInt}})
        m, n = size(B)
        By = hcat(B, y)
        R, pivs = rrefQQ(By)
        x = zeros(Rational{BigInt}, n)
        for (k, ck) in enumerate(pivs)
            if ck <= n
                x[ck] = R[k, n+1]
            end
        end
        x
    end
end

export rankQQ, nullspaceQQ, rrefQQ, solve_fullcolumnQQ, _HAVE_NEMO
end # module
