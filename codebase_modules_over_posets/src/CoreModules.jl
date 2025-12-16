module CoreModules
# -----------------------------------------------------------------------------
# Core prelude for this project
#  - QQ = Rational{BigInt} is the canonical exact scalar type
#  - Optional feature flags (e.g. for optional PL axis backend)
#  - Thin wrappers for exact linear algebra backends (optional Nemo)
#  - Exact rational <-> string helpers for serialization
# -----------------------------------------------------------------------------

using LinearAlgebra, SparseArrays

# ----- canonical field of scalars used everywhere --------------------------------
"Exact rationals used throughout (Rational{BigInt})."
const QQ = Rational{BigInt}
export QQ

# ----- feature flags --------------------------------------------------------------
"Enable optional axis-aligned PL backend if set to 1/true in ENV."
const ENABLE_PL_AXIS = get(ENV, "POSETMODULES_ENABLE_PL_AXIS", "false") in ("1","true","TRUE")
export ENABLE_PL_AXIS

# ----- (optional) thin wrappers so callers can hold 'a QQ matrix' abstractly -----
abstract type AbstractQQMatrix end

"Plain Julia dense QQ matrix wrapper."
struct QQDense <: AbstractQQMatrix
    data::Matrix{QQ}
end

"Extract a dense matrix of QQ from any AbstractQQMatrix."
to_dense(A::QQDense) = A.data

# Optional Nemo wrapper (no hard dependency)
const HAVE_NEMO = try
    @eval import Nemo
    true
catch
    false
end
if HAVE_NEMO
    "Nemo fmpq_mat wrapper; construct via QQNemo(A::Matrix{QQ})."
    struct QQNemo <: AbstractQQMatrix
        data::Nemo.fmpq_mat
    end
    QQNemo(A::Matrix{QQ}) = QQNemo(Nemo.fmpq_mat(A))
    to_dense(A::QQNemo) = Matrix{QQ}(A.data)
    export QQNemo
end
export AbstractQQMatrix, QQDense, to_dense, HAVE_NEMO

# ----- exact rational <-> string (for JSON round-trips) --------------------------
"Encode a rational as \"num/den\" so it survives JSON round-trips exactly."
rational_to_string(x::QQ) = string(numerator(x), "/", denominator(x))

"Inverse of `rational_to_string`."
function string_to_rational(s::AbstractString)::QQ
    t = split(strip(s), "/")
    length(t) == 2 || error("bad QQ string: $s")
    parse(BigInt, t[1]) // parse(BigInt, t[2])
end
export rational_to_string, string_to_rational

end # module


