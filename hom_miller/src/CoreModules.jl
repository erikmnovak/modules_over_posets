module CoreModules
# --------------------------------------------------------------------------------
# Core prelude for the PosetModules project
#
# What lives here:
#   - Common numeric constants / aliases used everywhere (e.g. QQ = Rational{BigInt})
#   - Feature flags for optional subsystems (axis-aligned PL backend)
#   - Optional *thin wrapper* structs to unify backends cleanly
#
# Keep this file dependency-free so it can be included at the top of the umbrella.
# --------------------------------------------------------------------------------

using LinearAlgebra, SparseArrays

# 1) Public numeric alias for exact rationals used across algebra code.
#    This keeps signatures uncluttered and centralizes exact arithmetic choices.
const QQ = Rational{BigInt}

export QQ

# 2) Feature flags (read once at load time). We gate optional backends via ENV.
#    Example usage: setenv POSETMODULES_ENABLE_PL_AXIS=true to include PLBackend.
const ENABLE_PL_AXIS = get(ENV, "POSETMODULES_ENABLE_PL_AXIS", "false") in ("1","true","TRUE")

export ENABLE_PL_AXIS

# 3) (Optional) thin wrappers for exact linear algebra backends.
#    These allow callers to hold a matrix without caring whether Nemo.jl is present.
abstract type AbstractQQMatrix end

"Plain Julia dense QQ matrix wrapper."
struct QQDense <: AbstractQQMatrix
    data::Matrix{QQ}
end

# Provide a conversion to dense QQ.
to_dense(A::QQDense) = A.data

# If Nemo is available, expose a wrapper. Consumers can dispatch on AbstractQQMatrix.
# (We don't import Nemo here to avoid a hard dependency; packages using it can.)
const HAVE_NEMO = try
    @eval import Nemo
    true
catch
    false
end

if HAVE_NEMO
    "Nemo fmpq_mat wrapper; constructed from a dense QQ matrix when desired."
    struct QQNemo <: AbstractQQMatrix
        data::Nemo.fmpq_mat
    end
    QQNemo(A::Matrix{QQ}) = QQNemo(Nemo.fmpq_mat(A))
    to_dense(A::QQNemo) = Matrix{QQ}(A.data)
end

export AbstractQQMatrix, QQDense, to_dense
if HAVE_NEMO
    export QQNemo
end

# NOTE: If you later add other *thin wrappers* (e.g., for Polyhedra backends or
#       axis-aligned shims), place their *struct definitions* here to avoid cycles.
#       Implementations that need other modules should live in those modules.






export QQ, rational_to_string, string_to_rational

"""
    rational_to_string(x::QQ) -> String

Encode a rational as `"num/den"` so it survives JSON round-trips exactly.
"""
function rational_to_string(x::QQ)
    # numerator/denominator are Base methods for Rational
    string(numerator(x), "/", denominator(x))
end

"""
    string_to_rational(s::AbstractString) -> QQ

Inverse of `rational_to_string`.
"""
function string_to_rational(s::AbstractString)
    # tolerate whitespace
    t = split(strip(s), "/")
    length(t) == 2 || error("bad QQ string: $s")
    parse(BigInt, t[1]) // parse(BigInt, t[2])
end

end # module



end # module CoreModules


