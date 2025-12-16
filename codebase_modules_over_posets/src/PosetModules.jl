module PosetModules
# =============================================================================
# Umbrella module for the library.
# Include order matters:
#   1) Core helpers (numeric aliases, feature flags, thin wrappers)
#   2) Finite poset layer (Fringe + indicator sets)
#   3) Presentation/copresentation record types used across subsystems
#   4) Exact rational linear algebra (exact RREF, rank, etc.)  <-- self-contained
#   5) Finite encodings (uptight poset)
#   6) Indicator resolutions and Hom/Ext assembly
#   7) Zn flange layer (flats/injectives) and CAS bridge
#   8) IO (JSON serializations) and 2D visualization
#   9) PL polyhedra backend (+ cross-validation against flange)
# =============================================================================

# 1) Core helpers and feature flags
include("CoreModules.jl")

# 2) Finite poset + indicator sets + fringe presentations
include("FiniteFringe.jl")

# 3) Shared record types for one-step indicator (co)presentations
include("IndicatorTypes.jl")

# 4) Exact rational linear algebra (Nemo if available, fallback otherwise)
#    IMPORTANT: our ExactQQ fallback is fully self-contained; no import from IndicatorResolutions.
include("ExactQQ.jl")

# 5) Finite encodings (uptight poset; image/preimage of up/downsets)
include("Encoding.jl")

# 6) Indicator resolutions + Hom/Ext (first page + general Tot^*)
include("HomExt.jl")
include("IndicatorResolutions.jl")

# 7) Zn flange data structure and CAS bridge
include("FlangeZn.jl")
include("M2SingularBridge.jl")

# 8) Serialization and lightweight visualization
include("Serialization.jl")
include("Viz2D.jl")

# 9) PL backend (general H-rep) and cross-validation helpers
include("PLPolyhedra.jl")
include("CrossValidateFlangePL.jl")

# Optional: axis-aligned PL backend (kept separate; off by default)
# Enable with ENV: POSETMODULES_ENABLE_PL_AXIS=true
if CoreModules.ENABLE_PL_AXIS
    @info "Including experimental PLBackend (axis-aligned) because ENABLE_PL_AXIS=true"
    include("PLBackend.jl")
end

# ----------------------- Re-exports for a clean user-facing API ----------------
using .CoreModules
using .FiniteFringe
using .IndicatorTypes
using .ExactQQ
using .Encoding
using .IndicatorResolutions
using .HomExt
using .FlangeZn
using .M2SingularBridge
using .Serialization
using .Viz2D
using .PLPolyhedra
using .CrossValidateFlangePL

if CoreModules.ENABLE_PL_AXIS
    using .PLBackend
end

# Bring specific symbols into our namespace so we can re-export them.
import .HomExt: build_hom_tot_complex, ext_dims_via_resolutions

# Finite poset + fringe presentation
export FinitePoset, Upset, Downset, FringeModule,
       upset_from_generators, downset_from_generators, cover_edges, fiber_dimension,
       hom_dimension, dense_to_sparse_K

# Encoding / uptight posets
export EncodingMap, UptightEncoding,
       build_uptight_encoding_from_fringe, pullback_fringe_along_encoding,
       image_upset, image_downset, preimage_upset, preimage_downset

# Indicator one-step data (types)
export UpsetPresentation, DownsetCopresentation

# Indicator resolutions + Hom/Ext (API)
export upset_presentation_one_step, downset_copresentation_one_step,
       hom_ext_first_page, build_hom_tot_complex, ext_dims_via_resolutions

# Zn flange + query + bridge
export Face, IndFlat, IndInj, Flange, canonical_matrix, dim_at, bounding_box,
       parse_flange_json, flange_from_m2

# Serialization and 2D viz
export save_flange_json, load_flange_json,
       save_encoding_json, load_encoding_json,
       draw_flange_regions2D, draw_constant_subdivision2D

# PL encoders + cross validation
export HPoly, PolyUnion, PLUpset, PLDownset, PLEncodingMap, locate, make_hpoly,
       encode_from_PL_fringe, cross_validate

# Optional experimental (axis-aligned PL) exports
if CoreModules.ENABLE_PL_AXIS
    export BoxUpset, BoxDownset, PLEncodingMapBoxes, encode_fringe_boxes
end

end # module
