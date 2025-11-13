module Viz2D
# ------------------------------------------------------------------------------
# Lightweight 2D visualization helpers for Z^n flanges.
# We expose an ASCII-only API and avoid heavy requirements; the only optional
# dependency used here is Plots.jl (for a heatmap). If it's missing, we throw
# a friendly error telling the user to `] add Plots`.
#
# What we draw: a heatmap of  g \mapsto dim M_g  on the integer grid inside a
# convex-projection bounding box.  On Z^2, this provides a quick visual sanity
# check for the module defined by the flange presentation (cf. Ex. 4.5 + section5).   [Miller]
# ------------------------------------------------------------------------------

import ..FlangeZn: Flange, bounding_box, dim_at

export heatmap_flange_2d

"""
    heatmap_flange_2d(fr::Flange; margin::Int=1, rankfun=rank) -> any

Compute a bounding box `[a,b]` (inflated by `margin`) and plot the heatmap of
`dim M_g = rank(\phi_g)` for `g \in Z^2 \cap [a,b]`.  Requires `Plots.jl`.
Returns the Plots.jl object.
"""
function heatmap_flange_2d(fr::Flange; margin::Int=1, rankfun=rank)
    try
        import Plots
    catch
        error("Viz2D.heatmap_flange_2d requires Plots.jl. Run:  ] add Plots")
    end

    fr.n == 2 || error("heatmap_flange_2d is implemented only for n = 2")

    a, b = bounding_box(fr; margin=margin)   # heuristic finite window
    xs = a[1]:b[1]; ys = a[2]:b[2]

    # Z[y,x] layout (Plots.heatmap expects row-major as y, column-major as x)
    Z = Array{Int}(undef, length(ys), length(xs))
    for (iy, y) in enumerate(ys), (ix, x) in enumerate(xs)
        Z[iy, ix] = dim_at(fr, [x,y]; rankfun=rankfun)
    end

    plt = Plots.heatmap(xs, ys, Z; xlabel="g1", ylabel="g2", colorbar_title="dim M_g",
                        title="Flange heatmap on Z^2 (dim image at each degree)")
    return plt
end

end # module
