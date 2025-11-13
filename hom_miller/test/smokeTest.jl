using PosetModules   # your umbrella module

# Build a tiny 2D flange over QQ by hand:
K = Rational{BigInt}
τfree_x  = Face(2, [true,  false])   # x free, y bounded
τfree_y  = Face(2, [false, true])    # y free, x bounded
τnone    = Face(2, [false,false])

F1 = IndFlat{K}([0,0], τnone,   :F1)    # g ≥ (0,0)
F2 = IndFlat{K}([2,0], τfree_y, :F2)    # x ≥ 2; y free
E1 = IndInj{K}([1,3], τnone,   :E1)     # g ≤ (1,3)
E2 = IndInj{K}([4,0], τfree_x, :E2)     # y ≤ 0; x free

Φ  = canonical_matrix([F1,F2], [E1,E2]) # 0/1 pattern allowed by intersections
FG = Flange{K}(2, [F1,F2], [E1,E2], Φ)

# Dimension at a few grades g ∈ Z^2:
@assert dim_at(FG, [0,0]) == 1
@assert dim_at(FG, [3,0]) ≥ 0

# Cross-validate with the PL backend on a small box:
ok, rep = CrossValidateFlangePL.cross_validate(FG; margin=0)
@show ok, rep
