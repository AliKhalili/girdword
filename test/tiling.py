from rl.common.Tiling import Tiling

tiling = Tiling(12, 12, number_of_tilling=4, bin=4, offset=(0, -3))
tiling.visualize_tilings(tiling.create_tilings())
