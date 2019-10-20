from rl.common.Tiling import Tiling

tiling = Tiling(10, 12, number_of_tilling=4, bin=4, offset=(-1, -1))
tiling.create_tilings()
tiling.visualize_tilings()
print(tiling.tile_encode(5, 6))
