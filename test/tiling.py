from rl.common.Tiling import Tiling

width, height = 13, 13
tiling = Tiling(width, height, number_of_tilling=4, bin=4, offset=(-3, -3))
# tiling.visualize_tilings()

state_number = 0
print(f'{state_number}=i:{11},j:{5} > {tiling.tiles(11, 5)}')
#
# for i in range(width):
#     for j in range(height):
#         print(f'{state_number}=i:{i},j:{j} > {tiling.tiles(i, j)}')
#         state_number += 1
