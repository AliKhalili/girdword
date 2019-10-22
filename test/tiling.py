from rl.common.Tiling import Tiling

width, height = 4, 4
tiling = Tiling(width, height, number_of_tilling=1, bin=2, offset=(0, 0))
#tiling.visualize_tilings()

state_number = 0
for i in range(width):
    for j in range(height):
        print(f'{state_number}=i:{i},j:{j} > {tiling.encode(i, j)}')
        state_number += 1
