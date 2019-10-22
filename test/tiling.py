from rl.common.Tiling import Tiling

tiling = Tiling(12, 12, number_of_tilling=1, bin=4, offset=(-3, -3))
tiling.create_tilings()
tiling.visualize_tilings()

for i in range(12):
    for j in range(12):
        print(f'i:{i},j:{j} > {tiling.encode(i, j)}')
