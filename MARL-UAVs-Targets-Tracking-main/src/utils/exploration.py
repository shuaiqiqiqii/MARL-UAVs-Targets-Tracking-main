import numpy as np

class GridExplorationBonus:
    def __init__(self, x_max, y_max, grid_size=100, bonus_weight=0.01, decay=0.9995):
        self.grid_size = grid_size
        self.bonus_weight = bonus_weight
        self.decay = decay
        # 直接初始化网格尺寸和计数器，不依赖 resize
        self.x_cells = int(x_max / self.grid_size) + 1
        self.y_cells = int(y_max / self.grid_size) + 1
        self.counts = np.zeros((self.x_cells, self.y_cells), dtype=np.float32)

    def get_bonus(self, x, y):
        ix = min(int(x / self.grid_size), self.x_cells - 1)
        iy = min(int(y / self.grid_size), self.y_cells - 1)
        count = self.counts[ix, iy]
        bonus = self.bonus_weight / (1.0 + count)
        return bonus

    def update(self, x, y):
        ix = min(int(x / self.grid_size), self.x_cells - 1)
        iy = min(int(y / self.grid_size), self.y_cells - 1)
        self.counts[ix, iy] += 1.0
        self.counts *= self.decay   # 全局衰减

    def resize(self, new_x_max, new_y_max):
        new_x_cells = int(new_x_max / self.grid_size) + 1
        new_y_cells = int(new_y_max / self.grid_size) + 1
        # 创建新数组，用默认值填充（可选2.0表示中等新颖度，或0.0表示完全新颖）
        new_counts = np.full((new_x_cells, new_y_cells), 2.0, dtype=np.float32)
        # 复制旧区域已有的访问计数
        old_x = min(self.x_cells, new_x_cells)
        old_y = min(self.y_cells, new_y_cells)
        new_counts[:old_x, :old_y] = self.counts[:old_x, :old_y]
        # 更新属性
        self.counts = new_counts
        self.x_cells = new_x_cells
        self.y_cells = new_y_cells

    def set_weight(self, weight):
        self.bonus_weight = weight