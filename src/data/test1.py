import pandas as pd
from pathlib import Path

# 1. 随便读取一个青铜层文件
f = next(Path('data/bronze/LEFT').glob('*.parquet'))
df = pd.read_parquet(f)

# 2. 模拟归一化
x0, y0 = df['position_x'].iloc[0], df['position_y'].iloc[0]
print(f"原始起点: ({x0:.2f}, {y0:.2f})")

# 3. 打印前 5 行的相对坐标
rel_df = df[['position_x', 'position_y']] - [x0, y0]
print("\n相对坐标前 5 行 (模型真正看到的数据):")
print(rel_df.head())