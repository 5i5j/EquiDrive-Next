import pandas as pd
from pathlib import Path

# 采样一个文件
sample_path = next(Path("data/landing/train").glob("*.parquet"))
df = pd.read_parquet(sample_path)

# 查看是否有与地图相关的 ID 字段
map_related_cols = [col for col in df.columns if 'lane' in col.lower() or 'segment' in col.lower()]
print(f"📄 文件: {sample_path.name}")
print(f"🧭 地图相关字段: {map_related_cols}")
print(f"📊 数据前五行:\n", df[map_related_cols].head())