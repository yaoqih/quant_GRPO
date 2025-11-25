import qlib
from qlib.data import D

# 1. 初始化 Qlib
# 注意：将 path/to/your/data 替换为你实际的数据目录路径
provider_uri = 'data/qlib_1d_c' 
qlib.init(provider_uri=provider_uri)

print("------ 初始化成功，开始测试 ------")

# 2. 测试日历读取
try:
    calendar = D.calendar(start_time='2023-01-01', end_time='2023-01-10')
    print(f"✅ 日历读取成功: 2023-01-01 到 2023-01-10 共 {len(calendar)} 个交易日")
except Exception as e:
    print(f"❌ 日历读取失败: {e}")

# 3. 测试具体数据读取 (以平安银行 sz000001 或 茅台 sh600519 为例)
test_stock = 'sh600519' # 确保这个代码在你的 instruments/all.txt 里
fields = ['$close', '$volume', '$open']

try:
    df = D.features([test_stock], fields, start_time='2023-01-01', end_time='2023-02-01')
    if not df.empty:
        print(f"✅ 数据读取成功 ({test_stock}):")
        print(df.head())
    else:
        print(f"⚠️ 数据为空: 请检查日期范围或股票代码是否存在。")
except Exception as e:
    print(f"❌ 数据读取报错: {e}")