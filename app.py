import yfinance as yf

t = yf.Ticker("NVDA")  # 或換成 2330.TW

# 測試 1: info 裡有哪些跟 PE/成長有關的 key
info = t.info
keywords = ['pe', 'growth', 'eps', 'earn', 'peg', 'revenue', 'profit']
print("=== info 相關欄位 ===")
for k, v in info.items():
    if any(kw in k.lower() for kw in keywords):
        print(f"  {k}: {v}")

# 測試 2: 財報 API 哪些有資料
print("\n=== quarterly_earnings ===")
try:
    print(t.quarterly_earnings)
except Exception as e:
    print(f"失敗: {e}")

print("\n=== income_stmt (前2行) ===")
try:
    print(t.income_stmt.iloc[:2])
except Exception as e:
    print(f"失敗: {e}")

print("\n=== quarterly_income_stmt (前2行) ===")
try:
    print(t.quarterly_income_stmt.iloc[:2])
except Exception as e:
    print(f"失敗: {e}")
