import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# --- 網頁配置 ---
st.set_page_config(page_title="AI 股票健診雷達", page_icon="🚀", layout="wide")
st.title("🚀 AI 股票智能健診雷達 (量化實戰特仕版)")
st.markdown("結合**動態 AI 評分、ATR 停損、MACD 動能、布林通道與 PEG 估值**，並嚴格執行時間與價格紀律。")

# --- 側邊欄：輸入參數與實戰紀律 ---
st.sidebar.header("🔍 查詢設定")
market_type = st.sidebar.selectbox("🌍 選擇市場", ["台股 (需加 .TW 或 .TWO)", "美股"])
ticker_input = st.sidebar.text_input("📝 輸入股票代號", value="2330.TW" if "台股" in market_type else "NVDA").upper()
period = st.sidebar.slider("📅 查看歷史期間 (月)", 3, 24, 6)

# 🚀 實戰紀律守則
st.sidebar.markdown("---")
st.sidebar.error("⚔️ **操盤手鐵律：資金與停損控管**")
st.sidebar.info(
    "1. **集中火力**：持股不超過 4 檔，挑選前四名操作。\n"
    "2. **資金配置**：每檔押 15%-20%，保留 30% 現金用來加碼最強勢個股。\n"
    "3. **價格停損**：跌破 5 週線 (約25MA) 警戒，底線為 10 週線 (約50MA)，破線無條件出場。\n"
    "4. **時間停損**：進場後 3 個月不漲，直接汰弱留強。"
)

# 動態決定 K 線顏色
if "台股" in market_type:
    color_up, color_down = '#FF3333', '#00AA00'
else:
    color_up, color_down = '#00AA00', '#FF3333'

# --- 核心邏輯函數 ---
@st.cache_data(ttl=900, show_spinner=False)
def get_stock_data(symbol, months):
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=months * 30 + 100)
        df = yf.download(symbol, start=start_date, end=end_date, progress=False)
       
        if df.empty: return None, None, None
       
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
           
        ticker_obj = yf.Ticker(symbol)
        try:
            info = ticker_obj.info
        except:
            info
