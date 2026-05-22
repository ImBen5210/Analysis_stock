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

# --- 側邊欄 ---
st.sidebar.header("🔍 查詢設定")
market_type = st.sidebar.selectbox("🌍 選擇市場", ["台股 (需加 .TW 或 .TWO)", "美股"])
ticker_input = st.sidebar.text_input("📝 輸入股票代號", value="2330.TW" if "台股" in market_type else "NVDA").upper().strip()
period = st.sidebar.slider("📅 查看歷史期間 (月)", 3, 24, 6)

# --- 核心函數 ---
@st.cache_data(ttl=600, show_spinner=False)
def get_stock_data(symbol):
    try:
        # 下載資料
        df = yf.download(symbol, period="2y", progress=False)
        if df.empty: return None, None, None

        # 降維：解決 MultiIndex KeyError 問題
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        ticker_obj = yf.Ticker(symbol)
        info = ticker_obj.info or {}
        
        # 強制計算 PEG
        price = info.get('currentPrice') or info.get('regularMarketPrice') or df['Close'].iloc[-1]
        pe = info.get('trailingPE') or info.get('forwardPE')
        growth = info.get('earningsGrowth') or info.get('revenueGrowth') or 0
        peg = info.get('pegRatio')
        
        if (peg is None or peg == 0) and pe and growth > 0:
            peg = pe / (growth * 100)
            
        valuation = {'pe': pe, 'peg': peg, 'growth': growth, 'price': price}
        return df, info, valuation
    except: return None, None, None

def calculate_indicators(df):
    df = df.copy()
    # 確保是單層索引且名稱正確
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    
    df['MA25'] = df['Close'].rolling(25).mean() # 5週
    df['MA50'] = df['Close'].rolling(50).mean() # 10週
    
    df['Resistance'] = df['High'].shift(1).rolling(20).max()
    df['Support'] = df['Low'].shift(1).rolling(20).min()
    
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + gain / (loss + 1e-9)))
    
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD_Hist'] = (exp1 - exp2) - (exp1 - exp2).ewm(span=9, adjust=False).mean()
    return df.dropna()

# --- 主程式 ---
if st.sidebar.button("啟動健診分析 🎯") or ticker_input:
    with st.spinner('分析中...'):
        df, info, val = get_stock_data(ticker_input)
        if df is not None:
            df = calculate_indicators(df)
            last = df.iloc[-1]
            
            st.subheader(f"📊 {info.get('longName', ticker_input)} 健診報告")
            
            # 第一排數據
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("收盤價", f"{last['Close']:.2f}")
            c2.metric("PEG 估值", f"{val['peg']:.2f}" if val['peg'] else "N/A")
            c3.metric("5週線", f"{last['MA25']:.2f}")
            c4.metric("10週線", f"{last['MA50']:.2f}")

            # 紀律審查
            st.markdown("---")
            st.subheader("⚔️ 嚴格紀律審查")
            if last['Close'] < last['MA50']:
                st.error("💀 跌破 10 週線底限，無條件停損！")
            elif last['Close'] < last['MA25']:
                st.warning("🚨 跌破 5 週線警戒，請提高警覺。")
            
            # 圖表
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='K線'))
            fig.add_trace(go.Scatter(x=df.index, y=df['MA25'], name='5週線', line=dict(color='orange')))
            fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], name='10週線', line=dict(color='red', dash='dot')))
            fig.update_layout(height=600, template="plotly_white", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("❌ 無法抓取資料，請確認代號（如 2330.TW）。")
