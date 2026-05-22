import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# -----------------------------
# 頁面設定
# -----------------------------
st.set_page_config(
    page_title="AI 股票健診雷達",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 AI 股票智能健診雷達")
st.markdown("整合 PEG、MACD、RSI、ATR、布林通道 的量化分析系統")

# -----------------------------
# Sidebar
# -----------------------------
market_type = st.sidebar.selectbox(
    "市場",
    ["台股", "美股"]
)

ticker_input = st.sidebar.text_input(
    "股票代號",
    value="2330.TW" if market_type == "台股" else "NVDA"
).upper()

period = st.sidebar.slider("歷史月份", 3, 24, 6)

# -----------------------------
# 抓資料
# -----------------------------
@st.cache_data(ttl=900)
def get_stock_data(symbol, months):

    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=months * 30 + 120)

        df = yf.download(
            symbol,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False
        )

        if df.empty:
            return None, None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        ticker = yf.Ticker(symbol)

        try:
            info = ticker.info
        except:
            info = {}

        return df, info

    except:
        return None, None

# -----------------------------
# 技術指標
# -----------------------------
def calculate_indicators(df):

    df = df.copy()

    # MA
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA25'] = df['Close'].rolling(25).mean()
    df['MA50'] = df['Close'].rolling(50).mean()

    # Volume
    df['Vol20'] = df['Volume'].rolling(20).mean()

    # RSI
    delta = df['Close'].diff()

    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / (avg_loss + 1e-9)

    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()

    df['MACD'] = ema12 - ema26
    df['Signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal']

    # ATR
    high_low = df['High'] - df['Low']
    high_close = abs(df['High'] - df['Close'].shift())
    low_close = abs(df['Low'] - df['Close'].shift())

    ranges = pd.concat([high_low, high_close, low_close], axis=1)

    true_range = ranges.max(axis=1)

    df['ATR'] = true_range.rolling(14).mean()

    # 布林
    std20 = df['Close'].rolling(20).std()

    df['BB_Upper'] = df['MA20'] + 2 * std20
    df['BB_Lower'] = df['MA20'] - 2 * std20

    return df.dropna()

# -----------------------------
# PEG 計算
# -----------------------------
def calculate_peg(info):

    # PEG
    peg_ratio = info.get("pegRatio")

    if peg_ratio is not None and peg_ratio > 0:
        return round(peg_ratio, 2)

    # PE fallback
    pe = (
        info.get("forwardPE")
        or info.get("trailingPE")
    )

    # 成長率 fallback
    growth = (
        info.get("earningsGrowth")
        or info.get("revenueGrowth")
    )

    # 防呆
    if pe is None:
        return None

    if growth is None:
        return None

    # Yahoo 是小數
    # 0.25 = 25%
    growth_percent = growth * 100

    # 避免負值與過小值
    if growth_percent <= 0:
        return None

    if growth_percent < 3:
        return None

    peg = pe / growth_percent

    # 避免異常值
    if peg <= 0 or peg > 10:
        return None

    return round(peg, 2)

# -----------------------------
# 主程式
# -----------------------------
if st.sidebar.button("開始分析"):

    with st.spinner("分析中..."):

        df, info = get_stock_data(ticker_input, period)

        if df is None:

            st.error("找不到股票資料")
            st.stop()

        df = calculate_indicators(df)

        last = df.iloc[-1]

        last_price = float(last['Close'])
        last_ma25 = float(last['MA25'])
        last_ma50 = float(last['MA50'])
        last_rsi = float(last['RSI'])
        last_macd_hist = float(last['MACD_Hist'])

        # PEG
        peg_ratio = calculate_peg(info)

        # 公司名
        company_name = (
            info.get("longName")
            or info.get("shortName")
            or ticker_input
        )

        st.subheader(f"{company_name} ({ticker_input})")

        # -----------------------------
        # Metrics
        # -----------------------------
        col1, col2, col3, col4 = st.columns(4)

        prev_close = float(df['Close'].iloc[-2])

        change = last_price - prev_close
        pct = (change / prev_close) * 100

        col1.metric(
            "股價",
            f"{last_price:.2f}",
            f"{pct:.2f}%"
        )

        col2.metric(
            "RSI",
            f"{last_rsi:.1f}"
        )

        col3.metric(
            "ATR",
            f"{float(last['ATR']):.2f}"
        )

        # PEG 顯示
        if peg_ratio is not None:

            if peg_ratio < 1:
                peg_text = "低估"
            elif peg_ratio < 1.5:
                peg_text = "合理"
            else:
                peg_text = "偏貴"

            col4.metric(
                "PEG",
                f"{peg_ratio:.2f}",
                peg_text
            )

        else:
            col4.metric(
                "PEG",
                "N/A",
                "無法計算"
            )

        # -----------------------------
        # AI Score
        # -----------------------------
        st.markdown("---")

        score = 0

        details = []

        # PEG
        if peg_ratio is not None:

            if peg_ratio < 1:
                score += 20
                details.append("✅ PEG 小於 1，估值具吸引力")

            elif peg_ratio < 1.5:
                score += 10
                details.append("🟡 PEG 合理")

            else:
                details.append("❌ PEG 偏高")

        # 趨勢
        if last_price > last_ma25 > last_ma50:
            score += 20
            details.append("✅ 多頭排列")

        # RSI
        if 45 <= last_rsi <= 65:
            score += 20
            details.append("✅ RSI 健康")

        # MACD
        if last_macd_hist > 0:
            score += 20
            details.append("✅ MACD 多頭")

        # Volume
        if last['Volume'] > last['Vol20']:
            score += 20
            details.append("✅ 量能增強")

        # Score Result
        if score >= 80:
            st.success(f"總分：{score}/100")
        elif score >= 60:
            st.warning(f"總分：{score}/100")
        else:
            st.error(f"總分：{score}/100")

        for d in details:
            st.write(d)

        # -----------------------------
        # 圖表
        # -----------------------------
        st.markdown("---")

        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.6, 0.2, 0.2]
        )

        # K線
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='K'
            ),
            row=1,
            col=1
        )

        # MA
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['MA25'],
                name='MA25'
            ),
            row=1,
            col=1
        )

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['MA50'],
                name='MA50'
            ),
            row=1,
            col=1
        )

        # Volume
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['Volume'],
                name='Volume'
            ),
            row=2,
            col=1
        )

        # MACD
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['MACD_Hist'],
                name='MACD Hist'
            ),
            row=3,
            col=1
        )

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['MACD'],
                name='MACD'
            ),
            row=3,
            col=1
        )

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['Signal'],
                name='Signal'
            ),
            row=3,
            col=1
        )

        fig.update_layout(
            height=850,
            template="plotly_white",
            xaxis_rangeslider_visible=False
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")

st.caption("本工具僅供研究參考，非投資建議")
