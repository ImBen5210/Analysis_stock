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
st.markdown("結合動態 AI 評分、技術指標與估值分析")

# --- 側邊欄 ---
st.sidebar.header("🔍 查詢設定")
market_type = st.sidebar.selectbox("🌍 選擇市場", ["台股 (需加 .TW 或 .TWO)", "美股"])
ticker_input = st.sidebar.text_input("📝 輸入股票代號", 
                                     value="2330.TW" if "台股" in market_type else "NVDA").upper()
period = st.sidebar.slider("📅 查看歷史期間 (月)", 3, 24, 6)

st.sidebar.markdown("---")
st.sidebar.error("⚔️ **操盤手鐵律**")
st.sidebar.info("1. 持股集中\n2. 資金控管\n3. 破5週線警戒、破10週線出場\n4. 3個月不漲汰弱")

if "台股" in market_type:
    color_up, color_down = '#FF3333', '#00AA00'
else:
    color_up, color_down = '#00AA00', '#FF3333'

# --- 核心函數 ---
@st.cache_data(ttl=600, show_spinner=False)
def get_stock_data(symbol, months):
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=months * 30 + 100)
        
        df = yf.download(symbol, start=start_date, end=end_date, progress=False)
        if df.empty:
            return None, None, None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        ticker_obj = yf.Ticker(symbol)
        info = ticker_obj.info or {}

        current_price = info.get('currentPrice') or info.get('regularMarketPrice') or df['Close'].iloc[-1]

        # 強化抓取
        trailing_pe = info.get('trailingPE')
        if not trailing_pe and current_price and info.get('trailingEps'):
            trailing_pe = current_price / info.get('trailingEps')

        forward_pe = info.get('forwardPE')
        if not forward_pe and current_price and info.get('forwardEps'):
            forward_pe = current_price / info.get('forwardEps')

        earnings_growth = (info.get('earningsGrowth') or 
                          info.get('earningsQuarterlyGrowth') or 
                          info.get('revenueGrowth') or None)

        peg_ratio = info.get('pegRatio')

        # 手動計算 PEG
        if peg_ratio is None and forward_pe and earnings_growth and earnings_growth > 0:
            growth_rate = earnings_growth * 100 if earnings_growth < 1 else earnings_growth
            peg_ratio = forward_pe / growth_rate

        valuation = {
            'trailing_pe': trailing_pe,
            'forward_pe': forward_pe,
            'peg_ratio': peg_ratio,
            'earnings_growth': earnings_growth,
            'current_price': current_price
        }

        return df, info, valuation

    except Exception as e:
        st.error(f"資料抓取錯誤: {str(e)}")
        return None, None, None


def calculate_indicators(df):
    df = df.copy()
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA25'] = df['Close'].rolling(window=25).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['Vol20'] = df['Volume'].rolling(window=20).mean()

    df['STD20'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['MA20'] + 2 * df['STD20']
    df['BB_Lower'] = df['MA20'] - 2 * df['STD20']

    df['Resistance'] = df['High'].shift(1).rolling(window=20).max()
    df['Support'] = df['Low'].shift(1).rolling(window=20).min()

    df['TR'] = np.maximum(df['High'] - df['Low'], 
               np.maximum(abs(df['High'] - df['Close'].shift(1)), 
                          abs(df['Low'] - df['Close'].shift(1))))
    df['ATR'] = df['TR'].rolling(window=14).mean()

    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))

    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal']

    return df.dropna()

# --- 主程式 ---
if st.sidebar.button("啟動健診分析 🎯", type="primary") or ticker_input:
    with st.spinner(f'分析 {ticker_input} 中...'):
        df, info, valuation = get_stock_data(ticker_input, period)
        
        if df is not None and not df.empty:
            df = calculate_indicators(df)
            
            last_price = float(df['Close'].iloc[-1])
            last_ma25 = float(df['MA25'].iloc[-1])
            last_ma50 = float(df['MA50'].iloc[-1])
            last_vol = float(df['Volume'].iloc[-1])
            last_vol20 = float(df['Vol20'].iloc[-1])
            last_rsi = float(df['RSI'].iloc[-1])
            last_macd = float(df['MACD'].iloc[-1])
            last_hist = float(df['MACD_Hist'].iloc[-1])

            past_60d_price = float(df['Close'].iloc[-61]) if len(df) > 60 else float(df['Close'].iloc[0])
            roc_3m = ((last_price - past_60d_price) / past_60d_price) * 100

            company_name = info.get('longName', info.get('shortName', ticker_input))

            st.subheader(f"📊 {company_name} ({ticker_input})")

            col1, col2, col3, col4 = st.columns(4)
            price_change = last_price - float(df['Close'].iloc[-2])
            pct_change = (price_change / float(df['Close'].iloc[-2])) * 100 if df['Close'].iloc[-2] != 0 else 0

            col1.metric("目前收盤價", f"{last_price:.2f}", f"{price_change:.2f} ({pct_change:.2f}%)")

            if last_price < last_ma50:
                col2.metric("10週線底限", f"{last_ma50:.2f}", "⚠️ 已破，請出場", delta_color="inverse")
            elif last_price < last_ma25:
                col2.metric("5週線警戒", f"{last_ma25:.2f}", "🚨 警戒", delta_color="inverse")
            else:
                col2.metric("5週線安全", f"{last_ma25:.2f}", f"↑ {((last_price/last_ma25)-1)*100:.1f}%")

            # === PEG / PE 顯示 ===
            peg_ratio = valuation.get('peg_ratio')
            trailing_pe = valuation.get('trailing_pe')

            if peg_ratio is not None:
                col3.metric("PEG 估值", f"{peg_ratio:.2f}", 
                           "便宜" if peg_ratio < 1 else "合理" if peg_ratio < 1.5 else "偏貴")
            elif trailing_pe is not None:
                col3.metric("Trailing P/E", f"{trailing_pe:.2f}", "PEG 無分析師預估")
            else:
                col3.metric("估值", "資料不足")

            col4.metric("近3月漲跌", f"{roc_3m:.1f}%")

            # 除錯資訊
            with st.expander("🔍 原始估值數據（除錯用） - 請展開查看"):
                st.json(valuation)

            # AI 評分（調整 PEG 部分）
            st.markdown("---")
            st.subheader("🤖 AI 動態評分")
            details = []

            if peg_ratio is not None:
                if peg_ratio <= 1.0:
                    peg_score = 20
                    details.append(f"✅ **PEG 極佳**：{peg_ratio:.2f}")
                elif peg_ratio <= 1.5:
                    peg_score = 12
                    details.append(f"🟡 **PEG 合理**：{peg_ratio:.2f}")
                else:
                    peg_score = 5
                    details.append(f"⚠️ **PEG 偏高**：{peg_ratio:.2f}")
            else:
                peg_score = 8 if trailing_pe and trailing_pe < 30 else 5
                details.append(f"🔵 **PEG 無法取得**（興櫃常見），參考 Trailing P/E = {trailing_pe:.2f if trailing_pe else 'N/A'}")

            # 趨勢、量能、MACD、RSI 評分（保持原邏輯）
            bias_25 = (last_price / last_ma25 - 1) * 100 if last_ma25 != 0 else 0
            trend_score = 18 if (last_price > last_ma25 and last_ma25 > last_ma50) else 8
            details.append(f"{'✅' if trend_score > 15 else '🟡'} **趨勢**：得分 {trend_score}/20")

            vol_ratio = last_vol / last_vol20 if last_vol20 > 0 else 1
            vol_score = 15 if vol_ratio > 1.2 else 8
            details.append(f"{'✅' if vol_score > 12 else '🔵'} **量能**：得分 {vol_score}/20")

            macd_score = 20 if last_hist > 0 and last_macd > 0 else 10 if last_hist > 0 else 4
            details.append(f"{'✅' if macd_score == 20 else '🟡'} **MACD**：得分 {macd_score}/20")

            rsi_score = 20 if 45 <= last_rsi <= 65 else 15 if last_rsi < 45 else 8
            details.append(f"{'✅' if rsi_score == 20 else '🔵'} **RSI**：{last_rsi:.1f} 分 {rsi_score}/20")

            total_score = peg_score + trend_score + vol_score + macd_score + rsi_score

            st.markdown("#### 🏆 綜合得分")
            if total_score >= 75:
                st.success(f"🔥 總分 {total_score:.1f} 分 - 強勢可操作")
            elif total_score >= 55:
                st.warning(f"⚖️ 總分 {total_score:.1f} 分 - 中性")
            else:
                st.error(f"⚠️ 總分 {total_score:.1f} 分 - 較弱")

            for d in details:
                st.write(d)

            # 圖表（簡化版，保留核心）
            st.markdown("---")
            st.subheader("📈 技術分析圖")
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.2, 0.2])
            
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='K線'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['MA25'], name='5週線', line=dict(color='orange')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], name='10週線', line=dict(color='red', dash='dot')), row=1, col=1)
            
            vol_colors = ['green' if c >= o else 'red' for c, o in zip(df['Close'], df['Open'])]
            fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=vol_colors, name='成交量'), row=2, col=1)
            
            fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], name='MACD Hist'), row=3, col=1)
            
            fig.update_layout(height=750, hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.error("無法取得資料，請確認代號正確")

st.caption("免責聲明：僅供參考，非投資建議。")
