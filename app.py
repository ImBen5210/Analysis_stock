import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# --- 網頁配置 ---
st.set_page_config(page_title="AI 股票健診雷達", page_icon="🚀", layout="wide")

st.title("🚀 AI 股票智能健診雷達 (Pro 版)")
st.markdown("輸入股票代號，系統將根據**技術趨勢、動能指標、量能、估值**進行綜合評分。")

# --- 側邊欄：輸入參數 ---
st.sidebar.header("🔍 查詢設定")
market_type = st.sidebar.selectbox("🌍 選擇市場", ["台股 (需加 .TW 或 .TWO)", "美股"])
ticker_input = st.sidebar.text_input("📝 輸入股票代號", value="2330.TW" if "台股" in market_type else "AAPL").upper()
period = st.sidebar.slider("📅 查看歷史期間 (月)", 3, 24, 6)

# 動態決定 K 線顏色 (台股紅漲綠跌；美股綠漲紅跌)
if "台股" in market_type:
    color_up, color_down = '#FF3333', '#00AA00'
else:
    color_up, color_down = '#00AA00', '#FF3333'

# --- 核心邏輯函數 (加入快取機制避免重複下載) ---
@st.cache_data(ttl=900, show_spinner=False)
def get_stock_data(symbol, months):
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=months * 30 + 100) # 多抓100天確保長天期均線準確
        df = yf.download(symbol, start=start_date, end=end_date, progress=False)
        
        if df.empty: return None, None
        
        # 展平 MultiIndex (如果 yfinance 返回多層索引)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
            
        info = yf.Ticker(symbol).info
        return df, info
    except Exception as e:
        return None, None

def calculate_indicators(df):
    # 均線計算
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA60'] = df['Close'].rolling(window=60).mean()
    df['Vol20'] = df['Volume'].rolling(window=20).mean()
    
    # RSI (14) 計算
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9) # 防除以零
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD 計算
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    return df.dropna() # 清除計算初期的空值

# --- 執行查詢 ---
if st.sidebar.button("啟動健診分析 🎯", type="primary") or ticker_input:
    with st.spinner(f'正在連線分析 {ticker_input} ...'):
        df, info = get_stock_data(ticker_input, period)
        
        if df is not None and not df.empty:
            df = calculate_indicators(df)
            
            # 取得最新數據
            last_price = float(df['Close'].iloc[-1])
            last_ma20 = float(df['MA20'].iloc[-1])
            last_ma60 = float(df['MA60'].iloc[-1])
            last_vol = float(df['Volume'].iloc[-1])
            last_vol20 = float(df['Vol20'].iloc[-1])
            last_rsi = float(df['RSI'].iloc[-1])
            last_macd = float(df['MACD'].iloc[-1])
            last_signal = float(df['Signal'].iloc[-1])
            
            # 基本面數據 (防呆處理)
            pe_ratio = info.get('trailingPE', "N/A")
            div_yield = info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0
            company_name = info.get('longName', info.get('shortName', ticker_input))

            st.subheader(f"📊 {company_name} ({ticker_input}) 核心數據")
            
            # --- 第一排：關鍵指標展示 ---
            col1, col2, col3, col4 = st.columns(4)
            price_change = float(df['Close'].iloc[-1]) - float(df['Close'].iloc[-2])
            pct_change = (price_change / float(df['Close'].iloc[-2])) * 100
            
            col1.metric("目前收盤價", f"{last_price:.2f}", f"{price_change:.2f} ({pct_change:.2f}%)")
            col2.metric("月線 (20MA) 乖離", f"{last_ma20:.2f}", f"{((last_price/last_ma20)-1)*100:.1f}%")
            col3.metric("本益比 (P/E)", f"{pe_ratio if pe_ratio == 'N/A' else round(pe_ratio, 2)}")
            col4.metric("殖利率", f"{div_yield:.2f}%")

            # --- 第二排：評分邏輯 ---
            st.markdown("---")
            st.subheader("🤖 AI 健診報告與建議")
            
            score = 0
            details = []
            
            # 條件 1: 趨勢 (短、中線判斷)
            if last_price > last_ma20 and last_ma20 > last_ma60:
                score += 25
                details.append("✅ **趨勢偏多**：股價站上月線，且月線大於季線，多頭排列健康。")
            elif last_price > last_ma20:
                score += 15
                details.append("🟡 **趨勢震盪**：股價站上月線，但均線尚未全面翻揚，處於整理或初升段。")
            else:
                details.append("❌ **趨勢偏空**：股價跌破月線防守，短線處於弱勢格局。")
                
            # 條件 2: 量能
            if last_vol > last_vol20 * 1.2:
                score += 25
                details.append("✅ **成交量能**：今日放量上漲 (大於月均量20%)，資金參與度極高。")
            elif last_vol > last_vol20 * 0.8:
                score += 15
                details.append("🟡 **成交量能**：量能平穩，維持平均水準。")
            else:
                details.append("❌ **成交量能**：量能顯著萎縮，市場觀望氣氛濃厚。")
                
            # 條件 3: MACD 動能
            if last_macd > last_signal and last_macd > 0:
                score += 25
                details.append("✅ **MACD 動能**：指標柱狀圖翻紅且大於零軸，上漲動能強烈。")
            elif last_macd > last_signal:
                score += 15
                details.append("🟡 **MACD 動能**：指標低檔黃金交叉，動能開始轉強。")
            else:
                details.append("❌ **MACD 動能**：指標死亡交叉或低於零軸，上漲動能疲弱。")

            # 條件 4: RSI 
            if 40 < last_rsi < 70:
                score += 25
                details.append(f"✅ **RSI 指標**：RSI 為 {last_rsi:.1f}，處於健康多空均衡區。")
            elif last_rsi >= 70:
                score += 10
                details.append(f"🔴 **RSI 指標**：RSI {last_rsi:.1f} 已進入超買過熱區，需留意追高回檔風險。")
            else:
                details.append(f"🔵 **RSI 指標**：RSI {last_rsi:.1f} 進入超賣區，可能有技術性反彈機會。")

            # 顯示評分結果
            if score >= 80:
                st.success(f"🏆 綜合評分：{score} 分 - 【強勢多頭，優先關注】")
            elif score >= 50:
                st.warning(f"⚖️ 綜合評分：{score} 分 - 【中立震盪，建議分批觀察】")
            else:
                st.error(f"⚠️ 綜合評分：{score} 分 - 【弱勢格局，暫不建議介入】")

            for detail in details:
                st.write(detail)

            # --- 第三排：專業股價 K 線與副圖 ---
            st.markdown("---")
            st.subheader("📈 專業技術分析圖表")
            
            # 使用 subplots 建立主圖(K線)與副圖(成交量)
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                vertical_spacing=0.05, row_heights=[0.7, 0.3])
            
            # 畫 K 線
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], 
                                         low=df['Low'], close=df['Close'], 
                                         name='K線', increasing_line_color=color_up, decreasing_line_color=color_down), 
                          row=1, col=1)
            
            # 畫均線
            fig.add_trace(go.Scatter(x=df.index, y=df['MA5'], name='5MA (週線)', line=dict(color='blue', width=1)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name='20MA (月線)', line=dict(color='orange', width=1.5)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['MA60'], name='60MA (季線)', line=dict(color='purple', width=1.5, dash='dot')), row=1, col=1)
            
            # 畫成交量副圖 (依據收盤價漲跌決定顏色)
            vol_colors = [color_up if row['Close'] >= row['Open'] else color_down for i, row in df.iterrows()]
            fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='成交量', marker_color=vol_colors), row=2, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['Vol20'], name='量能20MA', line=dict(color='orange', width=1.5)), row=2, col=1)

            # 隱藏非交易日的空白區間 (六日)
            fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
            
            # 版面設定
            fig.update_layout(height=650, template="plotly_white", 
                              hovermode="x unified", xaxis_rangeslider_visible=False,
                              margin=dict(l=0, r=0, t=30, b=0))
                              
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.error("❌ 找不到該股票數據，請確認代號是否正確（例如台股台積電需輸入 2330.TW）。")

# --- 免責聲明 ---
st.markdown("---")
st.caption("免責聲明：本工具透過技術與基本面指標自動生成，僅供參考，不構成任何投資建議。市場有風險，投資需謹慎。")