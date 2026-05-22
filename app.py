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

# 🚀 實戰紀律守則 (依據使用者設定)
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
        
        if df.empty: return None, None
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
            
        ticker_obj = yf.Ticker(symbol)
        try:
            info = ticker_obj.info
        except:
            info = {}
            
        return df, info
    except Exception as e:
        return None, None

def calculate_indicators(df):
    df = df.copy()
    # 均線計算 (包含 5週/25MA 與 10週/50MA 的防禦線)
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA25'] = df['Close'].rolling(window=25).mean() # 5週線
    df['MA50'] = df['Close'].rolling(window=50).mean() # 10週線
    df['MA60'] = df['Close'].rolling(window=60).mean()
    df['Vol20'] = df['Volume'].rolling(window=20).mean()
    
    # 布林通道
    df['STD20'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['MA20'] + 2 * df['STD20']
    df['BB_Lower'] = df['MA20'] - 2 * df['STD20']
    
    # 動態支撐壓力線 (Donchian Channel)
    df['Resistance'] = df['High'].shift(1).rolling(window=20).max()
    df['Support'] = df['Low'].shift(1).rolling(window=20).min()
    
    # ATR 波動率計算
    df['TR'] = np.maximum(df['High'] - df['Low'], 
               np.maximum(abs(df['High'] - df['Close'].shift(1)), 
                          abs(df['Low'] - df['Close'].shift(1))))
    df['ATR'] = df['TR'].rolling(window=14).mean()
    
    # RSI (14) 計算
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9) 
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD 計算
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal']
    
    return df.dropna()

# --- 執行查詢 ---
if st.sidebar.button("啟動健診分析 🎯", type="primary") or ticker_input:
    with st.spinner(f'正在進行量化分析 {ticker_input} ...'):
        df, info = get_stock_data(ticker_input, period)
        
        if df is not None and not df.empty:
            df = calculate_indicators(df)
            
            # 取得最新數據
            last_price = float(df['Close'].iloc[-1])
            last_ma20 = float(df['MA20'].iloc[-1])
            last_ma25 = float(df['MA25'].iloc[-1]) # 5週線
            last_ma50 = float(df['MA50'].iloc[-1]) # 10週線
            last_vol = float(df['Volume'].iloc[-1])
            last_vol20 = float(df['Vol20'].iloc[-1])
            last_rsi = float(df['RSI'].iloc[-1])
            last_macd = float(df['MACD'].iloc[-1])
            last_hist = float(df['MACD_Hist'].iloc[-1])
            last_atr = float(df['ATR'].iloc[-1])
            bb_upper = float(df['BB_Upper'].iloc[-1])
            bb_lower = float(df['BB_Lower'].iloc[-1])
            
            # 時間停損計算：過去 60 個交易日 (約 3 個月) 的漲跌幅
            past_60d_price = float(df['Close'].iloc[-61]) if len(df) > 60 else float(df['Close'].iloc[0])
            roc_3m = ((last_price - past_60d_price) / past_60d_price) * 100
            
            # 基本面數據與 PEG 計算
            forward_pe = info.get('forwardPE', None)
            earnings_growth = info.get('earningsGrowth', None)
            peg_ratio = info.get('pegRatio', None)
            
            # 如果 Yahoo 沒給 PEG，但有 PE 和 Growth，我們自己算
            if peg_ratio is None and forward_pe is not None and earnings_growth is not None and earnings_growth > 0:
                 peg_ratio = forward_pe / (earnings_growth * 100) # 把小數換成百分比數字
            
            company_name = info.get('longName', info.get('shortName', ticker_input))

            st.subheader(f"📊 {company_name} ({ticker_input}) 核心數據")
            
            # --- 第一排：關鍵指標展示 ---
            col1, col2, col3, col4 = st.columns(4)
            price_change = float(df['Close'].iloc[-1]) - float(df['Close'].iloc[-2])
            pct_change = (price_change / float(df['Close'].iloc[-2])) * 100
            
            col1.metric("目前收盤價", f"{last_price:.2f}", f"{price_change:.2f} ({pct_change:.2f}%)")
            
            # 判斷是否破防線，變換顏色
            if last_price < last_ma50:
                col2.metric("10週線防守 (底限)", f"{last_ma50:.2f}", "⚠️ 已破底限，請出場", delta_color="inverse")
            elif last_price < last_ma25:
                col2.metric("5週線防守 (警戒)", f"{last_ma25:.2f}", "🚨 跌破5週線警戒", delta_color="inverse")
            else:
                col2.metric("5週線防守 (安全)", f"{last_ma25:.2f}", f"距離 {((last_price/last_ma25)-1)*100:.1f}%")

            # 顯示 PEG
            if peg_ratio:
                col3.metric("PEG (本益成長比)", f"{peg_ratio:.2f}", "便宜 (高成長)" if peg_ratio < 1 else "偏貴 (低成長)", delta_color="inverse" if peg_ratio >= 1 else "normal")
            else:
                col3.metric("PEG (本益成長比)", "缺乏成長數據")
                
            col4.metric("近三個月漲跌幅", f"{roc_3m:.1f}%", "表現遲滯" if roc_3m < 0 else "趨勢向上", delta_color="normal")

            # --- 第二排：動態 AI 評分 ---
            st.markdown("---")
            st.subheader("🤖 AI 動態評分與實戰紀律")
            
            details = []
            
            # 1. PEG 估值分數 (0-20分)
            if peg_ratio is not None:
                if peg_ratio <= 1:
                    peg_score = 20
                    details.append(f"✅ **估值優勢 (PEG)**：PEG 為 {peg_ratio:.2f} (小於 1)，利潤成長足以支撐目前本益比，長線極具潛力！得分 20.0/20。")
                elif peg_ratio <= 1.5:
                    peg_score = 10
                    details.append(f"🟡 **估值中性 (PEG)**：PEG 為 {peg_ratio:.2f}，估值尚屬合理範圍。得分 10.0/20。")
                else:
                    peg_score = 0
                    details.append(f"❌ **估值偏高 (PEG)**：PEG 高達 {peg_ratio:.2f} (大於 1)，股價可能已透支未來成長，需留意高檔修正。得分 0.0/20。")
            else:
                peg_score = 10 # 無資料給基本分
                details.append("🔵 **估值評估 (PEG)**：系統無法取得預期利潤成長數據，給予中立分數 10.0/20。")

            # 2. 趨勢分數 (0-20分) - 結合 5週與10週線
            bias_25 = (last_price / last_ma25 - 1) * 100
            if last_price > last_ma25 and last_ma25 > last_ma50:
                trend_score = min(max(10 + bias_25 * 2, 0), 20)
                details.append(f"✅ **多頭排列**：股價穩站 5 週線之上，且 5 週線大於 10 週線，趨勢得分 {trend_score:.1f}/20。")
            else:
                trend_score = min(max(5 + bias_25 * 2, 0), 10)
                details.append(f"🟡 **趨勢震盪**：尚未形成完美多頭或跌破 5 週線，趨勢得分 {trend_score:.1f}/20。")
                
            # 3. 量能分數 (0-20分)
            vol_ratio = last_vol / last_vol20
            vol_score = min(max((vol_ratio - 0.5) * 15, 0), 20)
            if vol_ratio > 1.2 and last_price > float(df['Open'].iloc[-1]):
                vol_score = min(vol_score + 5, 20)
                details.append(f"✅ **價漲量增**：主力資金介入跡象明顯，量能得分 {vol_score:.1f}/20。")
            elif vol_ratio > 1.2:
                vol_score = max(vol_score - 10, 0)
                details.append(f"⚠️ **爆量收黑**：需留意主力出貨風險，量能得分 {vol_score:.1f}/20。")
            else:
                details.append(f"🔵 **量能平穩**：市場觀望氣氛濃厚，量能得分 {vol_score:.1f}/20。")

            # 4. MACD 動能分數 (0-20分)
            if last_macd > 0 and last_hist > 0:
                macd_score = 20
                details.append(f"✅ **動能強勁**：MACD 雙線均在零軸之上且發散，得分 20.0/20。")
            elif last_hist > 0:
                macd_score = 10 + min(last_hist / last_price * 1000, 10)
                details.append(f"🟡 **動能轉強**：MACD 出現紅柱，轉強得分 {macd_score:.1f}/20。")
            else:
                macd_score = max(10 - abs(last_hist) / last_price * 1000, 0)
                details.append(f"❌ **動能疲弱**：MACD 綠柱發散，動能得分 {macd_score:.1f}/20。")

            # 5. RSI 乖離分數 (0-20分)
            if 45 <= last_rsi <= 65:
                rsi_score = 20
                details.append(f"✅ **RSI 健康**：RSI={last_rsi:.1f}，無過熱風險，得分 20.0/20。")
            elif last_rsi > 65:
                rsi_score = max(20 - (last_rsi - 65) * 1.5, 0)
                details.append(f"🔴 **高檔過熱**：RSI={last_rsi:.1f}，隨時可能回檔，得分 {rsi_score:.1f}/20。")
            else:
                rsi_score = max(20 - (45 - last_rsi) * 1.5, 0)
                details.append(f"🔵 **低檔超賣**：RSI={last_rsi:.1f}，有技術反彈契機，得分 {rsi_score:.1f}/20。")

            total_score = peg_score + trend_score + vol_score + macd_score + rsi_score
            
            # --- 實戰紀律審查 ---
            st.markdown("#### ⚔️ 嚴格紀律審查")
            if last_price < last_ma50:
                st.error(f"💀 **破底限警告**：已跌破 10 週線 ({last_ma50:.2f})！無論是否虧損，請立刻執行出場紀律，保留資金。")
            elif last_price < last_ma25:
                st.warning(f"🚨 **跌破 5 週線**：股價落入警戒區 ({last_ma25:.2f})，請緊盯後續走勢，若無反彈準備減碼。")
            
            if roc_3m < 0:
                st.error(f"⏳ **時間停損觸發**：過去 3 個月累計報酬為負 ({roc_3m:.1f}%)，此檔股票已成為死水，建議換股操作！")
            
            # 顯示評分結果
            st.markdown("#### 🏆 綜合雷達總分")
            if total_score >= 80:
                st.success(f"🔥 綜合得分：{total_score:.1f} 分 - 【強勢且便宜，適合成為 15-20% 的核心持股】")
            elif total_score >= 60:
                st.warning(f"⚖️ 綜合得分：{total_score:.1f} 分 - 【中等水準，未破 5 週線前可續抱】")
            else:
                st.error(f"⚠️ 綜合得分：{total_score:.1f} 分 - 【體質轉弱，建議優先考慮汰除】")

            for detail in details:
                st.write(detail)

            # --- 第三排：專業股價 K 線與副圖 ---
            st.markdown("---")
            st.subheader("📈 實戰特仕版技術分析圖表 (內建 5週/10週線與支撐壓力)")
            
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                                vertical_spacing=0.03, row_heights=[0.6, 0.2, 0.2])
            
            # 畫 K 線
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], 
                                         low=df['Low'], close=df['Close'], 
                                         name='K線', increasing_line_color=color_up, decreasing_line_color=color_down), 
                          row=1, col=1)
            
            # 畫均線 (改為你的防守線：25MA 與 50MA)
            fig.add_trace(go.Scatter(x=df.index, y=df['MA25'], name='25MA (5週線警戒)', line=dict(color='orange', width=2)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], name='50MA (10週線底限)', line=dict(color='red', width=2, dash='dot')), row=1, col=1)
            
            # 畫動態支撐壓力線 (Donchian)
            fig.add_trace(go.Scatter(x=df.index, y=df['Resistance'], name='20日壓力線', line=dict(color='#FF1493', width=1.5, shape='hv', dash='dot')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['Support'], name='20日支撐線', line=dict(color='#00BFFF', width=1.5, shape='hv', dash='dot')), row=1, col=1)
            
            # 畫成交量副圖
            vol_colors = [color_up if row['Close'] >= row['Open'] else color_down for i, row in df.iterrows()]
            fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='成交量', marker_color=vol_colors), row=2, col=1)
            
            # 畫 MACD 副圖
            hist_colors = [color_up if val > 0 else color_down for val in df['MACD_Hist']]
            fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], name='MACD 柱狀圖', marker_color=hist_colors), row=3, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD 線', line=dict(color='blue', width=1.5)), row=3, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['Signal'], name='Signal 線', line=dict(color='orange', width=1.5)), row=3, col=1)

            fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
            
            fig.update_layout(height=800, template="plotly_white", 
                              hovermode="x unified", xaxis_rangeslider_visible=False,
                              margin=dict(l=0, r=0, t=30, b=0))
                              
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.error("❌ 找不到該股票數據，請確認代號是否正確（例如台股台積電需輸入 2330.TW）。")

# --- 免責聲明 ---
st.markdown("---")
st.caption("免責聲明：本工具透過技術與基本面指標自動生成，僅供參考，不構成任何投資建議。市場有風險，投資需謹慎。") 
