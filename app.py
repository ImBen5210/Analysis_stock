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

# --- 側邊欄 ---
st.sidebar.header("🔍 查詢設定")
market_type = st.sidebar.selectbox("🌍 選擇市場", ["台股 (需加 .TW 或 .TWO)", "美股"])
ticker_input = st.sidebar.text_input("📝 輸入股票代號", value="2330.TW" if "台股" in market_type else "NVDA").upper()
period = st.sidebar.slider("📅 查看歷史期間 (月)", 3, 24, 6)

st.sidebar.markdown("---")
st.sidebar.error("⚔️ **操盤手鐵律：資金與停損控管**")
st.sidebar.info(
    "1. **集中火力**：持股不超過 4 檔，挑選前四名操作。\n"
    "2. **資金配置**：每檔押 15%-20%，保留 30% 現金用來加碼最強勢個股。\n"
    "3. **價格停損**：跌破 5 週線 (約25MA) 警戒，底線為 10 週線 (約50MA)，破線無條件出場。\n"
    "4. **時間停損**：進場後 3 個月不漲，直接汰弱留強。"
)

if "台股" in market_type:
    color_up, color_down = '#FF3333', '#00AA00'
else:
    color_up, color_down = '#00AA00', '#FF3333'


# ─────────────────────────────────────────────
# 資料抓取
# ─────────────────────────────────────────────
@st.cache_data(ttl=900, show_spinner=False)
def get_stock_data(symbol, months):
    try:
        end_date   = datetime.now()
        start_date = end_date - timedelta(days=months * 30 + 100)
        df = yf.download(symbol, start=start_date, end=end_date,
                         progress=False, auto_adjust=True)
        if df.empty:
            return None, None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        ticker_obj = yf.Ticker(symbol)
        try:
            info = ticker_obj.info or {}
        except Exception:
            info = {}

        return df, info
    except Exception:
        return None, None


@st.cache_data(ttl=900, show_spinner=False)
def get_eps_growth(symbol):
    """
    從季度財報歷史直接計算 YoY EPS 成長率，
    不依賴 info['earningsGrowth']，台股美股均適用。
    回傳 (eps_growth_rate: float|None, source_desc: str)
    """
    try:
        t = yf.Ticker(symbol)

        # ── 方法 1：quarterly_earnings（較舊版 yfinance）──
        try:
            qe = t.quarterly_earnings
            if qe is not None and not qe.empty and len(qe) >= 5:
                # 最新季 vs 去年同季
                latest   = float(qe['Earnings'].iloc[0])
                year_ago = float(qe['Earnings'].iloc[4])
                if year_ago != 0:
                    rate = (latest - year_ago) / abs(year_ago)
                    return rate, "季度EPS YoY（quarterly_earnings）"
        except Exception:
            pass

        # ── 方法 2：income_stmt 裡的 Net Income（年度）──
        try:
            inc = t.income_stmt          # 年度，新版 API
            if inc is not None and not inc.empty:
                ni_row = None
                for key in ['Net Income', 'NetIncome', 'Net Income Common Stockholders']:
                    if key in inc.index:
                        ni_row = inc.loc[key]
                        break
                if ni_row is not None and len(ni_row) >= 2:
                    latest   = float(ni_row.iloc[0])
                    prev     = float(ni_row.iloc[1])
                    if prev != 0:
                        rate = (latest - prev) / abs(prev)
                        return rate, "淨利 YoY（income_stmt）"
        except Exception:
            pass

        # ── 方法 3：quarterly_income_stmt YoY ──
        try:
            qi = t.quarterly_income_stmt
            if qi is not None and not qi.empty:
                ni_row = None
                for key in ['Net Income', 'NetIncome', 'Net Income Common Stockholders']:
                    if key in qi.index:
                        ni_row = qi.loc[key]
                        break
                if ni_row is not None and len(ni_row) >= 5:
                    latest   = float(ni_row.iloc[0])
                    year_ago = float(ni_row.iloc[4])
                    if year_ago != 0:
                        rate = (latest - year_ago) / abs(year_ago)
                        return rate, "季度淨利 YoY（quarterly_income_stmt）"
        except Exception:
            pass

        # ── 方法 4：info 欄位 fallback（原始做法，多試幾個 key）──
        try:
            info = t.info or {}
            for key in ['earningsGrowth', 'earningsQuarterlyGrowth', 'revenueGrowth']:
                val = info.get(key)
                if val is not None and val > -1:   # -1 以下通常是異常值
                    return float(val), f"info['{key}']"
        except Exception:
            pass

        return None, "無法取得成長率數據"
    except Exception:
        return None, "無法取得成長率數據"


# ─────────────────────────────────────────────
# 技術指標
# ─────────────────────────────────────────────
def calculate_indicators(df):
    df = df.copy()
    df['MA5']  = df['Close'].rolling(5).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA25'] = df['Close'].rolling(25).mean()
    df['MA50'] = df['Close'].rolling(50).mean()
    df['MA60'] = df['Close'].rolling(60).mean()
    df['Vol20'] = df['Volume'].rolling(20).mean()

    df['STD20']    = df['Close'].rolling(20).std()
    df['BB_Upper'] = df['MA20'] + 2 * df['STD20']
    df['BB_Lower'] = df['MA20'] - 2 * df['STD20']

    df['Resistance'] = df['High'].shift(1).rolling(20).max()
    df['Support']    = df['Low'].shift(1).rolling(20).min()

    df['TR'] = np.maximum(df['High'] - df['Low'],
               np.maximum(abs(df['High'] - df['Close'].shift(1)),
                          abs(df['Low']  - df['Close'].shift(1))))
    df['ATR'] = df['TR'].rolling(14).mean()

    delta = df['Close'].diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + gain / (loss + 1e-9)))

    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD']      = exp1 - exp2
    df['Signal']    = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal']

    return df.dropna()


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────
if st.sidebar.button("啟動健診分析 🎯", type="primary") or ticker_input:
    with st.spinner(f'正在進行量化分析 {ticker_input} ...'):
        df, info = get_stock_data(ticker_input, period)

        if df is not None and not df.empty:
            df = calculate_indicators(df)

            last_price = float(df['Close'].iloc[-1])
            last_ma25  = float(df['MA25'].iloc[-1])
            last_ma50  = float(df['MA50'].iloc[-1])
            last_vol   = float(df['Volume'].iloc[-1])
            last_vol20 = float(df['Vol20'].iloc[-1])
            last_rsi   = float(df['RSI'].iloc[-1])
            last_macd  = float(df['MACD'].iloc[-1])
            last_hist  = float(df['MACD_Hist'].iloc[-1])

            past_60d_price = float(df['Close'].iloc[-61]) if len(df) > 60 else float(df['Close'].iloc[0])
            roc_3m = ((last_price - past_60d_price) / past_60d_price) * 100

            # ── PE：多來源取值 ──
            pe = info.get('forwardPE') or info.get('trailingPE')
            pe_label = "預期PE" if info.get('forwardPE') else "歷史PE"
            if not pe or pe <= 0:
                eps = info.get('trailingEps', 0)
                if eps and eps > 0:
                    pe = last_price / eps
                    pe_label = "自算PE"

            # ── EPS 成長率：從財報直接算 ──
            eps_growth, growth_source = get_eps_growth(ticker_input)

            # ── PEG 計算 ──
            peg_ratio  = info.get('pegRatio')
            peg_source = "Yahoo Finance"
            if (peg_ratio is None or peg_ratio <= 0) and pe and pe > 0 and eps_growth and eps_growth > 0:
                peg_ratio  = pe / (eps_growth * 100)
                peg_source = f"自算（{pe_label} ÷ 成長率 {eps_growth*100:.1f}%，來源：{growth_source}）"

            company_name = info.get('longName', info.get('shortName', ticker_input))

            # ════════════════════════════════
            # 第一排：關鍵指標
            # ════════════════════════════════
            st.subheader(f"📊 {company_name} ({ticker_input}) 核心數據")
            col1, col2, col3, col4 = st.columns(4)

            price_change = float(df['Close'].iloc[-1]) - float(df['Close'].iloc[-2])
            pct_change   = (price_change / float(df['Close'].iloc[-2])) * 100
            col1.metric("目前收盤價", f"{last_price:.2f}", f"{price_change:.2f} ({pct_change:.2f}%)")

            if last_price < last_ma50:
                col2.metric("10週線防守 (底限)", f"{last_ma50:.2f}", "⚠️ 已破底限，請出場", delta_color="inverse")
            elif last_price < last_ma25:
                col2.metric("5週線防守 (警戒)", f"{last_ma25:.2f}", "🚨 跌破5週線警戒", delta_color="inverse")
            else:
                col2.metric("5週線防守 (安全)", f"{last_ma25:.2f}", f"距離 {((last_price/last_ma25)-1)*100:.1f}%")

            if peg_ratio and peg_ratio > 0:
                col3.metric(
                    "PEG (本益成長比)",
                    f"{peg_ratio:.2f}",
                    "便宜 (高成長)" if peg_ratio < 1 else "偏貴 (低成長)",
                    delta_color="normal" if peg_ratio < 1 else "inverse"
                )
            else:
                col3.metric("PEG (本益成長比)", "缺乏成長數據")

            col4.metric("近三個月漲跌幅", f"{roc_3m:.1f}%",
                        "表現遲滯" if roc_3m < 0 else "趨勢向上", delta_color="normal")

            # ── 估值明細展開 ──
            with st.expander("🔍 估值計算明細"):
                c1, c2, c3 = st.columns(3)
                c1.metric(pe_label, f"{pe:.1f}x" if pe else "N/A")
                c2.metric(f"EPS成長率（{growth_source}）",
                          f"{eps_growth*100:.1f}%" if eps_growth else "N/A")
                c3.metric(f"PEG來源", peg_source if peg_ratio else "無法計算")

            # ════════════════════════════════
            # AI 評分
            # ════════════════════════════════
            st.markdown("---")
            st.subheader("🤖 AI 動態評分與實戰紀律")
            details = []

            # 1. PEG (0-20)
            if peg_ratio and peg_ratio > 0:
                if peg_ratio <= 1:
                    peg_score = 20
                    details.append(f"✅ **估值優勢 (PEG={peg_ratio:.2f})**：成長足以支撐本益比，長線極具潛力。得分 20/20。")
                elif peg_ratio <= 1.5:
                    peg_score = 10
                    details.append(f"🟡 **估值中性 (PEG={peg_ratio:.2f})**：估值尚屬合理。得分 10/20。")
                else:
                    peg_score = 0
                    details.append(f"❌ **估值偏高 (PEG={peg_ratio:.2f})**：股價可能已透支未來成長，留意回檔。得分 0/20。")
            else:
                peg_score = 10
                details.append("🔵 **估值評估**：無足夠數據計算 PEG，給予中立分數 10/20。")

            # 2. 趨勢 (0-20)
            bias_25 = (last_price / last_ma25 - 1) * 100
            if last_price > last_ma25 and last_ma25 > last_ma50:
                trend_score = min(max(10 + bias_25 * 2, 0), 20)
                details.append(f"✅ **多頭排列**：股價穩站 5 週線之上，趨勢得分 {trend_score:.1f}/20。")
            else:
                trend_score = min(max(5 + bias_25 * 2, 0), 10)
                details.append(f"🟡 **趨勢震盪**：尚未形成完美多頭或跌破 5 週線，趨勢得分 {trend_score:.1f}/20。")

            # 3. 量能 (0-20)
            vol_ratio = last_vol / last_vol20
            vol_score = min(max((vol_ratio - 0.5) * 15, 0), 20)
            if vol_ratio > 1.2 and last_price > float(df['Open'].iloc[-1]):
                vol_score = min(vol_score + 5, 20)
                details.append(f"✅ **價漲量增**：主力資金介入明顯，量能得分 {vol_score:.1f}/20。")
            elif vol_ratio > 1.2:
                vol_score = max(vol_score - 10, 0)
                details.append(f"⚠️ **爆量收黑**：留意主力出貨風險，量能得分 {vol_score:.1f}/20。")
            else:
                details.append(f"🔵 **量能平穩**：市場觀望，量能得分 {vol_score:.1f}/20。")

            # 4. MACD (0-20)
            if last_macd > 0 and last_hist > 0:
                macd_score = 20
                details.append("✅ **動能強勁**：MACD 雙線零軸之上且發散，得分 20/20。")
            elif last_hist > 0:
                macd_score = 10 + min(last_hist / last_price * 1000, 10)
                details.append(f"🟡 **動能轉強**：MACD 紅柱，得分 {macd_score:.1f}/20。")
            else:
                macd_score = max(10 - abs(last_hist) / last_price * 1000, 0)
                details.append(f"❌ **動能疲弱**：MACD 綠柱發散，得分 {macd_score:.1f}/20。")

            # 5. RSI (0-20)
            if 45 <= last_rsi <= 65:
                rsi_score = 20
                details.append(f"✅ **RSI 健康**：RSI={last_rsi:.1f}，無過熱風險，得分 20/20。")
            elif last_rsi > 65:
                rsi_score = max(20 - (last_rsi - 65) * 1.5, 0)
                details.append(f"🔴 **高檔過熱**：RSI={last_rsi:.1f}，得分 {rsi_score:.1f}/20。")
            else:
                rsi_score = max(20 - (45 - last_rsi) * 1.5, 0)
                details.append(f"🔵 **低檔超賣**：RSI={last_rsi:.1f}，有技術反彈契機，得分 {rsi_score:.1f}/20。")

            total_score = peg_score + trend_score + vol_score + macd_score + rsi_score

            # ── 紀律審查 ──
            st.markdown("#### ⚔️ 嚴格紀律審查")
            if last_price < last_ma50:
                st.error(f"💀 **破底限警告**：已跌破 10 週線 ({last_ma50:.2f})！請立刻執行出場紀律，保留資金。")
            elif last_price < last_ma25:
                st.warning(f"🚨 **跌破 5 週線**：落入警戒區 ({last_ma25:.2f})，請緊盯後續走勢，若無反彈準備減碼。")

            if roc_3m < 0:
                st.error(f"⏳ **時間停損觸發**：過去 3 個月累計報酬 {roc_3m:.1f}%，建議換股操作！")

            st.markdown("#### 🏆 綜合雷達總分")
            if total_score >= 80:
                st.success(f"🔥 綜合得分：{total_score:.1f} 分 — 強勢且便宜，適合成為 15-20% 的核心持股")
            elif total_score >= 60:
                st.warning(f"⚖️ 綜合得分：{total_score:.1f} 分 — 中等水準，未破 5 週線前可續抱")
            else:
                st.error(f"⚠️ 綜合得分：{total_score:.1f} 分 — 體質轉弱，建議優先考慮汰除")

            for d in details:
                st.write(d)

            # ════════════════════════════════
            # 技術分析圖表
            # ════════════════════════════════
            st.markdown("---")
            st.subheader("📈 實戰特仕版技術分析圖表 (內建 5週/10週線與支撐壓力)")

            fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                                vertical_spacing=0.03, row_heights=[0.6, 0.2, 0.2])

            fig.add_trace(go.Candlestick(
                x=df.index, open=df['Open'], high=df['High'],
                low=df['Low'], close=df['Close'], name='K線',
                increasing_line_color=color_up, decreasing_line_color=color_down), row=1, col=1)

            fig.add_trace(go.Scatter(x=df.index, y=df['MA25'],
                name='25MA (5週線警戒)', line=dict(color='orange', width=2)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['MA50'],
                name='50MA (10週線底限)', line=dict(color='red', width=2, dash='dot')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['Resistance'],
                name='20日壓力線', line=dict(color='#FF1493', width=1.5, shape='hv', dash='dot')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['Support'],
                name='20日支撐線', line=dict(color='#00BFFF', width=1.5, shape='hv', dash='dot')), row=1, col=1)

            vol_colors = [color_up if row['Close'] >= row['Open'] else color_down for _, row in df.iterrows()]
            fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='成交量',
                                 marker_color=vol_colors), row=2, col=1)

            hist_colors = [color_up if v > 0 else color_down for v in df['MACD_Hist']]
            fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], name='MACD柱',
                                 marker_color=hist_colors), row=3, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD',
                                     line=dict(color='blue', width=1.5)), row=3, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['Signal'], name='Signal',
                                     line=dict(color='orange', width=1.5)), row=3, col=1)

            fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
            fig.update_layout(height=800, template="plotly_white",
                              hovermode="x unified", xaxis_rangeslider_visible=False,
                              margin=dict(l=0, r=0, t=30, b=0))

            st.plotly_chart(fig, use_container_width=True)

        else:
            st.error("❌ 找不到該股票數據，請確認代號是否正確（例如台股台積電需輸入 2330.TW）。")

st.markdown("---")
st.caption("免責聲明：本工具透過技術與基本面指標自動生成，僅供參考，不構成任何投資建議。市場有風險，投資需謹慎。")
