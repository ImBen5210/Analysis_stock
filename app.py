import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="AI 股票健診雷達", page_icon="🚀", layout="wide")
st.title("🚀 AI 股票智能健診雷達 (量化實戰特仕版)")
st.markdown("結合**動態 AI 評分、ATR 停損、MACD 動能、布林通道與 PEG 估值**，並嚴格執行時間與價格紀律。")

st.sidebar.header("🔍 查詢設定")
market_type  = st.sidebar.selectbox("🌍 選擇市場", ["台股 (需加 .TW 或 .TWO)", "美股"])
ticker_input = st.sidebar.text_input("📝 輸入股票代號",
                  value="2330.TW" if "台股" in market_type else "NVDA").upper()
period = st.sidebar.slider("📅 查看歷史期間 (月)", 3, 24, 6)
st.sidebar.markdown("---")
st.sidebar.error("⚔️ **操盤手鐵律：資金與停損控管**")
st.sidebar.info(
    "1. **集中火力**：持股不超過 4 檔。\n"
    "2. **資金配置**：每檔 15-20%，留 30% 現金加碼最強勢個股。\n"
    "3. **價格停損**：跌破 25MA 警戒，跌破 50MA 無條件出場。\n"
    "4. **時間停損**：進場後 3 個月不漲，直接汰弱留強。"
)
color_up, color_down = ('#FF3333','#00AA00') if "台股" in market_type else ('#00AA00','#FF3333')


# ─────────────────────────────────────────────
# K 線資料
# ─────────────────────────────────────────────
@st.cache_data(ttl=900, show_spinner=False)
def get_price_data(symbol, months):
    try:
        end   = datetime.now()
        start = end - timedelta(days=months*30 + 100)
        df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        return df
    except Exception as e:
        st.error(f"K 線資料失敗：{e}")
        return None


# ─────────────────────────────────────────────
# 基本面：fast_info 優先，info 備用，財報兜底
# ─────────────────────────────────────────────
@st.cache_data(ttl=1800, show_spinner=False)
def get_fundamentals(symbol):
    res = {
        'company': symbol,
        'pe': None, 'pe_label': '',
        'growth': None, 'growth_label': '',
        'peg': None, 'peg_source': '',
        'debug': []
    }

    def log(msg):
        res['debug'].append(msg)

    for attempt in range(3):
        try:
            t = yf.Ticker(symbol)

            # ══ 第一層：fast_info（新版最穩定，不打 quote API）══
            try:
                fi = t.fast_info
                log(f"fast_info keys: {list(fi.__dict__.keys()) if hasattr(fi,'__dict__') else 'N/A'}")

                # fast_info 有 pe_forward / pe_trailing（新版欄位名）
                for attr in ['pe_forward', 'forward_pe']:
                    val = getattr(fi, attr, None)
                    if val and float(val) > 0:
                        res['pe'] = float(val)
                        res['pe_label'] = 'Forward PE (fast_info)'
                        log(f"PE from fast_info.{attr} = {val}")
                        break

                if not res['pe']:
                    for attr in ['pe_trailing', 'trailing_pe']:
                        val = getattr(fi, attr, None)
                        if val and float(val) > 0:
                            res['pe'] = float(val)
                            res['pe_label'] = 'Trailing PE (fast_info)'
                            log(f"PE from fast_info.{attr} = {val}")
                            break

                name = getattr(fi, 'display_name', None) or getattr(fi, 'shortName', None)
                if name:
                    res['company'] = name
            except Exception as e:
                log(f"fast_info 失敗: {e}")

            # ══ 第二層：info（完整但較慢，可能 rate limit）══
            if not res['pe']:
                try:
                    info = t.info
                    res['company'] = info.get('longName') or info.get('shortName', symbol)
                    log(f"info 取得成功，key 數量: {len(info)}")

                    for key in ['forwardPE', 'trailingPE']:
                        v = info.get(key)
                        if v and float(v) > 0:
                            res['pe'] = float(v)
                            res['pe_label'] = key
                            log(f"PE from info.{key} = {v}")
                            break

                    # PEG 直接從 info 拿
                    peg = info.get('pegRatio')
                    if peg and float(peg) > 0:
                        res['peg'] = float(peg)
                        res['peg_source'] = 'Yahoo Finance pegRatio'
                        log(f"PEG direct from info = {peg}")

                    # 成長率從 info
                    for key in ['earningsGrowth', 'earningsQuarterlyGrowth', 'revenueGrowth']:
                        v = info.get(key)
                        if v is not None and isinstance(v, (int,float)) and v > -0.99:
                            res['growth'] = float(v)
                            res['growth_label'] = key
                            log(f"growth from info.{key} = {v}")
                            break
                except Exception as e:
                    log(f"info 失敗: {e}")

            # ══ 第三層：income_stmt 直接算成長率 ══
            if res['growth'] is None and res['peg'] is None:
                for attr in ['income_stmt', 'financials', 'quarterly_income_stmt']:
                    try:
                        stmt = getattr(t, attr, None)
                        if stmt is None or stmt.empty:
                            log(f"{attr}: 空")
                            continue
                        log(f"{attr} index: {list(stmt.index[:5])}")
                        for row_key in ['Net Income','NetIncome',
                                        'Net Income Common Stockholders',
                                        'Diluted EPS','Basic EPS']:
                            if row_key not in stmt.index:
                                continue
                            row = stmt.loc[row_key].dropna()
                            log(f"  {row_key}: {list(row.values[:4])}")
                            n = 5 if 'quarterly' in attr else 2
                            if len(row) >= n:
                                v0 = float(row.iloc[0])
                                v1 = float(row.iloc[n-1])
                                if v1 != 0 and v0 * v1 > 0:  # 同號才算
                                    res['growth'] = (v0 - v1) / abs(v1)
                                    res['growth_label'] = f"{row_key} YoY ({attr})"
                                    log(f"  → growth = {res['growth']:.3f}")
                                    break
                        if res['growth'] is not None:
                            break
                    except Exception as e:
                        log(f"{attr} 失敗: {e}")

            # ══ 自算 PEG（公式：PE ÷ 成長率%）══
            if res['peg'] is None and res['pe'] and res['pe'] > 0 \
               and res['growth'] and res['growth'] > 0:
                # growth 是小數（0.25 = 25%），PEG = PE / growth%數字
                res['peg'] = res['pe'] / (res['growth'] * 100)
                res['peg_source'] = (
                    f"自算：{res['pe_label']} {res['pe']:.1f} ÷ "
                    f"成長率 {res['growth']*100:.1f}% ({res['growth_label']})"
                )
                log(f"PEG 自算 = {res['peg']:.2f}")

            break  # 成功，跳出 retry

        except Exception as e:
            log(f"attempt {attempt} 頂層失敗: {e}")
            if 'RateLimit' in str(e) or '429' in str(e):
                if attempt < 2:
                    time.sleep(4 + attempt * 3)
                    continue
            break

    return res


# ─────────────────────────────────────────────
# 技術指標
# ─────────────────────────────────────────────
def calculate_indicators(df):
    df = df.copy()
    df['MA5']   = df['Close'].rolling(5).mean()
    df['MA20']  = df['Close'].rolling(20).mean()
    df['MA25']  = df['Close'].rolling(25).mean()
    df['MA50']  = df['Close'].rolling(50).mean()
    df['MA60']  = df['Close'].rolling(60).mean()
    df['Vol20'] = df['Volume'].rolling(20).mean()
    df['STD20'] = df['Close'].rolling(20).std()
    df['BB_Upper'] = df['MA20'] + 2*df['STD20']
    df['BB_Lower'] = df['MA20'] - 2*df['STD20']
    df['Resistance'] = df['High'].shift(1).rolling(20).max()
    df['Support']    = df['Low'].shift(1).rolling(20).min()
    df['TR'] = np.maximum(df['High']-df['Low'],
               np.maximum(abs(df['High']-df['Close'].shift(1)),
                          abs(df['Low'] -df['Close'].shift(1))))
    df['ATR'] = df['TR'].rolling(14).mean()
    delta = df['Close'].diff()
    gain  = delta.where(delta>0,0).rolling(14).mean()
    loss  = (-delta.where(delta<0,0)).rolling(14).mean()
    df['RSI'] = 100 - (100/(1+gain/(loss+1e-9)))
    e1 = df['Close'].ewm(span=12,adjust=False).mean()
    e2 = df['Close'].ewm(span=26,adjust=False).mean()
    df['MACD']      = e1 - e2
    df['Signal']    = df['MACD'].ewm(span=9,adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal']
    return df.dropna()


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────
if st.sidebar.button("啟動健診分析 🎯", type="primary") or ticker_input:
    with st.spinner(f'正在分析 {ticker_input}...'):
        df   = get_price_data(ticker_input, period)
        fund = get_fundamentals(ticker_input)

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
        past_60    = float(df['Close'].iloc[-61]) if len(df)>60 else float(df['Close'].iloc[0])
        roc_3m     = (last_price - past_60) / past_60 * 100

        pe        = fund['pe']
        growth    = fund['growth']
        peg_ratio = fund['peg']
        peg_src   = fund['peg_source']
        company   = fund['company']

        # ── 指標列 ──
        st.subheader(f"📊 {company} ({ticker_input}) 核心數據")
        c1,c2,c3,c4 = st.columns(4)
        chg  = float(df['Close'].iloc[-1]) - float(df['Close'].iloc[-2])
        pchg = chg / float(df['Close'].iloc[-2]) * 100
        c1.metric("目前收盤價", f"{last_price:.2f}", f"{chg:.2f} ({pchg:.2f}%)")

        if last_price < last_ma50:
            c2.metric("10週線(底限)", f"{last_ma50:.2f}", "⚠️ 已破底限，請出場", delta_color="inverse")
        elif last_price < last_ma25:
            c2.metric("5週線(警戒)", f"{last_ma25:.2f}", "🚨 跌破5週線", delta_color="inverse")
        else:
            c2.metric("5週線(安全)", f"{last_ma25:.2f}", f"距離 {(last_price/last_ma25-1)*100:.1f}%")

        if peg_ratio and peg_ratio > 0:
            c3.metric("PEG (本益成長比)", f"{peg_ratio:.2f}",
                      "便宜(高成長)" if peg_ratio<1 else "偏貴(低成長)",
                      delta_color="normal" if peg_ratio<1 else "inverse")
        else:
            c3.metric("PEG (本益成長比)", "數據不足")

        c4.metric("近三月漲跌幅", f"{roc_3m:.1f}%",
                  "表現遲滯" if roc_3m<0 else "趨勢向上", delta_color="normal")

        # ── 估值明細 + debug ──
        with st.expander("🔍 估值計算明細（含除錯紀錄）"):
            dc1,dc2,dc3 = st.columns(3)
            dc1.metric(fund['pe_label'] or "本益比", f"{pe:.1f}x" if pe else "N/A")
            dc2.metric(f"成長率", f"{growth*100:.1f}% ({fund['growth_label']})" if growth else "N/A")
            dc3.metric("PEG 來源", peg_src if peg_ratio else "無法計算")
            st.markdown("**除錯紀錄：**")
            for line in fund['debug']:
                st.text(line)

        # ── AI 評分 ──
        st.markdown("---")
        st.subheader("🤖 AI 動態評分與實戰紀律")
        details = []

        if peg_ratio and peg_ratio > 0:
            if peg_ratio <= 1:
                peg_score = 20
                details.append(f"✅ **估值優勢 (PEG={peg_ratio:.2f})**：成長足以支撐本益比。得分 20/20。")
            elif peg_ratio <= 1.5:
                peg_score = 10
                details.append(f"🟡 **估值中性 (PEG={peg_ratio:.2f})**：估值合理。得分 10/20。")
            else:
                peg_score = 0
                details.append(f"❌ **估值偏高 (PEG={peg_ratio:.2f})**：股價可能透支成長。得分 0/20。")
        else:
            peg_score = 10
            details.append("🔵 **估值**：無足夠數據計算 PEG，中立 10/20。")

        bias_25 = (last_price/last_ma25-1)*100
        if last_price > last_ma25 and last_ma25 > last_ma50:
            trend_score = min(max(10+bias_25*2,0),20)
            details.append(f"✅ **多頭排列**：股價穩站5週線，趨勢得分 {trend_score:.1f}/20。")
        else:
            trend_score = min(max(5+bias_25*2,0),10)
            details.append(f"🟡 **趨勢震盪**：尚未完美多頭，趨勢得分 {trend_score:.1f}/20。")

        vol_ratio = last_vol/last_vol20
        vol_score = min(max((vol_ratio-0.5)*15,0),20)
        if vol_ratio>1.2 and last_price>float(df['Open'].iloc[-1]):
            vol_score = min(vol_score+5,20)
            details.append(f"✅ **價漲量增**：主力介入，量能得分 {vol_score:.1f}/20。")
        elif vol_ratio>1.2:
            vol_score = max(vol_score-10,0)
            details.append(f"⚠️ **爆量收黑**：留意出貨，量能得分 {vol_score:.1f}/20。")
        else:
            details.append(f"🔵 **量能平穩**：市場觀望，量能得分 {vol_score:.1f}/20。")

        if last_macd>0 and last_hist>0:
            macd_score = 20
            details.append("✅ **動能強勁**：MACD 零軸上方發散，得分 20/20。")
        elif last_hist>0:
            macd_score = 10+min(last_hist/last_price*1000,10)
            details.append(f"🟡 **動能轉強**：MACD 紅柱，得分 {macd_score:.1f}/20。")
        else:
            macd_score = max(10-abs(last_hist)/last_price*1000,0)
            details.append(f"❌ **動能疲弱**：MACD 綠柱，得分 {macd_score:.1f}/20。")

        if 45<=last_rsi<=65:
            rsi_score=20; details.append(f"✅ **RSI 健康**：RSI={last_rsi:.1f}，得分 20/20。")
        elif last_rsi>65:
            rsi_score=max(20-(last_rsi-65)*1.5,0); details.append(f"🔴 **高檔過熱**：RSI={last_rsi:.1f}，得分 {rsi_score:.1f}/20。")
        else:
            rsi_score=max(20-(45-last_rsi)*1.5,0); details.append(f"🔵 **低檔超賣**：RSI={last_rsi:.1f}，得分 {rsi_score:.1f}/20。")

        total = peg_score+trend_score+vol_score+macd_score+rsi_score

        st.markdown("#### ⚔️ 嚴格紀律審查")
        if last_price<last_ma50:
            st.error(f"💀 **破底限**：跌破10週線({last_ma50:.2f})！請立刻出場。")
        elif last_price<last_ma25:
            st.warning(f"🚨 **跌破5週線**：警戒區({last_ma25:.2f})，緊盯走勢。")
        if roc_3m<0:
            st.error(f"⏳ **時間停損**：3個月累計 {roc_3m:.1f}%，建議換股！")

        st.markdown("#### 🏆 綜合雷達總分")
        if total>=80: st.success(f"🔥 {total:.1f} 分 — 強勢便宜，適合15-20%核心持股")
        elif total>=60: st.warning(f"⚖️ {total:.1f} 分 — 中等，未破5週線可續抱")
        else: st.error(f"⚠️ {total:.1f} 分 — 體質轉弱，建議汰除")
        for d in details: st.write(d)

        # ── 圖表 ──
        st.markdown("---")
        st.subheader("📈 技術分析圖表")
        fig = make_subplots(rows=3,cols=1,shared_xaxes=True,
                            vertical_spacing=0.03,row_heights=[0.6,0.2,0.2])
        fig.add_trace(go.Candlestick(x=df.index,open=df['Open'],high=df['High'],
            low=df['Low'],close=df['Close'],name='K線',
            increasing_line_color=color_up,decreasing_line_color=color_down),row=1,col=1)
        fig.add_trace(go.Scatter(x=df.index,y=df['MA25'],name='25MA(5週線)',
            line=dict(color='orange',width=2)),row=1,col=1)
        fig.add_trace(go.Scatter(x=df.index,y=df['MA50'],name='50MA(10週線)',
            line=dict(color='red',width=2,dash='dot')),row=1,col=1)
        fig.add_trace(go.Scatter(x=df.index,y=df['Resistance'],name='壓力',
            line=dict(color='#FF1493',width=1.5,shape='hv',dash='dot')),row=1,col=1)
        fig.add_trace(go.Scatter(x=df.index,y=df['Support'],name='支撐',
            line=dict(color='#00BFFF',width=1.5,shape='hv',dash='dot')),row=1,col=1)
        vc = [color_up if r['Close']>=r['Open'] else color_down for _,r in df.iterrows()]
        fig.add_trace(go.Bar(x=df.index,y=df['Volume'],name='成交量',marker_color=vc),row=2,col=1)
        hc = [color_up if v>0 else color_down for v in df['MACD_Hist']]
        fig.add_trace(go.Bar(x=df.index,y=df['MACD_Hist'],name='MACD柱',marker_color=hc),row=3,col=1)
        fig.add_trace(go.Scatter(x=df.index,y=df['MACD'],name='MACD',
            line=dict(color='blue',width=1.5)),row=3,col=1)
        fig.add_trace(go.Scatter(x=df.index,y=df['Signal'],name='Signal',
            line=dict(color='orange',width=1.5)),row=3,col=1)
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat","mon"])])
        fig.update_layout(height=800,template="plotly_white",
                          hovermode="x unified",xaxis_rangeslider_visible=False,
                          margin=dict(l=0,r=0,t=30,b=0))
        st.plotly_chart(fig,use_container_width=True)
    else:
        st.error("❌ 找不到K線數據，請確認代號（台股如 2330.TW）。")

st.markdown("---")
st.caption("免責聲明：本工具僅供參考，不構成投資建議。市場有風險，投資需謹慎。")
