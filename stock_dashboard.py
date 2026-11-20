"""台股個股分析儀表板(Streamlit + FinMind + yfinance)

安裝:  pip install streamlit pandas numpy plotly requests yfinance
執行:  streamlit run stock_dashboard.py
FinMind 免費版不填 token 也能用(有次數限制),註冊後填 token 額度更高。
"""
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

API = "https://api.finmindtrade.com/api/v4/data"
UP, DN = "#ff4d5a", "#26c281"  # 台股:紅漲綠跌

st.set_page_config(page_title="台股儀表板", layout="wide")


# ---------- 資料 ----------
@st.cache_data(ttl=3600)
def finmind(dataset, stock, start, token=""):
    r = requests.get(API, params={"dataset": dataset, "data_id": stock,
                                  "start_date": start, "token": token}, timeout=30).json()
    if r.get("status") != 200:
        raise RuntimeError(r.get("msg", "FinMind 回傳錯誤"))
    return pd.DataFrame(r["data"])


def load_price(stock, days, token):
    start = (date.today() - timedelta(days=days + 400)).isoformat()  # 多抓一年給 MA240
    df = finmind("TaiwanStockPrice", stock, start, token)
    df = df.rename(columns={"max": "high", "min": "low", "Trading_Volume": "volume"})
    df["date"] = pd.to_datetime(df["date"])
    df["volume"] = df["volume"] / 1000  # 股 -> 張
    return df.set_index("date")[["open", "high", "low", "close", "volume"]]


@st.cache_data(ttl=3600)
def twse(path):
    """證交所 OpenAPI;若本機遇到 SSL 憑證問題,自動改用 verify=False 重試。"""
    url = f"https://openapi.twse.com.tw/v1/{path}"
    try:
        r = requests.get(url, timeout=30)
    except requests.exceptions.SSLError:
        r = requests.get(url, timeout=30, verify=False)
    r.raise_for_status()
    return pd.DataFrame(r.json())


def parse_twse(df):
    """證交所 STOCK_DAY 的民國年日期與千分位字串 -> 數值 DataFrame。"""
    num = lambda c: pd.to_numeric(df[c].astype(str).str.replace(",", ""), errors="coerce")
    def ad(t):
        y, m, d = t.strip().split("/")
        return pd.Timestamp(int(y) + 1911, int(m), int(d))
    out = pd.DataFrame({"open": num("開盤價"), "high": num("最高價"), "low": num("最低價"),
                        "close": num("收盤價"), "volume": num("成交股數") / 1000})
    out.index = pd.DatetimeIndex(df["日期"].map(ad), name="date")
    return out.dropna().sort_index()


@st.cache_data(ttl=3600)
def twse_month(stock, ym):
    url = "https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY"
    kw = dict(params={"date": ym + "01", "stockNo": stock, "response": "json"},
              headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    try:
        r = requests.get(url, **kw)
    except requests.exceptions.SSLError:
        r = requests.get(url, verify=False, **kw)
    j = r.json()
    if j.get("stat") != "OK":
        return pd.DataFrame()
    return pd.DataFrame(j["data"], columns=j["fields"])


def load_price_twse(stock, days):
    """逐月向證交所抓個股日成交資料(未還原權息)。證交所會限流,故每個月之間暫停。"""
    import time
    months = pd.period_range(date.today() - timedelta(days=days + 400), date.today(), freq="M")
    seen = st.session_state.setdefault("twse_seen", set())
    bar, frames = st.progress(0, text="向證交所逐月下載…(首次約需 20 秒)"), []
    for i, m in enumerate(months):
        key = m.strftime("%Y%m")
        if (stock, key) not in seen:
            time.sleep(0.6)
        frames.append(twse_month(stock, key))
        seen.add((stock, key))
        bar.progress((i + 1) / len(months))
    bar.empty()
    frames = [f for f in frames if not f.empty]
    return parse_twse(pd.concat(frames, ignore_index=True)) if frames else pd.DataFrame()


# ---------- 指標 ----------
def add_indicators(df):
    d = df.copy()
    for n in (5, 20, 60, 120, 240):
        d[f"ma{n}"] = d.close.rolling(n).mean()
    sd = d.close.rolling(20).std(ddof=0)
    d["bb_up"], d["bb_lo"] = d.ma20 + 2 * sd, d.ma20 - 2 * sd
    lo9, hi9 = d.low.rolling(9).min(), d.high.rolling(9).max()
    rsv = (d.close - lo9) / (hi9 - lo9).replace(0, np.nan) * 100
    d["k"] = rsv.ewm(alpha=1 / 3, adjust=False).mean()
    d["d"] = d.k.ewm(alpha=1 / 3, adjust=False).mean()
    delta = d.close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    d["rsi"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    d["dif"] = d.close.ewm(span=12, adjust=False).mean() - d.close.ewm(span=26, adjust=False).mean()
    d["dea"] = d.dif.ewm(span=9, adjust=False).mean()
    d["osc"] = d.dif - d.dea
    return d


# ---------- 筆記上的 MACD 規則 ----------
def macd_signal(osc: pd.Series, shrink_bars: int = 2):
    """綠柱連續縮小 + 當根轉紅 -> 看多(快速);紅柱縮小後轉綠 -> 看空。"""
    s = osc.dropna()
    if len(s) < shrink_bars + 3:
        return "資料不足", "off"
    cur, prev = s.iloc[-1], s.iloc[-2]

    def run_before_last(sign):
        vals = []
        for v in s.iloc[-2::-1]:
            if np.sign(v) != sign:
                break
            vals.append(abs(v))
        return vals[::-1]  # 由舊到新

    def shrinking(vals):
        return len(vals) > shrink_bars and all(vals[i] < vals[i - 1] for i in range(-shrink_bars, 0))

    if cur > 0 >= prev and shrinking(run_before_last(-1)):
        return f"綠柱連縮 {shrink_bars} 根後轉紅:看多(快速)", "bull"
    if cur < 0 <= prev and shrinking(run_before_last(1)):
        return f"紅柱連縮 {shrink_bars} 根後轉綠:看空", "bear"
    return ("紅柱中" if cur > 0 else "綠柱中") + ",無轉折訊號", "off"


def show(msg, kind):
    {"bull": st.success, "bear": st.error}.get(kind, st.info)(msg)


# ---------- 介面 ----------
with st.sidebar:
    stock = st.text_input("股票代號", "2454")
    days = st.slider("顯示天數", 60, 500, 250)
    source = st.radio("股價資料來源", ["FinMind", "證交所官方"])
    token = st.text_input("FinMind token(選填)", type="password")
    shrink = st.number_input("綠(紅)柱連續縮小幾根才算", 1, 5, 2)

st.title(f"{stock} 個股分析")
tab1, tab2, tab3, tab4, tab5 = st.tabs(["日 K 與指標", "籌碼", "月營收", "盤中訊號", "證交所官方快照"])

try:
    raw = load_price(stock, days, token) if source == "FinMind" else load_price_twse(stock, days)
    if raw.empty:
        raise RuntimeError("查無資料(證交所只含上市股票,上櫃請改用 FinMind)")
    px = add_indicators(raw)
except Exception as e:
    st.error(f"抓不到股價:{e}")
    st.stop()
view = px.tail(days)

with tab1:
    last, prev = px.iloc[-1], px.iloc[-2]
    c = st.columns(5)
    c[0].metric("收盤", f"{last.close:,.2f}", f"{last.close - prev.close:+.2f}")
    c[1].metric("成交量(張)", f"{last.volume:,.0f}")
    c[2].metric("RSI(14)", f"{last.rsi:.1f}")
    c[3].metric("K / D", f"{last.k:.0f} / {last.d:.0f}")
    c[4].metric("MACD 柱", f"{last.osc:.2f}")
    show(*macd_signal(px.osc, shrink))

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[.6, .2, .2], vertical_spacing=.03)
    fig.add_trace(go.Candlestick(x=view.index, open=view.open, high=view.high, low=view.low, close=view.close,
                                 increasing_line_color=UP, decreasing_line_color=DN, name="K 線"), 1, 1)
    for n, col in ((5, "#ffa94d"), (20, "#ff7a45"), (60, "#26c281"), (120, "#4da3ff"), (240, "#b07cff")):
        fig.add_trace(go.Scatter(x=view.index, y=view[f"ma{n}"], name=f"MA{n}",
                                 line=dict(width=1.2, color=col), visible=True if n <= 60 else "legendonly"), 1, 1)
    for col in ("bb_up", "bb_lo"):
        fig.add_trace(go.Scatter(x=view.index, y=view[col], name="布林帶", legendgroup="bb",
                                 showlegend=col == "bb_up", line=dict(width=1, dash="dot", color="#4d8dff")), 1, 1)
    fig.add_trace(go.Bar(x=view.index, y=view.osc, name="OSC",
                         marker_color=np.where(view.osc >= 0, UP, DN)), 2, 1)
    fig.add_trace(go.Scatter(x=view.index, y=view.dif, name="DIF", line=dict(color="#ffb84d", width=1)), 2, 1)
    fig.add_trace(go.Scatter(x=view.index, y=view.dea, name="DEA", line=dict(color="#4da3ff", width=1)), 2, 1)
    fig.add_trace(go.Scatter(x=view.index, y=view.k, name="K", line=dict(color="#ffb84d", width=1)), 3, 1)
    fig.add_trace(go.Scatter(x=view.index, y=view.d, name="D", line=dict(color="#4da3ff", width=1)), 3, 1)
    fig.add_trace(go.Scatter(x=view.index, y=view.rsi, name="RSI", line=dict(color="#b07cff", width=1)), 3, 1)
    fig.update_layout(template="plotly_dark", height=760, xaxis_rangeslider_visible=False,
                      margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    try:
        if source != "FinMind":
            raise RuntimeError("官方模式不含此資料,請在側邊欄改選 FinMind")
        start = (date.today() - timedelta(days=days)).isoformat()
        inst = finmind("TaiwanStockInstitutionalInvestorsBuySell", stock, start, token)
        inst["net"] = (inst.buy - inst.sell) / 1000
        grp = {"Foreign_Investor": "外資", "Foreign_Dealer_Self": "外資", "Investment_Trust": "投信",
               "Dealer_self": "自營商", "Dealer_Hedging": "自營商"}
        inst["who"] = inst.name.map(grp)
        pv = inst.dropna(subset=["who"]).pivot_table(index="date", columns="who", values="net", aggfunc="sum")
        st.subheader("三大法人累計買賣超(張)")
        st.line_chart(pv.cumsum())
        mg = finmind("TaiwanStockMarginPurchaseShortSale", stock, start, token).set_index("date")
        st.subheader("融資 / 融券餘額")
        st.line_chart(mg[["MarginPurchaseTodayBalance", "ShortSaleTodayBalance"]])
    except Exception as e:
        st.warning(f"籌碼資料抓取失敗:{e}")

with tab3:
    try:
        if source != "FinMind":
            raise RuntimeError("官方模式不含此資料,請在側邊欄改選 FinMind")
        rv = finmind("TaiwanStockMonthRevenue", stock, (date.today() - timedelta(days=1100)).isoformat(), token)
        rv["營收(億)"] = rv.revenue / 1e8
        rv["年增率(%)"] = rv.revenue.pct_change(12) * 100
        rv = rv.set_index("date").tail(24)
        st.bar_chart(rv["營收(億)"])
        st.line_chart(rv["年增率(%)"])
    except Exception as e:
        st.warning(f"月營收抓取失敗:{e}")

with tab4:
    st.caption("盤中資料用 yfinance(免費,1 分 K 只有最近約 7 天)。交易時間外會顯示最近一個交易日。")
    if st.button("更新盤中訊號"):
        import yfinance as yf

        def intraday(interval):
            d = yf.download(f"{stock}.TW", period="5d", interval=interval, progress=False, auto_adjust=False)
            d.columns = d.columns.get_level_values(0)
            d = d.rename(columns=str.lower).dropna()
            d["ma5"], d["ma20"] = d.close.rolling(5).mean(), d.close.rolling(20).mean()
            d["osc"] = (d.close.ewm(span=12, adjust=False).mean() - d.close.ewm(span=26, adjust=False).mean()
                        ).pipe(lambda x: x - x.ewm(span=9, adjust=False).mean())
            return d

        m5, m1 = intraday("5m"), intraday("1m")
        st.subheader("5 分 K")
        show(*macd_signal(m5.osc, shrink))
        below = m5.close.iloc[-1] < m5.ma5.iloc[-1]
        broke = below and m5.close.iloc[-2] >= m5.ma5.iloc[-2]
        show("收盤跌破 5MA(剛跌破)" if broke else "收盤在 5MA 下方" if below else "收盤在 5MA 上方",
             "bear" if below else "off")
        st.subheader("1 分 K")
        show(*macd_signal(m1.osc, shrink))
        x = (m1.ma5 - m1.ma20).dropna().tail(6)
        dead = bool(len(x) > 1 and (x.iloc[:-1] > 0).any() and x.iloc[-1] < 0 and
                    (np.sign(x).diff().dropna() < 0).any())
        show("MA5 近 5 根內向下穿越 MA20(死亡交叉)" if dead else "1 分 K 無死亡交叉",
             "bear" if dead else "off")
        st.subheader("綜合")
        if below and dead:
            st.error("5 分 K 跌破 5MA 且 1 分 K 死叉:同步看空")
        else:
            st.info("尚未同步看空")
        st.plotly_chart(go.Figure(go.Candlestick(
            x=m5.index, open=m5.open, high=m5.high, low=m5.low, close=m5.close,
            increasing_line_color=UP, decreasing_line_color=DN)).update_layout(
            template="plotly_dark", height=380, xaxis_rangeslider_visible=False,
            margin=dict(l=10, r=10, t=10, b=10)), use_container_width=True)
    st.caption("訊號規則依筆記整理,僅供研究,不構成投資建議。")

with tab5:
    st.caption("來源:證交所 OpenAPI(只含上市股票;多為最新一個交易日的全市場資料,不含歷史)")
    for title, path in [("每日收盤行情", "exchangeReport/STOCK_DAY_ALL"),
                        ("本益比 / 殖利率 / 股價淨值比", "exchangeReport/BWIBBU_ALL"),
                        ("融資融券彙總", "exchangeReport/MI_MARGN"),
                        ("最新月營收", "opendata/t187ap05_L")]:
        st.subheader(title)
        try:
            df = twse(path)
            col = next((c for c in df.columns if c in ("Code", "公司代號", "股票代號")), None)
            row = df[df[col].astype(str).str.strip() == stock] if col else df.head(0)
            if row.empty:
                st.info("這張表找不到這檔(可能是上櫃股票,或該表沒有個股欄位)")
            else:
                st.dataframe(row.T.rename(columns={row.index[0]: "值"}), use_container_width=True)
        except Exception as e:
            st.warning(f"抓取失敗:{e}")
