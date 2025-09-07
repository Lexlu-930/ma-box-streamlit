# -*- coding: utf-8 -*-
# 技術指標面板（MA / 量價 / KD / MACD）+ 多股票對比 + 箱型進出說明
# 新增 (v1.4.3):
#  - 個股分頁加入「當日數據／計算結果」文字說明（MA=建議買價、上軌=建議賣價、下軌、量能過濾）
#  - 可選填「持有成本/停利(%)/停損(%)」，顯示「依成本計算」（目前損益%、停利價/停損價）
# 保留:
#  - 相對表現期間切換（近1月 / 近3月 / 近半年 / YTD）
#  - 黃金交叉/死亡交叉（MA5×MA20、MACD×Signal）標註
#  - 下載 CSV / Excel
#  - yfinance 失敗自動 TWSE 備援
#  - 多股票排名表（近N天報酬、波動、量能達標次數）
# 作者: LexLu
# 版本: v1.4.3 (2025-09-07)

import os
import io
import requests
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# ====== 基本資訊 ======
AUTHOR = "LexLu"
VERSION = "v1.4.3 (2025-09-07)"
YEAR = datetime.now().year

# ====== 字型設定 ======
try:
    from matplotlib import font_manager
    FONT_PATH = os.path.join(os.path.dirname(__file__), "fonts", "NotoSansCJKtc-Regular.otf")
    if os.path.exists(FONT_PATH):
        font_manager.fontManager.addfont(FONT_PATH)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT_PATH).get_name()
    else:
        plt.rcParams["font.sans-serif"] = [
            "Microsoft JhengHei", "SimHei", "PMingLiU", "Noto Sans CJK TC",
            "Noto Sans CJK SC", "PingFang TC", "PingFang SC", "WenQuanYi Zen Hei"
        ]
    plt.rcParams["axes.unicode_minus"] = False
except Exception:
    pass

# ====== yfinance ======
try:
    import yfinance as yf
    HAS_YF = True
except Exception:
    HAS_YF = False


# ================= 工具 =================
def ensure_scalar(x):
    try:
        if isinstance(x, pd.DataFrame):
            if x.empty: return np.nan
            return ensure_scalar(x.iloc[-1].squeeze())
        elif isinstance(x, (pd.Series, list, tuple, np.ndarray)):
            if len(x) == 0: return np.nan
            return ensure_scalar(x[-1])
        else:
            v = pd.to_numeric(x, errors="coerce")
            return float(v) if pd.notna(v) else np.nan
    except Exception:
        return np.nan


def parse_end_date(end_date_str: str) -> pd.Timestamp | None:
    if not end_date_str.strip():
        return pd.Timestamp.today().normalize()
    try:
        return pd.to_datetime(end_date_str.strip())
    except Exception:
        return None


def period_start_from_choice(end_date: pd.Timestamp, choice: str) -> pd.Timestamp:
    choice = choice.strip()
    if choice == "近1月":   return end_date - pd.Timedelta(days=30)
    if choice == "近3月":   return end_date - pd.Timedelta(days=90)
    if choice == "近半年":  return end_date - pd.Timedelta(days=183)
    if choice == "YTD":    return pd.Timestamp(year=end_date.year, month=1, day=1)
    return end_date - pd.Timedelta(days=90)


# ====== TWSE 備援 ======
def _tw_yyyymmdd(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y%m%d")

def _roc_to_gregorian(date_str: str) -> pd.Timestamp | None:
    try:
        y, m, d = date_str.split("/")
        year = int(y) + 1911
        return pd.Timestamp(year=year, month=int(m), day=int(d))
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def twse_stock_day_range(stock_no: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    dfs = []
    months = pd.period_range(start=start, end=end, freq="M")
    if len(months) == 0:
        months = pd.period_range(start=start, end=end + pd.Timedelta(days=1), freq="M")
    for p in months:
        dt = pd.Timestamp(p.start_time)
        url = "https://www.twse.com.tw/exchangeReport/STOCK_DAY"
        params = {"response": "json", "date": _tw_yyyymmdd(dt), "stockNo": stock_no}
        try:
            r = requests.get(url, params=params, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
            js = r.json()
            if js.get("stat") != "OK":
                continue
            fields = js.get("fields", [])
            data = js.get("data", [])
            fidx = {name: i for i, name in enumerate(fields)}
            rows = []
            for row in data:
                dt_ = _roc_to_gregorian(row[fidx["日期"]]) if "日期" in fidx else None
                if dt_ is None or not (start <= dt_ <= end):
                    continue
                def to_float(s):
                    s = str(s).replace(",", "").replace("X", "").strip()
                    try: return float(s)
                    except: return np.nan
                def to_int(s):
                    s = str(s).replace(",", "").strip()
                    try: return int(s)
                    except: return np.nan
                openp = to_float(row[fidx.get("開盤價", -1)]) if "開盤價" in fidx else np.nan
                high  = to_float(row[fidx.get("最高價", -1)]) if "最高價" in fidx else np.nan
                low   = to_float(row[fidx.get("最低價", -1)]) if "最低價" in fidx else np.nan
                close = to_float(row[fidx.get("收盤價", -1)]) if "收盤價" in fidx else np.nan
                volume = to_int(row[fidx.get("成交股數", -1)]) if "成交股數" in fidx else np.nan
                rows.append((dt_, openp, high, low, close, volume))
            if rows:
                dfm = pd.DataFrame(rows, columns=["DATE", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]).set_index("DATE")
                dfs.append(dfm)
        except Exception:
            continue
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs).sort_index()


# ====== yfinance 多層欄位處理 ======
def _pick_series_any_level(df: pd.DataFrame, name: str, preferred_symbol: str | None = None):
    if isinstance(df.columns, pd.MultiIndex):
        for level in range(df.columns.nlevels):
            vals = df.columns.get_level_values(level)
            if name in vals:
                sub = df.xs(name, axis=1, level=level, drop_level=True)
                if isinstance(sub, pd.DataFrame):
                    if preferred_symbol and preferred_symbol in sub.columns:
                        return sub[preferred_symbol]
                    return sub.iloc[:, 0]
                return sub
        return None
    else:
        return df[name] if name in df.columns else None


@st.cache_data(show_spinner=False)
def load_price_data(ticker: str, end_date_str: str, lookback_days: int) -> pd.DataFrame:
    end = parse_end_date(end_date_str)
    if end is None:
        return pd.DataFrame()
    start = end - pd.Timedelta(days=max(lookback_days * 2, 120))
    yf_symbol = f"{ticker}.TW" if ticker.isdigit() else ticker

    # 1) yfinance
    if HAS_YF:
        try:
            df = yf.download(
                yf_symbol,
                start=start.strftime("%Y-%m-%d"),
                end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                progress=False,
                auto_adjust=True,
            )
            if not df.empty:
                close_s = _pick_series_any_level(df, "Close", preferred_symbol=yf_symbol) \
                          or _pick_series_any_level(df, "Adj Close", preferred_symbol=yf_symbol)
                vol_s = _pick_series_any_level(df, "Volume", preferred_symbol=yf_symbol)
                open_s = _pick_series_any_level(df, "Open", preferred_symbol=yf_symbol)
                high_s = _pick_series_any_level(df, "High", preferred_symbol=yf_symbol)
                low_s  = _pick_series_any_level(df, "Low",  preferred_symbol=yf_symbol)
                if close_s is not None and vol_s is not None:
                    out = pd.DataFrame({
                        "OPEN": open_s if open_s is not None else np.nan,
                        "HIGH": high_s if high_s is not None else np.nan,
                        "LOW":  low_s  if low_s  is not None else np.nan,
                        "CLOSE": close_s,
                        "VOLUME": vol_s,
                    }).dropna(subset=["CLOSE"])
                    out.attrs["simulated"] = False
                    out.attrs["source"] = "yfinance"
                    return out
        except Exception:
            pass

    # 2) TWSE 備援
    if ticker.isdigit():
        tw = twse_stock_day_range(ticker, start, end)
        if not tw.empty:
            tw.attrs["simulated"] = False
            tw.attrs["source"] = "twse"
            return tw

    # 3) 模擬
    rng = pd.date_range(end=end, periods=lookback_days, freq="B")
    close = np.linspace(100, 110, len(rng)) + np.random.normal(0, 1.5, len(rng))
    openp = close + np.random.normal(0, 0.6, len(rng))
    high  = np.maximum(openp, close) + np.random.uniform(0.1, 0.8, len(rng))
    low   = np.minimum(openp, close) - np.random.uniform(0.1, 0.8, len(rng))
    volume = np.random.randint(1200, 3000, size=len(rng))
    demo = pd.DataFrame({"OPEN": openp, "HIGH": high, "LOW": low, "CLOSE": close, "VOLUME": volume}, index=rng)
    demo.attrs["simulated"] = True
    demo.attrs["source"] = "simulated"
    return demo


# ================== 技術指標 ==================
def add_mas(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    for w in windows:
        df[f"MA{w}"] = df["CLOSE"].rolling(w).mean()
    return df

def add_vol_ma(df: pd.DataFrame, vol_win: int) -> pd.DataFrame:
    df["VOL_MA"] = df["VOLUME"].rolling(vol_win).mean()
    return df

def add_boll(df: pd.DataFrame, boll_win: int = 20, k: float = 2.0) -> pd.DataFrame:
    ma = df["CLOSE"].rolling(boll_win).mean()
    std = df["CLOSE"].rolling(boll_win).std(ddof=0)
    df["BOLL_MA"] = ma
    df["BOLL_UPPER"] = ma + k * std
    df["BOLL_LOWER"] = ma - k * std
    return df

def add_kd(df: pd.DataFrame, n: int = 9, k_smooth: int = 3, d_smooth: int = 3) -> pd.DataFrame:
    low_n = df["LOW"].rolling(n).min()
    high_n = df["HIGH"].rolling(n).max()
    denom = (high_n - low_n).replace(0, np.nan)
    rsv = 100 * (df["CLOSE"] - low_n) / denom
    k = rsv.rolling(k_smooth).mean()
    d = k.rolling(d_smooth).mean()
    df["%K"] = k
    df["%D"] = d
    return df

def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = df["CLOSE"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["CLOSE"].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - macd_signal
    df["MACD"] = macd
    df["MACD_SIGNAL"] = macd_signal
    df["MACD_HIST"] = hist
    return df


# =============== 布林 + 量能 ===============
def analyze_core(df: pd.DataFrame, vol_filter: bool, vol_win: int, k_boll: float, boll_win: int):
    df = df.copy()
    df = add_vol_ma(df, vol_win=vol_win)
    df = add_boll(df, boll_win=boll_win, k=k_boll)

    last = df.iloc[-1]
    close = ensure_scalar(last["CLOSE"])
    ma = ensure_scalar(df["BOLL_MA"].iloc[-1])
    std = ensure_scalar((df["BOLL_UPPER"].iloc[-1] - df["BOLL_MA"].iloc[-1]) / k_boll) if k_boll != 0 else np.nan
    vol = ensure_scalar(last["VOLUME"])
    volma = ensure_scalar(last["VOL_MA"])
    upper = ensure_scalar(last["BOLL_UPPER"])
    lower = ensure_scalar(last["BOLL_LOWER"])

    vol_ok = True
    vol_msg = ""
    if vol_filter:
        if pd.notna(vol) and pd.notna(volma):
            vol_ok = vol >= volma
            if not vol_ok: vol_msg = "量能不足（最後一日 < 量均）"
        else:
            vol_ok = False
            vol_msg = "量能資料不足，無法判斷"

    pos = "N/A"
    if pd.notna(close) and pd.notna(ma) and pd.notna(upper):
        if close < ma: pos = "低於中軌"
        elif close >= upper: pos = "接近/高於上軌"
        else: pos = "介於中軌與上軌"

    return df, dict(close=close, ma=ma, std=std, vol=vol, volma=volma,
                    upper=upper, lower=lower, vol_ok=vol_ok, vol_msg=vol_msg, pos=pos)


# ====== 交叉點偵測 ======
def detect_cross(x: pd.Series, y: pd.Series):
    x = x.astype(float); y = y.astype(float)
    prev = (x.shift(1) - y.shift(1))
    now = (x - y)
    gold = (prev <= 0) & (now > 0)
    death = (prev >= 0) & (now < 0)
    return x.index[gold.fillna(False)], x.index[death.fillna(False)]


# ====== 下載 ======
def make_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=True).encode("utf-8-sig")

def make_excel_bytes(df: pd.DataFrame) -> bytes | None:
    try:
        import openpyxl  # noqa
        bio = io.BytesIO()
        with pd.ExcelWriter(bio, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="sheet1")
        bio.seek(0)
        return bio.read()
    except Exception:
        return None


# ====== 箱型文字報告 ======
def build_box_report(m: dict, use_vol_filter: bool, cost: float | None, tp_pct: float | None, sl_pct: float | None) -> str:
    close = m["close"]; ma = m["ma"]; upper = m["upper"]; lower = m["lower"]
    vol = m["vol"]; volma = m["volma"]; vol_ok = m["vol_ok"]; vol_msg = m.get("vol_msg","")

    lines = []
    lines.append("— 當日數據 / 計算結果 —")
    lines.append(f"收盤價: {close:.2f}" if pd.notna(close) else "收盤價: N/A")
    lines.append(f"日線均價(MA) = 建議買價: {ma:.2f}" if pd.notna(ma) else "日線均價(MA) = 建議買價: N/A")
    lines.append(f"箱型上限 = 建議賣價: {upper:.2f}" if pd.notna(upper) else "箱型上限 = 建議賣價: N/A")
    lines.append(f"箱型下限: {lower:.2f}" if pd.notna(lower) else "箱型下限: N/A")
    lines.append(f"成交量 / 量均: "
                 f"{int(vol) if pd.notna(vol) else 'N/A'} / {int(volma) if pd.notna(volma) else 'N/A'}")
    if use_vol_filter:
        lines.append(f"量能過濾判定: {'通過' if vol_ok else '未通過'}{('（'+vol_msg+'）' if vol_msg else '')}")

    # 依成本計算
    if cost is not None and pd.notna(cost):
        lines.append("")
        lines.append("— 依成本計算 —")
        tp_price = cost * (1 + (tp_pct or 0)/100.0) if tp_pct is not None else np.nan
        sl_price = cost * (1 - (sl_pct or 0)/100.0) if sl_pct is not None else np.nan
        profit_pct = ((close - cost) / cost * 100.0) if (pd.notna(close) and cost != 0) else np.nan
        lines.append(f"目前損益: {profit_pct:.2f}%" if pd.notna(profit_pct) else "目前損益: N/A")
        lines.append(f"停利價（+{tp_pct}%）: {tp_price:.2f}" if (tp_pct is not None and pd.notna(tp_price)) else "停利價: （未設定）")
        lines.append(f"停損價（-{sl_pct}%）: {sl_price:.2f}" if (sl_pct is not None and pd.notna(sl_price)) else "停損價: （未設定）")

    # 交易提示
    advice = []
    if pd.notna(close) and pd.notna(ma):
        if close < ma: advice.append("價格低於MA，可觀察逢低/分批")
        elif pd.notna(upper) and close < upper: advice.append("價格位於MA~上軌之間，持有可續抱")
        elif pd.notna(upper) and close >= upper: advice.append("接近/突破上軌，考慮部分了結")
    if use_vol_filter and not vol_ok:
        advice.append("量能不足，保守應對")

    if advice:
        lines.append("")
        lines.append("— 交易提示（參考） —")
        for s in advice:
            lines.append("• " + s)

    return "\n".join(lines)


# ================= Streamlit UI =================
st.set_page_config(page_title="技術指標面板（含多股票對比）", layout="wide")

st.title("技術指標面板（MA / 量價 / KD / MACD）")
st.caption(f"作者: **{AUTHOR}** ｜ 版本: **{VERSION}**")

st.sidebar.title("ℹ️ 功能")
st.sidebar.markdown(
    "- 移動平均線（可多條）\n"
    "- 量價（含量均）\n"
    "- KD / MACD（含交叉點）\n"
    "- 多股票相對表現（近1月/3月/半年/YTD）\n"
    "- 下載 CSV/Excel\n"
    "- yfinance 斷線時自動 TWSE 備援\n"
)

with st.form("params"):
    c1, c2, c3, c4 = st.columns([1.2, 1.1, 1, 1])

    with c1:
        tickers_str = st.text_input("股票代碼（可多個，逗號分隔）", value="2330, 2317, 3481")
        end_dt_str = st.text_input("結束日期（YYYY-MM-DD，可留空=今天）", value="")
        lookback = st.number_input("觀察天數 N（近 N 天）", min_value=30, max_value=3650, value=180, step=1)
        compare_period = st.selectbox("相對表現期間", ["近1月", "近3月", "近半年", "YTD"], index=1)

    with c2:
        vol_win = st.number_input("量均視窗（天）", min_value=2, max_value=365, value=20, step=1)
        vol_filter = st.checkbox("啟用量能過濾（最後一日 成交量 ≥ 量均）", value=False)
        show_kd = st.checkbox("顯示 KD（個股分頁）", value=True)
        show_macd = st.checkbox("顯示 MACD（個股分頁）", value=True)

    with c3:
        ma_list_str = st.text_input("移動平均線天數（逗號分隔）", value="5,10,20,60")
        boll_win = st.number_input("布林帶 MA 天數", min_value=5, max_value=120, value=20, step=1)
        k_boll = st.number_input("布林帶 k", min_value=0.5, max_value=4.0, value=2.0, step=0.1)

    with c4:
        macd_fast = st.number_input("MACD 快線", min_value=5, max_value=20, value=12, step=1)
        macd_slow = st.number_input("MACD 慢線", min_value=10, max_value=40, value=26, step=1)
        macd_signal = st.number_input("MACD 訊號線", min_value=5, max_value=20, value=9, step=1)
        cost_str = st.text_input("持有成本（可留空）", value="")
        tp_pct = st.number_input("停利(%)", value=8.0, step=0.5)
        sl_pct = st.number_input("停損(%)", value=5.0, step=0.5)

    submitted = st.form_submit_button("開始分析")

if submitted:
    # 解析代碼
    tickers = [t.strip() for t in tickers_str.split(",") if t.strip()]
    if not tickers:
        st.error("請輸入至少一個代碼")
        st.stop()

    # 多條MA
    try:
        ma_windows = sorted({int(x.strip()) for x in ma_list_str.split(",") if x.strip().isdigit()})
        ma_windows = [w for w in ma_windows if w > 0]
    except Exception:
        ma_windows = [5, 10, 20, 60]

    # 成本/停利停損
    cost_val = ensure_scalar(cost_str) if cost_str.strip() else np.nan
    cost_for_report = cost_val if pd.notna(cost_val) else None

    results = {}
    for tk in tickers:
        df_raw = load_price_data(tk, end_dt_str, int(lookback))
        if df_raw.empty:
            continue
        df_proc, metrics = analyze_core(
            df=df_raw, vol_filter=vol_filter, vol_win=int(vol_win), k_boll=float(k_boll), boll_win=int(boll_win)
        )
        df_proc = add_mas(df_proc, ma_windows + [5, 20])  # 確保 MA5/MA20
        df_proc = add_kd(df_proc, n=9, k_smooth=3, d_smooth=3)
        df_proc = add_macd(df_proc, fast=int(macd_fast), slow=int(macd_slow), signal=int(macd_signal))

        vol_pass = (df_proc["VOLUME"] >= df_proc["VOL_MA"]).tail(int(lookback)).sum()

        results[tk] = {
            "df": df_proc,
            "metrics": metrics,
            "vol_pass": int(vol_pass),
            "source": df_raw.attrs.get("source", "unknown"),
            "sim": df_raw.attrs.get("simulated", False),
        }

    if not results:
        st.error("所有代碼皆讀不到資料，請確認輸入。")
        st.stop()

    # ---- 排名表 ----
    rows = []
    for tk, r in results.items():
        dfp = r["df"].tail(int(lookback))
        if dfp["CLOSE"].dropna().shape[0] < 2:
            ret = np.nan; vol_ = np.nan
        else:
            ret = dfp["CLOSE"].iloc[-1] / dfp["CLOSE"].iloc[0] - 1
            vol_ = dfp["CLOSE"].pct_change().dropna().std()
        rows.append({
            "代碼": tk,
            "近N天報酬(%)": f"{ret*100:.2f}%" if pd.notna(ret) else "N/A",
            "日波動(σ)": f"{vol_*100:.2f}%" if pd.notna(vol_) else "N/A",
            "量能達標次數": r["vol_pass"],
            "資料來源": r["source"] + (" (模擬)" if r["sim"] else ""),
        })
    st.subheader("多股票排名表（近 N 天）")
    st.dataframe(pd.DataFrame(rows))

    # ---- 相對表現圖 ----
    st.subheader(f"多股票相對表現（{compare_period}）")
    end_dt = parse_end_date(end_dt_str) or pd.Timestamp.today().normalize()
    start_cmp = period_start_from_choice(end_dt, compare_period)

    figC = plt.figure(figsize=(11, 4.2))
    axC = plt.gca()
    for tk, r in results.items():
        ser = r["df"].loc[start_cmp:end_dt]["CLOSE"].dropna()
        if ser.empty: continue
        base = ser.iloc[0]
        if base and not np.isnan(base) and base != 0:
            axC.plot(ser.index, ser / base * 100.0, label=tk)
    axC.legend(); axC.set_ylabel("相對表現（基準=100）"); axC.set_title("相對表現對比")
    locator = mdates.AutoDateLocator(); formatter = mdates.ConciseDateFormatter(locator)
    axC.xaxis.set_major_locator(locator); axC.xaxis.set_major_formatter(formatter)
    figC.autofmt_xdate(rotation=45)
    st.pyplot(figC, clear_figure=True)

    # ---- 個股分頁（含箱型文字報告 + 交叉標註 + 下載）----
    st.subheader("個股詳細圖表")
    tabs = st.tabs(tickers)
    for i, tk in enumerate(tickers):
        if tk not in results:
            with tabs[i]: st.warning(f"{tk} 無資料"); continue
        r = results[tk]; dfp = r["df"]; m = r["metrics"]
        with tabs[i]:
            if r["sim"]:
                st.warning("⚠️ 此檔目前使用模擬/替代資料（僅示範用途）。")

            # 頂部卡片
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("收盤價", f"{m['close']:.2f}" if pd.notna(m['close']) else "N/A")
            c2.metric("中軌MA(=建議買價)", f"{m['ma']:.2f}" if pd.notna(m['ma']) else "N/A")
            c3.metric("上軌(=建議賣價)", f"{m['upper']:.2f}" if pd.notna(m['upper']) else "N/A")
            c4.metric("下靶(箱型下限)", f"{m['lower']:.2f}" if pd.notna(m['lower']) else "N/A")

            # ---- 箱型文字說明（回來了！）----
            report = build_box_report(m, use_vol_filter=vol_filter,
                                      cost=cost_for_report, tp_pct=tp_pct, sl_pct=sl_pct)
            st.text(report)

            # ---- 價格 + 多MA + 布林 + MA5/20 交叉 ----
            fig1 = plt.figure(figsize=(10.8, 4.2)); ax1 = plt.gca()
            ax1.plot(dfp.index, dfp["CLOSE"], label="收盤")
            for w in ma_windows:
                col = f"MA{w}"
                if col in dfp: ax1.plot(dfp.index, dfp[col], label=f"MA{w}")
            ax1.plot(dfp.index, dfp["BOLL_MA"], label=f"BOLL_MA({int(boll_win)})")
            ax1.plot(dfp.index, dfp["BOLL_UPPER"], label="上軌")
            ax1.plot(dfp.index, dfp["BOLL_LOWER"], label="下軌")
            if "MA5" in dfp and "MA20" in dfp:
                g_idx, d_idx = detect_cross(dfp["MA5"], dfp["MA20"])
                ax1.scatter(g_idx, dfp.loc[g_idx, "CLOSE"], marker="^", s=60, label="MA5↑MA20", zorder=3)
                ax1.scatter(d_idx, dfp.loc[d_idx, "CLOSE"], marker="v", s=60, label="MA5↓MA20", zorder=3)
            ax1.legend(); ax1.set_title(f"{tk}｜價格 / 多MA / 布林（含 MA5×MA20 交叉）")
            locator = mdates.AutoDateLocator(); formatter = mdates.ConciseDateFormatter(locator)
            ax1.xaxis.set_major_locator(locator); ax1.xaxis.set_major_formatter(formatter)
            fig1.autofmt_xdate(rotation=45)
            st.pyplot(fig1, clear_figure=True)

            # ---- 量價 ----
            fig2 = plt.figure(figsize=(10.8, 2.8)); ax2 = plt.gca()
            ax2.bar(dfp.index, dfp["VOLUME"], width=0.8, label="成交量")
            if "VOL_MA" in dfp: ax2.plot(dfp.index, dfp["VOL_MA"], label=f"量均({int(vol_win)})")
            ax2.legend(); ax2.set_title("量價（成交量與量均）")
            ax2.xaxis.set_major_locator(locator); ax2.xaxis.set_major_formatter(formatter)
            fig2.autofmt_xdate(rotation=45)
            st.pyplot(fig2, clear_figure=True)

            # ---- KD ----
            if show_kd:
                fig3 = plt.figure(figsize=(10.8, 2.8)); ax3 = plt.gca()
                ax3.plot(dfp.index, dfp["%K"], label="%K"); ax3.plot(dfp.index, dfp["%D"], label="%D")
                ax3.axhline(80, linestyle="--", linewidth=1); ax3.axhline(20, linestyle="--", linewidth=1)
                ax3.legend(); ax3.set_title("KD")
                ax3.xaxis.set_major_locator(locator); ax3.xaxis.set_major_formatter(formatter)
                fig3.autofmt_xdate(rotation=45)
                st.pyplot(fig3, clear_figure=True)

            # ---- MACD + 交叉 ----
            if show_macd:
                fig4 = plt.figure(figsize=(10.8, 3.0)); ax4 = plt.gca()
                ax4.plot(dfp.index, dfp["MACD"], label="MACD")
                ax4.plot(dfp.index, dfp["MACD_SIGNAL"], label="Signal")
                ax4.bar(dfp.index, dfp["MACD_HIST"], width=0.8, alpha=0.5, label="Hist")
                g2, d2 = detect_cross(dfp["MACD"], dfp["MACD_SIGNAL"])
                ax4.scatter(g2, dfp.loc[g2, "MACD"], marker="^", s=50, label="MACD↑Signal", zorder=3)
                ax4.scatter(d2, dfp.loc[d2, "MACD"], marker="v", s=50, label="MACD↓Signal", zorder=3)
                ax4.legend(); ax4.set_title(f"MACD (f={int(macd_fast)}, s={int(macd_slow)}, sig={int(macd_signal)})")
                ax4.xaxis.set_major_locator(locator); ax4.xaxis.set_major_formatter(formatter)
                fig4.autofmt_xdate(rotation=45)
                st.pyplot(fig4, clear_figure=True)

            # ---- 下載 ----
            df_export = dfp.tail(int(lookback)).round(6)
            st.download_button("下載 CSV（最近 N 天指標）",
                               data=make_csv_bytes(df_export),
                               file_name=f"{tk}_indicators_last_{int(lookback)}d.csv",
                               mime="text/csv")
            xlsx = make_excel_bytes(df_export)
            if xlsx:
                st.download_button("下載 Excel（最近 N 天指標）",
                                   data=xlsx,
                                   file_name=f"{tk}_indicators_last_{int(lookback)}d.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

            with st.expander("原始＋指標資料（完整）"):
                st.dataframe(dfp.round(4))

# ====== 固定頁尾 ======
FOOTER_HTML = f"""
<style>
.footer {{
    position: fixed; left: 0; right: 0; bottom: 0; width: 100%;
    background: rgba(250, 250, 250, 0.92); backdrop-filter: blur(6px);
    border-top: 1px solid #e5e7eb; padding: 8px 16px;
    font-size: 12.5px; color: #4b5563; z-index: 9999;
}}
.footer .inner {{ max-width: 1200px; margin: 0 auto; display: flex; justify-content: space-between; gap: 12px; flex-wrap: wrap; }}
@media (max-width: 600px) {{ .footer {{ font-size: 12px; padding: 8px 10px; }} .footer .inner {{ padding-right: 62px; }} }}
</style>
<div class="footer">
  <div class="inner">
    <div>© {YEAR} {AUTHOR}</div>
    <div>版本：{VERSION}</div>
  </div>
</div>
"""
st.markdown(FOOTER_HTML, unsafe_allow_html=True)
