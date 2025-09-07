# -*- coding: utf-8 -*-
# 技術指標面板（MA / 量價 / KD / MACD）+ 多股票對比 + 箱型進出說明 + 個股名稱顯示
# 資料來源：yfinance(.TW / .TWO) → TWSE（上市）→ TPEX（櫃買 JSON/CSV）→ 模擬
# v1.4.9:
#   - 強化名稱抓取：TWSE ISIN 多表容錯；yfinance 依序 get_info() → fast_info → info
#   - 其餘功能沿用 v1.4.8 / v1.4.7 的修正
# 作者: LexLu   日期: 2025-09-07

import os, io, re, requests
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

AUTHOR  = "LexLu"
VERSION = "v1.4.9 (2025-09-07)"
YEAR    = datetime.now().year

# ---------------- 字型 ----------------
try:
    from matplotlib import font_manager
    FONT_PATH = os.path.join(os.path.dirname(__file__), "fonts", "NotoSansCJKtc-Regular.otf")
    if os.path.exists(FONT_PATH):
        font_manager.fontManager.addfont(FONT_PATH)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT_PATH).get_name()
    else:
        plt.rcParams["font.sans-serif"] = [
            "Microsoft JhengHei","SimHei","PMingLiU","Noto Sans CJK TC","PingFang TC","WenQuanYi Zen Hei"
        ]
    plt.rcParams["axes.unicode_minus"] = False
except Exception:
    pass

# ---------------- yfinance ----------------
try:
    import yfinance as yf
    HAS_YF = True
except Exception:
    HAS_YF = False


# ---------------- 小工具 ----------------
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

def parse_end_date(s: str) -> pd.Timestamp | None:
    if not s.strip(): return pd.Timestamp.today().normalize()
    try: return pd.to_datetime(s.strip())
    except Exception: return None

def period_start_from_choice(end_date: pd.Timestamp, choice: str) -> pd.Timestamp:
    if choice == "近1月":  return end_date - pd.Timedelta(days=30)
    if choice == "近3月":  return end_date - pd.Timedelta(days=90)
    if choice == "近半年": return end_date - pd.Timedelta(days=183)
    if choice == "YTD":   return pd.Timestamp(year=end_date.year, month=1, day=1)
    return end_date - pd.Timedelta(days=90)


# ---------------- 官方來源（TWSE / TPEX） ----------------
def _tw_yyyymmdd(ts: pd.Timestamp) -> str: return ts.strftime("%Y%m%d")
def _gregorian_to_roc_year_month(ts: pd.Timestamp) -> str: return f"{ts.year-1911}/{ts.strftime('%m')}"
def _roc_to_gregorian(date_str: str) -> pd.Timestamp | None:
    try:
        y, m, d = date_str.split("/")
        return pd.Timestamp(year=int(y)+1911, month=int(m), day=int(d))
    except Exception:
        return None

COMMON_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "X-Requested-With": "XMLHttpRequest",
}

@st.cache_data(show_spinner=False)
def twse_stock_day_range(stock_no: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    dfs = []
    months = pd.period_range(start=start, end=end, freq="M")
    if len(months) == 0:
        months = pd.period_range(start=start, end=end + pd.Timedelta(days=1), freq="M")
    for p in months:
        dt = pd.Timestamp(p.start_time)
        url = "https://www.twse.com.tw/exchangeReport/STOCK_DAY"
        params = {"response":"json","date":_tw_yyyymmdd(dt),"stockNo":stock_no}
        try:
            r = requests.get(url, params=params, timeout=12, headers=COMMON_HEADERS)
            js = r.json()
            if js.get("stat") != "OK":
                continue
            fields = js.get("fields", [])
            data   = js.get("data", [])
            fidx   = {name:i for i, name in enumerate(fields)}
            rows = []
            for row in data:
                day = _roc_to_gregorian(row[fidx.get("日期",0)])
                if day is None or not (start <= day <= end): continue
                def fnum(s):
                    s = str(s).replace(",", "").replace("X","").strip()
                    try: return float(s)
                    except: return np.nan
                def inum(s):
                    s = str(s).replace(",", "").strip()
                    try: return int(s)
                    except: return np.nan
                rows.append((day,
                             fnum(row[fidx.get("開盤價",-1)]),
                             fnum(row[fidx.get("最高價",-1)]),
                             fnum(row[fidx.get("最低價",-1)]),
                             fnum(row[fidx.get("收盤價",-1)]),
                             inum(row[fidx.get("成交股數",-1)])))
            if rows:
                dfs.append(pd.DataFrame(rows, columns=["DATE","OPEN","HIGH","LOW","CLOSE","VOLUME"]).set_index("DATE"))
        except Exception:
            continue
    return pd.concat(dfs).sort_index() if dfs else pd.DataFrame()

def _tpex_json_month(stock_no: str, month_ts: pd.Timestamp) -> pd.DataFrame:
    endpoints = [
        "https://www.tpex.org.tw/web/stock/aftertrading/daily_trading_info/st43_result.php",
        "https://www.tpex.org.tw/www/stock/aftertrading/daily_trading_info/st43_result.php",
    ]
    params = {"l":"zh-tw","d":_gregorian_to_roc_year_month(month_ts),"stkno":stock_no}
    for url in endpoints:
        try:
            headers = dict(COMMON_HEADERS)
            headers["Referer"] = "https://www.tpex.org.tw/web/stock/aftertrading/daily_trading_info/st43.php?l=zh-tw"
            r = requests.get(url, params=params, timeout=12, headers=headers)
            js = r.json()
            data = js.get("aaData") or js.get("data") or []
            if not data: 
                continue
            rows = []
            for row in data:
                day = _roc_to_gregorian(row[0])
                def fnum(x):
                    x = str(x).replace(",", "").replace("X","").replace("---","").strip()
                    if x in ["","—","－"]: return np.nan
                    try: return float(x)
                    except: return np.nan
                def inum(x):
                    x = str(x).replace(",", "").strip()
                    try: return int(x)
                    except: return np.nan
                rows.append((day, fnum(row[3]), fnum(row[4]), fnum(row[5]), fnum(row[6]), inum(row[1])))
            if rows:
                return pd.DataFrame(rows, columns=["DATE","OPEN","HIGH","LOW","CLOSE","VOLUME"]).set_index("DATE")
        except Exception:
            continue
    return pd.DataFrame()

def _tpex_csv_month(stock_no: str, month_ts: pd.Timestamp) -> pd.DataFrame:
    endpoints = [
        "https://www.tpex.org.tw/web/stock/aftertrading/daily_trading_info/st43_download.php",
        "https://www.tpex.org.tw/www/stock/aftertrading/daily_trading_info/st43_download.php",
    ]
    params = {"l":"zh-tw","d":_gregorian_to_roc_year_month(month_ts),"stkno":stock_no}
    for url in endpoints:
        try:
            headers = dict(COMMON_HEADERS)
            headers["Referer"] = "https://www.tpex.org.tw/web/stock/aftertrading/daily_trading_info/st43.php?l=zh-tw"
            r = requests.get(url, params=params, timeout=12, headers=headers)
            content = r.content
            try:
                df = pd.read_csv(io.BytesIO(content), encoding="utf-8-sig")
            except Exception:
                df = pd.read_csv(io.BytesIO(content), encoding="cp950")
            if "日期" not in df.columns:
                continue
            out = []
            for _, row in df.iterrows():
                day = _roc_to_gregorian(str(row["日期"]))
                if day is None: continue
                def pickf(name):
                    s = str(row.get(name, "")).replace(",", "").replace("X","").replace("---","").strip()
                    if s in ["","—","－"]: return np.nan
                    try: return float(s)
                    except: return np.nan
                def picki(name):
                    s = str(row.get(name, "")).replace(",", "").strip()
                    try: return int(s)
                    except: return np.nan
                out.append((day, pickf("開盤"), pickf("最高"), pickf("最低"), pickf("收盤"), picki("成交股數")))
            if out:
                return pd.DataFrame(out, columns=["DATE","OPEN","HIGH","LOW","CLOSE","VOLUME"]).set_index("DATE")
        except Exception:
            continue
    return pd.DataFrame()

@st.cache_data(show_spinner=False)
def tpex_stock_day_range(stock_no: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    dfs = []
    months = pd.period_range(start=start, end=end, freq="M")
    if len(months) == 0:
        months = pd.period_range(start=start, end=end + pd.Timedelta(days=1), freq="M")
    for p in months:
        dt = pd.Timestamp(p.start_time)
        dfm = _tpex_json_month(stock_no, dt)
        if dfm.empty:
            dfm = _tpex_csv_month(stock_no, dt)
        if not dfm.empty:
            dfs.append(dfm)
    return pd.concat(dfs).sort_index() if dfs else pd.DataFrame()


# ---------------- yfinance 多層欄位處理 ----------------
def _pick_series_any_level(df: pd.DataFrame, name: str, preferred_symbol: str | None = None):
    if isinstance(df.columns, pd.MultiIndex):
        for level in range(df.columns.nlevels):
            if name in df.columns.get_level_values(level):
                sub = df.xs(name, axis=1, level=level, drop_level=True)
                if isinstance(sub, pd.DataFrame):
                    if preferred_symbol and preferred_symbol in sub.columns:
                        return sub[preferred_symbol]
                    return sub.iloc[:, 0]
                return sub
        return None
    return df[name] if name in df.columns else None


# ---------------- 個股名稱：TWSE ISIN + yfinance 補強 ----------------
def _normalize_code(ticker: str) -> str:
    return "".join(ch for ch in ticker if ch.isdigit())

@st.cache_data(show_spinner=False, ttl=86400)
def load_tw_isin_mapping() -> dict:
    """
    從 TWSE ISIN 名稱表抓上市/上櫃所有代碼→名稱（多表容錯）
    """
    url = "https://isin.twse.com.tw/isin/C_public.jsp?strMode=2"
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, timeout=15, headers=headers)
    r.encoding = "utf-8"
    tables = pd.read_html(r.text)
    mapping = {}
    for df in tables:
        # 可能的欄名：['有價證券代號及名稱', '國際證券辨識號碼(ISIN Code)', ...]
        first_col = df.columns[0]
        for val in df[first_col].dropna().astype(str):
            s = val.replace("\u3000", " ").strip()
            # 允許前面帶英文字（市場別），找出第一個 4~6 位數字代碼
            m = re.search(r"(\d{4,6})\s+(.+)", s)
            if m:
                code = m.group(1)
                name = m.group(2).strip()
                # 去除常見註記
                name = re.sub(r"\s*\(.*?存託憑證.*?\)\s*", "", name)
                mapping[code] = name
    return mapping

@st.cache_data(show_spinner=False, ttl=86400)
def get_stock_name(ticker: str) -> str:
    code = _normalize_code(ticker)

    # 1) TWSE ISIN 名稱表（最快、覆蓋上市+上櫃）
    try:
        mp = load_tw_isin_mapping()
        if code in mp and isinstance(mp[code], str) and mp[code].strip():
            return mp[code].strip()
    except Exception:
        pass

    # 2) yfinance（新版優先 get_info()，再 fast_info，最後舊 .info）
    if HAS_YF:
        for sym in [f"{code}.TW", f"{code}.TWO", ticker]:
            try:
                tkr = yf.Ticker(sym)

                # 2-1 get_info()
                try:
                    info = tkr.get_info()
                    nm = (info or {}).get("shortName") or (info or {}).get("longName")
                    if nm and str(nm).strip():
                        return str(nm).strip()
                except Exception:
                    pass

                # 2-2 fast_info
                try:
                    fi = getattr(tkr, "fast_info", None)
                    if isinstance(fi, dict):
                        nm = fi.get("shortName") or fi.get("longName")
                        if nm and str(nm).strip():
                            return str(nm).strip()
                except Exception:
                    pass

                # 2-3 舊 .info
                try:
                    info2 = getattr(tkr, "info", None)
                    if isinstance(info2, dict):
                        nm = info2.get("shortName") or info2.get("longName")
                        if nm and str(nm).strip():
                            return str(nm).strip()
                except Exception:
                    pass

            except Exception:
                continue

    return ""


# ---------------- 資料載入（避免 Series 真值判定） ----------------
@st.cache_data(show_spinner=False)
def load_price_data(ticker: str, end_date_str: str, lookback_days: int) -> pd.DataFrame:
    """
    順序：yfinance(.TW→.TWO) → TWSE → TPEX → 模擬；所有嘗試記錄寫入 df.attrs['attempts']
    """
    attempts = []
    end = parse_end_date(end_date_str)
    if end is None: return pd.DataFrame()
    start = end - pd.Timedelta(days=max(lookback_days*2, 120))

    # 1) yfinance
    if HAS_YF:
        symbols = [ticker] if not ticker.isdigit() else [f"{ticker}.TW", f"{ticker}.TWO"]
        for sym in symbols:
            try:
                df = yf.download(sym,
                                 start=start.strftime("%Y-%m-%d"),
                                 end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                                 progress=False, auto_adjust=True)
                if df is None or df.empty:
                    attempts.append(f"yfinance({sym}): empty")
                    continue

                close_s = _pick_series_any_level(df, "Close", preferred_symbol=sym)
                if close_s is None:
                    close_s = _pick_series_any_level(df, "Adj Close", preferred_symbol=sym)
                vol_s = _pick_series_any_level(df, "Volume", preferred_symbol=sym)
                open_s = _pick_series_any_level(df, "Open", preferred_symbol=sym)
                high_s = _pick_series_any_level(df, "High", preferred_symbol=sym)
                low_s  = _pick_series_any_level(df, "Low",  preferred_symbol=sym)

                if close_s is None or vol_s is None:
                    attempts.append(f"yfinance({sym}): missing Close/Volume")
                    continue

                out = pd.DataFrame({
                    "OPEN": open_s if open_s is not None else np.nan,
                    "HIGH": high_s if high_s is not None else np.nan,
                    "LOW":  low_s  if low_s  is not None else np.nan,
                    "CLOSE": close_s,
                    "VOLUME": vol_s,
                }).dropna(subset=["CLOSE"])

                if out.empty:
                    attempts.append(f"yfinance({sym}): parsed empty")
                    continue

                out.attrs["simulated"] = False
                out.attrs["source"]    = f"yfinance({sym})"
                out.attrs["attempts"]  = attempts
                return out
            except Exception as e:
                attempts.append(f"yfinance({sym}) error: {type(e).__name__}: {str(e)[:120]}")

    # 2) TWSE
    if ticker.isdigit():
        tw = twse_stock_day_range(ticker, start, end)
        if not tw.empty:
            tw.attrs["simulated"] = False
            tw.attrs["source"]    = "twse"
            attempts.append(f"twse: ok rows={len(tw)}")
            tw.attrs["attempts"]  = attempts
            return tw
        attempts.append("twse: empty")

    # 3) TPEX
    if ticker.isdigit():
        tp = tpex_stock_day_range(ticker, start, end)
        if not tp.empty:
            tp.attrs["simulated"] = False
            tp.attrs["source"]    = "tpex"
            attempts.append(f"tpex: ok rows={len(tp)}")
            tp.attrs["attempts"]  = attempts
            return tp
        attempts.append("tpex: empty")

    # 4) 模擬
    rng = pd.date_range(end=end, periods=lookback_days, freq="B")
    close  = np.linspace(100, 110, len(rng)) + np.random.normal(0, 1.5, len(rng))
    openp  = close + np.random.normal(0, 0.6, len(rng))
    high   = np.maximum(openp, close) + np.random.uniform(0.1, 0.8, len(rng))
    low    = np.minimum(openp, close) - np.random.uniform(0.1, 0.8, len(rng))
    volume = np.random.randint(1200, 3000, size=len(rng))
    demo = pd.DataFrame({"OPEN": openp,"HIGH":high,"LOW":low,"CLOSE":close,"VOLUME":volume}, index=rng)
    demo.attrs["simulated"] = True
    demo.attrs["source"]    = "simulated"
    demo.attrs["attempts"]  = attempts
    return demo


# ---------------- 指標 ----------------
def add_mas(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    for w in windows: df[f"MA{w}"] = df["CLOSE"].rolling(w).mean()
    return df

def add_vol_ma(df: pd.DataFrame, vol_win: int) -> pd.DataFrame:
    df["VOL_MA"] = df["VOLUME"].rolling(vol_win).mean(); return df

def add_boll(df: pd.DataFrame, boll_win: int = 20, k: float = 2.0) -> pd.DataFrame:
    ma  = df["CLOSE"].rolling(boll_win).mean()
    std = df["CLOSE"].rolling(boll_win).std(ddof=0)
    df["BOLL_MA"]    = ma
    df["BOLL_UPPER"] = ma + k*std
    df["BOLL_LOWER"] = ma - k*std
    return df

def add_kd(df: pd.DataFrame, n: int = 9, k_smooth: int = 3, d_smooth: int = 3) -> pd.DataFrame:
    low_n  = df["LOW"].rolling(n).min()
    high_n = df["HIGH"].rolling(n).max()
    rsv = 100*(df["CLOSE"]-low_n) / (high_n-low_n).replace(0, np.nan)
    k = rsv.rolling(k_smooth).mean()
    d = k.rolling(d_smooth).mean()
    df["%K"] = k; df["%D"] = d
    return df

def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = df["CLOSE"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["CLOSE"].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig  = macd.ewm(span=signal, adjust=False).mean()
    df["MACD"] = macd; df["MACD_SIGNAL"] = sig; df["MACD_HIST"] = macd - sig
    return df

def analyze_core(df: pd.DataFrame, vol_filter: bool, vol_win: int, k_boll: float, boll_win: int):
    df = df.copy()
    df = add_vol_ma(df, vol_win)
    df = add_boll(df, boll_win, k_boll)
    last = df.iloc[-1]
    close = ensure_scalar(last["CLOSE"])
    ma    = ensure_scalar(df["BOLL_MA"].iloc[-1])
    std   = ensure_scalar((df["BOLL_UPPER"].iloc[-1]-df["BOLL_MA"].iloc[-1]) / (k_boll or np.nan))
    vol   = ensure_scalar(last["VOLUME"])
    volma = ensure_scalar(last["VOL_MA"])
    upper = ensure_scalar(last["BOLL_UPPER"])
    lower = ensure_scalar(last["BOLL_LOWER"])
    vol_ok=True; vol_msg=""
    if vol_filter:
        if pd.notna(vol) and pd.notna(volma):
            vol_ok = (vol >= volma)
            if not vol_ok: vol_msg = "量能不足（最後一日 < 量均）"
        else:
            vol_ok=False; vol_msg="量能資料不足，無法判斷"
    pos="N/A"
    if pd.notna(close) and pd.notna(ma) and pd.notna(upper):
        if close < ma: pos="低於中軌"
        elif close >= upper: pos="接近/高於上軌"
        else: pos="介於中軌與上軌"
    return df, dict(close=close, ma=ma, std=std, vol=vol, volma=volma,
                    upper=upper, lower=lower, vol_ok=vol_ok, vol_msg=vol_msg, pos=pos)

def detect_cross(x: pd.Series, y: pd.Series):
    x = x.astype(float); y = y.astype(float)
    prev = (x.shift(1) - y.shift(1))
    now  = (x - y)
    gold  = (prev <= 0) & (now > 0)
    death = (prev >= 0) & (now < 0)
    return x.index[gold.fillna(False)], x.index[death.fillna(False)]

def make_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=True).encode("utf-8-sig")

def make_excel_bytes(df: pd.DataFrame) -> bytes | None:
    try:
        import openpyxl  # noqa
        bio = io.BytesIO()
        with pd.ExcelWriter(bio, engine="openpyxl") as w:
            df.to_excel(w, sheet_name="sheet1")
        bio.seek(0); return bio.read()
    except Exception:
        return None

def build_box_report(m: dict, use_vol_filter: bool, cost: float | None, tp_pct: float | None, sl_pct: float | None) -> str:
    close, ma, upper, lower = m["close"], m["ma"], m["upper"], m["lower"]
    vol, volma, vol_ok, vol_msg = m["vol"], m["volma"], m["vol_ok"], m.get("vol_msg","")
    lines = [
        "— 當日數據 / 計算結果 —",
        f"收盤價: {close:.2f}" if pd.notna(close) else "收盤價: N/A",
        f"日線均價(MA) = 建議買價: {ma:.2f}" if pd.notna(ma) else "日線均價(MA) = 建議買價: N/A",
        f"箱型上限 = 建議賣價: {upper:.2f}" if pd.notna(upper) else "箱型上限 = 建議賣價: N/A",
        f"箱型下限: {lower:.2f}" if pd.notna(lower) else "箱型下限: N/A",
        f"成交量 / 量均: {int(vol) if pd.notna(vol) else 'N/A'} / {int(volma) if pd.notna(volma) else 'N/A'}"
    ]
    if use_vol_filter:
        lines.append(f"量能過濾判定: {'通過' if vol_ok else '未通過'}{('（'+vol_msg+'）' if vol_msg else '')}")
    if cost is not None and pd.notna(cost):
        tp = cost*(1+(tp_pct or 0)/100.0) if tp_pct is not None else np.nan
        sl = cost*(1-(sl_pct or 0)/100.0) if sl_pct is not None else np.nan
        pnl = ((close-cost)/cost*100.0) if (pd.notna(close) and cost!=0) else np.nan
        lines += ["", "— 依成本計算 —",
                  f"目前損益: {pnl:.2f}%" if pd.notna(pnl) else "目前損益: N/A",
                  f"停利價（+{tp_pct}%）: {tp:.2f}" if pd.notna(tp) else "停利價: （未設定）",
                  f"停損價（-{sl_pct}%）: {sl:.2f}" if pd.notna(sl) else "停損價: （未設定）"]
    advice=[]
    if pd.notna(close) and pd.notna(ma):
        if close<ma: advice.append("價格低於MA，可觀察逢低/分批")
        elif pd.notna(upper) and close<upper: advice.append("價格位於MA~上軌之間，持有可續抱")
        elif pd.notna(upper) and close>=upper: advice.append("接近/突破上軌，考慮部分了結")
    if use_vol_filter and not vol_ok: advice.append("量能不足，保守應對")
    if advice: lines += ["", "— 交易提示（參考） —"] + [f"• {s}" for s in advice]
    return "\n".join(lines)


# ---------------- Streamlit UI ----------------
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
    "- 個股名稱顯示（TWSE ISIN / yfinance）\n"
    "- yfinance 斷線 → 自動 TWSE/TPEX（JSON/CSV）備援\n"
)

with st.form("params"):
    c1,c2,c3,c4 = st.columns([1.2,1.1,1,1])
    with c1:
        tickers_str = st.text_input("股票代碼（可多個，逗號分隔）", value="2330, 5314, 3481")
        end_dt_str   = st.text_input("結束日期（YYYY-MM-DD，可留空=今天）", value="")
        lookback     = st.number_input("觀察天數 N（近 N 天）", min_value=30, max_value=3650, value=180, step=1)
        compare_period = st.selectbox("相對表現期間", ["近1月","近3月","近半年","YTD"], index=1)
    with c2:
        vol_win   = st.number_input("量均視窗（天）", min_value=2, max_value=365, value=20, step=1)
        vol_filter= st.checkbox("啟用量能過濾（最後一日 成交量 ≥ 量均）", value=False)
        show_kd   = st.checkbox("顯示 KD（個股分頁）", value=True)
        show_macd = st.checkbox("顯示 MACD（個股分頁）", value=True)
    with c3:
        ma_list_str = st.text_input("移動平均線天數（逗號分隔）", value="5,10,20,60")
        boll_win    = st.number_input("布林帶 MA 天數", min_value=5, max_value=120, value=20, step=1)
        k_boll      = st.number_input("布林帶 k", min_value=0.5, max_value=4.0, value=2.0, step=0.1)
    with c4:
        macd_fast  = st.number_input("MACD 快線", min_value=5, max_value=20, value=12, step=1)
        macd_slow  = st.number_input("MACD 慢線", min_value=10, max_value=40, value=26, step=1)
        macd_signal= st.number_input("MACD 訊號線", min_value=5, max_value=20, value=9, step=1)
        cost_str   = st.text_input("持有成本（可留空）", value="")
        tp_pct     = st.number_input("停利(%)", value=8.0, step=0.5)
        sl_pct     = st.number_input("停損(%)", value=5.0, step=0.5)
    submitted = st.form_submit_button("開始分析")

if not submitted:
    st.stop()

tickers = [t.strip() for t in tickers_str.split(",") if t.strip()]
if not tickers:
    st.error("請輸入至少一個代碼"); st.stop()

try:
    ma_windows = sorted({int(x.strip()) for x in ma_list_str.split(",") if x.strip().isdigit()})
    ma_windows = [w for w in ma_windows if w > 0]
except Exception:
    ma_windows = [5,10,20,60]

cost_val = ensure_scalar(cost_str) if cost_str.strip() else np.nan
cost_for_report = cost_val if pd.notna(cost_val) else None

# 先取得所有名稱（快取 1 天）
name_map = {tk: get_stock_name(tk) for tk in tickers}

# 下載 + 計算
results = {}
for tk in tickers:
    df_raw = load_price_data(tk, end_dt_str, int(lookback))
    if df_raw.empty: continue
    df_proc, metrics = analyze_core(df=df_raw, vol_filter=vol_filter, vol_win=int(vol_win), k_boll=float(k_boll), boll_win=int(boll_win))
    df_proc = add_mas(df_proc, ma_windows + [5,20])  # 確保 MA5/MA20
    df_proc = add_kd(df_proc, n=9, k_smooth=3, d_smooth=3)
    df_proc = add_macd(df_proc, fast=int(macd_fast), slow=int(macd_slow), signal=int(macd_signal))
    vol_pass = (df_proc["VOLUME"] >= df_proc["VOL_MA"]).tail(int(lookback)).sum()
    results[tk] = {
        "df": df_proc,
        "metrics": metrics,
        "vol_pass": int(vol_pass),
        "source": df_raw.attrs.get("source","unknown"),
        "sim": df_raw.attrs.get("simulated", False),
        "attempts": df_raw.attrs.get("attempts", []),
        "name": name_map.get(tk, ""),
    }

if not results:
    st.error("所有代碼皆讀不到資料"); st.stop()

# 排名表（加入名稱）
rows=[]
for tk,r in results.items():
    dfp=r["df"].tail(int(lookback))
    if dfp["CLOSE"].dropna().shape[0] < 2:
        ret=np.nan; vol_=np.nan
    else:
        ret=dfp["CLOSE"].iloc[-1]/dfp["CLOSE"].iloc[0]-1
        vol_=dfp["CLOSE"].pct_change().dropna().std()
    rows.append({
        "代碼": tk,
        "名稱": r.get("name",""),
        "近N天報酬(%)": f"{ret*100:.2f}%" if pd.notna(ret) else "N/A",
        "日波動(σ)": f"{vol_*100:.2f}%" if pd.notna(vol_) else "N/A",
        "量能達標次數": r["vol_pass"],
        "資料來源": r["source"]+(" (模擬)" if r["sim"] else "")
    })
st.subheader("多股票排名表（近 N 天）")
st.dataframe(pd.DataFrame(rows))

# 相對表現
st.subheader(f"多股票相對表現（{compare_period}）")
end_dt   = parse_end_date(end_dt_str) or pd.Timestamp.today().normalize()
start_cmp= period_start_from_choice(end_dt, compare_period)
figC = plt.figure(figsize=(11,4.2)); axC = plt.gca()
for tk,r in results.items():
    ser = r["df"].loc[start_cmp:end_dt]["CLOSE"].dropna()
    if ser.empty: continue
    base = ser.iloc[0]
    label = f"{tk} {r.get('name','')}".strip()
    if pd.notna(base) and base != 0:
        axC.plot(ser.index, ser / base * 100.0, label=label)
axC.legend(); axC.set_ylabel("相對表現（基準=100）"); axC.set_title("相對表現對比")
locator = mdates.AutoDateLocator(); formatter = mdates.ConciseDateFormatter(locator)
axC.xaxis.set_major_locator(locator); axC.xaxis.set_major_formatter(formatter)
figC.autofmt_xdate(rotation=45)
st.pyplot(figC, clear_figure=True)

# 個股分頁（標籤顯示代碼＋名稱）
tab_labels = [ (f"{tk} {results[tk].get('name','')}".strip()) for tk in tickers ]
tabs = st.tabs(tab_labels)

for i,tk in enumerate(tickers):
    if tk not in results:
        with tabs[i]: st.warning(f"{tk} 無資料"); continue
    r=results[tk]; dfp=r["df"]; m=r["metrics"]; nm=r.get("name","")
    title_prefix = f"{tk} {nm}".strip()
    with tabs[i]:
        if r["sim"]:
            st.warning("⚠️ 此檔目前使用模擬/替代資料（僅示範用途）。")
            if r.get("attempts"):
                with st.expander("資料來源嘗試紀錄（debug）"):
                    st.write("\n".join(r["attempts"]))

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("收盤價", f"{m['close']:.2f}" if pd.notna(m['close']) else "N/A")
        c2.metric("中軌MA(=建議買價)", f"{m['ma']:.2f}" if pd.notna(m['ma']) else "N/A")
        c3.metric("上軌(=建議賣價)", f"{m['upper']:.2f}" if pd.notna(m['upper']) else "N/A")
        c4.metric("下靶(箱型下限)", f"{m['lower']:.2f}" if pd.notna(m['lower']) else "N/A")
        st.text(build_box_report(m, vol_filter, cost_for_report, tp_pct, sl_pct))

        fig1 = plt.figure(figsize=(10.8,4.2)); ax1 = plt.gca()
        ax1.plot(dfp.index, dfp["CLOSE"], label="收盤")
        for w in ma_windows:
            col=f"MA{w}"
            if col in dfp: ax1.plot(dfp.index, dfp[col], label=f"MA{w}")
        ax1.plot(dfp.index, dfp["BOLL_MA"], label=f"BOLL_MA({int(boll_win)})")
        ax1.plot(dfp.index, dfp["BOLL_UPPER"], label="上軌"); ax1.plot(dfp.index, dfp["BOLL_LOWER"], label="下軌")
        if "MA5" in dfp and "MA20" in dfp:
            g,d=detect_cross(dfp["MA5"], dfp["MA20"])
            ax1.scatter(g, dfp.loc[g,"CLOSE"], marker="^", s=60, label="MA5↑MA20", zorder=3)
            ax1.scatter(d, dfp.loc[d,"CLOSE"], marker="v", s=60, label="MA5↓MA20", zorder=3)
        ax1.legend(); ax1.set_title(f"{title_prefix}｜價格 / 多MA / 布林（含 MA5×MA20 交叉）")
        ax1.xaxis.set_major_locator(locator); ax1.xaxis.set_major_formatter(formatter)
        fig1.autofmt_xdate(rotation=45); st.pyplot(fig1, clear_figure=True)

        fig2 = plt.figure(figsize=(10.8,2.8)); ax2 = plt.gca()
        ax2.bar(dfp.index, dfp["VOLUME"], width=0.8, label="成交量")
        if "VOL_MA" in dfp: ax2.plot(dfp.index, dfp["VOL_MA"], label=f"量均({int(vol_win)})")
        ax2.legend(); ax2.set_title(f"{title_prefix}｜量價（成交量與量均）")
        ax2.xaxis.set_major_locator(locator); ax2.xaxis.set_major_formatter(formatter)
        fig2.autofmt_xdate(rotation=45); st.pyplot(fig2, clear_figure=True)

        if show_kd:
            fig3=plt.figure(figsize=(10.8,2.8)); ax3=plt.gca()
            ax3.plot(dfp.index, dfp["%K"], label="%K"); ax3.plot(dfp.index, dfp["%D"], label="%D")
            ax3.axhline(80, linestyle="--", linewidth=1); ax3.axhline(20, linestyle="--", linewidth=1)
            ax3.legend(); ax3.set_title(f"{title_prefix}｜KD")
            ax3.xaxis.set_major_locator(locator); ax3.xaxis.set_major_formatter(formatter)
            fig3.autofmt_xdate(rotation=45); st.pyplot(fig3, clear_figure=True)

        if show_macd:
            fig4=plt.figure(figsize=(10.8,3.0)); ax4=plt.gca()
            ax4.plot(dfp.index, dfp["MACD"], label="MACD")
            ax4.plot(dfp.index, dfp["MACD_SIGNAL"], label="Signal")
            ax4.bar(dfp.index, dfp["MACD_HIST"], width=0.8, alpha=0.5, label="Hist")
            g2,d2=detect_cross(dfp["MACD"], dfp["MACD_SIGNAL"])
            ax4.scatter(g2, dfp.loc[g2,"MACD"], marker="^", s=50, label="MACD↑Signal", zorder=3)
            ax4.scatter(d2, dfp.loc[d2,"MACD"], marker="v", s=50, label="MACD↓Signal", zorder=3)
            ax4.legend(); ax4.set_title(f"{title_prefix}｜MACD (f={int(macd_fast)}, s={int(macd_slow)}, sig={int(macd_signal)})")
            ax4.xaxis.set_major_locator(locator); ax4.xaxis.set_major_formatter(formatter)
            fig4.autofmt_xdate(rotation=45); st.pyplot(fig4, clear_figure=True)

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

# ---------------- Footer ----------------
FOOTER_HTML = f"""
<style>
.footer {{ position: fixed; left:0; right:0; bottom:0; width:100%;
  background: rgba(250,250,250,.92); border-top:1px solid #e5e7eb;
  padding:8px 16px; font-size:12.5px; color:#4b5563; z-index:9999; }}
.footer .inner {{ max-width:1200px; margin:0 auto; display:flex; justify-content:space-between;
  gap:12px; flex-wrap:wrap; }}
@media (max-width:600px) {{ .footer {{ font-size:12px; padding:8px 10px; }} }}
</style>
<div class="footer"><div class="inner">
  <div>© {YEAR} {AUTHOR}</div><div>版本：{VERSION}</div>
</div></div>
"""
st.markdown(FOOTER_HTML, unsafe_allow_html=True)
