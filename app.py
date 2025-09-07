# -*- coding: utf-8 -*-
# 技術指標面板（MA / 量價 / KD / MACD）+ 多股票對比 + 箱型進出說明 + 個股名稱顯示
# 資料來源：yfinance(.TW / .TWO) → TWSE（上市）→ TPEX（櫃買 JSON/CSV）→ 模擬
# v1.4.8:
#   - 新增：顯示個股名稱（TWSE ISIN 名稱表優先，yfinance 補充），排名表與分頁/圖標題帶入名稱
#   - 延續 v1.4.7 修正：避免 Series 的真值判定錯誤
# 作者: LexLu   日期: 2025-09-07

import os, io, re, requests
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

AUTHOR  = "LexLu"
VERSION = "v1.4.8 (2025-09-07)"
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


# ---------------- 個股名稱：TWSE ISIN + yfinance 補充 ----------------
@st.cache_data(show_spinner=False, ttl=86400)
def load_tw_isin_mapping() -> dict:
    """
    從 TWSE ISIN 名稱表抓上市/上櫃所有代碼→名稱
    """
    url = "https://isin.twse.com.tw/isin/C_public.jsp?strMode=2"
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, timeout=15, headers=headers)
    r.encoding = "utf-8"
    tables = pd.read_html(r.text)
    df = pd.concat(tables, ignore_index=True)
    mapping = {}
    # 第一欄通常包含「代碼 名稱」或「代碼　名稱」（含全形空白）
    for val in df.iloc[:, 0].dropna().astype(str):
        s = val.replace("\u3000", " ").strip()
        m = re.match(r"^(\d{4,6})\s+(.+)$", s)
        if m:
            mapping[m.group(1)] = m.group(2).strip()
    return mapping

@st.cache_data(show_spinner=False, ttl=86400)
def get_stock_name(ticker: str) -> str:
    code = "".join(ch for ch in ticker if ch.isdigit())
    # 1) TWSE ISIN 名稱表
    try:
        mp = load_tw_isin_mapping()
        if code in mp:
            return mp[code]
    except Exception:
        pass
    # 2) yfinance 補充
    if HAS_YF:
        for sym in [f"{code}.TW", f"{code}.TWO", ticker]:
            try:
                tkr = yf.Ticker(sym)
                name = None
                # fast_info 可能沒有名稱，退而求其次用 info
                try:
                    info = tkr.info
                    name = info.get("shortName") or info.get("longName")
                except Exception:
                    name = None
                if name:
                    return str(name)
            except Exception:
                continue
    return ""


# ---------------- 資料載入（修正 Series 邏輯） ----------------
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
        tw = twse_stock_day_range(ticker, start, e_
