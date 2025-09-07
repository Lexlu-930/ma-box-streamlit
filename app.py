# -*- coding: utf-8 -*-
# Streamlit 版：MA + k×標準差（箱型）分析 + 指標面板 (MA/量價/KD/MACD)
# 精簡：拿掉不支援的基本面項目（個股獲利來源、毛利率、營收、EPS、合理本益比）
# 作者: LexLu
# 版本: v1.4 (2025-09-07)

import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# ====== 基本資訊 ======
AUTHOR = "LexLu"
VERSION = "v1.4 (2025-09-07)"
YEAR = datetime.now().year

# ====== 字型設定（雲端通常沒有中文字型，支援專案內 fonts/ 自帶字型）======
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

# ====== 嘗試載入 yfinance；失敗仍能用假資料 ======
try:
    import yfinance as yf
    HAS_YF = True
except Exception:
    HAS_YF = False


# ================= 工具函式 =================
def ensure_scalar(x):
    """把任何 DataFrame/Series/ndarray/字串壓成單一 float 或 np.nan，避免 ambiguous 真值判斷。"""
    try:
        if isinstance(x, pd.DataFrame):
            if x.empty:
                return np.nan
            return ensure_scalar(x.iloc[-1].squeeze())
        elif isinstance(x, (pd.Series, list, tuple, np.ndarray)):
            if len(x) == 0:
                return np.nan
            return ensure_scalar(x[-1])
        else:
            v = pd.to_numeric(x, errors="coerce")
            return float(v) if pd.notna(v) else np.nan
    except Exception:
        return np.nan


def parse_end_date(end_date_str: str) -> pd.Timestamp | None:
    """將 YYYY-MM-DD 或空字串 轉成 Timestamp（空字串=今天）；錯誤回傳 None。"""
    if not end_date_str.strip():
        return pd.Timestamp.today().normalize()
    try:
        return pd.to_datetime(end_date_str.strip())
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_price_data(ticker: str, end_date_str: str, lookback_days: int) -> pd.DataFrame:
    """
    下載 OHLCV（優先 yfinance；支持台股多層欄位），回傳含 OPEN/HIGH/LOW/CLOSE/VOLUME。
    失敗則回傳假資料，並在 df.attrs['simulated']=True 標註。
    """
    end = parse_end_date(end_date_str)
    if end is None:
        return pd.DataFrame()

    start = end - pd.Timedelta(days=max(lookback_days * 2, 120))
    yf_symbol = f"{ticker}.TW" if ticker.isdigit() else ticker

    def _to_ohlcv(df: pd.DataFrame) -> pd.DataFrame | None:
        open_s = high_s = low_s = close_s = vol_s = None
        if isinstance(df.columns, pd.MultiIndex):
            # 嘗試 level=0 為欄位名
            def pick(name: str):
                if name in df.columns.get_level_values(0):
                    sub = df.xs(name, axis=1, level=0, drop_level=True)
                    return sub.iloc[:, 0] if isinstance(sub, pd.DataFrame) else sub
                return None
            open_s  = pick("Open")
            high_s  = pick("High")
            low_s   = pick("Low")
            close_s = pick("Close") or pick("Adj Close")
            vol_s   = pick("Volume")
        else:
            cols = df.columns
            open_s  = df["Open"] if "Open" in cols else None
            high_s  = df["High"] if "High" in cols else None
            low_s   = df["Low"]  if "Low"  in cols else None
            close_s = df["Close"] if "Close" in cols else (df["Adj Close"] if "Adj Close" in cols else None)
            vol_s   = df["Volume"] if "Volume" in cols else None
            # 備援
            if close_s is None and df.shape[1] > 0: close_s = df.iloc[:, 0]
            if vol_s   is None and df.shape[1] > 1: vol_s   = df.iloc[:, -1]

        if close_s is None or vol_s is None:
            return None

        out = pd.DataFrame({
            "OPEN":  open_s if open_s is not None else np.nan,
            "HIGH":  high_s if high_s is not None else np.nan,
            "LOW":   low_s  if low_s  is not None else np.nan,
            "CLOSE": close_s,
            "VOLUME": vol_s
        }).dropna(subset=["CLOSE"])
        return out

    if HAS_YF:
        try:
            df = yf.download(
                yf_symbol,
                start=start.strftime("%Y-%m-%d"),
                end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                progress=False,
                auto_adjust=True,  # yfinance 新版預設 True
            )
            if not df.empty:
                ohlcv = _to_ohlcv(df)
                if ohlcv is not None and not ohlcv.empty:
                    ohlcv.attrs["simulated"] = False
                    return ohlcv
        except Exception:
            pass

    # fallback 假資料
    rng = pd.date_range(end=end, periods=lookback_days, freq="B")
    close = np.linspace(100, 110, len(rng)) + np.random.normal(0, 1.5, len(rng))
    openp = close + np.random.normal(0, 0.6, len(rng))
    high  = np.maximum(openp, close) + np.random.uniform(0.1, 0.8, len(rng))
    low   = np.minimum(openp, close) - np.random.uniform(0.1, 0.8, len(rng))
    volume = np.random.randint(1200, 3000, size=len(rng))
    demo = pd.DataFrame({"OPEN": openp, "HIGH": high, "LOW": low, "CLOSE": close, "VOLUME": volume}, index=rng)
    demo.attrs["simulated"] = True
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


# ================= 分析（布林 + 量能） =================
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
            if not vol_ok:
                vol_msg = "量能不符（最後一日量 < 量均）"
        else:
            vol_ok = False
            vol_msg = "量能資料不足，無法判斷"

    return df, dict(close=close, ma=ma, std=std, vol=vol, volma=volma, upper=upper, lower=lower,
                    vol_ok=vol_ok, vol_msg=vol_msg)


# ================= Streamlit UI =================
st.set_page_config(page_title="技術指標面板（MA / 量價 / KD / MACD）", layout="wide")

# 標題與版本
st.title("技術指標面板（MA / 量價 / KD / MACD）")
st.caption(f"作者: **{AUTHOR}** ｜ 版本: **{VERSION}**")

# Sidebar 簡介
st.sidebar.title("ℹ️ 功能")
st.sidebar.markdown(
    "- 移動平均線（可多條）\n"
    "- 量價（含量均）\n"
    "- KD 指標\n"
    "- MACD 指標\n"
)

# 參數表單
with st.form("params"):
    c1, c2, c3 = st.columns([1.1, 1.1, 1])

    with c1:
        ticker = st.text_input("股票代碼", value="3481")
        end_dt_str = st.text_input("結束日期（YYYY-MM-DD，可留空=今天）", value="")
        lookback = st.number_input("觀察天數 N（近 N 天）", min_value=30, max_value=3650, value=180, step=1)
        vol_win = st.number_input("量均視窗（天）", min_value=2, max_value=365, value=20, step=1)
        vol_filter = st.checkbox("啟用量能過濾（最後一日 成交量 ≥ 量均）", value=False)

    with c2:
        # 價格線與布林
        ma_list_str = st.text_input("移動平均線天數（逗號分隔）", value="5,10,20,60")
        boll_win = st.number_input("布林帶 MA 天數", min_value=5, max_value=120, value=20, step=1)
        k_boll = st.number_input("布林帶 k", min_value=0.5, max_value=4.0, value=2.0, step=0.1)

        # KD 參數
        kd_n = st.number_input("KD 期數 n", min_value=5, max_value=30, value=9, step=1)
        kd_k = st.number_input("K 平滑", min_value=1, max_value=9, value=3, step=1)
        kd_d = st.number_input("D 平滑", min_value=1, max_value=9, value=3, step=1)

    with c3:
        # MACD 參數
        macd_fast = st.number_input("MACD 快線", min_value=5, max_value=20, value=12, step=1)
        macd_slow = st.number_input("MACD 慢線", min_value=10, max_value=40, value=26, step=1)
        macd_signal = st.number_input("MACD 訊號線", min_value=5, max_value=20, value=9, step=1)
        show_kd = st.checkbox("顯示 KD", value=True)
        show_macd = st.checkbox("顯示 MACD", value=True)

    submitted = st.form_submit_button("開始分析")

if submitted:
    # 下載資料
    df_raw = load_price_data(ticker.strip(), end_dt_str, int(lookback))

    if df_raw.empty:
        st.error("讀不到任何價格資料，請確認代碼與日期。")
        st.stop()

    # 主分析（布林/量能）
    df_proc, metrics = analyze_core(
        df=df_raw, vol_filter=vol_filter, vol_win=int(vol_win), k_boll=float(k_boll), boll_win=int(boll_win)
    )

    # MA
    try:
        ma_windows = sorted({int(x.strip()) for x in ma_list_str.split(",") if x.strip().isdigit()})
        ma_windows = [w for w in ma_windows if w > 0]
    except Exception:
        ma_windows = [5, 10, 20, 60]
    df_proc = add_mas(df_proc, ma_windows)

    # KD / MACD
    df_proc = add_kd(df_proc, n=int(kd_n), k_smooth=int(kd_k), d_smooth=int(kd_d))
    df_proc = add_macd(df_proc, fast=int(macd_fast), slow=int(macd_slow), signal=int(macd_signal))

    # ----------- 頂部資訊卡 -----------
    top1, top2, top3, top4 = st.columns(4)
    top1.metric("收盤價", f"{metrics['close']:.2f}" if pd.notna(metrics['close']) else "N/A")
    top2.metric("布林中軌(MA)", f"{metrics['ma']:.2f}" if pd.notna(metrics['ma']) else "N/A")
    top3.metric("上軌", f"{metrics['upper']:.2f}" if pd.notna(metrics['upper']) else "N/A")
    top4.metric("下軌", f"{metrics['lower']:.2f}" if pd.notna(metrics['lower']) else "N/A")

    if df_raw.attrs.get("simulated", False):
        st.warning("⚠️ 注意：目前無法從資料源取得實際行情，以下為模擬資料（僅示範用途）。")

    # ----------- 圖1：價格 + MA + 布林 -----------
    fig1 = plt.figure(figsize=(11, 4.2))
    ax1 = plt.gca()
    ax1.plot(df_proc.index, df_proc["CLOSE"], label="收盤")
    # 多條 MA
    for w in ma_windows:
        col = f"MA{w}"
        if col in df_proc:
            ax1.plot(df_proc.index, df_proc[col], label=f"MA{w}")
    # 布林
    ax1.plot(df_proc.index, df_proc["BOLL_MA"], label=f"BOLL_MA({int(boll_win)})")
    ax1.plot(df_proc.index, df_proc["BOLL_UPPER"], label="上軌")
    ax1.plot(df_proc.index, df_proc["BOLL_LOWER"], label="下軌")
    ax1.legend()
    ax1.set_title("價格 / 多條MA / 布林帶")
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax1.xaxis.set_major_locator(locator)
    ax1.xaxis.set_major_formatter(formatter)
    fig1.autofmt_xdate(rotation=45)
    st.pyplot(fig1, clear_figure=True)

    # ----------- 圖2：成交量（量價） -----------
    fig2 = plt.figure(figsize=(11, 2.8))
    ax2 = plt.gca()
    ax2.bar(df_proc.index, df_proc["VOLUME"], width=0.8, label="成交量")
    if "VOL_MA" in df_proc:
        ax2.plot(df_proc.index, df_proc["VOL_MA"], label=f"量均({int(vol_win)})")
    ax2.legend()
    ax2.set_title("量價（成交量與量均）")
    ax2.xaxis.set_major_locator(locator)
    ax2.xaxis.set_major_formatter(formatter)
    fig2.autofmt_xdate(rotation=45)
    st.pyplot(fig2, clear_figure=True)

    # ----------- 圖3：KD -----------
    if show_kd:
        fig3 = plt.figure(figsize=(11, 2.8))
        ax3 = plt.gca()
        ax3.plot(df_proc.index, df_proc["%K"], label="%K")
        ax3.plot(df_proc.index, df_proc["%D"], label="%D")
        ax3.axhline(80, linestyle="--", linewidth=1)
        ax3.axhline(20, linestyle="--", linewidth=1)
        ax3.legend()
        ax3.set_title(f"KD 指標 (n={int(kd_n)}, K平滑={int(kd_k)}, D平滑={int(kd_d)})")
        ax3.xaxis.set_major_locator(locator)
        ax3.xaxis.set_major_formatter(formatter)
        fig3.autofmt_xdate(rotation=45)
        st.pyplot(fig3, clear_figure=True)

    # ----------- 圖4：MACD -----------
    if show_macd:
        fig4 = plt.figure(figsize=(11, 3.0))
        ax4 = plt.gca()
        ax4.plot(df_proc.index, df_proc["MACD"], label="MACD")
        ax4.plot(df_proc.index, df_proc["MACD_SIGNAL"], label="Signal")
        ax4.bar(df_proc.index, df_proc["MACD_HIST"], width=0.8, alpha=0.5, label="Hist")
        ax4.legend()
        ax4.set_title(f"MACD (fast={int(macd_fast)}, slow={int(macd_slow)}, signal={int(macd_signal)})")
        ax4.xaxis.set_major_locator(locator)
        ax4.xaxis.set_major_formatter(formatter)
        fig4.autofmt_xdate(rotation=45)
        st.pyplot(fig4, clear_figure=True)

# ====== 固定頁尾（Footer）======
FOOTER_HTML = f"""
<style>
.footer {{
    position: fixed; left: 0; right: 0; bottom: 0; width: 100%;
    background: rgba(250, 250, 250, 0.92); backdrop-filter: blur(6px);
    border-top: 1px solid #e5e7eb; padding: 8px 16px;
    font-size: 12.5px; color: #4b5563; z-index: 9999;
}}
.footer .inner {{ max-width: 1200px; margin: 0 auto; display: flex; justify-content: space-between; gap: 12px; flex-wrap: wrap; }}
@media (max-width: 600px) {{
    .footer {{ font-size: 12px; padding: 8px 10px; }}
    .footer .inner {{ padding-right: 62px; }}
}}
</style>
<div class="footer">
  <div class="inner">
    <div>© {YEAR} {AUTHOR}</div>
    <div>版本：{VERSION}</div>
  </div>
</div>
"""
st.markdown(FOOTER_HTML, unsafe_allow_html=True)
