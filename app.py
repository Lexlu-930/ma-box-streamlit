# -*- coding: utf-8 -*-
# Streamlit 版：MA + k×標準差（箱型）分析
# 修正：相容 yfinance 台股多層欄位（3481 等會正確顯示），加入作者/版本資訊與固定頁尾
# 作者: LexLu
# 版本: v1.2 (2025-09-07)

import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# ====== 基本資訊 ======
AUTHOR = "LexLu"
VERSION = "v1.2 (2025-09-07)"
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
    優先用 yfinance；成功則回傳含 CLOSE、VOLUME 欄位的 DataFrame。
    - 兼容 yfinance 可能回傳的 MultiIndex 欄位（台股常見）。
    - 若取不到資料，回傳假資料；並在 df.attrs['simulated'] = True 標示。
    """
    end = parse_end_date(end_date_str)
    if end is None:
        return pd.DataFrame()

    start = end - pd.Timedelta(days=max(lookback_days * 2, 120))  # 多抓避免滾動窗口 NaN
    yf_symbol = f"{ticker}.TW" if ticker.isdigit() else ticker

    if HAS_YF:
        try:
            df = yf.download(
                yf_symbol,
                start=start.strftime("%Y-%m-%d"),
                end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                progress=False,
            )
            if not df.empty:
                close_series = None
                volume_series = None

                if isinstance(df.columns, pd.MultiIndex):
                    # level 0: 'Close'/'Volume', level 1: ticker
                    if "Close" in df.columns.get_level_values(0):
                        close_df = df.xs("Close", axis=1, level=0, drop_level=True)
                        close_series = close_df.iloc[:, 0] if isinstance(close_df, pd.DataFrame) else close_df
                    if "Volume" in df.columns.get_level_values(0):
                        vol_df = df.xs("Volume", axis=1, level=0, drop_level=True)
                        volume_series = vol_df.iloc[:, 0] if isinstance(vol_df, pd.DataFrame) else vol_df
                else:
                    if "Close" in df.columns:
                        close_series = df["Close"]
                    elif "Adj Close" in df.columns:
                        close_series = df["Adj Close"]
                    else:
                        close_series = df.iloc[:, 0]

                    if "Volume" in df.columns:
                        volume_series = df["Volume"]
                    else:
                        volume_series = df.iloc[:, -1]

                if close_series is not None and volume_series is not None:
                    out = pd.DataFrame({"CLOSE": close_series, "VOLUME": volume_series}).dropna(how="all")
                    out.attrs["simulated"] = False
                    return out
        except Exception:
            pass

    # fallback 假資料
    rng = pd.date_range(end=end, periods=lookback_days, freq="B")
    close = np.linspace(100, 110, len(rng)) + np.random.normal(0, 1.5, len(rng))
    volume = np.random.randint(1200, 3000, size=len(rng))
    demo = pd.DataFrame({"CLOSE": close, "VOLUME": volume}, index=rng)
    demo.attrs["simulated"] = True
    return demo


def analyze(df: pd.DataFrame, lookback: int, ma_win: int, vol_win: int, k: float,
            use_vol_filter: bool, cost, tp_pct, sl_pct):
    """核心計算：回傳 (報告文字, 指標DataFrame, 錯誤訊息或 None)。"""
    if df.empty or "CLOSE" not in df or "VOLUME" not in df:
        return None, None, "資料讀取失敗或欄位不足（需要 CLOSE、VOLUME），或日期格式錯誤。"

    df = df.tail(lookback).copy()
    df["MA"] = df["CLOSE"].rolling(ma_win).mean()
    df["STD"] = df["CLOSE"].rolling(ma_win).std(ddof=0)
    df["VOL_MA"] = df["VOLUME"].rolling(vol_win).mean()
    df["UPPER"] = df["MA"] + k * df["STD"]
    df["LOWER"] = df["MA"] - k * df["STD"]

    last = df.iloc[-1]
    close = ensure_scalar(last["CLOSE"])
    ma    = ensure_scalar(last["MA"])
    std   = ensure_scalar(last["STD"])
    vol   = ensure_scalar(last["VOLUME"])
    volma = ensure_scalar(last["VOL_MA"])
    upper = ensure_scalar(last["UPPER"])
    lower = ensure_scalar(last["LOWER"])

    # 量能過濾
    vol_ok = True
    vol_msg = ""
    if use_vol_filter:
        if pd.notna(vol) and pd.notna(volma):
            vol_ok = vol >= volma
            if not vol_ok:
                vol_msg = "量能不符（最後一日量 < 量均）"
        else:
            vol_ok = False
            vol_msg = "量能資料不足，無法判斷"

    # 停利/停損、損益
    tp_price = sl_price = profit_pct = np.nan
    if pd.notna(cost):
        if pd.notna(tp_pct):
            tp_price = cost * (1 + tp_pct / 100.0)
        if pd.notna(sl_pct):
            sl_price = cost * (1 - sl_pct / 100.0)
        if pd.notna(close) and cost != 0:
            profit_pct = (close - cost) / cost * 100.0

    # 報告文字
    lines = []
    lines.append("— 當日數據 / 計算結果 —")
    lines.append(f"收盤價: {close if pd.notna(close) else 'N/A'}")
    lines.append(f"日線均價(MA) = 建議買價: {f'{ma:.2f}' if pd.notna(ma) else 'N/A'}")
    lines.append(f"標準差(STD): {f'{std:.4f}' if pd.notna(std) else 'N/A'}")
    lines.append(f"箱型上限 = 建議賣價: {f'{upper:.2f}' if pd.notna(upper) else 'N/A'}")
    lines.append(f"箱型下限: {f'{lower:.2f}' if pd.notna(lower) else 'N/A'}")
    lines.append(f"成交量 / 量均: "
                 f"{int(vol) if pd.notna(vol) else 'N/A'} / {int(volma) if pd.notna(volma) else 'N/A'}")
    if use_vol_filter:
        lines.append(f"量能過濾判定: {'通過' if vol_ok else '未通過'}{('（'+vol_msg+'）' if vol_msg else '')}")

    if pd.notna(cost):
        lines.append("")
        lines.append("— 依成本計算 —")
        lines.append(f"目前損益: {profit_pct:.2f}%" if pd.notna(profit_pct) else "目前損益: N/A")
        lines.append(f"停利價（+{tp_pct}%）: {f'{tp_price:.2f}' if pd.notna(tp_price) else '（未設定）'}")
        lines.append(f"停損價（-{sl_pct}%）: {f'{sl_price:.2f}' if pd.notna(sl_price) else '（未設定）'}")

    # 簡單提示
    suggest_buy = ma
    suggest_sell = upper
    advice = []
    if pd.notna(suggest_buy) and pd.notna(close):
        if close < suggest_buy:
            advice.append("價格低於MA，可觀察逢低/分批")
        elif close > suggest_buy and pd.notna(suggest_sell) and close < suggest_sell:
            advice.append("價格位於MA~上軌之間，持有可續抱")
        elif pd.notna(suggest_sell) and close >= suggest_sell:
            advice.append("接近/突破上軌，考慮部分了結")
    if use_vol_filter and not vol_ok:
        advice.append("量能不足，保守應對")

    if advice:
        lines.append("")
        lines.append("— 交易提示（參考） —")
        for s in advice:
            lines.append("• " + s)

    report = "\n".join(lines)
    return report, df, None


# ================= Streamlit UI =================
st.set_page_config(page_title="MA 規則判斷（網頁版）", layout="centered")

# 標題與版本
st.title("MA 規則判斷（網頁版）")
st.caption(f"作者: **{AUTHOR}** ｜ 版本: **{VERSION}**")

# Sidebar 關於
st.sidebar.title("ℹ️ 關於")
st.sidebar.markdown(f"**作者:** {AUTHOR}\n\n**版本:** {VERSION}\n\n這是一個基於 MA + k×標準差 的股票分析工具。")

with st.form("params"):
    c1, c2 = st.columns([1, 1])

    with c1:
        ticker = st.text_input("股票代碼", value="6672")
        end_dt_str = st.text_input("結束日期（YYYY-MM-DD，可留空=今天）", value="")
        lookback = st.number_input("觀察天數 N（近 N 天）", min_value=10, max_value=3650, value=90, step=1)
        ma_win = st.number_input("均線天數 MA", min_value=2, max_value=365, value=10, step=1)
        k = st.number_input("標準差倍數 k", min_value=0.1, max_value=10.0, value=2.0, step=0.1)

    with c2:
        tp_pct = st.number_input("獲利門檻（+%）", value=8.0, step=0.5)
        sl_pct = st.number_input("停損門檻（-%）", value=5.0, step=0.5)
        cost_str = st.text_input("持有成本（可留空）", value="")
        vol_win = st.number_input("量均視窗（天）", min_value=2, max_value=365, value=20, step=1)
        use_vol_filter = st.checkbox("啟用量能過濾（最後一日 成交量 ≥ 量均）", value=True)

    submitted = st.form_submit_button("開始分析")

if submitted:
    cost = ensure_scalar(cost_str) if cost_str.strip() else np.nan
    df_raw = load_price_data(ticker.strip(), end_dt_str, int(lookback))
    report, df_ind, err = analyze(
        df=df_raw,
        lookback=int(lookback),
        ma_win=int(ma_win),
        vol_win=int(vol_win),
        k=float(k),
        use_vol_filter=use_vol_filter,
        cost=cost,
        tp_pct=float(tp_pct) if tp_pct is not None else np.nan,
        sl_pct=float(sl_pct) if sl_pct is not None else np.nan,
    )

    if err:
        st.error(err)
    else:
        if df_raw.attrs.get("simulated", False):
            st.warning("⚠️ 注意：目前無法從資料源取得實際行情，以下為模擬資料（僅示範用途）。")

        st.text(report)

        # 圖表：收盤/MA/上下軌
        fig = plt.figure(figsize=(9.5, 4.2))
        ax = plt.gca()
        ax.plot(df_ind.index, df_ind["CLOSE"], label="收盤")
        ax.plot(df_ind.index, df_ind["MA"], label="MA")
        ax.plot(df_ind.index, df_ind["UPPER"], label="上軌")
        ax.plot(df_ind.index, df_ind["LOWER"], label="下軌")
        ax.legend()
        ax.set_title("價格 / MA / 上下軌")

        locator = mdates.AutoDateLocator()
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        fig.autofmt_xdate(rotation=45)

        st.pyplot(fig, clear_figure=True)

        with st.expander("查看最近 N 天原始與指標數據"):
            st.dataframe(df_ind.round(3))

# ====== 固定頁尾（Footer）======
FOOTER_HTML = f"""
<style>
.footer {{
    position: fixed;
    left: 0; right: 0; bottom: 0;
    width: 100%;
    background: rgba(250, 250, 250, 0.92);
    backdrop-filter: blur(6px);
    border-top: 1px solid #e5e7eb;
    padding: 8px 16px;
    font-size: 12.5px;
    color: #4b5563;
    z-index: 9999;
}}
.footer .inner {{
    max-width: 1100px;
    margin: 0 auto;
    display: flex;
    justify-content: space-between;
    gap: 12px;
    flex-wrap: wrap;
}}
@media (max-width: 600px) {{
    .footer {{ font-size: 12px; padding: 8px 10px; }}
    .footer .inner {{ padding-right: 62px; }} /* 避開 Streamlit 右下角圖示 */
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
