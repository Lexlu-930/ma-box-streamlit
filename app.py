# -*- coding: utf-8 -*-
# Streamlit 版：MA + k×標準差（箱型）分析
# 修正：相容 yfinance 對台股回傳的「多層欄位（MultiIndex）」格式，避免誤判為空而落回假資料。

import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

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


# ================= 工具函式 =============
