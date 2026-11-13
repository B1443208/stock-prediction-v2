
"""股票趨勢預測與應用｜V2 Streamlit 修正版。

使用歷史價格與成交量建立 8 項特徵，預測下一交易日報酬率。
研究／課堂展示用途，不構成投資建議。
"""

from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)
from sklearn.model_selection import TimeSeriesSplit

st.set_page_config(page_title="AI 股票趨勢預測", page_icon="📈", layout="wide")

STOCKS = {
    "台積電（2330）": "2330.TW",
    "鴻海（2317）": "2317.TW",
    "聯發科（2454）": "2454.TW",
}

FEATURES = [
    "RSI14", "Return1", "Return5", "VolumeRatio",
    "Close_MA5", "Close_MA20", "MA5_MA20", "DailyRange",
]


@st.cache_data(ttl=3600, show_spinner=False)
def download_stock(ticker: str) -> tuple[pd.DataFrame, bool]:
    """取得日線資料；若台灣當天尚未收盤，排除當日可能未完成的 K 棒。"""
    raw = yf.download(
        ticker,
        period="7y",
        interval="1d",
        auto_adjust=False,
        multi_level_index=False,
        progress=False,
    )
    if raw.empty:
        raise ValueError("無法下載股票資料，請檢查股票代碼或稍後再試。")

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [name for name in required if name not in raw.columns]
    if missing:
        raise ValueError(f"下載資料缺少欄位：{', '.join(missing)}")

    df = raw[required].copy()
    for name in required:
        df[name] = pd.to_numeric(df[name], errors="coerce")

    df = df.sort_index()
    df = df.loc[~df.index.duplicated(keep="last")]
    df = df.dropna(subset=required)
    df = df.loc[
        (df["Open"] > 0)
        & (df["High"] > 0)
        & (df["Low"] > 0)
        & (df["Close"] > 0)
        & (df["Volume"] > 0)
        & (df["High"] >= df["Low"])
    ].copy()

    # 台股一般交易時段為台灣時間 09:00～13:30；多留 30 分鐘緩衝。
    # 此檢查只能避免「當日尚未收盤」的明顯情況，不能保證行情完全即時。
    taipei_now = datetime.now(ZoneInfo("Asia/Taipei"))
    excluded_today = False
    if (
        not df.empty
        and df.index[-1].date() == taipei_now.date()
        and (taipei_now.hour, taipei_now.minute) < (14, 0)
    ):
        df = df.iloc[:-1].copy()
        excluded_today = True

    if len(df) < 200:
        raise ValueError("可用歷史資料不足 200 筆，無法完成訓練與測試。")
    return df, excluded_today


def make_features(raw: pd.DataFrame) -> pd.DataFrame:
    data = raw.copy()
    data["MA5"] = data["Close"].rolling(5).mean()
    data["MA20"] = data["Close"].rolling(20).mean()
    data["Return1"] = data["Close"].pct_change(1)
    data["Return5"] = data["Close"].pct_change(5)

    # RSI14：無下跌且有上漲設為 100；完全持平設為 50。
    delta = data["Close"].diff()
    gains = delta.clip(lower=0).rolling(14).mean()
    losses = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gains / losses.replace(0, np.nan)
    data["RSI14"] = (100 - 100 / (1 + rs)).where(losses != 0, 100)
    data.loc[(gains == 0) & (losses == 0), "RSI14"] = 50

    data["VolumeRatio"] = data["Volume"] / data["Volume"].rolling(5).mean()
    data["Close_MA5"] = data["Close"] / data["MA5"] - 1
    data["Close_MA20"] = data["Close"] / data["MA20"] - 1
    data["MA5_MA20"] = data["MA5"] / data["MA20"] - 1
    data["DailyRange"] = (data["High"] - data["Low"]) / data["Close"]

    # 每筆 X(t) 的答案是下一個實際交易日 t+1 的報酬率。
    data["NextClose"] = data["Close"].shift(-1)
    data["NextReturn"] = data["NextClose"] / data["Close"] - 1
    data["NextDate"] = data.index.to_series().shift(-1)
    data = data.replace([np.inf, -np.inf], np.nan)
    # 保留最後一筆尚未有隔日答案的資料，用於正式未來預測。
    return data.dropna(subset=FEATURES).copy()


def new_model() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
    )


def evaluate_model(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """固定同一訓練／測試範圍，比較 AI 與兩種簡單基準。"""
    model = new_model()
    model.fit(train[FEATURES], train["NextReturn"])

    pred_return = model.predict(test[FEATURES])
    actual_return = test["NextReturn"].to_numpy()
    today_close = test["Close"].to_numpy()
    actual_close = test["NextClose"].to_numpy()
    pred_close = today_close * (1 + pred_return)

    # 價格基準：下一日價格 = 今日收盤價。
    baseline_close = today_close
    # 方向基準：永遠猜訓練資料中占比較高的類別（平手則猜沒有上漲）。
    actual_up = (actual_return > 0).astype(int)
    pred_up = (pred_return > 0).astype(int)
    majority_up = int((train["NextReturn"] > 0).mean() > 0.5)
    baseline_up = np.full(len(test), majority_up, dtype=int)

    mae = mean_absolute_error(actual_close, pred_close)
    baseline_mae = mean_absolute_error(actual_close, baseline_close)
    rmse = np.sqrt(mean_squared_error(actual_close, pred_close))
    baseline_rmse = np.sqrt(mean_squared_error(actual_close, baseline_close))
    mape = float(np.mean(np.abs((actual_close - pred_close) / actual_close)) * 100)
    baseline_mape = float(np.mean(np.abs((actual_close - baseline_close) / actual_close)) * 100)

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": mape,
        "baseline_mae": float(baseline_mae),
        "baseline_rmse": float(baseline_rmse),
        "baseline_mape": baseline_mape,
        "accuracy": float(accuracy_score(actual_up, pred_up)),
        "baseline_accuracy": float(accuracy_score(actual_up, baseline_up)),
        "balanced_accuracy": float(balanced_accuracy_score(actual_up, pred_up)),
        "precision": float(precision_score(actual_up, pred_up, zero_division=0)),
        "recall": float(recall_score(actual_up, pred_up, zero_division=0)),
        "f1": float(f1_score(actual_up, pred_up, zero_division=0)),
        "results": pd.DataFrame(
            {
                "實際下一交易日收盤價": actual_close,
                "AI 預測收盤價": pred_close,
                "簡單基準（前日收盤價）": baseline_close,
            },
            index=pd.DatetimeIndex(test["NextDate"].to_numpy(), name="實際交易日期"),
        ),
    }


def time_series_validation(train_data: pd.DataFrame) -> pd.DataFrame:
    """只在開發／訓練資料上執行四折驗證，不使用最終 20% 測試資料。"""
    splitter = TimeSeriesSplit(n_splits=4, gap=1)
    rows = []
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(train_data), start=1):
        fold_train = train_data.iloc[train_idx]
        fold_valid = train_data.iloc[valid_idx]
        scores = evaluate_model(fold_train, fold_valid)
        rows.append(
            {
                "折數": f"第 {fold} 折",
                "驗證起始日": fold_valid.index[0].strftime("%Y-%m-%d"),
                "驗證結束日": fold_valid.index[-1].strftime("%Y-%m-%d"),
                "AI Accuracy": scores["accuracy"],
                "方向基準 Accuracy": scores["baseline_accuracy"],
                "AI MAE（元）": scores["mae"],
                "價格基準 MAE（元）": scores["baseline_mae"],
            }
        )
    return pd.DataFrame(rows)


@st.cache_data(ttl=3600, show_spinner=False)
def run_analysis(ticker: str) -> dict:
    """整個分析結果快取一小時，避免調整圖表就重新訓練模型。"""
    raw, excluded_today = download_stock(ticker)
    data = make_features(raw)
    labeled = data.dropna(subset=["NextClose", "NextReturn", "NextDate"]).copy()

    split = int(len(labeled) * 0.8)
    # 留下一筆邊界間隔：避免訓練標籤使用測試起始日的收盤價。
    train = labeled.iloc[:split - 1]
    test = labeled.iloc[split:]
    if len(train) < 100 or len(test) < 30:
        raise ValueError("訓練或測試資料不足，請選擇歷史資料較長的股票。")

    holdout = evaluate_model(train, test)
    cv_results = time_series_validation(train)

    # 最終未來預測：使用所有已知標籤資料重新訓練。
    final_model = new_model()
    final_model.fit(labeled[FEATURES], labeled["NextReturn"])
    latest = data.iloc[[-1]]
    future_return = float(final_model.predict(latest[FEATURES])[0])
    latest_close = float(latest["Close"].iloc[0])

    return {
        "data": data,
        "excluded_today": excluded_today,
        "n_train": len(train),
        "n_test": len(test),
        "test_start": test.index[0],
        "test_end": test.index[-1],
        "latest_date": data.index[-1],
        "latest_price": latest_close,
        "future_return": future_return,
        "future_price": latest_close * (1 + future_return),
        "importance": pd.Series(
            final_model.feature_importances_, index=FEATURES
        ).sort_values(ascending=False),
        "cv_results": cv_results,
        **holdout,
    }


st.title("📈 AI 股票趨勢預測與分析系統")
st.caption("V2 修正版｜Python · yfinance · Random Forest｜課堂示範，非投資建議")

with st.sidebar:
    st.header("股票設定")
    selected = st.selectbox("選擇股票", list(STOCKS.keys()))
    show_days = st.select_slider(
        "走勢圖顯示範圍（交易日）", options=[60, 120, 200, 300], value=200
    )
    start = st.button("開始分析與預測", type="primary", use_container_width=True)
    st.caption("下載和模型結果快取一小時；換股票或調整圖表範圍可繼續使用。")

# Streamlit 操作選單時會重新執行；用 session_state 保留已啟動狀態。
if "analysis_started" not in st.session_state:
    st.session_state["analysis_started"] = False
if start:
    st.session_state["analysis_started"] = True
if not st.session_state["analysis_started"]:
    st.info("請在左側選擇股票，再按『開始分析與預測』。")
    st.stop()

try:
    with st.spinner("正在取得資料、進行測試與訓練模型……"):
        summary = run_analysis(STOCKS[selected])
except Exception as error:
    st.error(f"分析未完成：{error}")
    st.stop()

data = summary["data"]
if summary["excluded_today"]:
    st.info("已排除台灣時間今日 14:00 前可能尚未收盤的日線資料。")
st.caption("資料來源：Yahoo Finance；可能延遲或修訂，收盤狀態仍應以交易所公告為準。")

st.subheader(f"{selected}｜歷史股價走勢")
st.line_chart(data[["Close", "MA5", "MA20"]].tail(show_days))
st.caption("Close：收盤價；MA5：5 日均線；MA20：20 日均線。")

st.subheader("下一交易日模型預測")
a, b, c = st.columns(3)
a.metric("最新收盤價", f"NT$ {summary['latest_price']:,.2f}")
b.metric(
    "AI 預測收盤價",
    f"NT$ {summary['future_price']:,.2f}",
    delta=f"{summary['future_return']:+.2%}",
)
c.metric("AI 預測方向", "上漲" if summary["future_return"] > 0 else "沒有上漲")
st.caption(
    f"最新資料日期：{summary['latest_date']:%Y-%m-%d}。"
    "預測下一個實際交易日，不保證為實際成交價格。"
)

st.subheader("獨立測試結果（依時間約 80% 訓練／20% 測試）")
st.write(
    f"訓練：{summary['n_train']} 筆｜測試：{summary['n_test']} 筆｜"
    f"測試特徵日期：{summary['test_start']:%Y-%m-%d}～"
    f"{summary['test_end']:%Y-%m-%d}"
)
a, b, c = st.columns(3)
a.metric("方向 Accuracy", f"{summary['accuracy']:.2%}")
b.metric("AI MAE", f"{summary['mae']:.2f} 元")
c.metric("AI RMSE", f"{summary['rmse']:.2f} 元")

st.write("**股價預測：與『下一交易日收盤價等於今天』比較**")
price_comp = pd.DataFrame(
    {
        "AI 模型": [summary["mae"], summary["rmse"], summary["mape"]],
        "簡單基準": [
            summary["baseline_mae"],
            summary["baseline_rmse"],
            summary["baseline_mape"],
        ],
    },
    index=["MAE（元）", "RMSE（元）", "MAPE（%）"],
)
st.dataframe(price_comp.style.format("{:.2f}"), use_container_width=True)
st.caption("MAPE 是平均絕對百分比誤差，可輔助比較不同價位股票；越低越好。")

st.write("**隔日漲跌方向評估（『上漲』視為正類）**")
direction_comp = pd.DataFrame(
    {
        "指標": ["Accuracy", "Balanced Accuracy", "Precision", "Recall", "F1"],
        "AI 模型": [
            summary["accuracy"],
            summary["balanced_accuracy"],
            summary["precision"],
            summary["recall"],
            summary["f1"],
        ],
    }
)
direction_comp["AI 模型"] = direction_comp["AI 模型"].map(lambda x: f"{x:.2%}")
st.dataframe(direction_comp, use_container_width=True, hide_index=True)
st.write(f"固定猜訓練集多數類別的方向基準 Accuracy：**{summary['baseline_accuracy']:.2%}**")

if summary["mae"] >= summary["baseline_mae"]:
    st.warning("本次 AI 的價格 MAE 尚未勝過簡單基準，不能宣稱預測能力較好。")
else:
    st.success("本次 AI 的價格 MAE 優於簡單基準，但仍須檢查其他期間與股票的穩定性。")

st.subheader("四折時間序列驗證（僅使用訓練資料）")
cv = summary["cv_results"]
st.dataframe(
    cv.style.format(
        {
            "AI Accuracy": "{:.2%}",
            "方向基準 Accuracy": "{:.2%}",
            "AI MAE（元）": "{:.2f}",
            "價格基準 MAE（元）": "{:.2f}",
        }
    ),
    use_container_width=True,
    hide_index=True,
)
st.write(
    f"四折平均：AI 方向 Accuracy **{cv['AI Accuracy'].mean():.2%}** "
    f"（基準 {cv['方向基準 Accuracy'].mean():.2%}）；"
    f"AI MAE **{cv['AI MAE（元）'].mean():.2f} 元** "
    f"（基準 {cv['價格基準 MAE（元）'].mean():.2f} 元）。"
)
st.caption("四折結果來自訓練期間的開發驗證，不是另一組獨立最終測試。")

st.subheader("實際股價、AI 預測與基準比較")
st.line_chart(summary["results"].tail(80))
st.caption("橫軸為被預測的下一個實際交易日，不是產生預測的當天。")

st.subheader("模型特徵重要性")
st.bar_chart(summary["importance"])
st.caption("特徵重要性代表模型分裂時的相對貢獻，不代表因果關係。")

with st.expander("顯示最近 10 筆測試結果"):
    st.dataframe(
        summary["results"].tail(10).style.format("{:.2f}"),
        use_container_width=True,
    )

st.divider()
st.caption(
    "限制：只使用歷史股價、成交量及其衍生特徵；"
    "沒有加入財報、新聞或突發事件。過去測試表現不保證未來結果。"
)
