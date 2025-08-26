import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")

import plotly.express as px
import plotly.graph_objects as go

import ta
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

def add_weeks_safe(date_like, weeks):
    date_like = pd.to_datetime(date_like)
    return date_like + pd.to_timedelta(int(weeks)*7, unit='d')

def add_days_safe(date_like, days):
    date_like = pd.to_datetime(date_like)
    return date_like + pd.to_timedelta(int(days), unit='d')

st.set_page_config(layout="wide", page_title="Stock Trend & Price Predictor")

TICKERS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "KOTAKBANK.NS", "LT.NS", "BHARTIARTL.NS"
]

TIME_FRAMES_WEEKS = {
    "4 Weeks (1 Month)": 4,
    "8 Weeks (2 Months)": 8,
    "26 Weeks (6 Months)": 26,
    "52 Weeks (1 Year)": 52
}

@st.cache_resource
def load_scaler(ticker: str):
    fn = f'scaler_{ticker.replace(".","")}.pkl'
    with open(fn, 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def load_model(model_type: str, timeframe_weeks: int):
    fn = f'lightgbm_{model_type}_{timeframe_weeks}w.pkl'
    with open(fn, 'rb') as f:
        return pickle.load(f)

@st.cache_data(ttl=3600)
def load_combined_data_and_features():
    df = pd.read_csv("combined_normalized_data.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    non_feature_cols = [
        'Date','Ticker','NSE','Sector','Stock Name','Market Cap (INR Billions)',
        'Open','High','Low','Close','Adj Close','Volume'
    ] + [f"FutureReturn_{w}w" for w in TIME_FRAMES_WEEKS.values()] + \
        [f"Trend_{w}w" for w in TIME_FRAMES_WEEKS.values()]
    feature_cols = [c for c in df.columns if c not in non_feature_cols and pd.api.types.is_numeric_dtype(df[c])]
    feature_cols.sort()
    return df, feature_cols

@st.cache_data(ttl=3600)
def load_stock_info():
    try:
        si = pd.read_csv("stock_info.csv")
        return si
    except Exception:
        return None

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_values('Date')
    if 'Close' in out.columns:
        out['SMA_20'] = out['Close'].rolling(window=20).mean()
        out['EMA_20'] = out['Close'].ewm(span=20, adjust=False).mean()
        bb_mid = out['Close'].rolling(20).mean()
        bb_std = out['Close'].rolling(20).std()
        out['BB_upper'] = bb_mid + 2*bb_std
        out['BB_lower'] = bb_mid - 2*bb_std
    try:
        out['RSI_14'] = ta.momentum.RSIIndicator(close=out['Close'], window=14).rsi()
    except Exception:
        out['RSI_14'] = np.nan
    try:
        macd_ind = ta.trend.MACD(close=out['Close'], window_slow=26, window_fast=12, window_sign=9)
        out['MACD'] = macd_ind.macd()
        out['MACD_Signal'] = macd_ind.macd_signal()
    except Exception:
        out['MACD'] = np.nan
        out['MACD_Signal'] = np.nan
    return out

def get_prediction_date_row(df_ticker: pd.DataFrame, as_of_date: pd.Timestamp):
    df_sub = df_ticker[df_ticker['Date'] <= as_of_date]
    if df_sub.empty:
        return None
    return df_sub.tail(1).copy()

def scale_features_for_row(row_df: pd.DataFrame, feature_cols: list, scaler: MinMaxScaler):
    X_raw = row_df[feature_cols].fillna(0)
    X_scaled = scaler.transform(X_raw)
    return pd.DataFrame(X_scaled, columns=feature_cols)

df_all, feature_cols = load_combined_data_and_features()
all_available_tickers = sorted(df_all['Ticker'].unique().tolist())
stock_info_df = load_stock_info()

st.title("📊 Stock Trend & Price Prediction Dashboard")

st.markdown("---")

with st.sidebar:
    st.header("⚙️ Prediction Settings")
    default_idx = all_available_tickers.index(TICKERS[0]) if TICKERS and TICKERS[0] in all_available_tickers else 0
    sel_ticker = st.selectbox("Select Stock:", all_available_tickers, index=default_idx)
    tf_label = st.selectbox("Prediction Timeframe:", list(TIME_FRAMES_WEEKS.keys()))
    tf_weeks = TIME_FRAMES_WEEKS[tf_label]

    noise_tolerance = 0.5

    st.markdown("---")
    st.header("📅 Historical Range")
    ticker_df = df_all[df_all['Ticker']==sel_ticker].sort_values('Date')
    min_date = ticker_df['Date'].min().date() if not ticker_df.empty else datetime(2000,1,1).date()
    max_date = ticker_df['Date'].max().date() if not ticker_df.empty else datetime.today().date()
    date_range = st.date_input("Select Date Range (charts & stats):", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    if isinstance(date_range, tuple) and len(date_range)==2:
        start_date, end_date = [pd.to_datetime(d) for d in date_range]
    else:
        start_date, end_date = pd.to_datetime(min_date), pd.to_datetime(max_date)

    as_of_date = st.date_input(
        "Predict as if at this Date (backtest view):",
        value=max_date, min_value=min_date, max_value=max_date,
        help="We will run the model using the latest row on/before this date."
    )
    as_of_date = pd.to_datetime(as_of_date)

df_ticker_all = df_all[df_all['Ticker']==sel_ticker].sort_values('Date').reset_index(drop=True)
row_for_pred = get_prediction_date_row(df_ticker_all, as_of_date)

if row_for_pred is None or row_for_pred.empty:
    st.error(f"No data available for {sel_ticker} on/before {as_of_date.date()}.")
    st.stop()

current_close_price = row_for_pred['Close'].iloc[0]
stock_full_name = row_for_pred['Stock Name'].iloc[0] if 'Stock Name' in row_for_pred.columns else sel_ticker

scaler = load_scaler(sel_ticker)
clf = load_model("classifier", tf_weeks)
reg = load_model("regressor", tf_weeks)

try:
    model_input_raw = row_for_pred[feature_cols]

    X_scaled_df = pd.DataFrame(scaler.transform(model_input_raw.fillna(0)), columns=feature_cols)

    trend_pred = clf.predict(X_scaled_df)[0]
    try:
        trend_prob_up = clf.predict_proba(X_scaled_df)[:,1][0]
    except Exception:
        trend_prob_up = np.nan

    pred_return_pct = reg.predict(X_scaled_df)[0]

    if trend_pred == 0 and pred_return_pct > 0:
        if pred_return_pct <= noise_tolerance:
            st.warning(f"Trend=DOWN, small positive return ({pred_return_pct:.2f}%) clipped to 0%.")
            pred_return_pct = 0.0
        else:
            st.warning(f"Trend=DOWN but regressor predicted +{pred_return_pct:.2f}%. Possible model mismatch.")

    predicted_price = current_close_price * (1 + pred_return_pct/100.0)
    predicted_date = add_weeks_safe(row_for_pred['Date'].iloc[0], tf_weeks)

except Exception as e:
    st.error(f"An error occurred during prediction or feature display: {e}. Please ensure the data and models are correctly generated and compatible.")
    st.stop()

@st.cache_data(ttl=3600)
def get_overall_recommendations(all_tickers, time_frames, df_full, feature_cols, noise_tol):
    recommendations = []

    for ticker in all_tickers:
        df_t = df_full[df_full['Ticker'] == ticker].sort_values('Date').reset_index(drop=True)
        if df_t.empty:
            continue

        latest_row = df_t.tail(1).copy()
        if latest_row.empty:
            continue

        current_close = latest_row['Close'].iloc[0]
        stock_name = latest_row['Stock Name'].iloc[0] if 'Stock Name' in latest_row.columns else ticker

        try:
            scaler = load_scaler(ticker)
            model_input_raw = latest_row[feature_cols]
            X_scaled_df = pd.DataFrame(scaler.transform(model_input_raw.fillna(0)), columns=feature_cols)
        except Exception:
            continue

        for tf_label, tf_weeks_val in time_frames.items():
            try:
                clf = load_model("classifier", tf_weeks_val)
                reg = load_model("regressor", tf_weeks_val)

                trend_pred_overall = clf.predict(X_scaled_df)[0]
                pred_return_overall = reg.predict(X_scaled_df)[0]

                if trend_pred_overall == 0 and pred_return_overall > 0 and pred_return_overall <= noise_tol:
                    pred_return_overall = 0.0

                predicted_price_overall = current_close * (1 + pred_return_overall / 100.0)
                predicted_date_overall = add_weeks_safe(latest_row['Date'].iloc[0], tf_weeks_val)

                recommendations.append({
                    'Ticker': ticker,
                    'Stock Name': stock_name,
                    'Timeframe': tf_label,
                    'Weeks': tf_weeks_val,
                    'Current Close': current_close,
                    'Predicted Return (%)': pred_return_overall,
                    'Predicted Price': predicted_price_overall,
                    'Predicted Date': predicted_date_overall.date()
                })
            except Exception:
                continue

    return pd.DataFrame(recommendations)

tab_info, tab_pred, tab_overall_conclusion, tab_adv, tab_hist, tab_backtest = st.tabs([
    "ℹ️ Stock Information",
    "🔮 Prediction",
    "🏆 Overall Recommendations",
    "📊 Advanced Charts",
    "📉 Performance Compare",
    "📈 Historical Accuracy"
])

with tab_info:
    st.subheader("📚 Explore Our Stock Universe")
    st.markdown("""
    Welcome to the **Stock Information Hub**! Here, you can get a snapshot of all the stocks currently
    available in our analysis. Dive into the market capitalization breakdown by sector and view detailed
    information about each company.
    """)

    if stock_info_df is not None and not stock_info_df.empty:
        total_market_cap = stock_info_df['Market Cap (INR Billions)'].sum() if 'Market Cap (INR Billions)' in stock_info_df.columns else None
        num_stocks = len(stock_info_df)

        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.metric(label="Total Stocks Analyzed", value=f"{num_stocks}")
        with col_info2:
            if total_market_cap is not None:
                st.metric(label="Total Market Cap (INR Billions)", value=f"₹ {total_market_cap:,.2f} B")
            else:
                st.info("Market Cap data not available for summary.")

        st.markdown("---")
        st.markdown("### Market Capitalization by Sector")
        st.write("Understand the distribution of market cap across different sectors in our dataset.")

        info_sun_tab, info_table_tab = st.tabs(["Sunburst Chart", "Detailed Table"])
        with info_sun_tab:
            if all(col in stock_info_df.columns for col in ['NSE','Sector','Stock Name']):
                if 'Market Cap (INR Billions)' in stock_info_df.columns:
                    fig_sun_top = px.sunburst(stock_info_df, path=['NSE','Sector','Stock Name'], values='Market Cap (INR Billions)', color='Sector',
                                             title="Market Capitalization Breakdown by Sector", height=600)
                else:
                    tmp_si = stock_info_df.copy()
                    tmp_si['Count'] = 1
                    fig_sun_top = px.sunburst(tmp_si, path=['NSE','Sector','Stock Name'], values='Count', color='Sector',
                                             title="Stock Count Breakdown by Sector", height=600)

                if fig_sun_top.data and len(fig_sun_top.data) > 0 and hasattr(fig_sun_top.data[0].marker, 'colors') and fig_sun_top.data[0].marker.colors:
                    current_colors = list(fig_sun_top.data[0].marker.colors)
                    current_colors[0] = 'lightblue'
                    fig_sun_top.update_traces(marker=dict(colors=current_colors))

                st.plotly_chart(fig_sun_top, use_container_width=True)
            else:
                st.info("Required columns ('NSE', 'Sector', 'Stock Name') for Sunburst Chart are missing in `stock_info.csv`.")
        with info_table_tab:
            st.markdown("#### All Stock Details")
            st.write("A comprehensive list of all included stocks and their core information.")
            try:
                st.dataframe(stock_info_df.set_index('Ticker'), use_container_width=True)
            except Exception:
                st.dataframe(stock_info_df, use_container_width=True)
    else:
        st.warning("No stock information available. Please ensure `stock_info.csv` is present and correctly formatted.")


with tab_overall_conclusion:
    st.subheader("🏆 Overall Stock Recommendations Across Timeframes")
    st.write("This section provides a general overview of predicted top-performing stocks for various investment horizons.")

    with st.spinner("Generating overall recommendations... This might take a moment."):
        all_recs_df = get_overall_recommendations(all_available_tickers, TIME_FRAMES_WEEKS, df_all, feature_cols, noise_tolerance)

    if not all_recs_df.empty:
        all_recs_df_sorted = all_recs_df.sort_values(by='Predicted Return (%)', ascending=False).reset_index(drop=True)

        for tf_label, tf_weeks_val in TIME_FRAMES_WEEKS.items():
            st.markdown(f"---")
            st.markdown(f"### Best Stock for **{tf_label}**")

            best_stock_tf = all_recs_df_sorted[
                (all_recs_df_sorted['Weeks'] == tf_weeks_val) &
                (all_recs_df_sorted['Predicted Return (%)'] > 0)
            ].head(1)

            if not best_stock_tf.empty:
                stock_name = best_stock_tf['Stock Name'].iloc[0]
                pred_return = best_stock_tf['Predicted Return (%)'].iloc[0]
                pred_price = best_stock_tf['Predicted Price'].iloc[0]
                pred_date = best_stock_tf['Predicted Date'].iloc[0]
                current_close = best_stock_tf['Current Close'].iloc[0]

                st.success(f"""
                **📈 Top Pick for {tf_label}: {stock_name}**
                * **Predicted Return:** `{pred_return:.2f}%`
                * **Current Close Price:** `₹ {current_close:,.2f}`
                * **Estimated Target Price:** `₹ {pred_price:,.2f}`
                * **Predicted Date:** `{pred_date}`
                """)

                st.markdown(f"#### Top 5 Performers for {tf_label}")
                top_5_stocks_tf = all_recs_df_sorted[
                    (all_recs_df_sorted['Weeks'] == tf_weeks_val) &
                    (all_recs_df_sorted['Predicted Return (%)'] > 0)
                ].head(5)

                if not top_5_stocks_tf.empty:
                    st.dataframe(
                        top_5_stocks_tf[['Stock Name', 'Predicted Return (%)', 'Current Close', 'Predicted Price', 'Predicted Date']]
                        .style.format({
                            'Predicted Return (%)': "{:.2f}%",
                            'Current Close': "₹ {:.2f}",
                            'Predicted Price': "₹ {:.2f}"
                        }),
                        hide_index=True,
                        use_container_width=True
                    )
                else:
                    st.info(f"No positive return predictions found for {tf_label} to list top 5.")

            else:
                st.info(f"No positive return predictions found for {tf_label} among the available stocks.")
    else:
        st.warning("Could not generate overall recommendations. Please check data and model files.")

    st.markdown("---")
    st.warning("""
    **Disclaimer**: These are AI-generated predictions and **not financial advice**. Investment decisions should always be based on thorough personal research, risk tolerance, and consultation with a qualified financial advisor. Market conditions can change rapidly, and past performance or model predictions do not guarantee future results.
    """)


with tab_pred:
    st.subheader(f"Data & Prediction for **{stock_full_name}**")
    c1, c2 = st.columns(2)
    with c1:
        st.metric(label=f"Latest Close (as of {row_for_pred['Date'].dt.strftime('%Y-%m-%d').iloc[0]})",
                     value=f"₹ {current_close_price:,.2f}")
        if pd.notna(trend_prob_up):
            st.success(f"Confidence (Prob. of Up): **{trend_prob_up:.2%}**")
    with c2:
        st.metric(label=f"Estimated Price in {tf_weeks} weeks ({predicted_date.date()})",
                     value=f"₹ {predicted_price:,.2f}")
        st.info(f"Predicted Return: **{pred_return_pct:.2f}%**")

    st.markdown("---")
    st.caption("Prediction uses the latest available row on/before the chosen 'Predict as if' date.")

    st.subheader(f"Investment Conclusion for **{stock_full_name}**")

    if pred_return_pct > 0:
        st.markdown(f"""
        Based on our model's prediction, **{stock_full_name}** shows a potential for growth.

        * **Predicted Return**: You could expect an approximate **{pred_return_pct:.2f}%** return over the next **{tf_weeks} weeks**.
        * **Estimated Target Price**: The model projects a price of **₹ {predicted_price:,.2f}** by **{predicted_date.date()}**.
        """)
        st.success("This prediction indicates a positive outlook for the selected stock.")
    elif pred_return_pct < 0:
        st.markdown(f"""
        Our model suggests caution for **{stock_full_name}**.

        * **Predicted Return**: The model anticipates a return of **{pred_return_pct:.2f}%** over the next **{tf_weeks} weeks**.
        * **Estimated Target Price**: The projected price is **₹ {predicted_price:,.2f}** by **{predicted_date.date()}**.
        """)
        st.warning("This prediction indicates a negative outlook, suggesting a potential decline.")
    else:
        st.markdown(f"""
        The model predicts a neutral outlook for **{stock_full_name}**.

        * **Predicted Return**: A **0.00%** return is anticipated over the next **{tf_weeks} weeks**.
        * **Estimated Target Price**: The price is projected to remain around **₹ {predicted_price:,.2f}** by **{predicted_date.date()}**.
        """)
        st.info("This prediction indicates a stable or flat outlook for the selected stock.")

    st.markdown("---")
    st.warning("""
    **Disclaimer**: This conclusion is based solely on algorithmic predictions from the provided models and historical data. It does not constitute financial advice. Stock market investments are subject to market risks, and you should perform your own due diligence or consult with a qualified financial advisor before making any investment decisions.
    """)


with tab_adv:
    st.subheader("Candlestick with Overlays, Volume & Prediction Marker")

    df_ticker = df_ticker_all[(df_ticker_all['Date']>=start_date)&(df_ticker_all['Date']<=end_date)].copy()
    if df_ticker.empty:
        st.warning("No data in selected date range.")
    else:
        df_ticker = add_indicators(df_ticker)
        df_ticker.set_index('Date', inplace=True)

        fig = go.Figure()

        fig.add_trace(go.Candlestick(
            x=df_ticker.index, open=df_ticker['Open'], high=df_ticker['High'],
            low=df_ticker['Low'], close=df_ticker['Close'], name='OHLC'
        ))

        if 'SMA_20' in df_ticker.columns:
            fig.add_trace(go.Scatter(x=df_ticker.index, y=df_ticker['SMA_20'], mode='lines', name='SMA 20'))
        if 'EMA_20' in df_ticker.columns:
            fig.add_trace(go.Scatter(x=df_ticker.index, y=df_ticker['EMA_20'], mode='lines', name='EMA 20'))
        if {'BB_upper','BB_lower'}.issubset(df_ticker.columns):
            fig.add_trace(go.Scatter(x=df_ticker.index, y=df_ticker['BB_upper'], mode='lines', name='BB Upper', opacity=0.4))
            fig.add_trace(go.Scatter(x=df_ticker.index, y=df_ticker['BB_lower'], mode='lines', name='BB Lower', opacity=0.4))

        fig.add_trace(go.Bar(x=df_ticker.index, y=df_ticker['Volume'], name='Volume', opacity=0.3, yaxis='y2'))

        pred_point_x = pd.to_datetime(predicted_date).to_pydatetime()
        as_of_x = pd.to_datetime(row_for_pred['Date'].iloc[0]).to_pydatetime()

        fig.add_shape(type="line", x0=as_of_x, x1=as_of_x, y0=df_ticker['Close'].min(), y1=df_ticker['Close'].max(), line=dict(color="gray", dash="dot"))
        fig.add_annotation(x=as_of_x, y=df_ticker['Close'].max(), text="Prediction date", showarrow=False, yshift=10)


        fig.add_trace(go.Scatter(
            x=[pred_point_x], y=[predicted_price], mode='markers+text',
            text=[f"Pred ₹{predicted_price:,.2f}"], textposition='top center',
            name='Predicted Price', marker=dict(size=10, color='purple', symbol='diamond')
        ))
        fig.add_trace(go.Scatter(
            x=[as_of_x, pred_point_x],
            y=[current_close_price, predicted_price],
            mode='lines', name='Projected Path', line=dict(dash='dash', color='purple')
        ))

        fig.update_layout(
            title=f"{stock_full_name} — OHLC with Indicators & Prediction",
            xaxis_rangeslider_visible=False,
            yaxis_title="Price",
            yaxis2=dict(title="Volume", overlaying='y', side='right', showgrid=False),
            height=650,
            xaxis_type='date'
        )
        st.plotly_chart(fig, use_container_width=True)

with tab_backtest:
    st.subheader(f"Historical Prediction Accuracy — {stock_full_name} ({tf_label})")

    df_bt = df_ticker_all.copy()
    if df_bt.empty:
        st.info("No historical data for backtest.")
    else:
        try:
            X_raw = df_bt[feature_cols].fillna(0)
            X_scaled = scaler.transform(X_raw)
            Xs = pd.DataFrame(X_scaled, columns=feature_cols)

            y_trend_pred = clf.predict(Xs)
            try:
                y_trend_proba = clf.predict_proba(Xs)[:,1]
            except Exception:
                y_trend_proba = np.full(len(Xs), np.nan)
            y_ret_pred = reg.predict(Xs)

            y_ret_pred_adj = y_ret_pred.copy()
            mask_down_pos = (y_trend_pred==0) & (y_ret_pred_adj>0)
            small_pos = (y_ret_pred_adj<=noise_tolerance)
            y_ret_pred_adj[np.where(mask_down_pos & small_pos)] = 0.0

            trend_col = f"Trend_{tf_weeks}w"
            ret_col = f"FutureReturn_{tf_weeks}w"
            if trend_col not in df_bt.columns or ret_col not in df_bt.columns:
                st.warning(f"Missing target columns '{trend_col}' or '{ret_col}' for backtest.")
            else:
                df_bt['PredTrend'] = y_trend_pred
                df_bt['PredTrendProbUp'] = y_trend_proba
                df_bt['PredReturn'] = y_ret_pred_adj
                df_bt['ActualTrend'] = df_bt[trend_col]
                df_bt['ActualReturn'] = df_bt[ret_col]

                valid_trend = df_bt.dropna(subset=['ActualTrend'])
                trend_acc = (valid_trend['PredTrend'] == valid_trend['ActualTrend']).mean() if not valid_trend.empty else np.nan
                valid_ret = df_bt.dropna(subset=['ActualReturn'])
                mae_ret = np.mean(np.abs(valid_ret['PredReturn'] - valid_ret['ActualReturn'])) if not valid_ret.empty else np.nan

                m1, m2 = st.columns(2)
                with m1:
                    st.metric("Trend Accuracy", f"{trend_acc*100:,.2f}%" if pd.notna(trend_acc) else "N/A")
                with m2:
                    st.metric("Return MAE (pp)", f"{mae_ret:,.2f}" if pd.notna(mae_ret) else "N/A")

                fig_ret = go.Figure()
                fig_ret.add_trace(go.Scatter(x=df_bt['Date'], y=df_bt['ActualReturn'], name="Actual Return (%)", mode='lines'))
                fig_ret.add_trace(go.Scatter(x=df_bt['Date'], y=df_bt['PredReturn'], name="Predicted Return (%)", mode='lines'))
                fig_ret.update_layout(title="Actual vs Predicted Future Return (%)", xaxis_title="Date", yaxis_title="Return (%)", height=400)
                st.plotly_chart(fig_ret, use_container_width=True)

                fig_tr = go.Figure()
                fig_tr.add_trace(go.Scatter(x=df_bt['Date'], y=df_bt['ActualTrend'], name="Actual Trend", mode='lines'))
                fig_tr.add_trace(go.Scatter(x=df_bt['Date'], y=df_bt['PredTrend'], name="Predicted Trend", mode='lines'))
                fig_tr.update_layout(title="Actual vs Predicted Trend (0=Down,1=Up)", xaxis_title="Date", yaxis=dict(tickmode='array', tickvals=[0,1]), height=300)
                st.plotly_chart(fig_tr, use_container_width=True)

        except Exception as e:
            st.error(f"Backtest failed: {e}")

with tab_hist:
    st.subheader("Historical Performance Comparison")

    if 'comparison_timeframe_label' not in st.session_state:
        st.session_state.comparison_timeframe_label = list(TIME_FRAMES_WEEKS.keys())[3]

    st.write("Select Timeframe for Comparison:")
    comparison_tf_options = list(TIME_FRAMES_WEEKS.keys())
    
    cols = st.columns(len(comparison_tf_options))
    for i, tf_option in enumerate(comparison_tf_options):
        with cols[i]:
            if st.button(tf_option, key=f"hist_tf_button_{i}", type="primary" if st.session_state.comparison_timeframe_label == tf_option else "secondary"):
                st.session_state.comparison_timeframe_label = tf_option
                st.rerun()

    comparison_tf_label = st.session_state.comparison_timeframe_label
    comparison_weeks_num = TIME_FRAMES_WEEKS[comparison_tf_label]
    comparison_col = f'FutureReturn_{comparison_weeks_num}w'

    selected_tickers_for_comparison = st.multiselect(
        "Select Stocks for Historical Comparison:",
        options=all_available_tickers,
        default=all_available_tickers
    )

    if selected_tickers_for_comparison:
        df_comp = df_all[df_all['Ticker'].isin(selected_tickers_for_comparison)].copy()
        df_comp = df_comp[(df_comp['Date']>=start_date)&(df_comp['Date']<=end_date)]
        if df_comp.empty:
            st.info("No historical data available for selected stocks and date range.")
        else:
            st.subheader(f"Historical {comparison_tf_label} Future Returns (%) — Latest per Ticker")
            if comparison_col in df_comp.columns:
                latest_returns_df = df_comp.groupby('Ticker').tail(1).reset_index(drop=True)
                latest_returns_df[comparison_col] = pd.to_numeric(latest_returns_df[comparison_col], errors='coerce')
                fig_cmp = px.bar(latest_returns_df, x='Stock Name', y=comparison_col, color='Ticker')
                st.plotly_chart(fig_cmp, use_container_width=True)
                st.markdown("---")
                st.subheader("Latest Recorded Historical Returns Table")
                display_cols = ['Ticker','Stock Name','Date','Adj Close', comparison_col]
                exist_cols = [c for c in display_cols if c in latest_returns_df.columns]
                latest_returns_df[comparison_col] = latest_returns_df[comparison_col].round(2)
                st.dataframe(latest_returns_df[exist_cols].set_index('Ticker'))
            else:
                st.warning(f"Column '{comparison_col}' not found in data.")
    else:
        st.info("Please select stocks for historical comparison above.")
    st.markdown("---")