"""ARTEMIS Streamlit dashboard: portfolio performance, robustness, replay, live diagnosis, reward history."""
import datetime
import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from agent.collect import collect_episodes
from agent.evaluate import benchmark_buy_and_hold
from data.fetcher import DataPipeline
from diagnostics.diagnose import diagnose_episodes, load_lstm, most_common_fix
from diagnostics.lstm_model import FAILURE_MODES
from reward.fix_applicators import DEFAULT_REWARD_CONFIG, apply_fix, describe_fix

SEED = 42

st.set_page_config(page_title="ARTEMIS", layout="wide")
st.title("ARTEMIS — Autonomous Reinforcement Trading Dashboard")

REGIMES = ["bull", "bear", "sideways"]


@st.cache_data
def load_regime_df(regime: str) -> pd.DataFrame:
    return DataPipeline().get_regime(regime)


@st.cache_resource
def load_ppo_model(regime: str):
    from stable_baselines3 import PPO
    path = os.path.join("models", f"{regime}_ppo.zip")
    if not os.path.exists(path):
        return None
    return PPO.load(path)


@st.cache_resource
def load_diagnostic_lstm():
    path = os.path.join("models", "lstm", "lstm_model.pth")
    if not os.path.exists(path):
        return None
    return load_lstm(path)


tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Portfolio Performance",
    "Regime Robustness Grid",
    "Episode Replay",
    "Live Diagnosis",
    "Reward Engineering History",
])

# ---------------------------------------------------------------------------
# Tab 1: Portfolio Performance
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Portfolio Performance vs. Buy & Hold")
    regime = st.selectbox("Select regime", REGIMES, key="perf_regime")
    model = load_ppo_model(regime)

    if model is None:
        st.info(f"No trained model found at models/{regime}_ppo.zip — "
                f"run `python -m agent.train --regime {regime}` first.")
    else:
        if "perf_episode_key" not in st.session_state:
            st.session_state["perf_episode_key"] = 0
        if st.button("Run New Episode"):
            st.session_state["perf_episode_key"] += 1

        df = load_regime_df(regime)
        trajectories = collect_episodes(model, df, n_episodes=1)
        traj = trajectories[0]

        bh = benchmark_buy_and_hold(df)

        steps = traj["steps"]
        portfolio_curve = [100.0]
        for s in steps:
            portfolio_curve.append(portfolio_curve[-1] * (1 + s["daily_return"]))

        closes = df["Close"].values[: len(portfolio_curve)]
        bh_curve = 100.0 * closes / closes[0]

        fig = go.Figure()
        fig.add_trace(go.Scatter(y=portfolio_curve, mode="lines", name="ARTEMIS Agent"))
        fig.add_trace(go.Scatter(y=bh_curve, mode="lines", name="NIFTY 50 Buy & Hold"))
        fig.update_layout(xaxis_title="Trading day", yaxis_title="Value (base=100)")
        st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Return", f"{traj['total_return']:.2f}%")
        c2.metric("Sharpe Ratio", f"{traj['sharpe']:.2f}")
        c3.metric("Max Drawdown", f"{traj['max_drawdown']:.2%}")
        c4.metric("Win Rate", "100%" if traj["total_return"] > 0 else "0%")
        c5.metric("Num Trades", traj["num_trades"])

# ---------------------------------------------------------------------------
# Tab 2: Regime Robustness Grid
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Regime Robustness Grid (Sharpe Ratio)")
    grid_path = "results/regime_grid.json"
    if not os.path.exists(grid_path):
        st.info("Run agent/evaluate.py --grid to generate this")
    else:
        with open(grid_path) as f:
            grid = json.load(f)
        train_regimes = list(grid.keys())
        test_regimes = REGIMES
        z = [[grid[tr].get(te, None) for te in test_regimes] for tr in train_regimes]
        fig = go.Figure(data=go.Heatmap(
            z=z, x=test_regimes, y=train_regimes,
            colorscale="RdYlGn", zmid=0,
            text=z, texttemplate="%{text:.2f}",
        ))
        fig.update_layout(xaxis_title="Test regime", yaxis_title="Train regime")
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 3: Episode Replay
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Episode Replay")
    demo_path = "logs/demo_episodes.json"
    if not os.path.exists(demo_path):
        st.info("No demo episodes found at logs/demo_episodes.json — generate some with agent.collect first.")
    else:
        with open(demo_path) as f:
            demo_episodes = json.load(f)
        idx = st.slider("Episode index", 0, min(9, len(demo_episodes) - 1), 0)
        episode = demo_episodes[idx]

        prices = episode.get("prices", [])
        trade_log = episode.get("trade_log", [])
        portfolio_values = episode.get("portfolio_values", [])

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4],
                             subplot_titles=("NIFTY 50 Price", "Portfolio Value"))
        fig.add_trace(go.Scatter(y=prices, mode="lines", name="Close"), row=1, col=1)

        buys = [t for t in trade_log if t["action"] == "BUY"]
        sells = [t for t in trade_log if t["action"] == "SELL"]
        fig.add_trace(go.Scatter(
            x=[t["step"] for t in buys], y=[t["price"] for t in buys],
            mode="markers", marker=dict(symbol="triangle-up", color="green", size=12), name="Buy",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=[t["step"] for t in sells], y=[t["price"] for t in sells],
            mode="markers", marker=dict(symbol="triangle-down", color="red", size=12), name="Sell",
        ), row=1, col=1)

        fig.add_trace(go.Scatter(y=portfolio_values, mode="lines", name="Portfolio Value"), row=2, col=1)
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(pd.DataFrame(trade_log))

# ---------------------------------------------------------------------------
# Tab 4: Live Diagnosis
# ---------------------------------------------------------------------------
with tab4:
    st.subheader("Live Diagnosis")
    if "diagnosis_history" not in st.session_state:
        st.session_state["diagnosis_history"] = []

    diag_regime = st.selectbox("Select regime", REGIMES, key="diag_regime")
    lstm = load_diagnostic_lstm()
    diag_model = load_ppo_model(diag_regime)

    if lstm is None:
        st.info("No trained LSTM found at models/lstm/lstm_model.pth — run `python -m diagnostics.train_lstm` first.")
    elif diag_model is None:
        st.info(f"No trained model found at models/{diag_regime}_ppo.zip — "
                f"run `python -m agent.train --regime {diag_regime}` first.")
    else:
        if st.button("Run Episode + Diagnose"):
            df = load_regime_df(diag_regime)
            trajectories = collect_episodes(diag_model, df, n_episodes=1)
            diagnoses = diagnose_episodes(lstm, trajectories)
            diagnosis = diagnoses[0]

            fix_type = diagnosis["fix_type"]
            new_config = apply_fix(fix_type, DEFAULT_REWARD_CONFIG)
            description = describe_fix(fix_type, DEFAULT_REWARD_CONFIG, new_config)

            st.session_state["diagnosis_history"].insert(0, {
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "failure_mode": diagnosis["failure_mode"],
                "fix_type": fix_type,
                "confidence": diagnosis["failure_confidence"],
            })
            st.session_state["diagnosis_history"] = st.session_state["diagnosis_history"][:5]
            st.session_state["last_diagnosis"] = diagnosis
            st.session_state["last_fix_description"] = description

        if "last_diagnosis" in st.session_state:
            diagnosis = st.session_state["last_diagnosis"]
            failure_mode = diagnosis["failure_mode"]
            color = "green" if failure_mode == "PROFITABLE" else ("red" if failure_mode == "MAX_DRAWDOWN" else "orange")
            st.markdown(
                f"<div style='background-color:{color};color:white;padding:16px;"
                f"border-radius:8px;font-size:24px;text-align:center;'>{failure_mode}</div>",
                unsafe_allow_html=True,
            )

            probs = diagnosis["failure_probs"]
            fig = go.Figure(go.Bar(x=list(probs.keys()), y=list(probs.values())))
            fig.update_layout(yaxis_title="Confidence", xaxis_title="Failure mode")
            st.plotly_chart(fig, use_container_width=True)

            st.text_area("Recommended fix", f"{diagnosis['fix_type']} — {st.session_state['last_fix_description']}")

        if st.session_state["diagnosis_history"]:
            st.write("Last 5 diagnoses")
            st.dataframe(pd.DataFrame(st.session_state["diagnosis_history"]))

# ---------------------------------------------------------------------------
# Tab 5: Reward Engineering History
# ---------------------------------------------------------------------------
with tab5:
    st.subheader("Reward Engineering History")
    fix_log_path = "logs/fix_loop.json"
    if not os.path.exists(fix_log_path):
        st.info("Run reward/fix_loop.py to generate this")
    else:
        with open(fix_log_path) as f:
            fix_log = json.load(f)

        components = ["return_weight", "transaction_cost", "drawdown_penalty",
                      "holding_bonus", "profit_take_bonus", "sharpe_bonus"]
        iterations = [entry["iteration"] for entry in fix_log]

        fig = go.Figure()
        for comp in components:
            fig.add_trace(go.Bar(
                x=iterations, y=[entry["reward_config"].get(comp, 0) for entry in fix_log], name=comp,
            ))
        fig.update_layout(barmode="group", xaxis_title="Iteration", yaxis_title="Weight")
        st.plotly_chart(fig, use_container_width=True)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=iterations, y=[e["sharpe_before"] for e in fix_log],
                                   mode="lines+markers", name="Sharpe before"))
        fig2.add_trace(go.Scatter(x=iterations, y=[e["sharpe_after"] for e in fix_log],
                                   mode="lines+markers", name="Sharpe after"))
        fig2.update_layout(xaxis_title="Iteration", yaxis_title="Sharpe ratio")
        st.plotly_chart(fig2, use_container_width=True)

        table_df = pd.DataFrame([{
            "iteration": e["iteration"], "fix_type": e["fix_type"], "outcome": e["outcome"],
            "sharpe_before": e["sharpe_before"], "sharpe_after": e["sharpe_after"],
        } for e in fix_log])
        st.dataframe(table_df)
