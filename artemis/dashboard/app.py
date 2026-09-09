"""ARTEMIS Research dashboard: performance, robustness, attention XAI, ablation, meta-policy, cross-market."""
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
from diagnostics.diagnose import diagnose_episodes, load_lstm
from diagnostics.lstm_model import FAILURE_MODES
from reward.fix_applicators import DEFAULT_REWARD_CONFIG, apply_fix, describe_fix

SEED = 42

st.set_page_config(page_title="ARTEMIS Research", layout="wide")
st.title("ARTEMIS Research — Autonomous Reinforcement Trading Dashboard")

REGIMES = ["bull", "bear", "sideways"]


@st.cache_data
def load_regime_df(regime: str) -> pd.DataFrame:
    return DataPipeline().get_regime(regime)


@st.cache_resource
def load_ppo_model(regime: str):
    from stable_baselines3 import PPO
    path = f"models/{regime}_ppo_seed42.zip"
    if not os.path.exists(path):
        return None
    return PPO.load(path)


@st.cache_resource
def load_diagnostic_lstm():
    path = "models/lstm/lstm_model.pth"
    if not os.path.exists(path):
        return None
    return load_lstm(path)


@st.cache_resource
def load_sac_meta_agent():
    from meta.meta_agent import load_meta_agent
    path = "models/meta/sac_meta_policy.zip"
    if not os.path.exists(path):
        return None
    return load_meta_agent(path)


tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Portfolio Performance",
    "Regime Robustness Grid",
    "Episode Replay + Attention",
    "Live Diagnosis",
    "Ablation Study Results",
    "Meta-Policy Training",
    "Cross-Market Generalization",
])

# ---------------------------------------------------------------------------
# Tab 1: Portfolio Performance
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Portfolio Performance vs. Buy & Hold")
    regime = st.selectbox("Select regime", REGIMES, key="perf_regime")
    model = load_ppo_model(regime)

    if model is None:
        st.info(f"No trained model found at models/{regime}_ppo_seed42.zip — "
                f"run `python -m agent.train --regime {regime}` first.")
    else:
        if "perf_episode_key" not in st.session_state:
            st.session_state["perf_episode_key"] = 0
        if st.button("Run New Episode"):
            st.session_state["perf_episode_key"] += 1

        df = load_regime_df(regime)
        trajectories = collect_episodes(model, df, n_episodes=1)
        traj = trajectories[0]

        steps = traj["steps"]
        portfolio_curve = [100.0]
        for s in steps:
            portfolio_curve.append(portfolio_curve[-1] * (1 + s["daily_return"]))

        from data.fetcher import raw_close
        closes = raw_close(df).values[: len(portfolio_curve)]
        bh_curve = 100.0 * closes / closes[0]

        fig = go.Figure()
        fig.add_trace(go.Scatter(y=portfolio_curve, mode="lines", name="ARTEMIS Agent"))
        fig.add_trace(go.Scatter(y=bh_curve, mode="lines", name="NIFTY 50 Buy & Hold"))
        fig.update_layout(xaxis_title="Trading day", yaxis_title="Value (base=100)")
        st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Return", f"{traj['total_return']*100:.2f}%")
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
        st.info("Run experiments/stats.py (compute_regime_grid) to generate this")
    else:
        with open(grid_path) as f:
            grid = json.load(f)
        train_regimes = list(grid.keys())
        test_regimes = REGIMES
        z = [[grid[tr].get(te, None) for te in test_regimes] for tr in train_regimes]
        fig = go.Figure(data=go.Heatmap(
            z=z, x=test_regimes, y=train_regimes, colorscale="RdYlGn", zmid=0,
            text=z, texttemplate="%{text:.2f}",
        ))
        fig.update_layout(xaxis_title="Test regime", yaxis_title="Train regime")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Diagonal cells (train regime == test regime) are the in-distribution baseline.")

# ---------------------------------------------------------------------------
# Tab 3: Episode Replay + Attention
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Episode Replay + Attention")
    demo_path = "logs/demo_episodes.json"
    if not os.path.exists(demo_path):
        st.info("No demo episodes found at logs/demo_episodes.json — generate some with agent.collect + diagnostics.diagnose first.")
    else:
        with open(demo_path) as f:
            demo_episodes = json.load(f)
        idx = st.slider("Episode index", 0, min(9, len(demo_episodes) - 1), 0)
        episode = demo_episodes[idx]

        prices = episode.get("prices", [])
        trade_log = episode.get("trade_log", [])
        portfolio_values = episode.get("portfolio_values", [])
        attention_weights = episode.get("attention_weights", [])
        failure_mode = episode.get("failure_mode", "UNKNOWN")

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

        st.markdown(f"**Attention Heatmap — Which moments caused this failure ({failure_mode})?**")
        if attention_weights:
            fig2 = go.Figure(data=go.Heatmap(
                z=[attention_weights], colorscale="Inferno", showscale=True,
            ))
            fig2.update_layout(xaxis_title="Timestep", yaxis=dict(showticklabels=False), height=200)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.caption("No attention weights recorded for this episode.")

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
        st.info(f"No trained model found at models/{diag_regime}_ppo_seed42.zip — "
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

            attn = diagnosis.get("attention_weights", [])
            if attn:
                fig3 = go.Figure(go.Scatter(y=attn, mode="lines", name="Attention"))
                fig3.update_layout(xaxis_title="Timestep", yaxis_title="Attention weight",
                                    title="Critical timesteps (attention over the episode)")
                st.plotly_chart(fig3, use_container_width=True)

            st.text_area("Recommended fix", f"{diagnosis['fix_type']} — {st.session_state['last_fix_description']}")

        if st.session_state["diagnosis_history"]:
            st.write("Last 5 diagnoses")
            st.dataframe(pd.DataFrame(st.session_state["diagnosis_history"]))

# ---------------------------------------------------------------------------
# Tab 5: Ablation Study Results
# ---------------------------------------------------------------------------
with tab5:
    st.subheader("Ablation Study Results")
    st.markdown(
        "**Condition A:** PPO only. **B:** Hand-coded fixes. **C:** Random fixes. "
        "**D:** ARTEMIS (SAC meta-policy)."
    )
    ablation_path = "results/ablation.json"
    stats_path = "results/stats_report.json"
    if not os.path.exists(ablation_path):
        st.info("Run experiments/ablation.py to generate this (expensive — see README for cost estimate)")
    else:
        with open(ablation_path) as f:
            ablation = json.load(f)
        conditions = list(ablation.keys())
        regimes = list(ablation[conditions[0]].keys())

        fig = go.Figure()
        for cond in conditions:
            means = [np.mean([e["sharpe_ratio"] for e in ablation[cond][r]]) for r in regimes]
            stds = [np.std([e["sharpe_ratio"] for e in ablation[cond][r]]) for r in regimes]
            fig.add_trace(go.Bar(x=regimes, y=means, name=cond, error_y=dict(type="data", array=stds)))
        fig.update_layout(barmode="group", xaxis_title="Regime", yaxis_title="Sharpe ratio (mean ± std)")
        st.plotly_chart(fig, use_container_width=True)

        if os.path.exists(stats_path):
            with open(stats_path) as f:
                stats_report = json.load(f)
            rows = []
            for cond, regime_stats in stats_report.items():
                for regime, s in regime_stats.items():
                    rows.append({"condition": cond, "regime": regime, "mean": s["mean"], "std": s["std"]})
            st.dataframe(pd.DataFrame(rows))
        else:
            st.caption("Run experiments/stats.py for Wilcoxon significance p-values.")

# ---------------------------------------------------------------------------
# Tab 6: Meta-Policy Training
# ---------------------------------------------------------------------------
with tab6:
    st.subheader("Meta-Policy Training")
    meta_log_path = "logs/meta_training.json"
    if not os.path.exists(meta_log_path):
        st.info("Run meta/meta_train.py to generate this")
    else:
        with open(meta_log_path) as f:
            meta_log = json.load(f)
        steps = [e["step"] for e in meta_log]
        meta_rewards = [e["sharpe_after"] - e["sharpe_before"] for e in meta_log]

        fig = go.Figure(go.Scatter(x=steps, y=meta_rewards, mode="lines+markers", name="Meta-reward"))
        fig.update_layout(xaxis_title="Meta-training step", yaxis_title="Meta-reward (Δ Sharpe)")
        st.plotly_chart(fig, use_container_width=True)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=steps, y=[e["sharpe_before"] for e in meta_log], mode="lines+markers", name="Sharpe before"))
        fig2.add_trace(go.Scatter(x=steps, y=[e["sharpe_after"] for e in meta_log], mode="lines+markers", name="Sharpe after"))
        fig2.update_layout(xaxis_title="Meta-training step", yaxis_title="Sharpe ratio")
        st.plotly_chart(fig2, use_container_width=True)

        final_config = meta_log[-1]["reward_config"]
        fig3 = go.Figure(go.Bar(x=list(final_config.keys()), y=list(final_config.values())))
        fig3.update_layout(xaxis_title="Reward component", yaxis_title="Learned weight")
        st.plotly_chart(fig3, use_container_width=True)
        st.caption("The SAC meta-policy learned these reward adjustments without any human specification.")

# ---------------------------------------------------------------------------
# Tab 7: Cross-Market Generalization
# ---------------------------------------------------------------------------
with tab7:
    st.subheader("Cross-Market Generalization")
    cross_path = "results/cross_market.json"
    if not os.path.exists(cross_path):
        st.info("Run experiments/cross_market.py to generate this")
    else:
        with open(cross_path) as f:
            cross = json.load(f)
        cols = st.columns(len(cross))
        for col, (market, data) in zip(cols, cross.items()):
            with col:
                st.markdown(f"**{market.upper()}**")
                labels = ["Buy & Hold", "PPO only", "ARTEMIS"]
                values = [data["buy_and_hold"]["sharpe_ratio"], data["ppo_only"]["sharpe_ratio"],
                          data["artemis"]["sharpe_ratio"]]
                fig = go.Figure(go.Bar(x=labels, y=values))
                fig.update_layout(yaxis_title="Sharpe ratio", height=350)
                st.plotly_chart(fig, use_container_width=True)

                meta_log = data.get("meta_log", [])
                failure_modes_seen = set()
                for entry in meta_log:
                    failure_modes_seen.update(entry.get("failure_distribution", {}).keys())
                st.text_area(
                    f"Failure modes identified ({market})",
                    ", ".join(sorted(failure_modes_seen)) or "none recorded",
                    key=f"failtext_{market}",
                )
