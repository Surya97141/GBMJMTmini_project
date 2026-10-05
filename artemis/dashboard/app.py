"""ARTEMIS Research dashboard: performance, robustness, attention XAI, ablation, meta-policy, cross-market."""
import datetime
import json
import os
import sys

# Makes `agent`, `data`, `diagnostics`, etc. importable, AND makes every
# relative path in this file ("models/...", "results/...", "logs/...")
# resolve correctly, regardless of the working directory the app was
# launched from -- locally we always `cd` into artemis/ first, but
# Streamlit Cloud runs from the repo root instead.
_ARTEMIS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ARTEMIS_DIR)
os.chdir(_ARTEMIS_DIR)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from agent.collect import collect_episodes
from agent.evaluate import benchmark_buy_and_hold
from data.fetcher import DataPipeline, raw_close
from diagnostics.diagnose import diagnose_episodes, load_lstm
from diagnostics.lstm_model import FAILURE_MODES
from env.trading_env import REWARD_CONFIG_BOUNDS, TradingEnv
from reward.fix_applicators import DEFAULT_REWARD_CONFIG, apply_fix, describe_fix

SEED = 42

st.set_page_config(page_title="ARTEMIS Research", layout="wide")
st.title("ARTEMIS Research : Autonomous Reinforcement Trading Dashboard")

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


tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "Portfolio Performance",
    "Regime Robustness Grid",
    "Episode Replay + Attention",
    "Live Diagnosis",
    "Ablation Study Results",
    "Meta-Policy Training",
    "Cross-Market Generalization",
    "🎛️ Reward Sandbox",
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

# ---------------------------------------------------------------------------
# Tab 8: Reward Sandbox -- live, interactive reward-shaping demo
# ---------------------------------------------------------------------------
REWARD_LABELS = {
    "return_weight": ("Return weight", "How much the AI is rewarded for making money. Higher = cares more about profit."),
    "transaction_cost": ("Transaction cost", "Penalty per trade. More negative = trading costs more (this actually reduces real rupee returns, not just the training signal)."),
    "drawdown_penalty": ("Drawdown penalty", "How harshly a dip below the portfolio's peak value is punished. More negative = more risk-averse, panickier agent."),
    "holding_bonus": ("Holding bonus", "Reward for continuing to hold a position that's currently winning."),
    "profit_take_bonus": ("Profit-take bonus", "Reward for selling while sitting on a profit, i.e. locking in gains."),
    "sharpe_bonus": ("Sharpe bonus", "Reward for smooth, consistent day-to-day returns rather than wild swings."),
}


def _run_sandbox_episode(model, df, reward_config):
    """Runs one real (deterministic) episode with the frozen model, using a custom
    reward_config. The agent's trading DECISIONS don't change (same model, same
    observations -- reward_config never feeds into what the policy sees), but the
    real cash transaction_cost and the recorded reward signal do change, which is
    exactly what lets this be a meaningful live demo without any retraining."""
    env = TradingEnv(df, reward_config=reward_config)
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(int(action))
        done = terminated or truncated
    traj = env.get_trajectory()
    traj["trade_log"] = env.trade_log
    traj["portfolio_values"] = [100.0]
    for s in traj["steps"]:
        traj["portfolio_values"].append(traj["portfolio_values"][-1] * (1 + s["daily_return"]))
    return traj


with tab8:
    st.subheader("🎛️ Reward Sandbox — tweak the reward weights and watch it play out live")
    st.caption(
        "The trained agent's trading decisions are frozen (same brain, same market signals in). "
        "What changes when you move these sliders: the real transaction cost deducted on every "
        "trade, and the reward signal fed to the diagnostic LSTM -- so you can watch the economic "
        "outcome AND the diagnosis change in response to the exact same trading session."
    )

    sb_regime = st.selectbox("Select regime", REGIMES, key="sandbox_regime")
    sb_model = load_ppo_model(sb_regime)
    sb_lstm = load_diagnostic_lstm()

    if sb_model is None:
        st.info(f"No trained model found at models/{sb_regime}_ppo_seed42.zip — train one first.")
    else:
        if "sandbox_config" not in st.session_state:
            st.session_state["sandbox_config"] = DEFAULT_REWARD_CONFIG.copy()

        col_sliders, col_results = st.columns([1, 2])

        with col_sliders:
            st.markdown("**Reward weights**")
            if st.button("Reset to default"):
                st.session_state["sandbox_config"] = DEFAULT_REWARD_CONFIG.copy()

            custom_config = {}
            for key, (label, help_text) in REWARD_LABELS.items():
                lo, hi = REWARD_CONFIG_BOUNDS[key]
                step = (hi - lo) / 100.0
                custom_config[key] = st.slider(
                    label, min_value=float(lo), max_value=float(hi),
                    value=float(st.session_state["sandbox_config"][key]),
                    step=float(step), help=help_text, key=f"sb_{key}",
                )
            st.session_state["sandbox_config"] = custom_config

            run_clicked = st.button("▶ Run Episode With These Settings", type="primary")

        with col_results:
            if run_clicked:
                df = load_regime_df(sb_regime)
                default_traj = _run_sandbox_episode(sb_model, df, DEFAULT_REWARD_CONFIG)
                custom_traj = _run_sandbox_episode(sb_model, df, custom_config)
                st.session_state["sandbox_results"] = (default_traj, custom_traj)

            if "sandbox_results" in st.session_state:
                default_traj, custom_traj = st.session_state["sandbox_results"]

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Return (default)", f"{default_traj['total_return']*100:.2f}%")
                c2.metric("Return (yours)", f"{custom_traj['total_return']*100:.2f}%",
                           delta=f"{(custom_traj['total_return']-default_traj['total_return'])*100:.2f}%")
                c3.metric("Max drawdown (yours)", f"{custom_traj['max_drawdown']:.2%}")
                c4.metric("Trades (yours)", custom_traj["num_trades"])

                fig = go.Figure()
                fig.add_trace(go.Scatter(y=default_traj["portfolio_values"], mode="lines",
                                          name="Default config", line=dict(dash="dash", color="gray")))
                fig.add_trace(go.Scatter(y=custom_traj["portfolio_values"], mode="lines",
                                          name="Your config", line=dict(color="royalblue")))
                fig.update_layout(title="Portfolio value: default vs. your settings (base=100)",
                                   xaxis_title="Trading day", yaxis_title="Value")
                st.plotly_chart(fig, use_container_width=True)

                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(y=[s["reward"] for s in default_traj["steps"]], mode="lines",
                                           name="Default reward signal", line=dict(dash="dash", color="gray")))
                fig2.add_trace(go.Scatter(y=[s["reward"] for s in custom_traj["steps"]], mode="lines",
                                           name="Your reward signal", line=dict(color="orange")))
                fig2.update_layout(title="Per-step reward signal (what the LSTM actually reads)",
                                    xaxis_title="Trading day", yaxis_title="Reward")
                st.plotly_chart(fig2, use_container_width=True)

                if sb_lstm is not None:
                    diag_default = diagnose_episodes(sb_lstm, [default_traj])[0]
                    diag_custom = diagnose_episodes(sb_lstm, [custom_traj])[0]
                    dc1, dc2 = st.columns(2)
                    for col, label, diag in [(dc1, "Default config diagnosis", diag_default),
                                              (dc2, "Your config diagnosis", diag_custom)]:
                        with col:
                            color = ("green" if diag["failure_mode"] == "PROFITABLE"
                                     else "red" if diag["failure_mode"] == "MAX_DRAWDOWN" else "orange")
                            st.markdown(
                                f"<div style='background-color:{color};color:white;padding:10px;"
                                f"border-radius:8px;text-align:center;'>{label}<br>"
                                f"<span style='font-size:20px'>{diag['failure_mode']}</span><br>"
                                f"confidence {diag['failure_confidence']:.0%}</div>",
                                unsafe_allow_html=True,
                            )
                    if diag_default["failure_mode"] != diag_custom["failure_mode"]:
                        st.success(
                            f"Diagnosis changed from **{diag_default['failure_mode']}** to "
                            f"**{diag_custom['failure_mode']}** purely from your slider settings — "
                            f"same agent, same market data, different reward shaping."
                        )
                else:
                    st.caption("Train the diagnostic LSTM to see live diagnosis here too.")
            else:
                st.info("Adjust the sliders and click Run Episode to see the effect.")
