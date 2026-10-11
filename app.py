
import math

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="GFIPS | Football Intelligence",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# GFIPS initial dashboard.
# All included matches are illustrative demonstration data.

st.markdown("""
<style>
.main .block-container {padding-top: 1.6rem; max-width: 1400px;}
div[data-testid="stMetric"] {
    border: 1px solid rgba(128,128,128,.25);
    padding: 16px;
    border-radius: 12px;
}
.hero {
    padding: 26px;
    border-radius: 16px;
    background: linear-gradient(120deg, #102b25, #176448);
    color: white;
    margin-bottom: 22px;
}
.small-note {font-size: 13px; opacity: .8;}
</style>
""", unsafe_allow_html=True)

st.sidebar.title("⚽ GFIPS")
st.sidebar.caption("Global Football Intelligence & Prediction System")
page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Match Predictor", "Fixtures & Results", "About GFIPS"],
)
st.sidebar.divider()
st.sidebar.caption("Prototype v0.1")
st.sidebar.caption("Demonstration data only")

st.markdown("""
<div class="hero">
<h1>Global Football Intelligence</h1>
<p>Football statistics, match analysis and probability-based predictions.</p>
<p class="small-note">A growing platform for football competitions around the world.</p>
</div>
""", unsafe_allow_html=True)

# Demonstration fixtures; not live or verified results.
matches = pd.DataFrame([
    ["Premier League", "Arsenal", "Chelsea", 2, 1],
    ["Premier League", "Liverpool", "Everton", 3, 0],
    ["La Liga", "Real Madrid", "Sevilla", 2, 0],
    ["La Liga", "Barcelona", "Valencia", 2, 1],
    ["Serie A", "Inter", "Roma", 1, 1],
    ["Bundesliga", "Bayern Munich", "Dortmund", 3, 2],
    ["Ligue 1", "PSG", "Lyon", 2, 0],
    ["Premier League", "Chelsea", "Liverpool", 1, 2],
], columns=["League", "Home", "Away", "Home Goals", "Away Goals"])

teams = sorted(set(matches["Home"]) | set(matches["Away"]))


def poisson_probability(goals, expected):
    return (math.exp(-expected) * expected**goals) / math.factorial(goals)


def estimate_goals(team, venue):
    if venue == "home":
        scored = matches.loc[matches["Home"] == team, "Home Goals"]
        conceded = matches.loc[matches["Home"] == team, "Away Goals"]
    else:
        scored = matches.loc[matches["Away"] == team, "Away Goals"]
        conceded = matches.loc[matches["Away"] == team, "Home Goals"]

    overall_scored = matches["Home Goals"].mean()
    overall_conceded = matches["Away Goals"].mean()

    attack = float(scored.mean()) if not scored.empty else float(overall_scored)
    defence = float(conceded.mean()) if not conceded.empty else float(overall_conceded)

    return max(0.15, min(4.5, (attack + overall_conceded) / 2)), max(
        0.15, min(4.5, (defence + overall_scored) / 2)
    )


def predict(home, away):
    # Illustrative baseline, not a trained predictive model.
    home_attack, _ = estimate_goals(home, "home")
    _, away_defence = estimate_goals(away, "away")
    away_attack, _ = estimate_goals(away, "away")
    _, home_defence = estimate_goals(home, "home")

    home_xg = max(0.15, (home_attack + away_defence) / 2)
    away_xg = max(0.15, (away_attack + home_defence) / 2)

    home_probs = [poisson_probability(i, home_xg) for i in range(9)]
    away_probs = [poisson_probability(i, away_xg) for i in range(9)]

    matrix = np.outer(home_probs, away_probs)
    home_win = float(np.tril(matrix, -1).sum())
    draw = float(np.trace(matrix))
    away_win = float(np.triu(matrix, 1).sum())

    # Recalculate using the correct home-goal row / away-goal column.
    home_win = float(sum(matrix[i, j] for i in range(9) for j in range(9) if i > j))
    draw = float(sum(matrix[i, i] for i in range(9)))
    away_win = float(sum(matrix[i, j] for i in range(9) for j in range(9) if i < j))

    over_25 = float(sum(
        matrix[i, j] for i in range(9) for j in range(9) if i + j > 2
    ))
    btts = float(sum(
        matrix[i, j] for i in range(1, 9) for j in range(1, 9)
    ))

    return home_xg, away_xg, home_win, draw, away_win, over_25, btts, matrix


if page == "Overview":
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Demo leagues", matches["League"].nunique())
    c2.metric("Demo matches", len(matches))
    c3.metric("Teams represented", len(teams))
    c4.metric("Live data feeds", "Not connected")

    st.subheader("Explore the platform")
    a, b = st.columns(2)
    with a:
        st.markdown("#### 📊 Statistics")
        st.write("Browse illustrative fixtures, scores and league groupings.")
    with b:
        st.markdown("#### 🧠 Match Predictor")
        st.write("Explore baseline probabilities for selected team matchups.")

    st.subheader("Demonstration fixtures")
    st.dataframe(matches, use_container_width=True, hide_index=True)
    st.info(
        "These are example records for interface testing. They are not a "
        "verified current fixture list or a reliable historical dataset."
    )

elif page == "Match Predictor":
    st.subheader("Match Probability Explorer")
    st.write("Select two teams to estimate possible match outcomes.")

    col1, col2 = st.columns(2)
    with col1:
        home = st.selectbox("Home team", teams, index=0)
    with col2:
        away_options = [t for t in teams if t != home]
        away = st.selectbox("Away team", away_options, index=0)

    if home == away:
        st.warning("Choose two different teams.")
    else:
        hxg, axg, hw, dr, aw, over25, btts, matrix = predict(home, away)

        st.caption("Baseline estimates from demonstration data only.")
        a, b = st.columns(2)
        a.metric(f"{home} expected goals", f"{hxg:.2f}")
        b.metric(f"{away} expected goals", f"{axg:.2f}")

        st.markdown("### 1X2 market")
        x, y, z = st.columns(3)
        x.metric("1 · Home win", f"{hw:.1%}")
        y.metric("X · Draw", f"{dr:.1%}")
        z.metric("2 · Away win", f"{aw:.1%}")

        st.markdown("### Goals markets")
        x, y = st.columns(2)
        x.metric("Over 2.5 goals", f"{over25:.1%}")
        y.metric("Both teams to score", f"{btts:.1%}")

        st.markdown("### Correct-score probability")
        scores = []
        for i in range(9):
            for j in range(9):
                scores.append((f"{i}-{j}", float(matrix[i, j])))
        scores.sort(key=lambda item: item[1], reverse=True)
        score_table = pd.DataFrame(
            [{"Score": score, "Probability": f"{prob:.2%}"}
             for score, prob in scores[:10]]
        )
        st.dataframe(score_table, use_container_width=True, hide_index=True)

        st.warning(
            "This is an experimental statistical baseline, not a trained "
            "or validated model. Do not use these estimates as betting advice."
        )

elif page == "Fixtures & Results":
    st.subheader("Fixtures & Results Explorer")
    league = st.selectbox("Filter by league", ["All leagues"] + sorted(matches["League"].unique()))
    view = matches if league == "All leagues" else matches[matches["League"] == league]
    st.dataframe(view, use_container_width=True, hide_index=True)
    st.caption("Live fixtures and real historical records will be connected later.")

elif page == "About GFIPS":
    st.subheader("About the project")
    st.write(
        "GFIPS aims to become a global football intelligence platform "
        "for football statistics, match analysis and probability modelling."
    )
    st.markdown("""
    **Planned development**
    - Global league and competition coverage
    - Verified fixtures and historical results
    - Team form, home/away performance and standings
    - Multiple match markets and calibrated probabilities
    - Machine-learning models with historical backtesting
    - Accessible, mobile-friendly web interface
    """)
    st.info(
        "The current version is a user-interface prototype. Global coverage, "
        "live data and trained prediction models have not been implemented yet."
    )

st.divider()
st.caption("GFIPS · Global Football Intelligence & Prediction System · Prototype")
