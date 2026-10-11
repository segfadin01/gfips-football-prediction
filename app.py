
import math
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st

# ============================================================
# GFIPS: Global Football Intelligence & Prediction System
# Integrated prototype with API-Football support
# ============================================================

st.set_page_config(
    page_title="GFIPS | Football Intelligence",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE = "https://v3.football.api-sports.io"

LEAGUES = {
    "English Premier League": 39,
    "Spanish La Liga": 140,
    "Italian Serie A": 135,
    "German Bundesliga": 78,
    "French Ligue 1": 61,
    "UEFA Champions League": 2,
    "Major League Soccer": 253,
}

DEMO_MATCHES = pd.DataFrame(
    [
        ["Premier League", "Arsenal", "Chelsea", 2, 1],
        ["Premier League", "Liverpool", "Everton", 3, 0],
        ["La Liga", "Real Madrid", "Sevilla", 2, 0],
        ["La Liga", "Barcelona", "Valencia", 2, 1],
        ["Serie A", "Inter", "Roma", 1, 1],
        ["Bundesliga", "Bayern Munich", "Dortmund", 3, 2],
        ["Ligue 1", "PSG", "Lyon", 2, 0],
        ["Premier League", "Chelsea", "Liverpool", 1, 2],
    ],
    columns=["League", "Home", "Away", "Home Goals", "Away Goals"],
)


# -------------------- API FUNCTIONS -------------------------

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_api_data(endpoint, params, api_key):
    """Fetch API-Football data and cache successful responses."""
    response = requests.get(
        f"{API_BASE}/{endpoint}",
        headers={"x-apisports-key": api_key},
        params=params,
        timeout=25,
    )
    response.raise_for_status()

    payload = response.json()

    if payload.get("errors"):
        raise ValueError(str(payload["errors"]))

    return payload


def get_api_key():
    """Read the API key from Streamlit Secrets."""
    try:
        key = st.secrets["API_FOOTBALL_KEY"]
        if not key or not str(key).strip():
            return None
        return str(key).strip()
    except Exception:
        return None


def show_api_error(error):
    """Display useful API errors without revealing the secret."""
    if isinstance(error, requests.Timeout):
        st.error("The football API timed out. Please try again later.")
    elif isinstance(error, requests.HTTPError):
        code = error.response.status_code if error.response is not None else None
        if code == 401:
            st.error("API authentication failed. Check your API key in Streamlit Secrets.")
        elif code == 403:
            st.error(
                "Access was denied. Check your API subscription and competition coverage."
            )
        elif code == 429:
            st.error(
                "The API request limit may have been reached. Wait before trying again."
            )
        else:
            st.error(f"The football API returned HTTP status {code}.")
    elif isinstance(error, requests.RequestException):
        st.error("Could not connect to the football API. Check the connection and retry.")
    elif isinstance(error, ValueError):
        st.error(f"The football API reported a problem: {error}")
    else:
        st.error("An unexpected error occurred while loading football data.")


# -------------------- PREDICTION BASELINE -------------------

def poisson_probability(goals, expected):
    return math.exp(-expected) * expected**goals / math.factorial(goals)


def demo_prediction(home, away):
    """Experimental estimates from demonstration data only."""
    home_rows = DEMO_MATCHES[DEMO_MATCHES["Home"] == home]
    away_rows = DEMO_MATCHES[DEMO_MATCHES["Away"] == away]

    league_home_avg = max(0.5, float(DEMO_MATCHES["Home Goals"].mean()))
    league_away_avg = max(0.5, float(DEMO_MATCHES["Away Goals"].mean()))

    home_scored = (
        float(home_rows["Home Goals"].mean())
        if not home_rows.empty else league_home_avg
    )
    home_conceded = (
        float(home_rows["Away Goals"].mean())
        if not home_rows.empty else league_away_avg
    )
    away_scored = (
        float(away_rows["Away Goals"].mean())
        if not away_rows.empty else league_away_avg
    )
    away_conceded = (
        float(away_rows["Home Goals"].mean())
        if not away_rows.empty else league_home_avg
    )

    home_xg = max(0.15, min(4.5, (home_scored + away_conceded) / 2))
    away_xg = max(0.15, min(4.5, (away_scored + home_conceded) / 2))

    home_probs = [poisson_probability(i, home_xg) for i in range(10)]
    away_probs = [poisson_probability(i, away_xg) for i in range(10)]
    matrix = np.outer(home_probs, away_probs)

    home_win = sum(matrix[i, j] for i in range(10) for j in range(10) if i > j)
    draw = sum(matrix[i, i] for i in range(10))
    away_win = sum(matrix[i, j] for i in range(10) for j in range(10) if i < j)
    over_25 = sum(matrix[i, j] for i in range(10) for j in range(10) if i + j > 2)
    btts = sum(matrix[i, j] for i in range(1, 10) for j in range(1, 10))

    return home_xg, away_xg, home_win, draw, away_win, over_25, btts, matrix


# -------------------- PAGE: LIVE FOOTBALL DATA --------------

def show_live_fixtures():
    st.subheader("Real Football Fixtures & Results")
    st.write(
        "Retrieve fixtures and scores from API-Football. "
        "The competitions and seasons available depend on your API plan."
    )

    api_key = get_api_key()
    if not api_key:
        st.error(
            "API_FOOTBALL_KEY was not found. Open your Streamlit app settings, "
            "go to Secrets, and check the key name."
        )
        return

    c1, c2 = st.columns(2)

    with c1:
        league_name = st.selectbox("Competition", list(LEAGUES.keys()))

    with c2:
        current_year = date.today().year
        season = st.selectbox(
            "Season start year",
            list(range(current_year, current_year - 4, -1)),
        )

    today = date.today()
    d1, d2 = st.columns(2)

    with d1:
        date_from = st.date_input("Start date", value=today)

    with d2:
        date_to = st.date_input(
            "End date",
            value=today + timedelta(days=7),
        )

    if date_from > date_to:
        st.warning("The start date cannot be later than the end date.")
        return

    if (date_to - date_from).days > 60:
        st.warning(
            "Please choose a date range of 60 days or less to conserve API requests."
        )
        return

    if st.button("Fetch fixtures and results", type="primary"):
        params = {
            "league": LEAGUES[league_name],
            "season": season,
            "from": date_from.isoformat(),
            "to": date_to.isoformat(),
        }

        try:
            with st.spinner("Contacting API-Football..."):
                payload = fetch_api_data("fixtures", params, api_key)

            fixtures = payload.get("response", [])
            paging = payload.get("paging", {})
            requests_info = payload.get("requests", {})

            st.caption(f"Competition: {league_name} · Season: {season}")

            if isinstance(requests_info, dict):
                current = requests_info.get("current")
                limit = requests_info.get("limit_day")
                if current is not None and limit is not None:
                    st.caption(f"API requests reported: {current} of {limit} today")

            if not fixtures:
                st.info(
                    "No fixtures were returned for this selection. Try another date "
                    "range or season, and verify the competition is available on your plan."
                )
                return

            rows = []
            for item in fixtures:
                fixture = item.get("fixture", {})
                league = item.get("league", {})
                teams = item.get("teams", {})
                goals = item.get("goals", {})
                status = fixture.get("status", {})

                rows.append(
                    {
                        "Kick-off (API time)": fixture.get("date", ""),
                        "Competition": league.get("name", ""),
                        "Home team": teams.get("home", {}).get("name", ""),
                        "Away team": teams.get("away", {}).get("name", ""),
                        "Home goals": goals.get("home"),
                        "Away goals": goals.get("away"),
                        "Status": status.get("long", ""),
                    }
                )

            results = pd.DataFrame(rows)
            st.success(f"Loaded {len(results)} fixtures from API-Football.")
            st.dataframe(results, use_container_width=True, hide_index=True)

            if isinstance(paging, dict) and paging.get("total", 1) > 1:
                st.caption(
                    "The provider reports multiple pages of results. "
                    "This prototype currently displays the first returned page."
                )

            csv = results.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download results as CSV",
                data=csv,
                file_name="gfips_fixtures.csv",
                mime="text/csv",
            )

        except Exception as error:
            show_api_error(error)


# -------------------- PAGE: OVERVIEW ------------------------

def show_overview():
    st.markdown(
        """
        <div style="
            padding:28px;
            border-radius:16px;
            background:linear-gradient(120deg,#102b25,#176448);
            color:white;
            margin-bottom:20px;">
            <h1>Global Football Intelligence</h1>
            <p>Football statistics, match analysis and probability modelling.</p>
            <p>Building a global football data and prediction platform.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Demo competitions", DEMO_MATCHES["League"].nunique())
    c2.metric("Demo matches", len(DEMO_MATCHES))
    c3.metric(
        "Demo teams",
        len(set(DEMO_MATCHES["Home"]) | set(DEMO_MATCHES["Away"])),
    )
    c4.metric("Live data", "API connected in Live Football Data")

    st.subheader("Demonstration matches")
    st.dataframe(DEMO_MATCHES, use_container_width=True, hide_index=True)
    st.info(
        "These sample matches are illustrative only. They are not current fixtures "
        "or verified historical data. Use Live Football Data for API results."
    )


# -------------------- PAGE: MATCH PREDICTOR -----------------

def show_predictor():
    st.subheader("Experimental Match Probability Explorer")
    st.warning(
        "These estimates use a small illustrative dataset, not verified current "
        "team statistics. They are not validated forecasts or betting advice."
    )

    teams = sorted(set(DEMO_MATCHES["Home"]) | set(DEMO_MATCHES["Away"]))
    c1, c2 = st.columns(2)

    with c1:
        home = st.selectbox("Home team", teams)

    away_options = [team for team in teams if team != home]

    with c2:
        away = st.selectbox("Away team", away_options)

    hxg, axg, hw, draw, aw, over25, btts, matrix = demo_prediction(home, away)

    c1, c2 = st.columns(2)
    c1.metric(f"{home} expected goals", f"{hxg:.2f}")
    c2.metric(f"{away} expected goals", f"{axg:.2f}")

    st.markdown("#### Match outcome probabilities")
    c1, c2, c3 = st.columns(3)
    c1.metric("1 · Home win", f"{hw:.1%}")
    c2.metric("X · Draw", f"{draw:.1%}")
    c3.metric("2 · Away win", f"{aw:.1%}")

    st.markdown("#### Goals markets")
    c1, c2 = st.columns(2)
    c1.metric("Over 2.5 goals", f"{over25:.1%}")
    c2.metric("Both teams to score", f"{btts:.1%}")

    scores = []
    for i in range(10):
        for j in range(10):
            scores.append((f"{i}-{j}", float(matrix[i, j])))

    scores.sort(key=lambda x: x[1], reverse=True)
    st.markdown("#### Highest estimated correct-score probabilities")
    st.dataframe(
        pd.DataFrame(
            [
                {"Score": score, "Probability": f"{probability:.2%}"}
                for score, probability in scores[:10]
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )


# -------------------- PAGE: FIXTURES & RESULTS --------------

def show_demo_results():
    st.subheader("Demonstration Fixtures & Results")
    league = st.selectbox(
        "Filter demonstration data",
        ["All competitions"] + sorted(DEMO_MATCHES["League"].unique()),
    )

    if league == "All competitions":
        view = DEMO_MATCHES
    else:
        view = DEMO_MATCHES[DEMO_MATCHES["League"] == league]

    st.dataframe(view, use_container_width=True, hide_index=True)
    st.caption("For real fixtures, select Live Football Data in the navigation.")


# -------------------- PAGE: ABOUT ---------------------------

def show_about():
    st.subheader("About GFIPS")
    st.write(
        "GFIPS is being developed as a global football intelligence platform "
        "for statistics, fixtures, results and experimental match probability modelling."
    )

    st.markdown(
        """
        **Development roadmap**
        - Verified fixtures and historical results
        - Global competition coverage, subject to data availability
        - Team form and home/away statistics
        - Statistical and machine-learning prediction models
        - Multiple markets and probability calibration
        - Historical backtesting and transparent performance reporting
        """
    )

    st.info(
        "GFIPS is currently a prototype. Real API fixtures are available through "
        "the Live Football Data page when supported by your API plan. The prediction "
        "page still uses demonstration data."
    )


# -------------------- APP NAVIGATION ------------------------

st.sidebar.title("⚽ GFIPS")
st.sidebar.caption("Global Football Intelligence & Prediction System")

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Live Football Data",
        "Fixtures & Results",
        "Match Predictor",
        "About GFIPS",
    ],
)

st.sidebar.divider()
st.sidebar.caption("Prototype v0.2")
st.sidebar.caption("Live data requires API-Football access.")

if page == "Overview":
    show_overview()
elif page == "Live Football Data":
    show_live_fixtures()
elif page == "Fixtures & Results":
    show_demo_results()
elif page == "Match Predictor":
    show_predictor()
elif page == "About GFIPS":
    show_about()

st.divider()
st.caption("GFIPS · Global Football Intelligence & Prediction System")
