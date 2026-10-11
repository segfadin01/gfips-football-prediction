
import math
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st


# ============================================================
# GFIPS — GLOBAL FOOTBALL INTELLIGENCE & PREDICTION SYSTEM
# ============================================================

st.set_page_config(
    page_title="GFIPS | Football Intelligence",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE_URL = "https://v3.football.api-sports.io"

LEAGUES = {
    "English Premier League": 39,
    "Spanish La Liga": 140,
    "Italian Serie A": 135,
    "German Bundesliga": 78,
    "French Ligue 1": 61,
    "UEFA Champions League": 2,
    "Major League Soccer": 253,
}

# The API-Football Free plan error indicated that seasons
# from 2022 through 2024 are available.
AVAILABLE_SEASONS = [2024, 2023, 2022]


# ============================================================
# DEMONSTRATION DATA
# These are illustrative examples, not live match records.
# ============================================================

DEMO_MATCHES = pd.DataFrame(
    [
        {
            "Date": "2025-05-25",
            "Competition": "English Premier League",
            "Home Team": "Liverpool",
            "Away Team": "Crystal Palace",
            "Home Goals": 1,
            "Away Goals": 1,
        },
        {
            "Date": "2025-05-25",
            "Competition": "English Premier League",
            "Home Team": "Manchester City",
            "Away Team": "Fulham",
            "Home Goals": 2,
            "Away Goals": 0,
        },
        {
            "Date": "2025-05-24",
            "Competition": "Spanish La Liga",
            "Home Team": "Real Madrid",
            "Away Team": "Real Sociedad",
            "Home Goals": 2,
            "Away Goals": 0,
        },
        {
            "Date": "2025-05-24",
            "Competition": "Italian Serie A",
            "Home Team": "Inter",
            "Away Team": "Lazio",
            "Home Goals": 2,
            "Away Goals": 2,
        },
        {
            "Date": "2025-05-24",
            "Competition": "German Bundesliga",
            "Home Team": "Bayern Munich",
            "Away Team": "Hoffenheim",
            "Home Goals": 4,
            "Away Goals": 0,
        },
        {
            "Date": "2025-05-18",
            "Competition": "French Ligue 1",
            "Home Team": "Marseille",
            "Away Team": "Rennes",
            "Home Goals": 4,
            "Away Goals": 2,
        },
        {
            "Date": "2025-05-17",
            "Competition": "English Premier League",
            "Home Team": "Arsenal",
            "Away Team": "Newcastle United",
            "Home Goals": 1,
            "Away Goals": 0,
        },
        {
            "Date": "2025-05-17",
            "Competition": "Spanish La Liga",
            "Home Team": "Barcelona",
            "Away Team": "Villarreal",
            "Home Goals": 2,
            "Away Goals": 3,
        },
    ]
)

DEMO_MATCHES["Date"] = pd.to_datetime(DEMO_MATCHES["Date"])


# ============================================================
# API CONFIGURATION AND DATA RETRIEVAL
# ============================================================

def get_api_key():
    """Read the API key securely from Streamlit Secrets."""
    try:
        return st.secrets["API_FOOTBALL_KEY"]
    except (KeyError, FileNotFoundError):
        return None


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_api_data(endpoint, params, api_key):
    """
    Request data from API-Football.
    Cache successful responses for 30 minutes.
    """
    if not api_key:
        raise ValueError(
            "API key not found. Add API_FOOTBALL_KEY "
            "under your Streamlit app's Secrets settings."
        )

    headers = {"x-apisports-key": api_key}

    response = requests.get(
        f"{API_BASE_URL}/{endpoint.lstrip('/')}",
        headers=headers,
        params=params,
        timeout=30,
    )

    if response.status_code == 401:
        raise ValueError(
            "API authentication failed. Check your API key "
            "in Streamlit Secrets."
        )

    if response.status_code == 403:
        raise ValueError(
            "Your API plan does not permit this request. "
            "Check the selected season and competition."
        )

    if response.status_code == 429:
        raise ValueError(
            "The API request limit has been reached. "
            "Wait before trying again."
        )

    response.raise_for_status()
    payload = response.json()

    api_errors = payload.get("errors")

    if api_errors:
        raise ValueError(f"The football API reported a problem: {api_errors}")

    return payload


def show_api_error(error):
    """Display understandable messages for API errors."""
    message = str(error)

    if "timed out" in message.lower():
        st.error(
            "The football data service took too long to respond. "
            "Please try again."
        )
    elif "connection" in message.lower():
        st.error(
            "Could not connect to the football data service. "
            "Please check your connection and try again."
        )
    else:
        st.error(message)


def get_fixture_dataframe(fixtures):
    """Convert API-Football fixture responses into a readable table."""
    rows = []

    for item in fixtures:
        fixture = item.get("fixture", {})
        league = item.get("league", {})
        teams = item.get("teams", {})
        goals = item.get("goals", {})
        status = fixture.get("status", {})

        rows.append(
            {
                "Date": fixture.get("date", ""),
                "Competition": league.get("name", ""),
                "Country": league.get("country", ""),
                "Home Team": teams.get("home", {}).get("name", ""),
                "Away Team": teams.get("away", {}).get("name", ""),
                "Home Goals": goals.get("home"),
                "Away Goals": goals.get("away"),
                "Status": status.get("long", ""),
                "Venue": fixture.get("venue", {}).get("name", ""),
            }
        )

    return pd.DataFrame(rows)


def get_team_names(fixtures):
    """Extract unique team names from a fixture response."""
    names = set()

    for item in fixtures:
        teams = item.get("teams", {})
        home = teams.get("home", {}).get("name")
        away = teams.get("away", {}).get("name")

        if home:
            names.add(home)
        if away:
            names.add(away)

    return sorted(names)


# ============================================================
# SIDEBAR AND NAVIGATION
# ============================================================

st.sidebar.title("⚽ GFIPS")
st.sidebar.caption("Global Football Intelligence & Prediction System")
st.sidebar.divider()

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
st.sidebar.caption("Football data powered by API-Football.")
st.sidebar.caption(
    "Predictions are estimates, not guarantees of match outcomes."
)


# ============================================================
# OVERVIEW
# ============================================================

def show_overview():
    st.title("⚽ GFIPS Dashboard")
    st.write(
        "Your football intelligence workspace for exploring fixtures, "
        "match results, team information and experimental predictions."
    )

    st.warning(
        "The summary figures and match records on this page are "
        "demonstration data. They are not live football statistics."
    )

    total_matches = len(DEMO_MATCHES)
    total_competitions = DEMO_MATCHES["Competition"].nunique()
    total_teams = pd.unique(
        DEMO_MATCHES[["Home Team", "Away Team"]].values.ravel()
    ).size

    col1, col2, col3 = st.columns(3)

    col1.metric("Sample Matches", total_matches)
    col2.metric("Sample Competitions", total_competitions)
    col3.metric("Sample Teams", total_teams)

    st.subheader("Sample Match Records")
    st.dataframe(
        DEMO_MATCHES.sort_values("Date", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Explore GFIPS")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            **Live Football Data**
            - Retrieve fixtures from API-Football
            - Select a supported competition and season
            - Export retrieved data to CSV
            """
        )

    with col2:
        st.markdown(
            """
            **Match Predictor**
            - Explore an illustrative Poisson model
            - Estimate home, draw and away probabilities
            - View the most likely scorelines
            """
        )


# ============================================================
# LIVE FOOTBALL DATA
# ============================================================

def show_live_fixtures():
    st.title("🌍 Live Football Data")
    st.write(
        "Retrieve fixture records from API-Football for a supported "
        "competition, season and date range."
    )

    st.info(
        "The API-Football Free plan response indicated that seasons "
        "2022, 2023 and 2024 should be used. This page therefore "
        "restricts the season dropdown to those years."
    )

    api_key = get_api_key()

    if not api_key:
        st.error(
            "Your API key has not been found. In Streamlit Community "
            "Cloud, open your app's Settings → Secrets and add:"
        )
        st.code('API_FOOTBALL_KEY = "YOUR_API_KEY_HERE"')
        st.warning(
            "Replace the example value with your actual API key in "
            "Streamlit Secrets only. Do not put the real key in this code."
        )
        return

    with st.form("fixture_search_form"):
        col1, col2 = st.columns(2)

        with col1:
            competition = st.selectbox(
                "Competition",
                list(LEAGUES.keys()),
                index=0,
            )

            season = st.selectbox(
                "Season start year",
                [2024, 2023, 2022],
                index=0,
            )

        with col2:
            today = date.today()
            default_start = date(2024, 8, 1)
            default_end = date(2024, 8, 31)

            start_date = st.date_input(
                "Start date",
                value=default_start,
                min_value=date(2000, 1, 1),
                max_value=date(2035, 12, 31),
            )

            end_date = st.date_input(
                "End date",
                value=default_end,
                min_value=date(2000, 1, 1),
                max_value=date(2035, 12, 31),
            )

        submitted = st.form_submit_button(
            "Fetch Fixtures and Results",
            use_container_width=True,
            type="primary",
        )

    if not submitted:
        st.caption(
            "Select a competition and season, set your date range, "
            "then click Fetch Fixtures and Results."
        )
        return

    if start_date > end_date:
        st.error("The start date must be on or before the end date.")
        return

    if start_date.year < season or end_date.year > season + 1:
        st.warning(
            f"You selected season {season}. For a typical European "
            f"football season, choose dates within the {season}/{season + 1} "
            "season. Cup competitions and some leagues may use different "
            "schedules."
        )

    params = {
        "league": LEAGUES[competition],
        "season": season,
        "from": start_date.isoformat(),
        "to": end_date.isoformat(),
    }

    try:
        with st.spinner("Retrieving football data..."):
            payload = fetch_api_data("fixtures", params, api_key)

        fixtures = payload.get("response", [])
        results = get_fixture_dataframe(fixtures)

        st.success(f"Retrieved {len(results)} fixture record(s).")

        if results.empty:
            st.info(
                "No fixtures were returned for this competition and "
                "date range. Try another date range or season."
            )
            return

        col1, col2, col3 = st.columns(3)
        col1.metric("Fixtures Retrieved", len(results))
        col2.metric("Teams Found", len(get_team_names(fixtures)))
        col3.metric(
            "Completed Matches",
            int(results["Home Goals"].notna().sum()),
        )

        st.subheader(f"{competition}: {season} season")
        st.dataframe(
            results,
            use_container_width=True,
            hide_index=True,
        )

        csv_data = results.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Results as CSV",
            data=csv_data,
            file_name=(
                f"gfips_{season}_{LEAGUES[competition]}_fixtures.csv"
            ),
            mime="text/csv",
            use_container_width=True,
        )

    except requests.exceptions.Timeout:
        show_api_error("The API request timed out. Please try again.")
    except requests.exceptions.RequestException as error:
        show_api_error(f"Could not retrieve football data: {error}")
    except ValueError as error:
        show_api_error(error)
    except Exception as error:
        show_api_error(f"An unexpected error occurred: {error}")


# ============================================================
# FIXTURES AND RESULTS — DEMONSTRATION PAGE
# ============================================================

def show_demo_fixtures():
    st.title("📅 Fixtures & Results")
    st.write(
        "Browse the sample match records currently included in GFIPS."
    )

    st.warning(
        "This page displays demonstration records only. For API-Football "
        "records, open Live Football Data in the navigation menu."
    )

    competitions = ["All"] + sorted(
        DEMO_MATCHES["Competition"].unique().tolist()
    )

    selected_competition = st.selectbox(
        "Filter by competition",
        competitions,
    )

    filtered = DEMO_MATCHES.copy()

    if selected_competition != "All":
        filtered = filtered[
            filtered["Competition"] == selected_competition
        ]

    st.dataframe(
        filtered.sort_values("Date", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download Demonstration Data",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="gfips_demonstration_matches.csv",
        mime="text/csv",
    )


# ============================================================
# MATCH PREDICTOR — ILLUSTRATIVE POISSON MODEL
# ============================================================

def poisson_probability(goals, expected_goals):
    """Calculate the Poisson probability of a goal count."""
    if expected_goals < 0:
        return 0.0

    return (
        math.exp(-expected_goals)
        * (expected_goals ** goals)
        / math.factorial(goals)
    )


def calculate_match_probabilities(home_xg, away_xg, max_goals=8):
    """
    Estimate home-win, draw and away-win probabilities using
    independent Poisson goal distributions.
    """
    home_probs = [
        poisson_probability(g, home_xg)
        for g in range(max_goals + 1)
    ]
    away_probs = [
        poisson_probability(g, away_xg)
        for g in range(max_goals + 1)
    ]

    home_win = 0.0
    draw = 0.0
    away_win = 0.0
    scorelines = []

    for home_goals in range(max_goals + 1):
        for away_goals in range(max_goals + 1):
            probability = (
                home_probs[home_goals] * away_probs[away_goals]
            )

            scorelines.append(
                {
                    "Score": f"{home_goals}-{away_goals}",
                    "Probability": probability,
                }
            )

            if home_goals > away_goals:
                home_win += probability
            elif home_goals == away_goals:
                draw += probability
            else:
                away_win += probability

    total = home_win + draw + away_win

    if total > 0:
        home_win /= total
        draw /= total
        away_win /= total

    scorelines = sorted(
        scorelines,
        key=lambda item: item["Probability"],
        reverse=True,
    )

    return home_win, draw, away_win, scorelines


def show_predictor():
    st.title("🔮 Match Predictor")
    st.write(
        "Explore an illustrative Poisson-based model for estimating "
        "football scorelines and match outcomes."
    )

    st.warning(
        "This is an experimental demonstration, not a validated "
        "prediction engine. The expected-goal values are entered manually; "
        "they are not automatically calculated from real team statistics."
    )

    col1, col2 = st.columns(2)

    with col1:
        home_team = st.text_input(
            "Home team",
            value="Liverpool",
        )
        home_xg = st.slider(
            "Home team's expected goals",
            min_value=0.1,
            max_value=4.0,
            value=1.6,
            step=0.1,
        )

    with col2:
        away_team = st.text_input(
            "Away team",
            value="Arsenal",
        )
        away_xg = st.slider(
            "Away team's expected goals",
            min_value=0.1,
            max_value=4.0,
            value=1.2,
            step=0.1,
        )

    if st.button("Generate Estimate", type="primary"):
        if not home_team.strip() or not away_team.strip():
            st.error("Enter both team names.")
            return

        if home_team.strip().casefold() == away_team.strip().casefold():
            st.error("Choose two different teams.")
            return

        home_prob, draw_prob, away_prob, scorelines = (
            calculate_match_probabilities(home_xg, away_xg)
        )

        st.subheader(f"{home_team} vs {away_team}")

        c1, c2, c3 = st.columns(3)
        c1.metric("Home Win", f"{home_prob * 100:.1f}%")
        c2.metric("Draw", f"{draw_prob * 100:.1f}%")
        c3.metric("Away Win", f"{away_prob * 100:.1f}%")

        st.subheader("Most Likely Scorelines")

        score_df = pd.DataFrame(scorelines[:5])
        score_df["Probability (%)"] = (
            score_df["Probability"] * 100
        ).round(2)
        score_df = score_df[["Score", "Probability (%)"]]

        st.dataframe(
            score_df,
            use_container_width=True,
            hide_index=True,
        )

        most_likely = scorelines[0]

        st.info(
            f"Most likely scoreline in this illustrative model: "
            f"{home_team} {most_likely['Score']} {away_team} "
            f"({most_likely['Probability'] * 100:.1f}% estimated probability)."
        )

        st.caption(
            "The probabilities are conditional on the manually supplied "
            "expected-goal values and the model's assumptions. They do not "
            "represent measured team form or guaranteed outcomes."
        )


# ============================================================
# ABOUT GFIPS
# ============================================================

def show_about():
    st.title("About GFIPS")

    st.markdown(
        """
        **Global Football Intelligence & Prediction System (GFIPS)** is
        being developed as a football data and analysis platform.

        ### Current features
        - Football fixture retrieval through API-Football
        - Competition and season selection
        - Date-range filtering
        - CSV export for retrieved fixtures
        - Demonstration match records
        - An experimental Poisson scoreline estimator

        ### Current limitations
        - The dashboard's demonstration records are not live records.
        - The Match Predictor uses manually entered expected-goal values.
        - The current predictor is not trained or validated against
          historical football results.
        - Live fixture retrieval depends on API-Football plan permissions
          and request limits.

        ### Planned development
        1. Retrieve team and match statistics.
        2. Build historical datasets for supported competitions.
        3. Develop data-driven prediction models.
        4. Evaluate predictions using historical backtesting.
        5. Add transparent performance metrics and model comparisons.
        """
    )

    st.caption(
        "GFIPS is an analytical project. Its estimates should not be "
        "treated as certainty or as a guarantee of financial returns."
    )


# ============================================================
# PAGE ROUTING
# ============================================================

if page == "Overview":
    show_overview()

elif page == "Live Football Data":
    show_live_fixtures()

elif page == "Fixtures & Results":
    show_demo_fixtures()

elif page == "Match Predictor":
    show_predictor()

elif page == "About GFIPS":
    show_about()
