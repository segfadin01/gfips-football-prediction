
import math
from datetime import date, timedelta

import pandas as pd
import requests
import streamlit as st

# =========================================================
# GFIPS — GLOBAL FOOTBALL INTELLIGENCE & PREDICTION SYSTEM
# =========================================================

st.set_page_config(
    page_title="GFIPS | Football Intelligence",
    page_icon="⚽",
    layout="wide",
)

# 5DollarFootballAPI's API-Football-compatible endpoint
API_BASE_URL = "https://api-football.5dollarfootballapi.com"


# =========================================================
# API CONNECTION
# =========================================================

def get_api_key():
    try:
        return st.secrets["FOOTBALL_API_KEY"]
    except (KeyError, FileNotFoundError):
        return None


@st.cache_data(ttl=300, show_spinner=False)
def api_request(endpoint, params, api_key):
    if not api_key:
        raise ValueError(
            "FOOTBALL_API_KEY was not found in Streamlit Secrets."
        )

    response = requests.get(
        f"{API_BASE_URL}/{endpoint.lstrip('/')}",
        headers={"x-apisports-key": api_key},
        params=params,
        timeout=30,
    )

    if response.status_code == 401:
        raise ValueError(
            "Authentication failed. Check that your new API key "
            "was copied correctly into Streamlit Secrets."
        )

    if response.status_code == 403:
        raise ValueError(
            "Your account plan does not cover this competition, "
            "season or endpoint. Try a supported league or date."
        )

    if response.status_code == 429:
        raise ValueError(
            "The API request limit has been reached. Wait before retrying."
        )

    response.raise_for_status()
    payload = response.json()

    # API-Football-compatible responses report many errors here.
    if payload.get("errors"):
        raise ValueError(str(payload["errors"]))

    # Also handle errors from other response formats.
    if payload.get("success") == 0:
        error = payload.get("error", {})
        raise ValueError(error.get("message", str(error)))

    return payload


def api_rows(payload):
    """Extract the list of records from compatible API responses."""
    return payload.get("response", payload.get("data", [])) or []


def display_api_error(error):
    st.error(str(error))


# =========================================================
# LEAGUE LOOKUP
# =========================================================

@st.cache_data(ttl=3600, show_spinner=False)
def search_leagues(search_term, api_key):
    payload = api_request(
        "leagues",
        {"search": search_term},
        api_key,
    )
    return api_rows(payload)


# Search terms are used to find the new provider's own league IDs.
LEAGUE_SEARCHES = {
    "English Premier League": "Premier League",
    "Spanish La Liga": "La Liga",
    "Italian Serie A": "Serie A",
    "German Bundesliga": "Bundesliga",
    "French Ligue 1": "Ligue 1",
    "UEFA Champions League": "Champions League",
    "Major League Soccer": "Major League Soccer",
}


def league_label(item):
    league = item.get("league", {})
    country = item.get("country", {})
    name = league.get("name", "Unknown competition")
    country_name = country.get("name", "")

    return f"{name} — {country_name}" if country_name else name


def find_selected_league(search_term, selected_label, api_key):
    matches = search_leagues(search_term, api_key)

    for item in matches:
        if league_label(item) == selected_label:
            return item

    return None


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("⚽ GFIPS")
st.sidebar.caption(
    "Global Football Intelligence & Prediction System"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Live Football Data",
        "Match Predictor",
        "About GFIPS",
    ],
)

st.sidebar.divider()
st.sidebar.caption("Football data by 5DollarFootballAPI")
st.sidebar.caption(
    "Predictions are estimates, not guaranteed outcomes."
)


# =========================================================
# OVERVIEW
# =========================================================

def show_overview():
    st.title("⚽ GFIPS Dashboard")

    st.write(
        "Explore football fixtures and results, retrieve available "
        "competition data, and experiment with match probabilities."
    )

    st.info(
        "Live Football Data retrieves records from the new API. "
        "The predictor below is experimental and does not yet train "
        "on real team statistics."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Live Football Data")
        st.write(
            "Search competitions, retrieve fixture records and "
            "download results as CSV."
        )

    with col2:
        st.subheader("Match Predictor")
        st.write(
            "Explore estimated home-win, draw and away-win probabilities "
            "using a simple Poisson model."
        )


# =========================================================
# LIVE FOOTBALL DATA
# =========================================================

def show_live_data():
    st.title("🌍 Live Football Data")

    st.write(
        "Retrieve football fixtures from 5DollarFootballAPI. "
        "The available competitions depend on your account plan."
    )

    api_key = get_api_key()

    if not api_key:
        st.error(
            "The FOOTBALL_API_KEY secret is missing. Open "
            "Streamlit Settings → Secrets and add your new key."
        )
        st.code('FOOTBALL_API_KEY = "YOUR_NEW_API_KEY"')
        return

    search_options = list(LEAGUE_SEARCHES.keys())

    with st.form("gfips_fixture_search"):
        selected_competition = st.selectbox(
            "Competition",
            search_options,
        )

        today = date.today()

        start_date = st.date_input(
            "Start date",
            value=today,
        )

        end_date = st.date_input(
            "End date",
            value=today + timedelta(days=7),
        )

        submitted = st.form_submit_button(
            "Fetch Fixtures",
            type="primary",
            use_container_width=True,
        )

    if not submitted:
        st.caption(
            "Choose a competition and date range, then click Fetch Fixtures."
        )
        return

    if start_date > end_date:
        st.error("The start date must not be after the end date.")
        return

    if (end_date - start_date).days > 30:
        st.error(
            "Please choose a date range of 30 days or less. "
            "You can search another period afterward."
        )
        return

    search_term = LEAGUE_SEARCHES[selected_competition]

    try:
        with st.spinner("Finding the competition in the new API..."):
            matches = search_leagues(search_term, api_key)

        if not matches:
            st.warning(
                "No matching competition was returned. Try another "
                "competition or check the provider's league coverage."
            )
            return

        # Prefer an exact competition-name match.
        target_name = search_term.casefold()
        exact_matches = [
            item for item in matches
            if item.get("league", {}).get("name", "").casefold()
            == target_name
        ]

        choices = exact_matches if exact_matches else matches

        labels = [league_label(item) for item in choices]
        selected_label = st.selectbox(
            "Choose the matching competition",
            labels,
        )

        chosen = choices[labels.index(selected_label)]
        league = chosen.get("league", {})
        country = chosen.get("country", {})

        league_id = league.get("id")

        if not league_id:
            st.error(
                "The API did not return a league ID for this competition."
            )
            return

        # Fetch using the new provider's league ID, not the old ID.
        # The compatible host supports the same from/to fixture filters.
        params = {
            "league": league_id,
            "from": start_date.isoformat(),
            "to": end_date.isoformat(),
        }

        seasons = chosen.get("seasons", [])
        current_seasons = [
            item.get("year")
            for item in seasons
            if item.get("year") is not None
        ]

        if current_seasons:
            season_options = sorted(
                set(current_seasons),
                reverse=True,
            )
            selected_season = st.selectbox(
                "Season start year",
                season_options,
            )
            params["season"] = selected_season

        with st.spinner("Retrieving fixtures and results..."):
            payload = api_request("fixtures", params, api_key)

        fixtures = api_rows(payload)

        if not fixtures:
            st.info(
                "No fixtures were returned for that competition and "
                "date range. Try a different date range or competition."
            )
            return

        rows = []

        for item in fixtures:
            fixture = item.get("fixture", {})
            teams = item.get("teams", {})
            goals = item.get("goals", {})
            status = fixture.get("status", {})
            fixture_league = item.get("league", {})
            venue = fixture.get("venue", {})

            rows.append(
                {
                    "Date": fixture.get("date", ""),
                    "Competition": fixture_league.get(
                        "name", league.get("name", "")
                    ),
                    "Country": fixture_league.get(
                        "country", country.get("name", "")
                    ),
                    "Home Team": teams.get("home", {}).get("name", ""),
                    "Away Team": teams.get("away", {}).get("name", ""),
                    "Home Goals": goals.get("home"),
                    "Away Goals": goals.get("away"),
                    "Status": status.get("long", ""),
                    "Venue": venue.get("name", ""),
                }
            )

        df = pd.DataFrame(rows)

        st.success(f"Retrieved {len(df)} fixture record(s).")

        st.metric("Fixtures retrieved", len(df))

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download fixtures as CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="gfips_football_fixtures.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.caption(
            "Football data by 5DollarFootballAPI"
        )

    except requests.exceptions.Timeout:
        display_api_error(
            "The API took too long to respond. Please try again."
        )
    except requests.exceptions.RequestException as error:
        display_api_error(
            f"Could not connect to the football API: {error}"
        )
    except (ValueError, KeyError, TypeError) as error:
        display_api_error(error)
    except Exception as error:
        display_api_error(
            f"An unexpected error occurred: {error}"
        )


# =========================================================
# EXPERIMENTAL POISSON MATCH PREDICTOR
# =========================================================

def poisson_probability(goals, expected_goals):
    return (
        math.exp(-expected_goals)
        * expected_goals ** goals
        / math.factorial(goals)
    )


def predict_match(home_xg, away_xg, max_goals=8):
    home_win = 0.0
    draw = 0.0
    away_win = 0.0
    scorelines = []

    for home_goals in range(max_goals + 1):
        for away_goals in range(max_goals + 1):
            probability = (
                poisson_probability(home_goals, home_xg)
                * poisson_probability(away_goals, away_xg)
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

    if total:
        home_win /= total
        draw /= total
        away_win /= total

    scorelines.sort(
        key=lambda row: row["Probability"],
        reverse=True,
    )

    return home_win, draw, away_win, scorelines


def show_predictor():
    st.title("🔮 Match Predictor")

    st.warning(
        "This is an experimental model. Expected goals are entered "
        "manually and are not derived from live team statistics."
    )

    col1, col2 = st.columns(2)

    with col1:
        home_team = st.text_input("Home team", "Arsenal")
        home_xg = st.slider(
            "Home expected goals",
            0.1, 4.0, 1.6, 0.1,
        )

    with col2:
        away_team = st.text_input("Away team", "Chelsea")
        away_xg = st.slider(
            "Away expected goals",
            0.1, 4.0, 1.2, 0.1,
        )

    if st.button("Generate prediction", type="primary"):
        if not home_team.strip() or not away_team.strip():
            st.error("Enter both team names.")
            return

        if home_team.strip().casefold() == away_team.strip().casefold():
            st.error("Choose two different teams.")
            return

        home, draw, away, scores = predict_match(home_xg, away_xg)

        a, b, c = st.columns(3)
        a.metric("Home win", f"{home * 100:.1f}%")
        b.metric("Draw", f"{draw * 100:.1f}%")
        c.metric("Away win", f"{away * 100:.1f}%")

        score_df = pd.DataFrame(scores[:5])
        score_df["Probability (%)"] = (
            score_df["Probability"] * 100
        ).round(2)

        st.subheader("Most likely scorelines")
        st.dataframe(
            score_df[["Score", "Probability (%)"]],
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "These estimates are not guarantees and should not be "
            "treated as validated betting predictions."
        )


# =========================================================
# ABOUT
# =========================================================

def show_about():
    st.title("About GFIPS")

    st.markdown(
        """
        **Global Football Intelligence & Prediction System**

        GFIPS is a developing football-data platform intended to support
        fixture exploration, historical analysis and data-driven predictions.

        **Current features**
        - Fixture retrieval through 5DollarFootballAPI
        - Competition search using the new provider's IDs
        - Date filtering and CSV export
        - Experimental Poisson match predictor

        **Planned improvements**
        - Historical results and team-form datasets
        - Team and match statistics where available
        - Odds and additional markets
        - Model training and historical backtesting
        - Prediction-performance reporting
        """
    )

    st.markdown(
        "Football data by [5DollarFootballAPI](https://5dollarfootballapi.com)"
    )


# =========================================================
# PAGE ROUTING
# =========================================================

if page == "Overview":
    show_overview()
elif page == "Live Football Data":
    show_live_data()
elif page == "Match Predictor":
    show_predictor()
elif page == "About GFIPS":
    show_about()
