import math
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st


# ============================================================
# GFIPS — Global Football Intelligence & Prediction System
# ============================================================

st.set_page_config(
    page_title="GFIPS | Football Intelligence",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE = "https://api-football.5dollarfootballapi.com"
REQUEST_TIMEOUT = 25
MAX_GOALS = 10

# Confirmed from your API response.
DEFAULT_LEAGUE_ID = 4160026622
DEFAULT_LEAGUE_NAME = "England Premier League"


# ============================================================
# Page styling
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1500px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    .gfips-hero {
        padding: 28px;
        border-radius: 18px;
        background: linear-gradient(125deg, #102b25, #176448, #248b68);
        color: white;
        margin-bottom: 20px;
    }

    .gfips-hero h1 {
        color: white;
        font-size: 2.3rem;
        margin-bottom: 8px;
    }

    .gfips-hero p {
        color: #e3f7ed;
        margin-bottom: 5px;
    }

    div[data-testid="stMetric"] {
        border: 1px solid rgba(128, 128, 128, 0.25);
        padding: 15px;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# API configuration and helpers
# ============================================================

def get_api_key():
    """Read the key from Streamlit Secrets without displaying it."""
    try:
        key = st.secrets.get("FOOTBALL_API_KEY", "")
        return str(key).strip() if key else ""
    except Exception:
        return ""


def api_request(endpoint, params=None, api_key=None):
    """Make a request to the configured football API."""
    key = api_key or get_api_key()

    if not key:
        raise ValueError(
            "FOOTBALL_API_KEY is missing. Add it under "
            "Streamlit Settings > Secrets."
        )

    endpoint = "/" + endpoint.lstrip("/")

    response = requests.get(
        API_BASE + endpoint,
        headers={"x-apisports-key": key},
        params=params or {},
        timeout=REQUEST_TIMEOUT,
    )

    if response.status_code in (401, 403):
        raise ValueError(
            "The API rejected the request. Check your key and "
            "subscription permissions."
        )

    if response.status_code == 429:
        raise ValueError(
            "The API request limit has been reached. Please wait and retry."
        )

    response.raise_for_status()

    try:
        payload = response.json()
    except ValueError as exc:
        raise ValueError("The API returned invalid JSON.") from exc

    if not isinstance(payload, dict):
        raise ValueError("The API returned an unexpected response format.")

    errors = payload.get("errors")
    if errors:
        raise ValueError(f"API error: {errors}")

    data = payload.get("response")
    if data is None:
        data = payload.get("data")
    if data is None:
        data = []

    return data, payload


@st.cache_data(ttl=900, show_spinner=False)
def cached_request(endpoint, params_tuple, api_key):
    """Cache successful data retrieval for 15 minutes."""
    data, _ = api_request(
        endpoint,
        params=dict(params_tuple),
        api_key=api_key,
    )
    return data


def fetch_data(endpoint, params=None, use_cache=True):
    params = params or {}
    key = get_api_key()

    if not key:
        raise ValueError("FOOTBALL_API_KEY is missing from Streamlit Secrets.")

    if use_cache:
        return cached_request(
            endpoint,
            tuple(sorted(params.items())),
            key,
        )

    data, _ = api_request(endpoint, params=params, api_key=key)
    return data


# ============================================================
# General data helpers
# ============================================================

def safe_dict(value):
    return value if isinstance(value, dict) else {}


def safe_list(value):
    return value if isinstance(value, list) else []


def number_or_none(value):
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def format_score(home, away):
    if home is None or away is None:
        return "—"
    return f"{home}-{away}"


def get_fixture_parts(item):
    item = safe_dict(item)

    return (
        safe_dict(item.get("fixture")),
        safe_dict(item.get("league")),
        safe_dict(item.get("teams")),
        safe_dict(item.get("goals")),
        safe_dict(item.get("score")),
    )


def fixture_to_row(item):
    fixture, league, teams, goals, score = get_fixture_parts(item)

    home = safe_dict(teams.get("home"))
    away = safe_dict(teams.get("away"))
    status = safe_dict(fixture.get("status"))
    halftime = safe_dict(score.get("halftime"))

    return {
        "Fixture ID": fixture.get("id"),
        "Date": fixture.get("date", ""),
        "League": league.get("name", "Unknown"),
        "Country": league.get("country", ""),
        "Home": home.get("name", "Unknown"),
        "Away": away.get("name", "Unknown"),
        "Home Goals": goals.get("home"),
        "Away Goals": goals.get("away"),
        "Half-time": format_score(
            halftime.get("home"),
            halftime.get("away"),
        ),
        "Status": status.get("long") or status.get("short", "Unknown"),
    }


def extract_fixture_id(item):
    fixture, _, _, _, _ = get_fixture_parts(item)
    return fixture.get("id")


def extract_team_name(item, side):
    _, _, teams, _, _ = get_fixture_parts(item)
    return safe_dict(teams.get(side)).get("name", "Unknown")


def extract_team_id(item, side):
    _, _, teams, _, _ = get_fixture_parts(item)
    return safe_dict(teams.get(side)).get("id")


def extract_status(item):
    fixture, _, _, _, _ = get_fixture_parts(item)
    return safe_dict(fixture.get("status")).get("short", "")


def download_dataframe(df, filename):
    st.download_button(
        "Download CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
    )


# ============================================================
# League search and selection
# ============================================================

@st.cache_data(ttl=1800, show_spinner=False)
def get_leagues(api_key, search_text=""):
    params = {}

    if search_text.strip():
        params["search"] = search_text.strip()

    data, _ = api_request(
        "/leagues",
        params=params,
        api_key=api_key,
    )

    return safe_list(data)


def normalize_league(item):
    """Support the nested league format shown in your API response."""
    item = safe_dict(item)
    league = safe_dict(item.get("league"))
    country = safe_dict(item.get("country"))

    # Also accept a flat league object if the API returns one.
    if not league and item.get("id") is not None:
        league = item

    seasons = safe_list(item.get("seasons"))

    current_season = None
    if seasons:
        current_season = next(
            (
                season for season in seasons
                if safe_dict(season).get("current") is True
            ),
            seasons[0],
        )

    season = safe_dict(current_season)

    return {
        "id": league.get("id"),
        "name": league.get("name", "Unknown league"),
        "country": country.get("name", ""),
        "season": season.get("year"),
        "seasons": seasons,
    }


def get_league_options(api_key, search_text=""):
    raw_leagues = get_leagues(api_key, search_text)
    normalized = []

    for item in raw_leagues:
        league = normalize_league(item)
        if league.get("id") is not None:
            normalized.append(league)

    # Ensure the confirmed default league remains selectable if returned.
    if not any(
        str(item["id"]) == str(DEFAULT_LEAGUE_ID)
        for item in normalized
    ):
        normalized.insert(
            0,
            {
                "id": DEFAULT_LEAGUE_ID,
                "name": DEFAULT_LEAGUE_NAME,
                "country": "England",
                "season": None,
                "seasons": [],
            },
        )

    return normalized


def league_selector(label, key):
    """
    Render a dynamic league selector using IDs returned by the API.
    The default league is available as a fallback.
    """
    if not get_api_key():
        st.warning("Configure FOOTBALL_API_KEY to load league options.")
        return DEFAULT_LEAGUE_ID, DEFAULT_LEAGUE_NAME, None

    search_text = st.text_input(
        "Optional league search",
        placeholder="Search for a competition",
        key=f"{key}_search",
    )

    try:
        leagues = get_league_options(get_api_key(), search_text)
    except Exception as exc:
        st.warning(
            f"Could not load league options: {exc}. "
            "The default league will be used."
        )
        leagues = [{
            "id": DEFAULT_LEAGUE_ID,
            "name": DEFAULT_LEAGUE_NAME,
            "country": "England",
            "season": None,
            "seasons": [],
        }]

    labels = [
        (
            f"{item['name']} — {item['country']}"
            if item.get("country")
            else item["name"]
        )
        for item in leagues
    ]

    default_index = next(
        (
            i for i, item in enumerate(leagues)
            if str(item["id"]) == str(DEFAULT_LEAGUE_ID)
        ),
        0,
    )

    selected_index = st.selectbox(
        label,
        options=list(range(len(leagues))),
        index=default_index,
        format_func=lambda i: labels[i],
        key=key,
    )

    selected = leagues[selected_index]

    seasons = selected.get("seasons", [])
    selected_season = None

    if seasons:
        season_options = [
            safe_dict(item).get("year")
            for item in seasons
            if safe_dict(item).get("year") is not None
        ]

        if season_options:
            default_season = selected.get("season")
            season_index = (
                season_options.index(default_season)
                if default_season in season_options
                else 0
            )

            selected_season = st.selectbox(
                "Season",
                options=season_options,
                index=season_index,
                key=f"{key}_season",
            )

    return selected["id"], selected["name"], selected_season


# ============================================================
# Fixture retrieval — CORRECTED
# ============================================================

@st.cache_data(ttl=900, show_spinner=False)
def get_fixtures_for_date_range(
    from_date,
    to_date,
    api_key,
    league_id=None,
    team_id=None,
    season=None,
):
    """
    Retrieve fixtures for a date range.

    IMPORTANT:
    The provider requires a league or team ID when using from/to.
    This function enforces that requirement.
    """
    if not league_id and not team_id:
        raise ValueError(
            "Select a league or provide a team ID before searching "
            "fixtures by date."
        )

    params = {
        "from": from_date,
        "to": to_date,
    }

    # The API requires at least one of these identifiers.
    if team_id:
        params["team"] = team_id
    else:
        params["league"] = league_id

    if season is not None:
        params["season"] = season

    data, _ = api_request(
        "/fixtures",
        params=params,
        api_key=api_key,
    )

    return safe_list(data)


@st.cache_data(ttl=1800, show_spinner=False)
def get_team_search(search_text, api_key):
    data, _ = api_request(
        "/teams",
        params={"search": search_text},
        api_key=api_key,
    )
    return safe_list(data)


@st.cache_data(ttl=900, show_spinner=False)
def get_fixture_details(fixture_id, api_key):
    data, _ = api_request(
        "/fixtures",
        params={"id": fixture_id},
        api_key=api_key,
    )
    return safe_list(data)


@st.cache_data(ttl=900, show_spinner=False)
def get_fixture_statistics(fixture_id, api_key):
    data, _ = api_request(
        "/fixtures/statistics",
        params={"fixture": fixture_id},
        api_key=api_key,
    )
    return safe_list(data)


# ============================================================
# Goal probability model
# ============================================================

def poisson_probability(goals, expected):
    return (
        math.exp(-expected)
        * (expected ** goals)
        / math.factorial(goals)
    )


def make_score_matrix(home_xg, away_xg, max_goals=MAX_GOALS):
    home_probs = [
        poisson_probability(i, home_xg)
        for i in range(max_goals + 1)
    ]

    away_probs = [
        poisson_probability(i, away_xg)
        for i in range(max_goals + 1)
    ]

    matrix = np.outer(home_probs, away_probs)

    total = matrix.sum()
    if total > 0:
        matrix = matrix / total

    return matrix


def probabilities_from_matrix(matrix):
    result = {
        "Home win": 0.0,
        "Draw": 0.0,
        "Away win": 0.0,
        "Over 0.5 goals": 0.0,
        "Over 1.5 goals": 0.0,
        "Over 2.5 goals": 0.0,
        "Over 3.5 goals": 0.0,
        "Over 4.5 goals": 0.0,
        "Under 2.5 goals": 0.0,
        "BTTS — Yes": 0.0,
        "BTTS — No": 0.0,
        "Home clean sheet": 0.0,
        "Away clean sheet": 0.0,
    }

    for home_goals in range(matrix.shape[0]):
        for away_goals in range(matrix.shape[1]):
            p = float(matrix[home_goals, away_goals])
            total_goals = home_goals + away_goals

            if home_goals > away_goals:
                result["Home win"] += p
            elif home_goals == away_goals:
                result["Draw"] += p
            else:
                result["Away win"] += p

            for threshold in (1, 2, 3, 4, 5):
                if total_goals >= threshold:
                    key = f"Over {threshold - 0.5:g} goals"
                    result[key] += p

            if total_goals <= 2:
                result["Under 2.5 goals"] += p

            if home_goals > 0 and away_goals > 0:
                result["BTTS — Yes"] += p

            if away_goals == 0:
                result["Home clean sheet"] += p

            if home_goals == 0:
                result["Away clean sheet"] += p

    result["BTTS — No"] = 1.0 - result["BTTS — Yes"]
    return result


def top_correct_scores(matrix, count=10):
    rows = []

    for home_goals in range(matrix.shape[0]):
        for away_goals in range(matrix.shape[1]):
            rows.append({
                "Correct score": f"{home_goals}-{away_goals}",
                "Probability": float(matrix[home_goals, away_goals]),
            })

    rows.sort(key=lambda item: item["Probability"], reverse=True)
    return rows[:count]


def calculate_goal_expectations(home_team, away_team, fixtures):
    """
    Estimate goal expectations from the loaded completed fixture records.
    The result is only a simple baseline, not a validated prediction.
    """
    home_scored = []
    home_conceded = []
    away_scored = []
    away_conceded = []

    for item in fixtures:
        if extract_status(item) not in {"FT", "AET", "PEN"}:
            continue

        _, _, teams, goals, _ = get_fixture_parts(item)

        home = safe_dict(teams.get("home")).get("name")
        away = safe_dict(teams.get("away")).get("name")

        hg = number_or_none(goals.get("home"))
        ag = number_or_none(goals.get("away"))

        if hg is None or ag is None:
            continue

        if home == home_team:
            home_scored.append(hg)
            home_conceded.append(ag)

        if away == home_team:
            home_scored.append(ag)
            home_conceded.append(hg)

        if home == away_team:
            away_scored.append(hg)
            away_conceded.append(ag)

        if away == away_team:
            away_scored.append(ag)
            away_conceded.append(hg)

    if len(home_scored) < 2 or len(away_scored) < 2:
        return None

    home_attack = float(np.mean(home_scored[-10:]))
    home_defence = float(np.mean(home_conceded[-10:]))
    away_attack = float(np.mean(away_scored[-10:]))
    away_defence = float(np.mean(away_conceded[-10:]))

    home_xg = max(0.10, min(5.0, (home_attack + away_defence) / 2))
    away_xg = max(0.10, min(5.0, (away_attack + home_defence) / 2))

    return home_xg, away_xg


# ============================================================
# Match statistics
# ============================================================

def normalize_stat_name(value):
    return str(value or "").strip().lower()


def extract_statistics_map(statistics_data):
    output = {}

    for item in safe_list(statistics_data):
        item = safe_dict(item)
        team = safe_dict(item.get("team"))
        team_name = team.get("name", "Unknown")
        stats = {}

        for stat in safe_list(item.get("statistics")):
            stat = safe_dict(stat)
            name = stat.get("type")
            stats[normalize_stat_name(name)] = stat.get("value")

        output[team_name] = stats

    return output


def display_statistics_table(statistics_data):
    stats = extract_statistics_map(statistics_data)

    if not stats:
        st.info("No statistics were returned for this fixture.")
        return

    team_names = list(stats.keys())

    if len(team_names) < 2:
        st.info("Statistics for both teams were not returned.")
        return

    home_name, away_name = team_names[0], team_names[1]

    keys = sorted(set(stats[home_name]) | set(stats[away_name]))

    rows = [
        {
            "Statistic": key.title(),
            home_name: stats[home_name].get(key),
            away_name: stats[away_name].get(key),
        }
        for key in keys
    ]

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# Navigation
# ============================================================

st.sidebar.title("⚽ GFIPS")
st.sidebar.caption("Global Football Intelligence & Prediction System")

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Fixtures & Results",
        "Global Search",
        "Match Analysis",
        "Prediction Lab",
        "API Diagnostics",
        "About GFIPS",
    ],
)

st.sidebar.divider()

api_key = get_api_key()

if api_key:
    st.sidebar.success("API secret detected")
else:
    st.sidebar.error("API secret not configured")

st.sidebar.caption("Data coverage depends on your API subscription.")


st.markdown(
    """
    <div class="gfips-hero">
        <h1>Global Football Intelligence</h1>
        <p>Fixtures. Statistics. Match intelligence. Probability modelling.</p>
        <p>Explore football data from competitions supported by your provider.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Overview
# ============================================================

if page == "Overview":
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Platform", "GFIPS")
