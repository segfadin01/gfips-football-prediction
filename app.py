
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


# ============================================================
# Styling
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

    .gfips-note {
        font-size: 0.85rem;
        opacity: 0.8;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# API helpers
# ============================================================

def get_api_key():
    """Read the secret without exposing its value."""
    try:
        key = st.secrets.get("FOOTBALL_API_KEY", "")
        return str(key).strip() if key else ""
    except Exception:
        return ""


def api_request(endpoint, params=None, api_key=None):
    """
    Send a request to the API-Football-compatible endpoint.

    The API key is sent as a request header and is never displayed.
    """
    key = api_key or get_api_key()

    if not key:
        raise ValueError(
            "FOOTBALL_API_KEY is missing. Add your key under "
            "Streamlit app Settings > Secrets."
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
            "The API rejected the request. Check your secret, "
            "API access permissions and subscription."
        )

    if response.status_code == 429:
        raise ValueError(
            "The API request limit has been reached. Wait before retrying."
        )

    response.raise_for_status()

    try:
        payload = response.json()
    except ValueError as exc:
        raise ValueError("The API returned an invalid JSON response.") from exc

    if not isinstance(payload, dict):
        raise ValueError("The API returned an unexpected response format.")

    errors = payload.get("errors")

    if errors:
        raise ValueError(f"API error: {errors}")

    # API-Football-compatible responses usually use "response".
    # The alternative supports providers that use "data".
    data = payload.get("response")

    if data is None:
        data = payload.get("data")

    if data is None:
        data = []

    return data, payload


@st.cache_data(ttl=900, show_spinner=False)
def cached_api_request(endpoint, params_tuple, api_key):
    """Cache API responses for 15 minutes."""
    params = dict(params_tuple)
    data, _ = api_request(endpoint, params=params, api_key=api_key)
    return data


def fetch_data(endpoint, params=None, use_cache=True):
    """Convenience wrapper around the API client."""
    params = params or {}

    if use_cache:
        key = get_api_key()
        if not key:
            raise ValueError(
                "FOOTBALL_API_KEY is missing from Streamlit Secrets."
            )

        return cached_api_request(
            endpoint,
            tuple(sorted(params.items())),
            key,
        )

    data, _ = api_request(endpoint, params=params)
    return data


# ============================================================
# Data helpers
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
    """Convert an API fixture object to a table row."""
    fixture, league, teams, goals, score = get_fixture_parts(item)

    home = safe_dict(teams.get("home"))
    away = safe_dict(teams.get("away"))
    status = safe_dict(fixture.get("status"))

    elapsed = status.get("elapsed")
    status_text = status.get("long") or status.get("short") or "Unknown"

    if elapsed is not None:
        status_text = f"{status_text} ({elapsed}')"

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
            safe_dict(score.get("halftime")).get("home"),
            safe_dict(score.get("halftime")).get("away"),
        ),
        "Status": status_text,
        "Venue": safe_dict(fixture.get("venue")).get("name", ""),
    }


def format_score(home, away):
    if home is None or away is None:
        return "—"
    return f"{home}-{away}"


def extract_fixture_id(item):
    fixture, _, _, _, _ = get_fixture_parts(item)
    return fixture.get("id")


def extract_team_name(item, side):
    _, _, teams, _, _ = get_fixture_parts(item)
    return safe_dict(teams.get(side)).get("name", "Unknown")


def extract_team_id(item, side):
    _, _, teams, _, _ = get_fixture_parts(item)
    return safe_dict(teams.get(side)).get("id")


def extract_league_id(item):
    _, league, _, _, _ = get_fixture_parts(item)
    return league.get("id")


def extract_league_name(item):
    _, league, _, _, _ = get_fixture_parts(item)
    return league.get("name", "Unknown")


def extract_status(item):
    fixture, _, _, _, _ = get_fixture_parts(item)
    return safe_dict(fixture.get("status")).get("short", "")


def dataframe_download(df, filename="gfips_export.csv"):
    """Offer a CSV download when a table is available."""
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download CSV",
        data=csv_bytes,
        file_name=filename,
        mime="text/csv",
    )


# ============================================================
# Fixture retrieval
# ============================================================

@st.cache_data(ttl=900, show_spinner=False)
def get_fixtures_for_date_range(from_date, to_date, api_key):
    """
    Retrieve fixtures for a date range.

    Results depend on provider coverage and subscription limits.
    """
    all_fixtures = []

    # Query the date range. If the provider rejects a date range,
    # the error will be displayed by the caller.
    data, _ = api_request(
        "/fixtures",
        params={
            "from": from_date,
            "to": to_date,
        },
        api_key=api_key,
    )

    all_fixtures.extend(safe_list(data))
    return all_fixtures


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


@st.cache_data(ttl=1800, show_spinner=False)
def get_team_recent_fixtures(team_id, api_key, count=10):
    data, _ = api_request(
        "/fixtures",
        params={
            "team": team_id,
            "last": count,
        },
        api_key=api_key,
    )
    return safe_list(data)


# ============================================================
# Probability model
# ============================================================

def poisson_probability(goals, expected):
    if expected < 0:
        return 0.0

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

    # Normalize the truncated matrix so displayed probabilities sum to 1.
    total = matrix.sum()
    if total > 0:
        matrix = matrix / total

    return matrix


def probabilities_from_matrix(matrix):
    home_win = float(
        sum(
            matrix[i, j]
            for i in range(matrix.shape[0])
            for j in range(matrix.shape[1])
            if i > j
        )
    )

    draw = float(np.trace(matrix))

    away_win = float(
        sum(
            matrix[i, j]
            for i in range(matrix.shape[0])
            for j in range(matrix.shape[1])
            if i < j
        )
    )

    over_05 = 0.0
    over_15 = 0.0
    over_25 = 0.0
    over_35 = 0.0
    over_45 = 0.0
    under_25 = 0.0
    btts_yes = 0.0
    clean_home = 0.0
    clean_away = 0.0

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            probability = float(matrix[i, j])
            total_goals = i + j

            if total_goals >= 1:
                over_05 += probability
            if total_goals >= 2:
                over_15 += probability
            if total_goals >= 3:
                over_25 += probability
            if total_goals >= 4:
                over_35 += probability
            if total_goals >= 5:
                over_45 += probability
            if total_goals <= 2:
                under_25 += probability
            if i > 0 and j > 0:
                btts_yes += probability
            if j == 0:
                clean_home += probability
            if i == 0:
                clean_away += probability

    return {
        "Home win": home_win,
        "Draw": draw,
        "Away win": away_win,
        "Over 0.5 goals": over_05,
        "Over 1.5 goals": over_15,
        "Over 2.5 goals": over_25,
        "Over 3.5 goals": over_35,
        "Over 4.5 goals": over_45,
        "Under 2.5 goals": under_25,
        "BTTS — Yes": btts_yes,
        "BTTS — No": 1.0 - btts_yes,
        "Home clean sheet": clean_home,
        "Away clean sheet": clean_away,
    }


def top_correct_scores(matrix, count=10):
    rows = []

    for home_goals in range(matrix.shape[0]):
        for away_goals in range(matrix.shape[1]):
            rows.append({
                "Score": f"{home_goals}-{away_goals}",
                "Probability": float(matrix[home_goals, away_goals]),
            })

    rows.sort(
        key=lambda item: item["Probability"],
        reverse=True,
    )

    return rows[:count]


def calculate_goal_expectations(home_team, away_team, fixtures):
    """
    Estimate goal expectations from completed matches in the
    currently loaded fixture dataset.

    This function deliberately does not invent team statistics.
    If the sample is insufficient, the prediction is unavailable.
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

    home_xg = max(
        0.10,
        min(5.0, (home_attack + away_defence) / 2),
    )
    away_xg = max(
        0.10,
        min(5.0, (away_attack + home_defence) / 2),
    )

    return home_xg, away_xg


# ============================================================
# Statistics helpers
# ============================================================

def normalize_stat_name(value):
    return str(value or "").strip().lower()


def extract_statistics_map(statistics_item):
    """
    Extract statistics in the API-Football statistics format.

    Example statistic entries can include Total Shots, Shots on Goal,
    Corner Kicks, Yellow Cards and Fouls.
    """
    output = {}

    for item in safe_list(statistics_item):
        team = safe_dict(item)
        team_name = team.get("team", {}).get("name", "Unknown")
        stats = {}

        for stat in safe_list(team.get("statistics")):
            stat = safe_dict(stat)
            name = stat.get("type")
            value = stat.get("value")
            stats[normalize_stat_name(name)] = value

        output[team_name] = stats

    return output


def find_stat(stats, *possible_names):
    for name in possible_names:
        key = normalize_stat_name(name)
        if key in stats:
            return stats[key]
    return None


def display_statistics_table(statistics_item):
    data = extract_statistics_map(statistics_item)

    if not data:
        st.info("No match statistics were returned by the API.")
        return

    team_names = list(data.keys())

    if len(team_names) < 2:
        st.info("The API did not return statistics for both teams.")
        return

    home_name, away_name = team_names[0], team_names[1]
    home_stats = data[home_name]
    away_stats = data[away_name]

    all_keys = sorted(set(home_stats) | set(away_stats))

    rows = []
    for key in all_keys:
        rows.append({
            "Statistic": key.title(),
            home_name: home_stats.get(key),
            away_name: away_stats.get(key),
        })

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# Sidebar and navigation
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

today = date.today()
default_from = today
default_to = today + timedelta(days=7)

st.sidebar.caption("Data provider: 5DollarFootballAPI")
st.sidebar.caption("Model: experimental statistical baseline")

api_key = get_api_key()
if api_key:
    st.sidebar.success("API secret detected")
else:
    st.sidebar.error("API secret not configured")


# ============================================================
# Page header
# ============================================================

st.markdown(
    """
    <div class="gfips-hero">
        <h1>Global Football Intelligence</h1>
        <p>Fixtures. Statistics. Match intelligence. Probability modelling.</p>
        <p class="gfips-note">
            A data-driven football analysis platform.
            Data availability depends on the API provider and subscription.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Overview
# ============================================================

if page == "Overview":
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Platform", "GFIPS")
    col2.metric("Data connection", "Configured" if api_key else "Missing key")
    col3.metric("Prediction engine", "Experimental")
    col4.metric("Data freshness", "API dependent")

    st.subheader("Welcome to GFIPS")

    st.write(
        "Use the navigation panel to search fixtures, inspect match data, "
        "review available statistics and explore experimental goal-market "
        "probabilities."
    )

    st.markdown("### Platform modules")

    a, b, c = st.columns(3)

    with a:
        st.markdown("#### 🌍 Global Search")
        st.write(
            "Search competitions, teams and fixtures where your API plan "
            "provides the required coverage."
        )

    with b:
        st.markdown("#### 📊 Match Analysis")
        st.write(
            "Inspect fixture details and the match statistics returned "
            "by the data provider."
        )

    with c:
        st.markdown("#### 🧠 Prediction Lab")
        st.write(
            "Explore goal-based probability estimates from available "
            "completed match records."
        )

    st.warning(
        "GFIPS does not guarantee outcomes. Estimates may be unreliable "
        "when historical samples are small, stale or incomplete. "
        "Markets requiring historical shot, card or corner data are not "
        "estimated unless the required data is available."
    )

    st.markdown("### Quick fixture lookup")

    q1, q2 = st.columns(2)

    with q1:
        quick_from = st.date_input(
            "Start date",
            value=default_from,
            key="overview_start",
        )

    with q2:
        quick_to = st.date_input(
            "End date",
            value=default_to,
            key="overview_end",
        )

    if st.button("Load fixtures", type="primary", key="overview_load"):
        if quick_from > quick_to:
            st.error("The start date must not be after the end date.")
        elif not api_key:
            st.error("Add FOOTBALL_API_KEY to Streamlit Secrets first.")
        else:
            try:
                with st.spinner("Retrieving fixtures..."):
                    fixtures = get_fixtures_for_date_range(
                        quick_from.isoformat(),
                        quick_to.isoformat(),
                        api_key,
                    )

                if fixtures:
                    df = pd.DataFrame(
                        [fixture_to_row(item) for item in fixtures]
                    )
                    st.dataframe(
                        df,
                        use_container_width=True,
                        hide_index=True,
                    )
                    dataframe_download(df, "gfips_fixtures.csv")
                else:
                    st.info(
                        "No fixtures were returned for this date range. "
                        "Check API coverage and query restrictions."
                    )

            except Exception as exc:
                st.error(f"Could not load fixtures: {exc}")


# ============================================================
# Fixtures and results
# ============================================================

elif page == "Fixtures & Results":
    st.subheader("Fixtures & Results Explorer")

    left, right = st.columns(2)

    with left:
        start_date = st.date_input(
            "From date",
            value=default_from,
            key="fixtures_start",
        )

    with right:
        end_date = st.date_input(
            "To date",
            value=default_to,
            key="fixtures_end",
        )

    if start_date > end_date:
        st.warning("The start date must not be after the end date.")

    elif st.button("Search fixtures", type="primary"):
        if not api_key:
            st.error("Add FOOTBALL_API_KEY to Streamlit Secrets.")
        else:
            try:
                with st.spinner("Searching fixtures..."):
                    fixtures = get_fixtures_for_date_range(
                        start_date.isoformat(),
                        end_date.isoformat(),
                        api_key,
                    )

                if not fixtures:
                    st.info(
                        "No fixtures found. The provider may have no "
                        "records for these dates or may restrict this query."
                    )
                else:
                    df = pd.DataFrame(
                        [fixture_to_row(item) for item in fixtures]
                    )

                    search = st.text_input(
                        "Filter displayed fixtures by team or league"
                    )

                    if search.strip():
                        mask = df.astype(str).apply(
                            lambda column: column.str.contains(
                                search,
                                case=False,
                                na=False,
                            )
                        ).any(axis=1)
                        df = df[mask]

                    st.caption(f"{len(df)} fixture records displayed")
                    st.dataframe(
                        df,
                        use_container_width=True,
                        hide_index=True,
                    )
                    dataframe_download(df, "gfips_fixtures_results.csv")

            except Exception as exc:
                st.error(f"Fixture search failed: {exc}")


# ============================================================
# Global search
# ============================================================

elif page == "Global Search":
    st.subheader("Global Football Search")

    search_type = st.radio(
        "Search for",
        ["Teams", "Leagues / Competitions"],
        horizontal=True,
    )

    search_text = st.text_input(
        "Enter a team or competition name",
        placeholder="For example: Arsenal or Premier League",
    )

    if st.button("Search", type="primary"):
        if not api_key:
            st.error("Add FOOTBALL_API_KEY to Streamlit Secrets.")
        elif not search_text.strip():
            st.warning("Enter a search term first.")
        else:
            try:
                with st.spinner("Searching the provider..."):
                    if search_type == "Teams":
                        results = get_team_search(search_text, api_key)
                    else:
                        results = get_leagues(api_key, search_text)

                if not results:
                    st.info(
                        "No results returned. Check the spelling, "
                        "provider coverage and API subscription."
                    )
                else:
                    rows = []

                    for item in results:
                        item = safe_dict(item)

                        if search_type == "Teams":
                            team = safe_dict(item.get("team"))
                            venue = safe_dict(item.get("venue"))

                            rows.append({
                                "Team ID": team.get("id"),
                                "Team": team.get("name"),
                                "Country": team.get("country"),
                                "Founded": team.get("founded"),
                                "National team": team.get("national"),
                                "Venue": venue.get("name"),
                                "City": venue.get("city"),
                            })
                        else:
                            league = safe_dict(item.get("league"))
                            country = safe_dict(item.get("country"))

                            if not league:
                                league = item

                            rows.append({
                                "League ID": league.get("id"),
                                "Competition": league.get("name"),
                                "Type": league.get("type"),
                                "Country": country.get("name"),
                                "Country Code": country.get("code"),
                            })

                    df = pd.DataFrame(rows)
                    st.dataframe(
                        df,
                        use_container_width=True,
                        hide_index=True,
                    )
                    dataframe_download(df, "gfips_search_results.csv")

                    st.caption(
                        "League and team IDs are supplied by the provider. "
                        "Coverage varies by competition and subscription."
                    )

            except Exception as exc:
                st.error(f"Search failed: {exc}")


# ============================================================
# Match analysis
# ============================================================

elif page == "Match Analysis":
    st.subheader("Match Analysis")

    fixture_id = st.number_input(
        "Fixture ID",
        min_value=1,
        step=1,
        value=1,
        help="Enter a fixture ID returned by your API provider.",
    )

    if st.button("Load match analysis", type="primary"):
        if not api_key:
            st.error("Add FOOTBALL_API_KEY to Streamlit Secrets.")
        else:
            try:
                with st.spinner("Loading fixture details..."):
                    details = get_fixture_details(int(fixture_id), api_key)

                if not details:
                    st.warning(
                        "No fixture details were returned for this ID."
                    )
                else:
                    item = details[0]
                    fixture, league, teams, goals, score = get_fixture_parts(
                        item
                    )

                    home = safe_dict(teams.get("home"))
                    away = safe_dict(teams.get("away"))
                    status = safe_dict(fixture.get("status"))
                    venue = safe_dict(fixture.get("venue"))

                    st.markdown("### Match overview")

                    st.markdown(
                        f"## {home.get('name', 'Home')} "
                        f"{format_score(goals.get('home'), goals.get('away'))} "
                        f"{away.get('name', 'Away')}"
                    )

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Competition", league.get("name", "Unknown"))
                    m2.metric("Status", status.get("long", "Unknown"))
                    m3.metric("Kick-off", fixture.get("date", "Unknown"))

                    st.write(f"**Country:** {league.get('country', 'Unknown')}")
                    st.write(f"**Venue:** {venue.get('name', 'Unknown')}")
                    st.write(
                        "**Half-time score:** "
                        + format_score(
                            safe_dict(score.get("halftime")).get("home"),
                            safe_dict(score.get("halftime")).get("away"),
                        )
                    )

                    st.markdown("### Match statistics")

                    try:
                        stats = get_fixture_statistics(
                            int(fixture_id),
                            api_key,
                        )
                        display_statistics_table(stats)
                    except Exception as exc:
                        st.info(
                            "Statistics could not be loaded. Your plan may "
                            "not include this endpoint or this fixture may "
                            "not have statistics available. "
                            f"Details: {exc}"
                        )

                    st.markdown("### Events")

                    events = safe_list(item.get("events"))

                    if events:
                        event_rows = []

                        for event in events:
                            event = safe_dict(event)
                            time_info = safe_dict(event.get("time"))
                            team = safe_dict(event.get("team"))
                            player = safe_dict(event.get("player"))
                            assist = safe_dict(event.get("assist"))

                            event_rows.append({
                                "Minute": time_info.get("elapsed"),
                                "Team": team.get("name"),
                                "Player": player.get("name"),
                                "Event": event.get("type"),
                                "Detail": event.get("detail"),
                                "Assist": assist.get("name"),
                            })

                        st.dataframe(
                            pd.DataFrame(event_rows),
                            use_container_width=True,
                            hide_index=True,
                        )
                    else:
                        st.info(
                            "Events were not included in this response. "
                            "The provider may require a separate request."
                        )

            except Exception as exc:
                st.error(f"Match analysis failed: {exc}")


# ============================================================
# Prediction lab
# ============================================================

elif page == "Prediction Lab":
    st.subheader("GFIPS Prediction Lab")

    st.write(
        "Select a home team and an away team from completed fixtures "
        "loaded for your chosen date range. The baseline uses goal records "
        "from that dataset; it is not a trained or validated model."
    )

    p1, p2 = st.columns(2)

    with p1:
        history_from = st.date_input(
            "History from",
            value=today - timedelta(days=180),
            key="prediction_from",
        )

    with p2:
        history_to = st.date_input(
            "History to",
            value=today,
            key="prediction_to",
        )

    if history_from > history_to:
        st.warning("The start date must not be after the end date.")

    elif st.button("Load prediction dataset", type="primary"):
        if not api_key:
            st.error("Add FOOTBALL_API_KEY to Streamlit Secrets.")
        else:
            try:
                with st.spinner("Loading historical fixtures..."):
                    fixtures = get_fixtures_for_date_range(
                        history_from.isoformat(),
                        history_to.isoformat(),
                        api_key,
                    )

                completed = [
                    item for item in fixtures
                    if extract_status(item) in {"FT", "AET", "PEN"}
                ]

                st.session_state["gfips_prediction_fixtures"] = completed

                if completed:
                    st.success(
                        f"Loaded {len(completed)} completed fixtures."
                    )
                else:
                    st.warning(
                        "No completed fixtures were returned. "
                        "Prediction requires suitable historical records."
                    )

            except Exception as exc:
                st.error(f"Could not load historical fixtures: {exc}")

    fixtures = st.session_state.get("gfips_prediction_fixtures", [])

    if fixtures:
        available_teams = sorted({
            name
            for item in fixtures
            for name in [
                extract_team_name(item, "home"),
                extract_team_name(item, "away"),
            ]
            if name and name != "Unknown"
        })

        if len(available_teams) < 2:
            st.warning("The returned dataset has too few teams.")
        else:
            a, b = st.columns(2)

            with a:
                home_team = st.selectbox(
                    "Home team",
                    available_teams,
                    key="prediction_home",
                )

            away_options = [
                team for team in available_teams
                if team != home_team
            ]

            with b:
                away_team = st.selectbox(
                    "Away team",
                    away_options,
                    key="prediction_away",
                )

            if st.button("Calculate goal probabilities"):
                result = calculate_goal_expectations(
                    home_team,
                    away_team,
                    fixtures,
                )

                if result is None:
                    st.warning(
                        "There are not enough completed matches for both "
                        "selected teams in the loaded dataset. Extend the "
                        "history date range and load more fixtures."
                    )
                else:
                    home_xg, away_xg = result
                    matrix = make_score_matrix(home_xg, away_xg)
                    markets = probabilities_from_matrix(matrix)

                    st.markdown("### Expected goals")

                    c1, c2 = st.columns(2)
                    c1.metric(
                        f"{home_team} expected goals",
                        f"{home_xg:.2f}",
                    )
                    c2.metric(
                        f"{away_team} expected goals",
                        f"{away_xg:.2f}",
                    )

                    st.markdown("### Goal-market estimates")

                    market_rows = [
                        {
                            "Market": market,
                            "Probability": probability,
                            "Probability (%)": f"{probability:.1%}",
                        }
                        for market, probability in markets.items()
                    ]

                    market_rows.sort(
                        key=lambda item: item["Probability"],
                        reverse=True,
                    )

                    market_df = pd.DataFrame(market_rows)
                    st.dataframe(
                        market_df[["Market", "Probability (%)"]],
                        use_container_width=True,
                        hide_index=True,
                    )

                    st.markdown("### Highest estimated correct scores")

                    scores = top_correct_scores(matrix, count=10)
                    score_df = pd.DataFrame([
                        {
                            "Correct score": item["Score"],
                            "Probability": f"{item['Probability']:.2%}",
                        }
                        for item in scores
                    ])

                    st.dataframe(
                        score_df,
                        use_container_width=True,
                        hide_index=True,
                    )

                    st.markdown("### Probability matrix")

                    matrix_df = pd.DataFrame(
                        matrix,
                        index=[
                            f"Home {i}"
                            for i in range(matrix.shape[0])
                        ],
                        columns=[
                            f"Away {i}"
                            for i in range(matrix.shape[1])
                        ],
                    )

                    st.dataframe(
                        matrix_df.style.format("{:.1%}"),
                        use_container_width=True,
                    )

                    st.warning(
                        "These probabilities are experimental estimates "
                        "based on a simple Poisson goal model and the "
                        "currently loaded fixture sample. They have not "
                        "been calibrated or backtested. They are not "
                        "guarantees or betting advice."
                    )

                    st.info(
                        "First-half winner, HT/FT, corners, cards, shots, "
                        "offsides, fouls and goalkeeper saves are not "
                        "predicted by this goal-only model. They require "
                        "appropriate historical data and a separately "
                        "validated model."
                    )


# ============================================================
# API diagnostics
# ============================================================

elif page == "API Diagnostics":
    st.subheader("API Connection Diagnostics")

    st.write(
        "Use this page to check whether GFIPS can communicate with the "
        "configured data provider. The API key itself is never displayed."
    )

    st.write("**API host:**", API_BASE)
    st.write(
        "**Secret status:**",
        "Present" if api_key else "Missing",
    )

    if st.button("Test API connection", type="primary"):
        if not api_key:
            st.error(
                "No secret found. In Streamlit Settings > Secrets, add:"
            )
            st.code('FOOTBALL_API_KEY = "your_actual_api_key"')
        else:
            try:
                with st.spinner("Testing the leagues endpoint..."):
                    data, payload = api_request(
                        "/leagues",
                        params={},
                        api_key=api_key,
                    )

                st.success("The provider returned a valid response.")
                st.write("Number of league records returned:", len(data))

                # Show limited metadata only, never the request headers/key.
                st.json({
                    "response_type": type(data).__name__,
                    "records_returned": len(data),
                    "results_field": payload.get("results"),
                    "paging": payload.get("paging"),
                    "errors": payload.get("errors"),
                })

                if data:
                    st.write("Sample response record:")
                    st.json(data[0])

                st.caption(
                    "A successful connection does not mean every endpoint "
                    "or statistic is included in your subscription."
                )

            except Exception as exc:
                st.error(f"API connection test failed: {exc}")

    st.markdown("### Troubleshooting")

    st.markdown(
        """
        - **401 / 403:** check the secret and your API access permissions.
        - **429:** the provider may have rate-limited your requests.
        - **Empty response:** check the endpoint, search parameters,
          dates and competition coverage.
        - **Statistics missing:** not all fixtures or subscription plans
          provide every statistical field.
        - **Unexpected response format:** check the provider's current API
          documentation and response schema.
        """
    )


# ============================================================
# About
# ============================================================

elif page == "About GFIPS":
    st.subheader("About GFIPS")

    st.write(
        "GFIPS — Global Football Intelligence & Prediction System — is "
        "designed to bring football fixtures, statistics, match analysis "
        "and probability modelling into one interface."
    )

    st.markdown("### Current capabilities")

    st.markdown(
        """
        - API connection through a Streamlit secret
        - Fixture and result search
        - Team and league search
        - Match details and available statistics
        - CSV export
        - Experimental goal-based probability estimates
        - Correct-score ranking and probability matrix
        - API connection diagnostics
        """
    )

    st.markdown("### Important limitations")

    st.markdown(
        """
        - Competition coverage depends on the data provider.
        - The goal model is a baseline, not a validated machine-learning model.
        - Prediction quality depends on the quantity and quality of historical data.
        - Markets such as corners, cards and shots need suitable historical data
          and dedicated models before reliable forecasts can be produced.
        - No prediction is certain, and displayed probabilities are not guarantees.
        """
    )

    st.markdown("### Future development")

    st.markdown(
        """
        - Historical backtesting and calibration
        - Team-specific home and away form
        - First-half and half-time/full-time modelling
        - Dedicated corner, card and shot models
        - Model evaluation and reliability tracking
        - Improved league discovery and data pagination
        """
    )


# ============================================================
# Footer
# ============================================================

st.divider()

st.caption(
    "GFIPS · Global Football Intelligence & Prediction System · "
    "Experimental release"
)
