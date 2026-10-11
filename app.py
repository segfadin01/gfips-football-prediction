import math
import json
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

API_BASE_URL = "https://api-football.5dollarfootballapi.com"
API_TIMEOUT = 25


# ------------------------- DESIGN ---------------------------

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #08111f 0%, #0d1727 100%);
        color: #f4f7fb;
    }
    [data-testid="stSidebar"] {
        background: #0a1322;
        border-right: 1px solid #223047;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }
    .gfips-header {
        padding: 24px 26px;
        border-radius: 18px;
        background: linear-gradient(120deg, #12352f, #10253c 65%, #17213c);
        border: 1px solid #294b51;
        margin-bottom: 22px;
    }
    .gfips-title {
        font-size: 34px;
        font-weight: 800;
        letter-spacing: -1px;
        margin: 0;
    }
    .gfips-subtitle {
        color: #b5c7d8;
        margin-top: 7px;
        font-size: 14px;
    }
    .gfips-card {
        padding: 17px;
        border: 1px solid #27374d;
        border-radius: 14px;
        background: #101c2d;
        min-height: 110px;
    }
    .gfips-card-label {
        font-size: 12px;
        color: #9aacc0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .gfips-card-value {
        font-size: 26px;
        font-weight: 800;
        margin-top: 8px;
        color: #ffffff;
    }
    .gfips-card-note {
        font-size: 12px;
        color: #91d6b3;
        margin-top: 5px;
    }
    .section-heading {
        font-size: 21px;
        font-weight: 750;
        margin: 8px 0 14px 0;
    }
    .muted {
        color: #a8b6c8;
        font-size: 13px;
    }
    div[data-testid="stMetric"] {
        background: #101c2d;
        padding: 14px;
        border: 1px solid #27374d;
        border-radius: 12px;
    }
    div[data-testid="stMetricLabel"] {
        color: #a8b6c8;
    }
    div[data-testid="stMetricValue"] {
        color: #ffffff;
    }
    .stButton > button {
        border-radius: 9px;
        font-weight: 650;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------- API HELPERS ------------------------

def get_api_key():
    """Read the API key from Streamlit Secrets; never hardcode it."""
    try:
        return st.secrets["FOOTBALL_API_KEY"]
    except Exception:
        return None


def api_request(endpoint, params=None, use_cache=True):
    """
    Call the API-Football-compatible endpoint.
    Returns a dictionary with response data and error information.
    """
    key = get_api_key()

    if not key:
        return {
            "ok": False,
            "data": [],
            "error": (
                "API key not found. Add FOOTBALL_API_KEY to "
                "Streamlit app Settings → Secrets."
            ),
            "status": None,
        }

    url = f"{API_BASE_URL}/{endpoint.lstrip('/')}"
    headers = {"x-apisports-key": key}

    try:
        response = requests.get(
            url,
            headers=headers,
            params=params or {},
            timeout=API_TIMEOUT,
        )

        try:
            payload = response.json()
        except ValueError:
            payload = {"message": response.text[:500]}

        if not response.ok:
            return {
                "ok": False,
                "data": [],
                "error": payload.get(
                    "message",
                    payload.get("errors", f"HTTP {response.status_code}"),
                ),
                "status": response.status_code,
                "raw": payload,
            }

        errors = payload.get("errors", [])
        if errors:
            return {
                "ok": False,
                "data": payload.get("response", []),
                "error": errors,
                "status": response.status_code,
                "raw": payload,
            }

        return {
            "ok": True,
            "data": payload.get("response", []),
            "error": None,
            "status": response.status_code,
            "raw": payload,
        }

    except requests.Timeout:
        return {
            "ok": False,
            "data": [],
            "error": "The API request timed out. Try again shortly.",
            "status": None,
        }
    except requests.RequestException as exc:
        return {
            "ok": False,
            "data": [],
            "error": str(exc),
            "status": None,
        }


@st.cache_data(ttl=600, show_spinner=False)
def load_leagues():
    return api_request("leagues")


@st.cache_data(ttl=180, show_spinner=False)
def load_fixtures(params):
    return api_request("fixtures", dict(params))


@st.cache_data(ttl=300, show_spinner=False)
def load_teams(params):
    return api_request("teams", dict(params))


@st.cache_data(ttl=180, show_spinner=False)
def load_standings(params):
    return api_request("standings", dict(params))


@st.cache_data(ttl=180, show_spinner=False)
def load_odds(params):
    return api_request("odds", dict(params))


@st.cache_data(ttl=180, show_spinner=False)
def load_fixture_statistics(params):
    return api_request("fixtures/statistics", dict(params))


# --------------------- DATA NORMALIZATION -------------------

def safe_dict(value):
    return value if isinstance(value, dict) else {}


def safe_list(value):
    return value if isinstance(value, list) else []


def extract_league(item):
    item = safe_dict(item)
    league = safe_dict(item.get("league"))
    country = safe_dict(item.get("country"))
    seasons = safe_list(item.get("seasons"))

    current_season = next(
        (s for s in seasons if safe_dict(s).get("current") is True),
        seasons[0] if seasons else {},
    )

    return {
        "id": league.get("id"),
        "name": league.get("name", "Unknown competition"),
        "country": country.get("name", "Unknown country"),
        "season": safe_dict(current_season).get("year"),
        "seasons": seasons,
        "raw": item,
    }


def normalize_fixture(item):
    item = safe_dict(item)
    fixture = safe_dict(item.get("fixture"))
    league = safe_dict(item.get("league"))
    teams = safe_dict(item.get("teams"))
    home = safe_dict(teams.get("home"))
    away = safe_dict(teams.get("away"))
    goals = safe_dict(item.get("goals"))
    score = safe_dict(item.get("score"))
    status = safe_dict(fixture.get("status"))

    return {
        "fixture_id": fixture.get("id"),
        "date": fixture.get("date"),
        "timestamp": fixture.get("timestamp"),
        "venue": safe_dict(fixture.get("venue")).get("name"),
        "status": status.get("long", status.get("short", "Unknown")),
        "status_short": status.get("short", ""),
        "elapsed": status.get("elapsed"),
        "league_id": league.get("id"),
        "league": league.get("name", "Unknown competition"),
        "country": league.get("country", ""),
        "season": league.get("season"),
        "home_id": home.get("id"),
        "home": home.get("name", "Home team"),
        "home_logo": home.get("logo"),
        "home_winner": home.get("winner"),
        "away_id": away.get("id"),
        "away": away.get("name", "Away team"),
        "away_logo": away.get("logo"),
        "away_winner": away.get("winner"),
        "home_goals": goals.get("home"),
        "away_goals": goals.get("away"),
        "halftime": score.get("halftime", {}),
        "raw": item,
    }


def fixture_dataframe(items):
    return pd.DataFrame([normalize_fixture(x) for x in items])


def parse_api_error(result):
    error = result.get("error")
    if isinstance(error, (dict, list)):
        return json.dumps(error, ensure_ascii=False)
    return str(error or "Unknown API error")


def display_api_error(result, context="API request"):
    if not result.get("ok"):
        st.warning(f"{context}: {parse_api_error(result)}")


def format_kickoff(value):
    if not value:
        return "Date unavailable"
    try:
        return pd.to_datetime(value, utc=True).strftime("%d %b %Y · %H:%M UTC")
    except Exception:
        return str(value)


def status_label(fixture):
    short = str(fixture.get("status_short", "")).upper()
    if short in {"1H", "2H", "HT", "ET", "P", "LIVE"}:
        return "🔴 LIVE"
    if short in {"FT", "AET", "PEN"}:
        return "🏁 FINISHED"
    if short in {"PST", "CANC", "ABD", "AWD", "WO"}:
        return "⚠️ " + str(fixture.get("status", "Unavailable"))
    return "🗓️ " + str(fixture.get("status", "Scheduled"))


def result_text(fixture):
    hg = fixture.get("home_goals")
    ag = fixture.get("away_goals")
    if hg is None or ag is None:
        return "vs"
    return f"{hg} – {ag}"


# ----------------------- POISSON MODEL ----------------------

def poisson_probability(goals, expected_goals):
    """Poisson probability for a specified number of goals."""
    if expected_goals < 0:
        return 0.0
    return (
        math.exp(-expected_goals)
        * (expected_goals ** goals)
        / math.factorial(goals)
    )


def build_goal_matrix(home_xg, away_xg, max_goals=8):
    matrix = np.zeros((max_goals + 1, max_goals + 1))
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            matrix[h, a] = (
                poisson_probability(h, home_xg)
                * poisson_probability(a, away_xg)
            )
    total = matrix.sum()
    if total > 0:
        matrix = matrix / total
    return matrix


def calculate_markets(home_xg, away_xg):
    matrix = build_goal_matrix(home_xg, away_xg)
    home_win = float(np.tril(matrix, -1).sum())
    draw = float(np.trace(matrix))
    away_win = float(np.triu(matrix, 1).sum())

    total_goals = np.add.outer(
        np.arange(matrix.shape[0]),
        np.arange(matrix.shape[1]),
    )

    over_15 = float(matrix[total_goals >= 2].sum())
    over_25 = float(matrix[total_goals >= 3].sum())
    over_35 = float(matrix[total_goals >= 4].sum())
    under_25 = 1.0 - over_25
    btts = float(matrix[1:, 1:].sum())

    score_probs = []
    for h in range(matrix.shape[0]):
        for a in range(matrix.shape[1]):
            score_probs.append((h, a, float(matrix[h, a])))

    score_probs.sort(key=lambda x: x[2], reverse=True)
    most_likely = score_probs[:5]

    return {
        "home_win": home_win,
        "draw": draw,
        "away_win": away_win,
        "over_15": over_15,
        "over_25": over_25,
        "over_35": over_35,
        "under_25": under_25,
        "btts": btts,
        "most_likely_scores": most_likely,
    }


def pct(value):
    return f"{value * 100:.1f}%"


# ------------------------- SIDEBAR --------------------------

with st.sidebar:
    st.markdown("## ⚽ GFIPS")
    st.caption("Global Football Intelligence & Prediction System")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Intelligence Dashboard",
            "Global Fixtures",
            "Match Centre",
            "Odds & Markets",
            "Prediction Lab",
            "Historical Analytics",
            "API & Settings",
        ],
    )

    st.divider()
    st.markdown("### Match search")

    search_team = st.text_input(
        "Team name",
        placeholder="e.g. Arsenal",
        key="sidebar_team_search",
    )

    st.caption("Data source: 5DollarFootballAPI")


# ---------------------- SHARED DATA ------------------------

leagues_result = load_leagues()
league_records = [
    extract_league(x) for x in safe_list(leagues_result.get("data"))
]
league_records = [
    x for x in league_records if x.get("id") is not None
]

league_lookup = {}
for item in league_records:
    label = f"{item['name']} · {item['country']} · {item['id']}"
    league_lookup[label] = item

default_from = date.today()
default_to = date.today() + timedelta(days=7)

with st.sidebar:
    st.markdown("### Competition & dates")

    if league_lookup:
        league_labels = list(league_lookup.keys())
        selected_league_label = st.selectbox(
            "Competition",
            league_labels,
            index=0,
        )
        selected_league = league_lookup[selected_league_label]
    else:
        selected_league = None
        st.info("Competition list unavailable. Check API diagnostics.")

    start_date = st.date_input("From", value=default_from)
    end_date = st.date_input("To", value=default_to)

    if end_date < start_date:
        st.error("The end date must be on or after the start date.")
        end_date = start_date

    refresh = st.button("↻ Refresh data", use_container_width=True)
    if refresh:
        st.cache_data.clear()
        st.rerun()


def selected_season():
    if selected_league:
        return selected_league.get("season")
    return None


def selected_fixture_params():
    if not selected_league:
        return None
    params = {
        "league": selected_league["id"],
        "from": start_date.isoformat(),
        "to": end_date.isoformat(),
    }
    season = selected_season()
    if season is not None:
        params["season"] = season
    return params


@st.cache_data(ttl=180, show_spinner=False)
def fetch_selected_fixtures(params):
    if not params:
        return {"ok": False, "data": [], "error": "No competition selected."}
    return api_request("fixtures", dict(params))


selected_fixtures_result = fetch_selected_fixtures(
    selected_fixture_params() or {}
)
selected_fixture_items = safe_list(selected_fixtures_result.get("data"))
selected_fixtures_df = fixture_dataframe(selected_fixture_items)


# ------------------------ APP HEADER ------------------------

st.markdown(
    """
    <div class="gfips-header">
        <div class="gfips-title">GFIPS <span style="color:#69e0ad;">2.0</span></div>
        <div class="gfips-subtitle">
            Global football intelligence · Match analytics · Probability modelling
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    f"Competition: {selected_league['name'] if selected_league else 'Unavailable'}"
    f"  |  Period: {start_date} to {end_date}"
)


# ============================================================
# PAGE 1 — INTELLIGENCE DASHBOARD
# ============================================================

if page == "Intelligence Dashboard":
    st.markdown('<div class="section-heading">Intelligence Overview</div>',
                unsafe_allow_html=True)

    if not selected_fixtures_result.get("ok"):
        display_api_error(selected_fixtures_result, "Fixtures")
    else:
        total_fixtures = len(selected_fixture_items)
        live_count = sum(
            1 for x in selected_fixture_items
            if str(normalize_fixture(x).get("status_short", "")).upper()
            in {"1H", "2H", "HT", "ET", "P", "LIVE"}
        )
        finished_count = sum(
            1 for x in selected_fixture_items
            if str(normalize_fixture(x).get("status_short", "")).upper()
            in {"FT", "AET", "PEN"}
        )
        upcoming_count = max(0, total_fixtures - live_count - finished_count)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Fixtures returned", total_fixtures)
        with c2:
            st.metric("Live matches", live_count)
        with c3:
            st.metric("Finished", finished_count)
        with c4:
            st.metric("Upcoming / other", upcoming_count)

    st.divider()
    st.markdown("### Selected competition fixtures")

    if selected_fixtures_df.empty:
        st.info(
            "No fixtures were returned for this competition and date range. "
            "Try a different date range or competition."
        )
    else:
        show_cols = [
            "date", "home", "home_goals", "away_goals",
            "away", "status", "venue"
        ]
        show_cols = [c for c in show_cols if c in selected_fixtures_df.columns]
        dashboard_df = selected_fixtures_df[show_cols].copy()
        dashboard_df["date"] = dashboard_df["date"].apply(format_kickoff)
        st.dataframe(dashboard_df, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("### Competition coverage")

    if league_records:
        league_df = pd.DataFrame([
            {
                "Competition": x["name"],
                "Country": x["country"],
                "League ID": x["id"],
                "Current season": x["season"],
            }
            for x in league_records
        ])
        st.dataframe(league_df, use_container_width=True, hide_index=True)
    else:
        display_api_error(leagues_result, "Leagues")


# ============================================================
# PAGE 2 — GLOBAL FIXTURES
# ============================================================

elif page == "Global Fixtures":
    st.markdown('<div class="section-heading">Global Fixtures Explorer</div>',
                unsafe_allow_html=True)

    if not selected_fixtures_result.get("ok"):
        display_api_error(selected_fixtures_result, "Fixtures")
    elif selected_fixtures_df.empty:
        st.info("No fixtures found for the selected competition and dates.")
    else:
        df = selected_fixtures_df.copy()

        status_filter = st.selectbox(
            "Filter by match status",
            ["All", "Upcoming", "Live", "Finished"],
        )

        live_codes = {"1H", "2H", "HT", "ET", "P", "LIVE"}
        finished_codes = {"FT", "AET", "PEN"}

        if status_filter == "Live":
            df = df[df["status_short"].astype(str).str.upper().isin(live_codes)]
        elif status_filter == "Finished":
            df = df[df["status_short"].astype(str).str.upper().isin(finished_codes)]
        elif status_filter == "Upcoming":
            df = df[
                ~df["status_short"].astype(str).str.upper().isin(
                    live_codes | finished_codes
                )
            ]

        df["Kick-off (UTC)"] = df["date"].apply(format_kickoff)
        df["Score"] = df.apply(
            lambda row: (
                f"{row['home_goals']} – {row['away_goals']}"
                if pd.notna(row["home_goals"]) and pd.notna(row["away_goals"])
                else "vs"
            ),
            axis=1,
        )
        df["Match"] = df["home"].astype(str) + "  vs  " + df["away"].astype(str)

        display_df = df[
            ["Kick-off (UTC)", "Match", "Score", "status", "venue"]
        ].rename(columns={"status": "Status", "venue": "Venue"})

        st.dataframe(display_df, use_container_width=True, hide_index=True)

        csv_data = display_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download fixtures CSV",
            data=csv_data,
            file_name="gfips_fixtures.csv",
            mime="text/csv",
        )


# ============================================================
# PAGE 3 — MATCH CENTRE
# ============================================================

elif page == "Match Centre":
    st.markdown('<div class="section-heading">Match Centre</div>',
                unsafe_allow_html=True)

    if not selected_fixtures_result.get("ok"):
        display_api_error(selected_fixtures_result, "Fixtures")
    elif not selected_fixture_items:
        st.info("No fixtures are available to open in Match Centre.")
    else:
        normalized = [
            normalize_fixture(x) for x in selected_fixture_items
        ]
        fixture_options = {}
        for f in normalized:
            label = (
                f"{format_kickoff(f['date'])} | "
                f"{f['home']} vs {f['away']} | "
                f"ID {f['fixture_id']}"
            )
            fixture_options[label] = f

        selected_match_label = st.selectbox(
            "Select a match",
            list(fixture_options.keys()),
        )
        match = fixture_options[selected_match_label]

        st.markdown(f"## {match['home']}  vs  {match['away']}")
        st.caption(
            f"{match['league']} · {format_kickoff(match['date'])} · "
            f"{status_label(match)}"
        )

        left, middle, right = st.columns([2, 1, 2])
        with left:
            st.markdown(f"### 🏠 {match['home']}")
        with middle:
            st.markdown(
                f"<h2 style='text-align:center;'>{result_text(match)}</h2>",
                unsafe_allow_html=True,
            )
        with right:
            st.markdown(f"### ✈️ {match['away']}")

        st.markdown("### Match information")
        info1, info2, info3 = st.columns(3)
        info1.metric("Fixture ID", match["fixture_id"] or "Unavailable")
        info2.metric("Venue", match["venue"] or "Not provided")
        info3.metric("Season", match["season"] or "Not provided")

        st.divider()
        st.markdown("### Match statistics")

        if match["fixture_id"]:
            if st.button("Load match statistics"):
                with st.spinner("Requesting match statistics..."):
                    stats_result = load_fixture_statistics(
                        {"fixture": match["fixture_id"]}
                    )

                if not stats_result.get("ok"):
                    display_api_error(stats_result, "Match statistics")
                else:
                    stats_items = safe_list(stats_result.get("data"))
                    if not stats_items:
                        st.info(
                            "No match statistics were returned. "
                            "Statistics may be unavailable for this match "
                            "or unsupported by your API plan."
                        )
                    else:
                        rows = []
                        for team_stats in stats_items:
                            team = safe_dict(team_stats.get("team"))
                            for stat in safe_list(team_stats.get("statistics")):
                                stat = safe_dict(stat)
                                rows.append({
                                    "Team": team.get("name", "Unknown"),
                                    "Statistic": stat.get("type", "Unknown"),
                                    "Value": stat.get("value"),
                                })
                        if rows:
                            stats_df = pd.DataFrame(rows)
                            pivot = stats_df.pivot_table(
                                index="Statistic",
                                columns="Team",
                                values="Value",
                                aggfunc="first",
                            ).reset_index()
                            st.dataframe(
                                pivot, use_container_width=True, hide_index=True
                            )
                        else:
                            st.info("The API returned no usable statistic rows.")
        else:
            st.info("This fixture does not have a usable fixture ID.")


# ============================================================
# PAGE 4 — ODDS & MARKETS
# ============================================================

elif page == "Odds & Markets":
    st.markdown('<div class="section-heading">Odds & Markets Intelligence</div>',
                unsafe_allow_html=True)

    st.info(
        "Odds are supplied by the data provider when available. "
        "Coverage depends on competition, match, bookmaker and subscription plan. "
        "GFIPS does not guarantee that odds or markets are available for every fixture."
    )

    if not selected_fixture_items:
        st.warning("Load fixtures by selecting a competition and date range.")
    else:
        normalized = [
            normalize_fixture(x) for x in selected_fixture_items
            if normalize_fixture(x).get("fixture_id") is not None
        ]
        options = {
            (
                f"{x['home']} vs {x['away']} · "
                f"{format_kickoff(x['date'])} · ID {x['fixture_id']}"
            ): x
            for x in normalized
        }

        if options:
            odds_match_label = st.selectbox(
                "Select fixture for odds",
                list(options.keys()),
                key="odds_fixture_select",
            )
            odds_match = options[odds_match_label]

            if st.button("Fetch available odds", type="primary"):
                with st.spinner("Checking available odds..."):
                    odds_result = load_odds(
                        {"fixture": odds_match["fixture_id"]}
                    )

                if not odds_result.get("ok"):
                    display_api_error(odds_result, "Odds request")
                else:
                    odds_items = safe_list(odds_result.get("data"))
                    if not odds_items:
                        st.warning(
                            "No odds were returned for this fixture. "
                            "This may mean odds are unavailable for the match "
                            "or not included in your plan."
                        )
                    else:
                        st.success(f"Received {len(odds_items)} odds record(s).")

                        # Preserve the provider response so unsupported
                        # market structures are not silently misrepresented.
                        st.json(odds_items)

                        st.download_button(
                            "Download raw odds JSON",
                            data=json.dumps(
                                odds_items, ensure_ascii=False, indent=2
                            ),
                            file_name=f"gfips_odds_{odds_match['fixture_id']}.json",
                            mime="application/json",
                        )


# ============================================================
# PAGE 5 — PREDICTION LAB
# ============================================================

elif page == "Prediction Lab":
    st.markdown('<div class="section-heading">Prediction Lab</div>',
                unsafe_allow_html=True)

    st.write(
        "Explore goal-market probabilities using an independent Poisson model. "
        "Enter expected goals for each team. These inputs are assumptions, not "
        "official predictions or verified team-strength estimates."
    )

    st.warning(
        "Model limitation: this basic version does not automatically train on "
        "historical results. Do not interpret these estimates as betting advice "
        "or guaranteed outcomes."
    )

    p1, p2 = st.columns(2)
    with p1:
        home_name = st.text_input("Home team", value="Home team")
        home_xg = st.slider(
            "Home expected goals",
            min_value=0.1,
            max_value=4.5,
            value=1.5,
            step=0.1,
        )
    with p2:
        away_name = st.text_input("Away team", value="Away team")
        away_xg = st.slider(
            "Away expected goals",
            min_value=0.1,
            max_value=4.5,
            value=1.1,
            step=0.1,
        )

    markets = calculate_markets(home_xg, away_xg)

    st.divider()
    st.markdown("### 1X2 probabilities")
    c1, c2, c3 = st.columns(3)
    c1.metric(f"{home_name} win", pct(markets["home_win"]))
    c2.metric("Draw", pct(markets["draw"]))
    c3.metric(f"{away_name} win", pct(markets["away_win"]))

    st.markdown("### Goals & BTTS markets")
    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Over 1.5 goals", pct(markets["over_15"]))
    g2.metric("Over 2.5 goals", pct(markets["over_25"]))
    g3.metric("Under 2.5 goals", pct(markets["under_25"]))
    g4.metric("Both teams to score", pct(markets["btts"]))

    st.metric("Over 3.5 goals", pct(markets["over_35"]))

    st.divider()
    st.markdown("### Most likely scorelines")

    score_rows = [
        {
            "Scoreline": f"{home_name} {h} – {a} {away_name}",
            "Model probability": pct(prob),
        }
        for h, a, prob in markets["most_likely_scores"]
    ]
    st.dataframe(
        pd.DataFrame(score_rows),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "The Poisson distribution is truncated at 8 goals per team and "
        "renormalized. Real match outcomes may differ substantially."
    )


# ============================================================
# PAGE 6 — HISTORICAL ANALYTICS
# ============================================================

elif page == "Historical Analytics":
    st.markdown('<div class="section-heading">Historical Analytics</div>',
                unsafe_allow_html=True)

    st.write(
        "Review results returned by the provider for the selected competition "
        "and date range. This page analyses only the records currently loaded."
    )

    if not selected_fixtures_result.get("ok"):
        display_api_error(selected_fixtures_result, "Historical fixtures")
    elif selected_fixtures_df.empty:
        st.info(
            "No fixture records were returned. Choose a historical date range "
            "supported by your subscription and competition."
        )
    else:
        df = selected_fixtures_df.copy()
        df["home_goals"] = pd.to_numeric(df["home_goals"], errors="coerce")
        df["away_goals"] = pd.to_numeric(df["away_goals"], errors="coerce")

        finished_codes = {"FT", "AET", "PEN"}
        completed = df[
            df["status_short"].astype(str).str.upper().isin(finished_codes)
            & df["home_goals"].notna()
            & df["away_goals"].notna()
        ].copy()

        if completed.empty:
            st.info(
                "No completed matches with scores were returned in this range."
            )
        else:
            completed["total_goals"] = (
                completed["home_goals"] + completed["away_goals"]
            )
            completed["btts"] = (
                (completed["home_goals"] > 0)
                & (completed["away_goals"] > 0)
            )
            completed["over_25"] = completed["total_goals"] > 2.5
            completed["result"] = np.select(
                [
                    completed["home_goals"] > completed["away_goals"],
                    completed["home_goals"] == completed["away_goals"],
                    completed["home_goals"] < completed["away_goals"],
                ],
                ["Home win", "Draw", "Away win"],
                default="Unknown",
            )

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Completed matches", len(completed))
            m2.metric(
                "Average total goals",
                f"{completed['total_goals'].mean():.2f}",
            )
            m3.metric(
                "Over 2.5 rate",
                pct(float(completed["over_25"].mean())),
            )
            m4.metric(
                "BTTS rate",
                pct(float(completed["btts"].mean())),
            )

            st.markdown("### Results distribution")
            result_counts = (
                completed["result"]
                .value_counts()
                .rename_axis("Result")
                .reset_index(name="Matches")
            )
            st.bar_chart(result_counts.set_index("Result"))

            st.markdown("### Goal distribution")
            goal_counts = (
                completed["total_goals"]
                .value_counts()
                .sort_index()
                .rename_axis("Total goals")
                .reset_index(name="Matches")
            )
            st.bar_chart(goal_counts.set_index("Total goals"))

            st.markdown("### Completed match records")
            result_view = completed[
                ["date", "home", "home_goals", "away_goals", "away", "total_goals"]
            ].copy()
            result_view["date"] = result_view["date"].apply(format_kickoff)
            st.dataframe(
                result_view,
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "Download historical results CSV",
                data=result_view.to_csv(index=False).encode("utf-8"),
                file_name="gfips_historical_results.csv",
                mime="text/csv",
            )


# ============================================================
# PAGE 7 — API & SETTINGS
# ============================================================

elif page == "API & Settings":
    st.markdown('<div class="section-heading">API Health & Settings</div>',
                unsafe_allow_html=True)

    key = get_api_key()

    c1, c2 = st.columns(2)
    with c1:
        st.metric("API key configured", "Yes" if key else "No")
    with c2:
        st.metric("API host", "5DollarFootballAPI")

    st.caption(
        "Your API key is read from Streamlit Secrets. It is not displayed "
        "on this page."
    )

    st.markdown("### Test API connection")

    if st.button("Run connection diagnostics", type="primary"):
        with st.spinner("Testing the leagues endpoint..."):
            diagnostic = api_request("leagues")

        if diagnostic.get("ok"):
            st.success("The API request succeeded.")
            st.write(f"Records returned: {len(safe_list(diagnostic.get('data')))}")

            raw = safe_dict(diagnostic.get("raw"))
            if raw:
                st.write("Provider metadata")
                st.json({
                    "results": raw.get("results"),
                    "paging": raw.get("paging"),
                    "errors": raw.get("errors"),
                })
        else:
            st.error("The API request failed.")
            st.code(parse_api_error(diagnostic))

            st.markdown(
                """
                **Troubleshooting**
                - Confirm `FOOTBALL_API_KEY` exists in Streamlit Secrets.
                - Check that the key is active and your subscription is valid.
                - Check whether the requested endpoint is supported by your plan.
                - If the error mentions a required `league` or `team`, the API
                  needs one of those filters for that request.
                - If the error is HTTP 429, you may have reached a rate limit.
                """
            )

    st.divider()
    st.markdown("### Current configuration")
    st.json({
        "provider": "5DollarFootballAPI",
        "api_compatibility_host": API_BASE_URL,
        "fixtures_endpoint": "/fixtures",
        "leagues_endpoint": "/leagues",
        "teams_endpoint": "/teams",
        "statistics_endpoint": "/fixtures/statistics",
        "odds_endpoint": "/odds",
        "key_storage": "Streamlit Secrets",
        "prediction_method": "Independent Poisson goal model",
    })

    st.markdown("### Data and model transparency")
    st.write(
        "- Fixtures, scores, statistics and odds are provider data when returned.\n"
        "- Historical summaries use the fixture records actually received.\n"
        "- Prediction Lab percentages are model estimates based on entered "
        "expected-goal values.\n"
        "- Odds and statistics can be unavailable depending on coverage and plan.\n"
        "- The current version does not automatically train or backtest a "
        "machine-learning model."
    )


# -------------------------- FOOTER --------------------------

st.divider()
st.markdown(
    """
    <div style="text-align:center;color:#91a2b8;font-size:12px;padding:8px;">
        <b>GFIPS 2.0</b> · Global Football Intelligence & Prediction System<br>
        Football data by 5DollarFootballAPI · Model estimates are not guarantees.
    </div>
    """,
    unsafe_allow_html=True,
)
