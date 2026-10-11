import math
from datetime import date, datetime, time, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st

API_BASE = "https://api.5dollarfootballapi.com/v1"

st.set_page_config(
    page_title="GFIPS | Football Intelligence",
    page_icon="⚽",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1450px;
        padding-top: 1.2rem;
    }
    [data-testid="stMetric"] {
        background: linear-gradient(
            135deg,
            rgba(15,62,44,.6),
            rgba(20,28,34,.8)
        );
        border: 1px solid #28674d;
        border-radius: 14px;
        padding: 14px;
    }
    .hero {
        padding: 24px;
        border-radius: 18px;
        color: white;
        background: linear-gradient(
            115deg,
            #0a211c,
            #13583e,
            #172e37
        );
        margin-bottom: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title("⚽ GFIPS")
st.sidebar.caption(
    "Global Football Intelligence & Prediction System"
)

page = st.sidebar.radio(
    "Workspace",
    [
        "Overview",
        "Match Explorer",
        "Fixtures & Results",
        "API Diagnostics",
        "About GFIPS",
    ],
)

st.markdown(
    """
    <div class="hero">
        <h1>Global Football Intelligence</h1>
        <p>
        Fixture discovery · Match intelligence · Probability modelling
        · Transparent evidence
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# API CONNECTION
# ---------------------------------------------------------

def api_key():
    """Read the secret without displaying it."""
    try:
        return str(
            st.secrets.get("FOOTBALL_API_KEY", "")
        ).strip()
    except Exception:
        return ""


def api_get(path, params=None):
    """Send a request to the football data provider."""
    key = api_key()

    if not key:
        raise RuntimeError(
            'FOOTBALL_API_KEY is missing. Add '
            'FOOTBALL_API_KEY = "your_actual_key" '
            'in Streamlit Settings → Secrets, then save and reboot.'
        )

    response = requests.get(
        API_BASE + path,
        headers={
            "Authorization": f"Bearer {key}",
            "Accept": "application/json",
        },
        params=params or {},
        timeout=25,
    )

    if response.status_code in (401, 403):
        raise RuntimeError(
            f"API rejected the key or plan "
            f"(HTTP {response.status_code}). "
            "Check your key and subscription."
        )

    response.raise_for_status()
    payload = response.json()

    if payload.get("success") == 0 or "data" not in payload:
        raise RuntimeError(
            str(payload.get("error") or payload)[:500]
        )

    return payload


@st.cache_data(ttl=900, show_spinner=False)
def api_status():
    return api_get("/status")


@st.cache_data(ttl=1800, show_spinner=False)
def search_leagues(query):
    return api_get(
        "/leagues",
        {
            "search": query,
            "per_page": 50,
        },
    ).get("data", [])


@st.cache_data(ttl=600, show_spinner=False)
def fixtures_for_day(day_string, league_id="", status="all"):
    day = date.fromisoformat(day_string)

    start = int(
        datetime.combine(
            day,
            time.min,
            tzinfo=timezone.utc,
        ).timestamp()
    )

    params = {
        "start_time": start,
        "end_time": start + 86400,
        "status": status,
        "per_page": 100,
    }

    if league_id:
        params["league"] = int(league_id)

    return api_get(
        "/fixtures",
        params,
    ).get("data", [])


@st.cache_data(ttl=900, show_spinner=False)
def team_fixtures(team_id):
    return api_get(
        f"/teams/{int(team_id)}/fixtures",
        {
            "status": "finished",
            "page": 1,
            "per_page": 50,
            "order": "desc",
        },
    ).get("data", [])


@st.cache_data(ttl=600, show_spinner=False)
def fixture_details(fixture_id):
    return api_get(
        f"/fixtures/{int(fixture_id)}",
        {
            "include": "events,stats",
        },
    ).get("data", {})


# ---------------------------------------------------------
# HISTORICAL DATA PROCESSING
# ---------------------------------------------------------

def avg(values):
    values = [
        float(value)
        for value in values
        if value is not None
    ]

    return float(np.mean(values)) if values else None


def hist_features(fixtures, team_id, n=10):
    """
    Calculate recent scoring, conceding, corner and yellow-card
    indicators from finished matches returned by the provider.
    """

    scored = []
    conceded = []
    corners_for = []
    corners_against = []
    yellows_for = []
    yellows_against = []

    for fixture in fixtures:
        if fixture.get("status") != "finished":
            continue

        teams = fixture.get("teams") or {}
        goals = fixture.get("goals") or {}

        home = teams.get("home") or {}
        away = teams.get("away") or {}

        corners = fixture.get("corners") or {}
        cards = fixture.get("cards") or {}

        home_cards = cards.get("home") or {}
        away_cards = cards.get("away") or {}

        if home.get("id") == team_id:
            values = [
                goals.get("home"),
                goals.get("away"),
                corners.get("home"),
                corners.get("away"),
                home_cards.get("yellow"),
                away_cards.get("yellow"),
            ]

        elif away.get("id") == team_id:
            values = [
                goals.get("away"),
                goals.get("home"),
                corners.get("away"),
                corners.get("home"),
                away_cards.get("yellow"),
                home_cards.get("yellow"),
            ]

        else:
            continue

        targets = [
            scored,
            conceded,
            corners_for,
            corners_against,
            yellows_for,
            yellows_against,
        ]

        for target, value in zip(targets, values):
            target.append(value)

    def recent(values):
        return [
            value
            for value in values
            if value is not None
        ][-n:]

    return {
        "played": min(n, len(recent(scored))),
        "scored": avg(recent(scored)),
        "conceded": avg(recent(conceded)),
        "corners_for": avg(recent(corners_for)),
        "corners_against": avg(recent(corners_against)),
        "yellow_for": avg(recent(yellows_for)),
        "yellow_against": avg(recent(yellows_against)),
    }


# ---------------------------------------------------------
# GOAL PROBABILITY MODEL
# ---------------------------------------------------------

def poisson_pmf(goals, expected_goals):
    expected_goals = max(0.05, float(expected_goals))

    return (
        math.exp(-expected_goals)
        * expected_goals**goals
        / math.factorial(goals)
    )


def score_matrix(home_xg, away_xg):
    """Create a normalized home-goals by away-goals matrix."""

    home_probabilities = np.array([
        poisson_pmf(i, home_xg)
        for i in range(11)
    ])

    away_probabilities = np.array([
        poisson_pmf(i, away_xg)
        for i in range(11)
    ])

    matrix = np.outer(
        home_probabilities,
        away_probabilities,
    )

    total = matrix.sum()

    if total:
        matrix = matrix / total

    return matrix


# ---------------------------------------------------------
# PREDICTION MARKETS
# ---------------------------------------------------------

def predict_markets(
    home,
    away,
    home_features,
    away_features,
    matrix,
    home_xg,
    away_xg,
):
    rows = []

    p_home = sum(
        matrix[i, j]
        for i in range(11)
        for j in range(11)
        if i > j
    )

    p_draw = sum(
        matrix[i, j]
        for i in range(11)
        for j in range(11)
        if i == j
    )

    p_away = sum(
        matrix[i, j]
        for i in range(11)
        for j in range(11)
        if i < j
    )

    markets = [
        (
            f"{home} win",
            "Match result",
            p_home,
            "Poisson score matrix",
        ),
        (
            "Draw",
            "Match result",
            p_draw,
            "Poisson score matrix",
        ),
        (
            f"{away} win",
            "Match result",
            p_away,
            "Poisson score matrix",
        ),
        (
            "Over 1.5 total goals",
            "Goals",
            sum(
                matrix[i, j]
                for i in range(11)
                for j in range(11)
                if i + j >= 2
            ),
            "Poisson score matrix",
        ),
        (
            "Over 2.5 total goals",
            "Goals",
            sum(
                matrix[i, j]
                for i in range(11)
                for j in range(11)
                if i + j >= 3
            ),
            "Poisson score matrix",
        ),
        (
            "Under 4.5 total goals",
            "Goals",
            sum(
                matrix[i, j]
                for i in range(11)
                for j in range(11)
                if i + j <= 4
            ),
            "Poisson score matrix",
        ),
        (
            "Both teams to score — Yes",
            "Goals",
            sum(
                matrix[i, j]
                for i in range(1, 11)
                for j in range(1, 11)
            ),
            "Poisson score matrix",
        ),
        (
            f"{home} to score 1+",
            "Team goals",
            1 - math.exp(-home_xg),
            f"Expected goals {home_xg:.2f}",
        ),
        (
            f"{away} to score 1+",
            "Team goals",
            1 - math.exp(-away_xg),
            f"Expected goals {away_xg:.2f}",
        ),
    ]

    for name, group, probability, evidence in markets:
        rows.append({
            "Selection": name,
            "Market": group,
            "Probability": float(probability),
            "Evidence": evidence,
            "Data status": "Model estimate",
        })

    # Corners are estimated only when the required historical fields exist.
    if all(
        value is not None
        for value in [
            home_features["corners_for"],
            away_features["corners_for"],
            home_features["corners_against"],
            away_features["corners_against"],
        ]
    ):
        home_corners = (
            home_features["corners_for"]
            + away_features["corners_against"]
        ) / 2

        away_corners = (
            away_features["corners_for"]
            + home_features["corners_against"]
        ) / 2

        total_corners = home_corners + away_corners

        p_over_corners = 1 - sum(
            poisson_pmf(k, total_corners)
            for k in range(9)
        )

        p_home_most_corners = (
            0.5
            + 0.5 * math.erf(
                (home_corners - away_corners)
                / math.sqrt(
                    2 * max(0.1, home_corners + away_corners)
                )
            )
        )

        rows.extend([
            {
                "Selection": "Over 8.5 total corners",
                "Market": "Corners",
                "Probability": float(
                    max(0, min(1, p_over_corners))
                ),
                "Evidence": (
                    f"Estimated total corners "
                    f"{total_corners:.1f}"
                ),
                "Data status": "Historical estimate",
            },
            {
                "Selection": "Most corners: home",
                "Market": "Corners",
                "Probability": float(
                    max(0, min(1, p_home_most_corners))
                ),
                "Evidence": (
                    f"Home {home_corners:.1f} vs "
                    f"away {away_corners:.1f}"
                ),
                "Data status": "Historical estimate",
            },
        ])

    # Yellow-card estimates also require historical card data.
    if all(
        value is not None
        for value in [
            home_features["yellow_for"],
            away_features["yellow_for"],
            home_features["yellow_against"],
            away_features["yellow_against"],
        ]
    ):
        home_yellows = (
            home_features["yellow_for"]
            + away_features["yellow_against"]
        ) / 2

        away_yellows = (
            away_features["yellow_for"]
            + home_features["yellow_against"]
        ) / 2

        total_yellows = home_yellows + away_yellows

        p_under_cards = sum(
            poisson_pmf(k, total_yellows)
            for k in range(7)
        )

        p_home_most_cards = (
            0.5
            + 0.5 * math.erf(
                (home_yellows - away_yellows)
                / math.sqrt(
                    2 * max(0.1, total_yellows)
                )
            )
        )

        rows.extend([
            {
                "Selection": "Under 6.5 total yellow cards",
                "Market": "Cards",
                "Probability": float(p_under_cards),
                "Evidence": (
                    f"Estimated yellow cards "
                    f"{total_yellows:.1f}"
                ),
                "Data status": "Historical estimate",
            },
            {
                "Selection": "Most yellow cards: home",
                "Market": "Cards",
                "Probability": float(
                    max(0, min(1, p_home_most_cards))
                ),
                "Evidence": (
                    f"Home {home_yellows:.1f} vs "
                    f"away {away_yellows:.1f}"
                ),
                "Data status": "Historical estimate",
            },
        ])

    return (
        pd.DataFrame(rows)
        .sort_values("Probability", ascending=False)
        .reset_index(drop=True)
    )


def fixture_label(fixture):
    teams = fixture.get("teams") or {}

    home = (teams.get("home") or {}).get("name", "Home")
    away = (teams.get("away") or {}).get("name", "Away")

    league = (
        fixture.get("league") or {}
    ).get("name", "Competition")

    return (
        f"{fixture.get('kickoff_utc', 'time unknown')} · "
        f"{league} · {home} vs {away} · "
        f"{fixture.get('status', 'unknown')}"
    )


# ---------------------------------------------------------
# OVERVIEW
# ---------------------------------------------------------

if page == "Overview":
    col1, col2, col3 = st.columns(3)

    col1.metric(
        "API provider",
        "5DollarFootballAPI",
    )

    col2.metric(
        "API version",
        "Native /v1",
    )

    col3.metric(
        "Prediction model",
        "Poisson baseline",
    )

    st.subheader("Start analysing a match")

    st.write(
        "Open **Match Explorer**, search for a competition, "
        "load fixtures for a date, select a match, and analyse "
        "recent team history."
    )

    st.info(
        "The app does not present demonstration fixtures as live data. "
        "Markets are only produced when the required statistics exist."
    )

    st.markdown(
        """
        **Current model scope**

        - Goal probabilities use recent scoring and conceding records.
        - Corners and yellow cards are estimated only when historical
          fixture records contain those fields.
        - Historical shots on/off target are not assumed to exist.
        - Probabilities are experimental and require backtesting.
        """
    )


# ---------------------------------------------------------
# MATCH EXPLORER
# ---------------------------------------------------------

elif page == "Match Explorer":
    st.subheader("Match Explorer")

    st.caption(
        "Fixture times are queried in UTC. Select a fixture "
        "and run its analysis."
    )

    col1, col2, col3 = st.columns(3)

    fixture_day = col1.date_input(
        "Fixture date (UTC)",
        date.today(),
    )

    competition_query = col2.text_input(
        "Competition search (optional)",
        placeholder="Premier League",
    )

    status = col3.selectbox(
        "Status",
        ["all", "scheduled", "live", "finished"],
    )

    league = None

    if competition_query.strip():
        try:
            leagues = search_leagues(competition_query)

            if leagues:
                league_options = {
                    (
                        f"{item.get('
