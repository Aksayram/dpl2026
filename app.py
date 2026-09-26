"""
Over Utilization | Data Premier League 2026
Where captains actually use each bowler across the four phases of a T20 innings.

Run locally:  streamlit run app.py
Data:         put Data_Premier_League_Prelims_Dataset.csv next to this file,
              or upload it from the sidebar.
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Config — edit these to change the metric
# ---------------------------------------------------------------------------
DATA_FILE = Path(__file__).parent / "Data_Premier_League_Prelims_Dataset.csv"

# (name, first over, last over)
PHASES = [("PP", 1, 6), ("Mid 1", 7, 11), ("Mid 2", 12, 16), ("Death", 17, 20)]
PHASE_NAMES = [p[0] for p in PHASES]
PHASE_RANGES = {p[0]: f"{p[1]}–{p[2]}" for p in PHASES}

# Innings runs from early (cool) to late (hot): colour follows the clock.
PHASE_COLORS = {"PP": "#6FA8B8", "Mid 1": "#3D7A5F", "Mid 2": "#D29B3C", "Death": "#A4243B"}

DEFAULT_CUTOFF = 15.0   # % of a bowler's overs a phase needs to count as "used"
DEFAULT_MIN_OVERS = 20  # sample-size floor

# Your role names. Key = phases at/above the cutoff, joined with "+",
# always in PP, Mid 1, Mid 2, Death order. Anything not listed shows as the
# raw combination (e.g. "PP + Death").
LABEL_NAMES = {
    "PP+Mid 1+Mid 2+Death": "1-to-20 Bowler",
    # "PP+Death": "",
    # "PP+Mid 2+Death": "",
    # "Mid 1+Mid 2": "",
    # "PP+Mid 1+Mid 2": "",
    # "Mid 1+Mid 2+Death": "",
    # "PP": "",
}

USE_COLS = ["p_match", "over_num", "wide", "noball", "p_bowl", "bowl",
            "team_bowl", "year", "bowl_kind"]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading deliveries…")
def load_deliveries(source) -> pd.DataFrame:
    df = pd.read_csv(
        source,
        usecols=USE_COLS,
        dtype={"bowl": "category", "team_bowl": "category", "bowl_kind": "category",
               "over_num": "int8", "wide": "int8", "noball": "int8", "year": "int16"},
    )
    # Legal balls only: wides and no-balls don't use up the over.
    df = df[(df["wide"] == 0) & (df["noball"] == 0)].copy()
    df["phase"] = pd.cut(df["over_num"], bins=[0, 6, 11, 16, 20], labels=PHASE_NAMES)
    return df.drop(columns=["wide", "noball"])


def role_label(combo: str) -> str:
    if not combo:
        return "No phase at cutoff"
    return LABEL_NAMES.get(combo) or combo.replace("+", " + ")


def utilization(df: pd.DataFrame, cutoff: float) -> pd.DataFrame:
    """One row per bowler: overs, % of legal balls in each phase, role label."""
    if df.empty:
        return pd.DataFrame()

    balls = (df.groupby(["p_bowl", "phase"], observed=False).size()
               .unstack(fill_value=0).reindex(columns=PHASE_NAMES, fill_value=0))
    balls = balls[balls.sum(axis=1) > 0]
    total = balls.sum(axis=1)
    pct = balls.div(total, axis=0) * 100

    g = df.groupby("p_bowl", observed=True)
    info = pd.DataFrame({
        "Bowler": g["bowl"].agg(lambda s: s.mode().iat[0]),
        "Team": g["team_bowl"].agg(lambda s: " / ".join(sorted(s.astype(str).unique()))),
        "Type": g["bowl_kind"].agg(lambda s: s.mode().iat[0] if s.notna().any() else "unknown"),
        "Matches": g["p_match"].nunique(),
    })

    out = info.join(pd.DataFrame({"Overs": total / 6}))
    for p in PHASE_NAMES:
        out[f"{p} %"] = pct[p]
        out[f"{p} overs"] = balls[p] / 6

    combo = pct.apply(lambda r: "+".join(p for p in PHASE_NAMES if r[p] >= cutoff), axis=1)
    out["Phases used"] = combo.map(lambda c: c.count("+") + 1 if c else 0)
    out["Role"] = combo.map(role_label)
    out["Main phase"] = pct.idxmax(axis=1)
    return out.reset_index()


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def _base_layout(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, title=None),
        hoverlabel=dict(bgcolor="white"),
    )
    return fig


def role_map(table: pd.DataFrame) -> go.Figure:
    """100% stacked bar per bowler: how his overs split across the innings."""
    t = table.sort_values(["Death %", "PP %"], ascending=[True, False])
    fig = go.Figure()
    for p in PHASE_NAMES:
        fig.add_bar(
            y=t["Bowler"], x=t[f"{p} %"], name=f"{p} ({PHASE_RANGES[p]})",
            orientation="h", marker_color=PHASE_COLORS[p],
            customdata=t[[f"{p} overs", "Overs", "Role"]],
            hovertemplate=("<b>%{y}</b><br>" + p + ": %{x:.1f}% "
                           "(%{customdata[0]:.1f} of %{customdata[1]:.1f} overs)"
                           "<br>%{customdata[2]}<extra></extra>"),
        )
    fig.update_layout(barmode="stack", bargap=0.25)
    fig.update_xaxes(range=[0, 100], ticksuffix="%", showgrid=False)
    fig.update_yaxes(showgrid=False)
    return _base_layout(fig, max(320, 26 * len(t) + 80))


def phase_profile(row: pd.Series, cutoff: float) -> go.Figure:
    vals = [row[f"{p} %"] for p in PHASE_NAMES]
    colors = [PHASE_COLORS[p] if v >= cutoff else "#C9CFCB" for p, v in zip(PHASE_NAMES, vals)]
    fig = go.Figure(go.Bar(
        x=[f"{p}<br>overs {PHASE_RANGES[p]}" for p in PHASE_NAMES], y=vals,
        marker_color=colors, text=[f"{v:.1f}%" for v in vals], textposition="outside",
        customdata=[row[f"{p} overs"] for p in PHASE_NAMES],
        hovertemplate="%{y:.1f}% (%{customdata:.1f} overs)<extra></extra>",
    ))
    fig.add_hline(y=cutoff, line_dash="dash", line_color="#1C2B24",
                  annotation_text=f"{cutoff:g}% cutoff", annotation_position="top left")
    fig.update_yaxes(range=[0, max(vals + [cutoff]) * 1.2], ticksuffix="%", showgrid=False)
    return _base_layout(fig, 360)


def compare_chart(table: pd.DataFrame, names: list[str], cutoff: float) -> go.Figure:
    fig = go.Figure()
    for name in names:
        r = table.loc[table["Bowler"] == name].iloc[0]
        fig.add_bar(name=name, x=PHASE_NAMES, y=[r[f"{p} %"] for p in PHASE_NAMES],
                    hovertemplate=name + ": %{y:.1f}%<extra></extra>")
    fig.add_hline(y=cutoff, line_dash="dash", line_color="#1C2B24")
    fig.update_layout(barmode="group", colorway=["#1C2B24", "#A4243B", "#3D7A5F", "#D29B3C"])
    fig.update_yaxes(ticksuffix="%", showgrid=False)
    return _base_layout(fig, 380)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Over Utilization | DPL 2026", page_icon="🏏", layout="wide")

st.title("Over Utilization")
st.caption("Where captains actually use each bowler. Share of a bowler's legal balls "
           "bowled in each phase of the innings.")

with st.sidebar:
    st.header("Data")
    if DATA_FILE.exists():
        source = DATA_FILE
        st.caption(f"Using `{DATA_FILE.name}`")
    else:
        source = st.file_uploader("Upload the DPL dataset (CSV)", type="csv")
        if source is None:
            st.info("Upload Data_Premier_League_Prelims_Dataset.csv to start.")
            st.stop()

    deliveries = load_deliveries(source)

    st.header("Filters")
    seasons = sorted(deliveries["year"].unique())
    pick_seasons = st.multiselect("Season", seasons, default=seasons)
    teams = sorted(deliveries["team_bowl"].astype(str).unique())
    pick_teams = st.multiselect("Team", teams, placeholder="All teams")
    kinds = sorted(deliveries["bowl_kind"].dropna().astype(str).unique())
    pick_kinds = st.multiselect("Bowler type", kinds, placeholder="All types")

    st.header("Metric")
    min_overs = st.slider("Minimum overs", 5, 60, DEFAULT_MIN_OVERS, step=5)
    cutoff = st.number_input("Phase cutoff (%)", 5.0, 30.0, DEFAULT_CUTOFF, step=1.0)

f = deliveries[deliveries["year"].isin(pick_seasons)]
if pick_teams:
    f = f[f["team_bowl"].astype(str).isin(pick_teams)]
if pick_kinds:
    f = f[f["bowl_kind"].astype(str).isin(pick_kinds)]

table = utilization(f, cutoff)
if table.empty:
    st.warning("No deliveries match these filters. Widen the season, team or bowler type.")
    st.stop()
table = table[table["Overs"] >= min_overs].reset_index(drop=True)
if table.empty:
    st.warning(f"No bowler has {min_overs}+ overs under these filters. Lower the minimum overs.")
    st.stop()

# Summary strip
all_phase = (table["Phases used"] == 4).sum()
c1, c2, c3 = st.columns(3)
c1.metric("Bowlers qualifying", len(table), help=f"{min_overs}+ overs")
c2.metric("1-to-20 bowlers", all_phase, help=f"{cutoff:g}%+ of overs in every phase")
c3.metric("Most common role", table["Role"].mode().iat[0])

tab_map, tab_profile, tab_compare, tab_table = st.tabs(
    ["Role map", "Bowler profile", "Compare", "Full table"])

with tab_map:
    counts = table["Role"].value_counts()
    left, right = st.columns([1, 3])
    with left:
        st.subheader("Roles")
        st.dataframe(counts.rename("Bowlers").to_frame(), width="stretch")
        role_pick = st.selectbox("Show role", ["All roles"] + counts.index.tolist())
    with right:
        shown = table if role_pick == "All roles" else table[table["Role"] == role_pick]
        st.subheader(f"{role_pick} ({len(shown)})")
        st.plotly_chart(role_map(shown), width="stretch")

with tab_profile:
    names = table.sort_values("Overs", ascending=False)["Bowler"].tolist()
    who = st.selectbox("Bowler", names)
    row = table.loc[table["Bowler"] == who].iloc[0]

    a, b = st.columns([2, 1])
    with a:
        st.subheader(row["Role"])
        st.caption(f"{row['Team']} | {row['Type']} | {row['Overs']:.1f} overs in "
                   f"{row['Matches']} matches. Grey bars sit below the {cutoff:g}% cutoff.")
        st.plotly_chart(phase_profile(row, cutoff), width="stretch")
    with b:
        st.subheader("By season")
        mine = f[f["p_bowl"] == row["p_bowl"]]
        rows = []
        for yr in sorted(mine["year"].unique()):
            s = utilization(mine[mine["year"] == yr], cutoff)
            if not s.empty:
                r = s.iloc[0]
                rows.append({"Season": yr, "Overs": r["Overs"],
                             **{p: r[f"{p} %"] for p in PHASE_NAMES}, "Role": r["Role"]})
        if rows:
            st.dataframe(
                pd.DataFrame(rows).set_index("Season"),
                column_config={"Overs": st.column_config.NumberColumn(format="%.1f"),
                               **{p: st.column_config.NumberColumn(format="%.1f%%")
                                  for p in PHASE_NAMES}},
                width="stretch",
            )
            st.caption("A role that changes between seasons shows the captain "
                       "re-thinking how to use him.")

with tab_compare:
    default = names[:2]
    picks = st.multiselect("Pick up to 4 bowlers", names, default=default, max_selections=4)
    if picks:
        st.plotly_chart(compare_chart(table, picks, cutoff), width="stretch")
    else:
        st.info("Pick at least one bowler to compare.")

with tab_table:
    view_cols = (["Bowler", "Team", "Type", "Matches", "Overs"]
                 + [f"{p} %" for p in PHASE_NAMES] + ["Phases used", "Main phase", "Role"])
    out = table[view_cols].sort_values("Overs", ascending=False)
    st.dataframe(
        out,
        hide_index=True,
        width="stretch",
        column_config={
            "Overs": st.column_config.NumberColumn(format="%.1f"),
            **{f"{p} %": st.column_config.ProgressColumn(
                f"{p} ({PHASE_RANGES[p]})", format="%.1f%%", min_value=0, max_value=100)
               for p in PHASE_NAMES},
        },
    )
    st.download_button("Download CSV", out.round(2).to_csv(index=False),
                       file_name="over_utilization.csv", mime="text/csv")

st.caption("Method: legal balls only (wides and no-balls excluded), so shared or "
           "incomplete overs are split correctly. Phases: PP 1–6, Mid 1 7–11, "
           "Mid 2 12–16, Death 17–20.")
