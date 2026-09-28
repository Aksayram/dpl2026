"""
Bowler Pace & Release Point | Data Premier League 2026 (IPL 2023–24)

Tab 1  Pace & Speed    : Pace Effectiveness Index (PEI), Speed Dynamic Score (SDS),
                         Pace On vs Pace Off
Tab 2  Release Point   : bowler profile — best release + speed combination, best release per length

Rules used everywhere
  • Runs   = runs charged to the bowler (wides and no-balls count, byes and leg-byes don't)
  • Balls  = legal balls (wides and no-balls don't use up a ball)
  • Wickets= bowler's wickets only (caught, bowled, LBW, stumped, hit wicket)
  • Impractical speed readings are left out: pace bowlers below 100 km/h, spinners at 120 km/h+

Run: streamlit run bowler_pace.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Bowler Pace & Release | DPL 2026", layout="wide", page_icon="🏏")

DATA_FILE = Path(__file__).parent / "Data_Premier_League_Prelims_Dataset.csv"
PHASES = ['Powerplay (1–6)', 'Middle (7–16)', 'Death (17–20)']
USE_COLS = ['p_match', 'over_num', 'bowl', 'team_bowl', 'bowl_kind', 'bowl_type', 'bat_hand',
            'bowl_speed_category', 'bowl_release_point', 'length', 'bowlruns', 'wide', 'noball',
            'out', 'dismissal', 'year']

SPEED = {1: '140+', 2: '130–140', 3: '120–130', 4: '110–120',
         5: '100–110', 6: '85–100', 7: '70–85', 8: 'Below 70'}
RELEASE = {1: 'Above 2.0 m', 2: '1.8–2.0 m', 3: '1.6–1.8 m', 4: 'Below 1.6 m'}
LENGTHS = ['YORKER', 'FULL_TOSS', 'FULL', 'GOOD_LENGTH', 'SHORT_OF_A_GOOD_LENGTH', 'SHORT']
LENGTH_NAME = {'YORKER': 'Yorker', 'FULL_TOSS': 'Full toss', 'FULL': 'Full',
               'GOOD_LENGTH': 'Good length', 'SHORT_OF_A_GOOD_LENGTH': 'Short of good length',
               'SHORT': 'Short'}
SLOWER_TYPES = ['Knuckleball', 'OFF CUTTER', 'SLOWER BALL', 'SLOW BOUNCER', 'BACKHAND SLOWER', 'SLOW YORKER']
BOWLING_DISMISSALS = ['caught', 'bowled', 'leg before wicket', 'stumped', 'hit wicket']


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading deliveries…")
def load_data(source):
    df = pd.read_csv(source, usecols=USE_COLS).rename(columns={'over_num': 'over'})
    df['bowl'] = df['bowl'].str.strip()
    df['team_bowl'] = df['team_bowl'].str.strip().str.replace('Royal Challengers Bengaluru', 'RCB')
    df['bowl_kind'] = df['bowl_kind'].fillna('mixture/unknown')
    for c in ['bowlruns', 'wide', 'noball']:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype(int)
    df['phase'] = pd.cut(df['over'], bins=[0, 6, 16, 20], labels=PHASES)
    df['legal'] = ((df['wide'] == 0) & (df['noball'] == 0)).astype(int)
    df['wkt'] = (df['out'].astype(bool) & df['dismissal'].isin(BOWLING_DISMISSALS)).astype(int)
    df['dot'] = ((df['legal'] == 1) & (df['bowlruns'] == 0)).astype(int)

    band = df['bowl_speed_category']
    impractical = (((df['bowl_kind'] == 'pace bowler') & (band >= 6)) |
                   ((df['bowl_kind'] == 'spin bowler') & (band <= 3)))
    df['impractical'] = impractical
    df['speed_band'] = band.where(band.notna() & ~impractical).astype('Int64')   # usable speed only
    df['release'] = df['bowl_release_point'].astype('Int64')
    return df


def summarise(d, by):
    """Balls, runs, wickets and the rates we use, grouped by `by`."""
    g = d.groupby(by, observed=True).agg(Balls=('legal', 'sum'), Runs=('bowlruns', 'sum'),
                                         Wickets=('wkt', 'sum'), Dots=('dot', 'sum')).reset_index()
    g = g[g['Balls'] > 0]
    g['Runs_per_Ball'] = (g['Runs'] / g['Balls']).round(2)
    g['Economy'] = (g['Runs'] / g['Balls'] * 6).round(2)
    g['Balls_per_Wkt'] = (g['Balls'] / g['Wickets'].replace(0, np.nan)).round(1)
    g['Dot%'] = (g['Dots'] / g['Balls'] * 100).round(1)
    return g


def mode_or_na(s):
    s = s.dropna()
    return s.mode().iat[0] if len(s) else pd.NA


if DATA_FILE.exists():
    df = load_data(DATA_FILE)
else:
    uploaded = st.file_uploader("Upload Data_Premier_League_Prelims_Dataset.csv", type=["csv"])
    if uploaded is None:
        st.info("Please upload the DPL dataset.")
        st.stop()
    df = load_data(uploaded)

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("🏏 Filters")
years     = sorted(df['year'].unique())
sel_year  = st.sidebar.multiselect("Year", years, default=years)
sel_phase = st.sidebar.multiselect("Phase", PHASES, default=PHASES)
kinds     = sorted(df['bowl_kind'].unique())
sel_kind  = st.sidebar.multiselect("Bowler Kind", kinds, default=kinds)
hands     = sorted(df['bat_hand'].dropna().unique())
sel_hand  = st.sidebar.multiselect("Batter Hand", hands, default=hands)
teams     = sorted(df['team_bowl'].unique())
sel_team  = st.sidebar.multiselect("Bowling Team", teams, default=teams)

dff = df[df['year'].isin(sel_year) & df['phase'].isin(sel_phase) & df['bowl_kind'].isin(sel_kind) &
         df['bat_hand'].isin(sel_hand) & df['team_bowl'].isin(sel_team)]
st.sidebar.caption(f"{int(dff['impractical'].sum()):,} balls with impractical speed readings are "
                   "left out of every speed-based view.")

st.title("🎯 Bowler Pace & Release Point")
st.caption(f"IPL {'–'.join(str(y) for y in sel_year)}  |  **{dff['p_match'].nunique()}** matches  |  "
           f"**{dff['bowl'].nunique()}** bowlers  |  Runs = charged to the bowler (no byes / leg-byes)  |  "
           "Wickets = bowler's wickets")
if dff.empty:
    st.warning("No data matches current filters.")
    st.stop()

tab1, tab2 = st.tabs(["⚡ Pace & Speed", "📍 Release Point Profile"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Pace & Speed
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    spd = dff[dff['speed_band'].notna()].copy()
    spd['speed_band'] = spd['speed_band'].astype(int)
    p1, p2, p3 = st.tabs(["📈 Pace Effectiveness Index", "🌀 Speed Dynamic Score", "🐢 Pace On vs Pace Off"])

    # ── PEI ──────────────────────────────────────────────────────────────────
    with p1:
        st.subheader("📈 Pace Effectiveness Index (PEI)")
        st.caption("Pick a speed range and see which bowlers are most effective in it. "
                   "PEI = wickets per over ÷ economy (the same as wickets ÷ runs). Higher is better.")
        bands_slow_to_fast = [SPEED[b] for b in sorted(SPEED, reverse=True)]
        lo_lbl, hi_lbl = st.select_slider("Speed range (km/h)", options=bands_slow_to_fast,
                                          value=(bands_slow_to_fast[0], bands_slow_to_fast[-1]), key='pei_rng')
        lbl_to_band = {v: k for k, v in SPEED.items()}
        b_slow, b_fast = lbl_to_band[lo_lbl], lbl_to_band[hi_lbl]
        pace_df = spd[spd['speed_band'].between(b_fast, b_slow)]
        min_balls = st.slider("Minimum balls in this range", 5, 100, 20, key='pei_min')

        if pace_df.empty:
            st.warning("No deliveries in this speed range.")
        else:
            st.markdown(f"**{int(pace_df['legal'].sum()):,} balls** at **{lo_lbl} to {hi_lbl} km/h**")
            pg = summarise(pace_df, 'bowl')
            kind = pace_df.groupby('bowl')['bowl_kind'].agg(mode_or_na)
            pg['Type'] = pg['bowl'].map(kind)
            pg = pg[pg['Balls'] >= min_balls].copy()
            pg['Avg'] = (pg['Runs'] / pg['Wickets'].replace(0, np.nan)).round(1)
            pg['SR'] = pg['Balls_per_Wkt']
            pg['PEI'] = ((pg['Wickets'] / pg['Balls'] * 6) / pg['Economy'].replace(0, np.nan)).round(3).fillna(0)
            pg = pg.sort_values('PEI', ascending=False).reset_index(drop=True)
            pg.index += 1

            rank_by = st.radio("Rank by", ['PEI', 'Economy', 'Wickets', 'Dot%', 'SR'], horizontal=True, key='pei_rank')
            asc = rank_by in ['Economy', 'SR']
            c1, c2 = st.columns([1.2, 1])
            with c1:
                top = pg.dropna(subset=[rank_by])
                top = top.nsmallest(15, rank_by) if asc else top.nlargest(15, rank_by)
                fig = px.bar(top, x='bowl', y=rank_by, color=rank_by, text=rank_by,
                             color_continuous_scale='RdYlGn_r' if asc else 'RdYlGn',
                             title=f"Top 15 – {rank_by} at {lo_lbl} to {hi_lbl} km/h", height=420)
                fig.update_layout(xaxis_tickangle=-40)
                st.plotly_chart(fig, width="stretch")
            with c2:
                st.dataframe(pg[['bowl', 'Type', 'Balls', 'Runs', 'Economy', 'Wickets', 'SR', 'Dot%', 'PEI']],
                             width="stretch", height=400)

            if not pg.empty:
                fig2 = px.scatter(pg, x='Economy', y='Wickets', size='Balls', color='PEI', text='bowl',
                                  color_continuous_scale='RdYlGn', height=450,
                                  title="Economy vs Wickets (size = balls, colour = PEI)")
                fig2.update_traces(textposition='top center', textfont_size=9)
                st.plotly_chart(fig2, width="stretch")

    # ── SDS ──────────────────────────────────────────────────────────────────
    with p2:
        st.subheader("🌀 Speed Dynamic Score (pace bowlers)")
        st.caption("SDS = spread of a bowler's speed bands × his range of bands (slowest band minus fastest). "
                   "Higher = he mixes his pace more. Runs, wickets and PEI sit alongside to show whether it pays off.")
        min_sds = st.slider("Minimum balls", 10, 150, 60, key='sds_min')
        pace_only = spd[spd['bowl_kind'] == 'pace bowler']
        if pace_only.empty:
            st.info("No pace bowlers in the current filters.")
        else:
            band_stats = pace_only.groupby('bowl')['speed_band'].agg(
                Spread='std', Fastest='min', Slowest='max', Stock=mode_or_na).reset_index()
            sg = summarise(pace_only, 'bowl').merge(band_stats, on='bowl')
            sg = sg[sg['Balls'] >= min_sds].copy()
            sg['Spread'] = sg['Spread'].round(2)
            sg['Range'] = sg['Slowest'] - sg['Fastest']
            sg['SDS'] = (sg['Spread'] * sg['Range']).round(2)
            sg['PEI'] = ((sg['Wickets'] / sg['Balls'] * 6) / sg['Economy'].replace(0, np.nan)).round(3).fillna(0)
            sg['Stock Pace'] = sg['Stock'].map(SPEED)
            sg['Fastest Band'] = sg['Fastest'].map(SPEED)
            sg['Slowest Band'] = sg['Slowest'].map(SPEED)
            sg = sg.sort_values('SDS', ascending=False).reset_index(drop=True)
            sg.index += 1

            s1, s2 = st.columns([1.2, 1])
            with s1:
                fig = px.bar(sg.head(15), x='bowl', y='SDS', color='Economy', text='SDS',
                             color_continuous_scale='RdYlGn_r', height=420,
                             title="Top 15 – Speed Dynamic Score (colour = economy)")
                fig.update_layout(xaxis_tickangle=-40)
                st.plotly_chart(fig, width="stretch")
            with s2:
                st.dataframe(sg[['bowl', 'Balls', 'Stock Pace', 'Fastest Band', 'Slowest Band', 'Spread',
                                 'Range', 'SDS', 'Economy', 'Wickets', 'PEI']], width="stretch", height=400)

            fig2 = px.scatter(sg, x='SDS', y='Economy', size='Balls', color='PEI', text='bowl',
                              color_continuous_scale='RdYlGn', height=450,
                              title="Does mixing pace pay off? SDS vs Economy (colour = PEI)")
            fig2.update_traces(textposition='top center', textfont_size=9)
            st.plotly_chart(fig2, width="stretch")

    # ── Pace On vs Pace Off ─────────────────────────────────────────────────
    with p3:
        st.subheader("🐢 Pace On vs Pace Off (pace bowlers)")
        st.caption("Stock pace = the bowler's most common speed band. **Pace Off** = at least 2 bands slower "
                   "than his stock **and** labelled as a slower ball (Knuckleball, Off cutter, Slower ball, "
                   "Slow bouncer, Backhand slower, Slow yorker). Everything else = **Pace On**.")
        po = spd[spd['bowl_kind'] == 'pace bowler'].copy()
        if po.empty:
            st.info("No pace bowlers in the current filters.")
        else:
            stock = po.groupby('bowl')['speed_band'].agg(mode_or_na)
            po['stock'] = po['bowl'].map(stock).astype(int)
            po['Delivery'] = np.where((po['speed_band'] - po['stock'] >= 2) & po['bowl_type'].isin(SLOWER_TYPES),
                                      'Pace Off', 'Pace On')

            allp = summarise(po, 'Delivery')
            a1, a2 = st.columns([1, 1.3])
            with a1:
                st.markdown("**All pace bowlers**")
                st.dataframe(allp[['Delivery', 'Balls', 'Runs_per_Ball', 'Economy', 'Wickets', 'Balls_per_Wkt', 'Dot%']],
                             hide_index=True, width="stretch")
            with a2:
                fig = px.bar(allp, x='Delivery', y='Economy', color='Delivery', text='Economy', height=280,
                             color_discrete_map={'Pace On': '#636EFA', 'Pace Off': '#EF553B'},
                             title="Economy – Pace On vs Pace Off")
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, width="stretch")

            st.divider()
            min_off = st.slider("Minimum Pace Off balls per bowler", 5, 60, 20, key='po_min')
            pb = summarise(po, ['bowl', 'Delivery'])
            wide_t = pb.pivot(index='bowl', columns='Delivery',
                              values=['Balls', 'Economy', 'Wickets', 'Balls_per_Wkt'])
            wide_t.columns = [f"{d}_{m}" for m, d in wide_t.columns]
            wide_t = wide_t.reset_index()
            for c in ['Pace On_Balls', 'Pace Off_Balls', 'Pace On_Wickets', 'Pace Off_Wickets']:
                if c not in wide_t:
                    wide_t[c] = 0
                wide_t[c] = wide_t[c].fillna(0).astype(int)
            wide_t = wide_t[wide_t['Pace Off_Balls'] >= min_off].copy()
            if wide_t.empty:
                st.info("No bowler has that many Pace Off balls under these filters. Lower the minimum.")
            else:
                wide_t['Stock Pace'] = wide_t['bowl'].map(stock).map(SPEED)
                wide_t['Pace Off %'] = (wide_t['Pace Off_Balls'] /
                                        (wide_t['Pace Off_Balls'] + wide_t['Pace On_Balls']) * 100).round(1)
                wide_t['Econ Change'] = (wide_t['Pace Off_Economy'] - wide_t['Pace On_Economy']).round(2)
                wide_t = wide_t.sort_values('Econ Change').reset_index(drop=True)
                wide_t.index += 1
                st.caption("Econ Change = Pace Off economy minus Pace On economy. Negative = his slower ball "
                           "is cheaper than his stock ball.")
                b1, b2 = st.columns([1.2, 1])
                with b1:
                    melt = wide_t.melt(id_vars='bowl', value_vars=['Pace On_Economy', 'Pace Off_Economy'],
                                       var_name='Delivery', value_name='Economy')
                    melt['Delivery'] = melt['Delivery'].str.replace('_Economy', '')
                    fig = px.bar(melt, x='bowl', y='Economy', color='Delivery', barmode='group', height=420,
                                 color_discrete_map={'Pace On': '#636EFA', 'Pace Off': '#EF553B'},
                                 title="Economy – Pace On vs Pace Off by bowler")
                    fig.update_layout(xaxis_tickangle=-40)
                    st.plotly_chart(fig, width="stretch")
                with b2:
                    st.dataframe(wide_t[['bowl', 'Stock Pace', 'Pace On_Balls', 'Pace On_Economy', 'Pace On_Wickets',
                                         'Pace Off_Balls', 'Pace Off_Economy', 'Pace Off_Wickets', 'Pace Off %',
                                         'Econ Change']], width="stretch", height=400)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Release Point Profile
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("📍 Release Point Profile")
    st.caption("Pick a bowler to see where he usually operates, how effective he is at each release height "
               "and speed, and which combination works best. **Main measure: runs per ball** (lower is better). "
               "Wickets are shown alongside and break ties.")

    rel = dff[dff['release'].notna()].copy()
    rel['release'] = rel['release'].astype(int)
    counts = rel.groupby('bowl')['legal'].sum().sort_values(ascending=False)
    counts = counts[counts >= 30]
    if counts.empty:
        st.info("No bowler has enough release-point data under these filters.")
    else:
        r1, r2 = st.columns([2, 1])
        who = r1.selectbox("Bowler", counts.index.tolist(), key='rp_bowler',
                           format_func=lambda b: f"{b}  ({int(counts[b])} balls)")
        min_cell = r2.slider("Minimum balls to recommend a combination", 5, 60, 20, key='rp_min')

        me = rel[rel['bowl'] == who]
        me_sp = me[me['speed_band'].notna()].copy()
        me_sp['speed_band'] = me_sp['speed_band'].astype(int)
        overall = summarise(me.assign(k=1), 'k').iloc[0]

        # 1 — usual operating zone
        st.markdown("### 1️⃣ Usual operating zone")
        rel_mode = int(me['release'].mode().iat[0])
        rel_share = (me['release'] == rel_mode).mean() * 100
        z1, z2, z3, z4 = st.columns(4)
        z1.metric("Usual release", RELEASE[rel_mode], f"{rel_share:.0f}% of his balls", delta_color="off")
        if not me_sp.empty:
            sp_mode = int(me_sp['speed_band'].mode().iat[0])
            sp_share = (me_sp['speed_band'] == sp_mode).mean() * 100
            z2.metric("Usual speed", f"{SPEED[sp_mode]} km/h", f"{sp_share:.0f}% of his balls", delta_color="off")
        else:
            z2.metric("Usual speed", "–")
        z3.metric("Runs per ball (overall)", f"{overall['Runs_per_Ball']:.2f}", f"Economy {overall['Economy']:.2f}",
                  delta_color="off")
        z4.metric("Wickets", int(overall['Wickets']),
                  f"1 every {overall['Balls_per_Wkt']:.0f} balls" if overall['Wickets'] else "none", delta_color="off")

        # 2 — by release and by speed
        st.markdown("### 2️⃣ Effectiveness by release height and by speed")
        cols_show = ['Balls', 'Share%', 'Runs_per_Ball', 'Economy', 'Wickets', 'Balls_per_Wkt', 'Dot%']
        e1, e2 = st.columns(2)
        with e1:
            by_rel = summarise(me, 'release').sort_values('release')
            by_rel['Share%'] = (by_rel['Balls'] / by_rel['Balls'].sum() * 100).round(1)
            by_rel['Release'] = by_rel['release'].map(RELEASE)
            fig = px.bar(by_rel, x='Release', y='Runs_per_Ball', text='Runs_per_Ball', color='Runs_per_Ball',
                         color_continuous_scale='RdYlGn_r', height=300, title="Runs per ball by release height",
                         hover_data={'Balls': True, 'Wickets': True})
            st.plotly_chart(fig, width="stretch")
            st.dataframe(by_rel[['Release'] + cols_show], hide_index=True, width="stretch")
        with e2:
            if me_sp.empty:
                st.info("No usable speed readings for this bowler.")
            else:
                by_sp = summarise(me_sp, 'speed_band').sort_values('speed_band')
                by_sp['Share%'] = (by_sp['Balls'] / by_sp['Balls'].sum() * 100).round(1)
                by_sp['Speed'] = by_sp['speed_band'].map(SPEED)
                fig = px.bar(by_sp, x='Speed', y='Runs_per_Ball', text='Runs_per_Ball', color='Runs_per_Ball',
                             color_continuous_scale='RdYlGn_r', height=300, title="Runs per ball by speed (km/h)",
                             hover_data={'Balls': True, 'Wickets': True})
                st.plotly_chart(fig, width="stretch")
                st.dataframe(by_sp[['Speed'] + cols_show], hide_index=True, width="stretch")

        # 3 — best release + speed combination
        st.markdown("### 3️⃣ Best release + speed combination")
        if me_sp.empty:
            st.info("No usable speed readings for this bowler.")
        else:
            combo = summarise(me_sp, ['release', 'speed_band'])
            ok = combo[combo['Balls'] >= min_cell]
            if ok.empty:
                st.info(f"No release + speed combination has {min_cell}+ balls. Lower the minimum.")
            else:
                best = ok.sort_values(['Runs_per_Ball', 'Wickets'], ascending=[True, False]).iloc[0]
                usual = combo.sort_values('Balls', ascending=False).iloc[0]
                diff = overall['Runs_per_Ball'] - best['Runs_per_Ball']
                st.success(
                    f"**{who} is most effective at {SPEED[int(best['speed_band'])]} km/h, releasing "
                    f"{RELEASE[int(best['release'])].lower()}**: {best['Runs_per_Ball']:.2f} runs per ball "
                    f"(economy {best['Economy']:.2f}) over {int(best['Balls'])} balls, with {int(best['Wickets'])} "
                    f"wicket(s). That is {abs(diff):.2f} runs per ball {'better' if diff >= 0 else 'worse'} "
                    f"than his overall {overall['Runs_per_Ball']:.2f}.")
                if (usual['release'], usual['speed_band']) != (best['release'], best['speed_band']):
                    st.caption(f"His most-used combination is {SPEED[int(usual['speed_band'])]} km/h from "
                               f"{RELEASE[int(usual['release'])].lower()} ({int(usual['Balls'])} balls, "
                               f"{usual['Runs_per_Ball']:.2f} runs per ball).")
                else:
                    st.caption("This is also his most-used combination.")

            rels = sorted(combo['release'].unique())
            sps = sorted(combo['speed_band'].unique())
            z = combo.pivot(index='release', columns='speed_band', values='Runs_per_Ball').reindex(index=rels, columns=sps)
            n = combo.pivot(index='release', columns='speed_band', values='Balls').reindex(index=rels, columns=sps)
            zc = z.where(n >= min_cell)
            text = [[(f"{z.iat[i, j]:.2f}<br>({int(n.iat[i, j])} balls)" if pd.notna(n.iat[i, j]) else "")
                     for j in range(len(sps))] for i in range(len(rels))]
            fig = go.Figure(go.Heatmap(z=zc.values, x=[SPEED[s] for s in sps], y=[RELEASE[r] for r in rels],
                                       text=text, texttemplate="%{text}", colorscale='RdYlGn_r',
                                       colorbar=dict(title='Runs/ball'), hoverongaps=False))
            fig.update_layout(height=320, title=f"Runs per ball: release × speed (grey = fewer than {min_cell} balls)",
                              xaxis_title="Speed (km/h)", yaxis_title="Release height",
                              plot_bgcolor='#e9e9e9', margin=dict(t=50, l=10, r=10, b=10))
            st.plotly_chart(fig, width="stretch")

        # 4 — best release for each length
        st.markdown("### 4️⃣ Best release height for each length")
        ml = me[me['length'].isin(LENGTHS)]
        if ml.empty:
            st.info("No length data for this bowler.")
        else:
            rl = summarise(ml, ['length', 'release'])
            per_len = summarise(ml, 'length').set_index('length')
            rows = []
            for L in LENGTHS:
                if L not in per_len.index:
                    continue
                cand = rl[(rl['length'] == L) & (rl['Balls'] >= min_cell)]
                row = {'Length': LENGTH_NAME[L], 'Balls': int(per_len.at[L, 'Balls']),
                       'Runs/Ball (all releases)': per_len.at[L, 'Runs_per_Ball']}
                if cand.empty:
                    row.update({'Best Release': f"Not enough balls ({min_cell}+ needed)",
                                'Runs/Ball there': np.nan, 'Balls there': np.nan, 'Wickets there': np.nan})
                else:
                    b = cand.sort_values(['Runs_per_Ball', 'Wickets'], ascending=[True, False]).iloc[0]
                    row.update({'Best Release': RELEASE[int(b['release'])], 'Runs/Ball there': b['Runs_per_Ball'],
                                'Balls there': int(b['Balls']), 'Wickets there': int(b['Wickets'])})
                rows.append(row)
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

            rels_l = sorted(rl['release'].unique())
            lens = [L for L in LENGTHS if L in rl['length'].unique()]
            z = rl.pivot(index='release', columns='length', values='Runs_per_Ball').reindex(index=rels_l, columns=lens)
            n = rl.pivot(index='release', columns='length', values='Balls').reindex(index=rels_l, columns=lens)
            zc = z.where(n >= min_cell)
            text = [[(f"{z.iat[i, j]:.2f}<br>({int(n.iat[i, j])} balls)" if pd.notna(n.iat[i, j]) else "")
                     for j in range(len(lens))] for i in range(len(rels_l))]
            fig = go.Figure(go.Heatmap(z=zc.values, x=[LENGTH_NAME[L] for L in lens],
                                       y=[RELEASE[r] for r in rels_l], text=text, texttemplate="%{text}",
                                       colorscale='RdYlGn_r', colorbar=dict(title='Runs/ball'), hoverongaps=False))
            fig.update_layout(height=320, title=f"Runs per ball: release × length (grey = fewer than {min_cell} balls)",
                              xaxis_title="Length", yaxis_title="Release height", plot_bgcolor='#e9e9e9',
                              margin=dict(t=50, l=10, r=10, b=10))
            st.plotly_chart(fig, width="stretch")
