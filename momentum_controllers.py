"""
Momentum Controllers | Data Premier League 2026 (IPL 2023–24)
Ported from the Momentum Controllers tab in ipl_dashboard.py:
  1. Over Start Dominance   2. Bowler Resilience
  3. Post-Wicket Scoring    4. Team Momentum

Run: streamlit run momentum_controllers.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Momentum Controllers | DPL 2026", layout="wide", page_icon="🏏")

DATA_FILE = Path(__file__).parent / "Data_Premier_League_Prelims_Dataset.csv"
PHASES = ['Powerplay (1–6)', 'Middle (7–16)', 'Death (17–20)']
USE_COLS = ['p_match', 'inns', 'over_num', 'ball', 'bat', 'p_bat', 'bowl', 'team_bat', 'team_bowl',
            'score', 'batruns', 'bowlruns', 'ballfaced', 'out', 'p_out', 'wide', 'noball', 'dismissal',
            'bowl_kind', 'bat_hand', 'year']


@st.cache_data(show_spinner="Loading deliveries…")
def load_data(source):
    df = pd.read_csv(source, usecols=USE_COLS).rename(columns={'over_num': 'over'})
    for c in ['bat', 'bowl']:
        df[c] = df[c].str.strip()
    for c in ['team_bat', 'team_bowl']:
        df[c] = df[c].str.strip().str.replace('Royal Challengers Bengaluru', 'RCB')
    for c in ['score', 'batruns', 'bowlruns', 'ballfaced', 'ball', 'wide', 'noball']:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype(int)
    df['out'] = df['out'].astype(int)
    df['bowl_kind'] = df['bowl_kind'].fillna('mixture/unknown')
    df['phase'] = pd.cut(df['over'], bins=[0, 6, 16, 20], labels=PHASES)
    # Over Start Dominance: nth ball faced in the over (wides skip the count)
    order = df.sort_values(['p_match', 'inns', 'over', 'ball']).index
    df['faced_no'] = (df.loc[order].groupby(['p_match', 'inns', 'over'])['ballfaced'].cumsum()
                        .reindex(df.index))
    # Valid bowling dismissals — exclude run outs and retired (same as ipl_dashboard.py)
    bowling_dismissals = ['caught', 'bowled', 'leg before wicket', 'stumped', 'hit wicket']
    df['bowl_wicket'] = ((df['out'] == 1) & (df['dismissal'].isin(bowling_dismissals))).astype(int)
    return df


@st.cache_data(show_spinner=False)
def bowler_over_splits(d):
    """First 3 balls vs last 3 balls of each bowler-over.
    Runs = runs charged to the bowler (wides and no-balls count; byes and leg-byes don't).
    Boundaries = fours/sixes off the bat."""
    keys = ['p_match', 'inns', 'over', 'bowl', 'phase']
    f3 = d['ball'] <= 3
    l3 = ~f3
    h = d[keys].copy()
    h['first3_runs']      = d['bowlruns'].where(f3, 0)
    h['last3_runs']       = d['bowlruns'].where(l3, 0)
    h['last3_balls']      = l3.astype(int)
    h['last3_fours']      = (l3 & (d['batruns'] == 4)).astype(int)
    h['last3_sixes']      = (l3 & (d['batruns'] == 6)).astype(int)
    h['last3_dots']       = (l3 & (d['bowlruns'] == 0) & (d['wide'] == 0)).astype(int)
    h['last3_boundaries'] = (l3 & d['batruns'].isin([4, 6])).astype(int)
    return h.groupby(keys, observed=True).sum().reset_index()


def build_post_wicket(df, n):
    """Walk every innings ball by ball and build post-wicket records.

    Returns (records, members):
      records : one row per record  (role = 'new' | 'nd_overall' | 'nd_event' | 'before')
      members : (rec_id, row) pairs — which deliveries belong to which record
    Deliveries are counted later, after the sidebar filters are applied.
    """
    d = df.sort_values(['p_match', 'inns', 'over', 'ball'])
    legal = ((d['wide'] == 0) & (d['noball'] == 0)).astype(int)
    d = d.assign(_row=d.index, _leg=legal.groupby([d['p_match'], d['inns']]).cumsum(),
                 _leg_ov=legal.groupby([d['p_match'], d['inns'], d['over']]).cumsum(), _legal=legal)

    recs, members = [], []

    def add_rec(**kw):
        kw['rec_id'] = len(recs)
        recs.append(kw)
        return kw['rec_id']

    for (pm, inn), g in d.groupby(['p_match', 'inns'], sort=False):
        rows = g['_row'].to_numpy(); strk = g['p_bat'].to_numpy(); leg = g['_leg'].to_numpy()
        outs = g['out'].to_numpy(); pout = g['p_out'].to_numpy(); bf = g['ballfaced'].to_numpy()
        dism = g['dismissal'].to_numpy(); legal_k = g['_legal'].to_numpy(); leg_ov = g['_leg_ov'].to_numpy()
        team, year = g['team_bat'].iat[0], g['year'].iat[0]
        ov_lbl = (g['over'] - 1).astype(str).to_numpy()

        # crease slots: each slot is one batter; id is None until he faces a ball
        crease, slot_of_row, wickets = [], [], []
        expect = None      # which unseen batter should face next (strike rules after a wicket)
        for k in range(len(g)):
            s = strk[k]
            slot = next((c for c in crease if c['id'] == s), None)
            if slot is None:
                if expect is not None and expect in crease and expect['id'] is None:
                    slot = expect                                            # decided by strike rules
                else:
                    slot = next((c for c in crease if c['id'] is None), None)   # oldest unseen batter
                expect = None
                if slot is None:
                    slot = {'id': None}
                    if len(crease) >= 2:           # data gap: drop the least recent occupant
                        crease.pop(0)
                    crease.append(slot)
                slot['id'] = s
            slot_of_row.append(slot)
            if len(crease) < 2 and not wickets and len(crease) == 1:
                crease.append({'id': None})         # second opener, not yet on strike
            if outs[k] == 1:
                x = pout[k]
                gone = next((c for c in crease if c['id'] == x), None) \
                    or next((c for c in crease if c['id'] is None), None)
                if gone is not None:
                    crease.remove(gone)
                survivors = list(crease)
                new_slot = {'id': None}
                crease.append(new_slot)
                uncertain = False
                if any(c['id'] is None for c in survivors):
                    # the other batter hasn't faced yet, so two batters are still unseen.
                    # IPL 2023-24: the new batter takes strike, unless the wicket fell on the
                    # last ball of the over (then the other batter faces next). Run-outs can't
                    # be resolved this way (we don't know which end the new batter went to).
                    if dism[k] == 'run out':
                        uncertain = True
                    elif legal_k[k] == 1 and leg_ov[k] >= 6:
                        expect = next(c for c in survivors if c['id'] is None)
                    else:
                        expect = new_slot
                wickets.append({'k': k, 'leg': leg[k], 'survivors': survivors, 'new': new_slot,
                                'uncertain': uncertain,
                                'label': f"{ov_lbl[k]}.{int(g['ball'].iat[k])}"})

        if not wickets:
            continue                                # openers batted through: nothing to count

        # group wickets into events: next wicket within n legal balls of the previous one
        events = []
        for w in wickets:
            if events and w['leg'] - events[-1]['wk'][-1]['leg'] <= n:
                events[-1]['wk'].append(w)
            else:
                events.append({'wk': [w]})
        for e in events:
            e['end_leg'] = e['wk'][-1]['leg'] + n
            e['size'] = len(e['wk'])

        def rows_of(slot, start_k, stop_leg=None, max_balls=None, before=False):
            out_rows, balls = [], 0
            rng = range(start_k - 1, -1, -1) if before else range(start_k + 1, len(g))
            for k in rng:
                if not before and stop_leg is not None and leg[k] > stop_leg:
                    break
                if slot_of_row[k] is slot:
                    if max_balls is not None and balls >= max_balls:
                        break
                    out_rows.append(rows[k]); balls += bf[k]
            return out_rows

        base = dict(p_match=pm, inns=inn, team_bat=team, year=year)
        overall_done = set()
        for ei, e in enumerate(events):
            seen_in_event = set()
            for w in e['wk']:
                # new batter: his own first n balls
                rid = add_rec(role='new', slot=w['new'], wicket=w['label'], uncertain=w['uncertain'], **base)
                members += [(rid, r) for r in rows_of(w['new'], w['k'], max_balls=n)]
                for sv in w['survivors']:
                    sid = id(sv)
                    if sid not in overall_done:     # overall: from first wicket he survives to the end
                        overall_done.add(sid)
                        rid = add_rec(role='nd_overall', slot=sv, wicket=w['label'], uncertain=w['uncertain'], **base)
                        members += [(rid, r) for r in rows_of(sv, w['k'])]
                        rid = add_rec(role='before', link=rid, slot=sv, **base)
                        members += [(rid, r) for r in rows_of(sv, w['k'] + 1, max_balls=n, before=True)]
                    if sid not in seen_in_event:    # event: from first wicket he survives in it
                        seen_in_event.add(sid)
                        rid = add_rec(role='nd_event', slot=sv, wicket=w['label'], uncertain=w['uncertain'],
                                      event=f"{pm}-{inn}-{ei}", size=e['size'], **base)
                        members += [(rid, r) for r in rows_of(sv, w['k'], stop_leg=e['end_leg'])]
                        rid = add_rec(role='before', link=rid, slot=sv, **base)
                        members += [(rid, r) for r in rows_of(sv, e['wk'][0]['k'] + 1, max_balls=n, before=True)]

    import pandas as pd
    for r in recs:
        r['p_bat'] = r.pop('slot')['id']
    return pd.DataFrame(recs), pd.DataFrame(members, columns=['rec_id', '_row'])


@st.cache_data(show_spinner="Working out every wicket…")
def post_wicket_engine(df, n):
    return build_post_wicket(df, n)


def post_wicket_totals(df, recs, mem, keep_idx):
    """Add up each record using only deliveries that pass the sidebar filters."""
    m = mem[mem['_row'].isin(keep_idx)]
    x = df.loc[m['_row'], ['batruns', 'ballfaced', 'out', 'p_out', 'p_bat']]
    m = m.assign(runs=x['batruns'].to_numpy(), balls=x['ballfaced'].to_numpy(),
                 dots=((x['batruns'] == 0) & (x['ballfaced'] == 1)).to_numpy().astype(int),
                 fours=(x['batruns'] == 4).to_numpy().astype(int),
                 sixes=(x['batruns'] == 6).to_numpy().astype(int),
                 got_out=((x['out'] == 1) & (x['p_out'] == x['p_bat'])).to_numpy().astype(int))
    tot = m.groupby('rec_id')[['runs', 'balls', 'dots', 'fours', 'sixes', 'got_out']].sum()
    R = recs.join(tot, on='rec_id').fillna({c: 0 for c in tot.columns})
    before = (R[R['role'] == 'before'].groupby('link')[['runs', 'balls']].sum()
                .rename(columns={'runs': 'before_runs', 'balls': 'before_balls'}))
    R = R[R['role'] != 'before'].join(before, on='rec_id').fillna({'before_runs': 0, 'before_balls': 0})
    num = ['runs', 'balls', 'dots', 'fours', 'sixes', 'got_out', 'before_runs', 'before_balls']
    R[num] = R[num].astype(int)
    return R[R['balls'] > 0]


if DATA_FILE.exists():
    df = load_data(DATA_FILE)
else:
    uploaded = st.file_uploader("Upload Data_Premier_League_Prelims_Dataset.csv", type=["csv"])
    if uploaded is None:
        st.info("Please upload the DPL dataset.")
        st.stop()
    df = load_data(uploaded)

# ── Sidebar (same filters as ipl_dashboard.py) ────────────────────────────────
st.sidebar.title("🏏 Filters")
years     = sorted(df['year'].unique())
sel_year  = st.sidebar.multiselect("Year", years, default=years)
sel_phase = st.sidebar.multiselect("Phase", PHASES, default=PHASES)
kinds     = sorted(df['bowl_kind'].unique())
sel_kind  = st.sidebar.multiselect("Bowler Kind", kinds, default=kinds)
hands     = sorted(df['bat_hand'].dropna().unique())
sel_hand  = st.sidebar.multiselect("Batter Hand", hands, default=hands)
teams     = sorted(df['team_bat'].unique())
sel_team  = st.sidebar.multiselect("Batting Team", teams, default=teams)

dff = df[df['year'].isin(sel_year) & df['phase'].isin(sel_phase) &
         df['bowl_kind'].isin(sel_kind) & df['bat_hand'].isin(sel_hand) &
         df['team_bat'].isin(sel_team)]

st.title("⚡ Momentum Controllers")
st.caption(f"IPL {'–'.join(str(y) for y in sel_year)}  |  **{dff['p_match'].nunique()}** matches  |  "
           f"**{dff['bat'].nunique()}** batters  |  **{dff['bowl'].nunique()}** bowlers")
if dff.empty:
    st.warning("No data matches current filters.")
    st.stop()

mc1, mc2, mc3, mc4 = st.tabs([
    "🏏 Over Start Dominance", "🎳 Bowler Resilience", "🩹 Post-Wicket Scoring", "📊 Team Momentum"
])

# ── MC TAB 1: Over Start Dominance ────────────────────────────────────────
with mc1:
    st.subheader("🏏 Over Start Dominance – First 3 Balls")
    st.caption("First 3 balls faced in the over (wides skipped). Runs off the bat only.")
    start_thr = st.slider("Dominant start threshold (runs in first 3 balls)", 3, 10, 4, key='st')
    f3 = dff[(dff['ballfaced'] == 1) & (dff['faced_no'] <= 3)].copy()
    os_grp = f3.groupby(['p_match','inns','over','bat','team_bat','phase'],observed=True).agg(
        Runs=('batruns','sum'), Balls=('batruns','count'),
        Fours=('batruns', lambda x:(x==4).sum()), Sixes=('batruns', lambda x:(x==6).sum())
    ).reset_index()
    os_grp['SR'] = (os_grp['Runs']/os_grp['Balls']*100).round(1)
    dom = os_grp[os_grp['Runs'] >= start_thr]

    st.markdown(f"**Dominant starts ({start_thr}+ in first 3 balls): {len(dom):,}**")
    st.divider()

    st.markdown("### 🏆 Batter Leaderboard")
    bat_s = dom.groupby('bat').agg(
        Dom_Starts=('Runs','count'), Total_Runs=('Runs','sum'),
        Avg_Runs=('Runs','mean'), Fours=('Fours','sum'), Sixes=('Sixes','sum')
    ).reset_index()
    bat_s_tot = os_grp.groupby('bat').agg(Total_Starts=('Runs','count')).reset_index()
    bat_s = pd.merge(bat_s, bat_s_tot, on='bat', how='left')
    bat_s['Freq%']    = (bat_s['Dom_Starts']/bat_s['Total_Starts']*100).round(1)
    bat_s['Avg_Runs'] = bat_s['Avg_Runs'].round(1)
    bat_s = bat_s.sort_values('Dom_Starts',ascending=False).reset_index(drop=True); bat_s.index += 1

    cs1,cs2 = st.columns([1.2,1])
    with cs1:
        fig_s1 = px.bar(bat_s.head(15), x='bat', y='Dom_Starts',
                        color='Dom_Starts', color_continuous_scale='Plasma',
                        text='Dom_Starts', title=f"Most {start_thr}+ Run Starts", height=400)
        fig_s1.update_layout(xaxis_tickangle=-40)
        st.plotly_chart(fig_s1, width="stretch")
    with cs2:
        fig_s2 = px.bar(bat_s.sort_values('Freq%',ascending=False).head(15),
                        x='bat', y='Freq%', color='Freq%',
                        color_continuous_scale='RdYlGn', text='Freq%',
                        title="Start Frequency %", height=400)
        fig_s2.update_layout(xaxis_tickangle=-40)
        st.plotly_chart(fig_s2, width="stretch")
    st.dataframe(bat_s[['bat','Dom_Starts','Total_Starts','Freq%','Total_Runs','Avg_Runs','Fours','Sixes']],
                 width="stretch")

    st.divider()
    st.markdown("### 🏆 Team Leaderboard")
    team_s = dom.groupby('team_bat').agg(Dom_Starts=('Runs','count'), Total_Runs=('Runs','sum')).reset_index()
    team_s_tot = os_grp.groupby('team_bat').agg(Total_Starts=('Runs','count')).reset_index()
    team_s = pd.merge(team_s, team_s_tot, on='team_bat', how='left')
    team_s['Freq%'] = (team_s['Dom_Starts']/team_s['Total_Starts']*100).round(1)
    team_s = team_s.sort_values('Dom_Starts',ascending=False).reset_index(drop=True); team_s.index += 1
    ct1,ct2 = st.columns(2)
    with ct1:
        fig_t1 = px.bar(team_s, x='team_bat', y='Dom_Starts',
                        color='Dom_Starts', color_continuous_scale='Teal',
                        text='Dom_Starts', title="Team – Dominant Starts", height=360)
        fig_t1.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig_t1, width="stretch")
    with ct2:
        fig_t2 = px.bar(team_s, x='team_bat', y='Freq%',
                        color='Freq%', color_continuous_scale='RdYlGn',
                        text='Freq%', title="Team – Start Freq%", height=360)
        fig_t2.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig_t2, width="stretch")

    st.divider()
    st.markdown("### 📊 Phase-wise")
    ph_s = dom.groupby(['bat','phase'],observed=True).agg(Dom_Starts=('Runs','count')).reset_index()
    top12_s = bat_s.head(12)['bat'].tolist()
    fig_ps = px.bar(ph_s[ph_s['bat'].isin(top12_s)], x='bat', y='Dom_Starts',
                    color='phase', barmode='group',
                    color_discrete_map={'Powerplay (1–6)':'#636EFA','Middle (7–16)':'#EF553B','Death (17–20)':'#00CC96'},
                    title="Dominant Starts by Phase – Top 12", height=400)
    fig_ps.update_layout(xaxis_tickangle=-40)
    st.plotly_chart(fig_ps, width="stretch")

    st.divider()
    st.markdown("### 🔍 Individual Batter Start Profile")
    sel_sb = st.selectbox("Select Batter", sorted(os_grp['bat'].unique()), key='sb')
    sb_all = os_grp[os_grp['bat']==sel_sb]
    sb_dom = dom[dom['bat']==sel_sb]
    sg1,sg2,sg3,sg4 = st.columns(4)
    sg1.metric("Total Starts", len(sb_all)); sg2.metric("Dominant Starts", len(sb_dom))
    sg3.metric("Freq%", f"{len(sb_dom)/max(len(sb_all),1)*100:.1f}%")
    sg4.metric("Avg in Dominant", f"{sb_dom['Runs'].mean():.1f}" if len(sb_dom) > 0 else "N/A")
    si1,si2 = st.columns(2)
    with si1:
        ph_sb = sb_dom.groupby('phase',observed=True).agg(Count=('Runs','count')).reset_index()
        fig_si1 = px.bar(ph_sb, x='phase', y='Count', color='phase', text='Count',
                         color_discrete_map={'Powerplay (1–6)':'#636EFA','Middle (7–16)':'#EF553B','Death (17–20)':'#00CC96'},
                         title=f"{sel_sb} – Dominant Starts by Phase", height=320)
        st.plotly_chart(fig_si1, width="stretch")
    with si2:
        fig_si2 = px.histogram(sb_all, x='Runs', nbins=10,
                               title=f"{sel_sb} – First 3 Ball Score Distribution",
                               color_discrete_sequence=['#636EFA'], height=320)
        st.plotly_chart(fig_si2, width="stretch")

# ── MC TAB 2: Bowler Resilience (REBUILT) ─────────────────────────────────
with mc2:
    st.subheader("🎳 Bowler Resilience – Who Controls After a Bad Start?")
    st.caption("Bad start = first 3 balls go for X+ runs. Then see how the bowler responds in last 3 balls.")

    bad_thr = st.slider("Bad start threshold (first 3 balls runs)", 4, 15, 6, key='bt')

    # Build over-level data with ball-level detail
    bowl_ov = bowler_over_splits(dff)

    bad = bowl_ov[bowl_ov['first3_runs'] >= bad_thr].copy()

    # Three comeback tags
    bad['Strict']  = ((bad['last3_fours']==0) & (bad['last3_sixes']==0)).astype(int)
    bad['Good']    = ((bad['last3_fours']==1) & (bad['last3_sixes']==0)).astype(int)
    bad['Blown']   = ((bad['last3_sixes']>=1) | (bad['last3_fours']>=2)).astype(int)

    # Dot% and Boundary% in last 3
    bad['Last3_Dot%']      = (bad['last3_dots']/bad['last3_balls'].replace(0,np.nan)*100).round(1)
    bad['Last3_Boundary%'] = (bad['last3_boundaries']/bad['last3_balls'].replace(0,np.nan)*100).round(1)

    st.markdown(f"**Overs with {bad_thr}+ in first 3 balls: {len(bad):,}**")
    st.divider()

    st.markdown("### 🏆 Resilience Leaderboard")
    res = bad.groupby('bowl').agg(
        Bad_Starts   =('first3_runs','count'),
        Strict_Count =('Strict','sum'),
        Good_Count   =('Good','sum'),
        Blown_Count  =('Blown','sum'),
        Avg_First3   =('first3_runs','mean'),
        Avg_Last3    =('last3_runs','mean'),
        Dot_Pct      =('Last3_Dot%','mean'),
        Boundary_Pct =('Last3_Boundary%','mean'),
        Last3_Wkts   =('last3_dots','count')   # placeholder, fix below
    ).reset_index()

    # Recalculate last3 wickets properly
    last3_wkts = bad.groupby('bowl').apply(
        lambda x: x['last3_dots'].sum()  # dummy, replace
    ).reset_index()

    # Fix wickets - recompute from raw
    bowl_wkts = dff[dff['ball']>3].groupby('bowl').agg(Wkts=('bowl_wicket','sum')).reset_index()
    res = res.drop(columns=['Last3_Wkts'])
    res = pd.merge(res, bowl_wkts, on='bowl', how='left')

    res['Strict%']       = (res['Strict_Count']/res['Bad_Starts']*100).round(1)
    res['Good%']         = (res['Good_Count']/res['Bad_Starts']*100).round(1)
    res['Blown%']        = (res['Blown_Count']/res['Bad_Starts']*100).round(1)
    res['Avg_First3']    = res['Avg_First3'].round(1)
    res['Avg_Last3']     = res['Avg_Last3'].round(1)
    res['Dot_Pct']       = res['Dot_Pct'].round(1)
    res['Boundary_Pct']  = res['Boundary_Pct'].round(1)
    res = res[res['Bad_Starts']>=2].sort_values('Strict%',ascending=False).reset_index(drop=True)
    res.index += 1

    cr1,cr2 = st.columns([1.2,1])
    with cr1:
        fig_r = px.bar(res.head(15), x='bowl', y='Strict%',
                       color='Strict%', color_continuous_scale='RdYlGn',
                       text='Strict%', title=f"Strict Comeback % (0 boundaries in last 3)", height=400)
        fig_r.update_layout(xaxis_tickangle=-40)
        st.plotly_chart(fig_r, width="stretch")
    with cr2:
        fig_r2 = px.bar(res.head(15).sort_values('Good%',ascending=False),
                        x='bowl', y='Good%', color='Good%',
                        color_continuous_scale='Blues', text='Good%',
                        title="Good Comeback % (max 1 four, no sixes)", height=400)
        fig_r2.update_layout(xaxis_tickangle=-40)
        st.plotly_chart(fig_r2, width="stretch")

    st.markdown("**Full Resilience Table**")
    st.dataframe(res[['bowl','Bad_Starts','Strict_Count','Strict%','Good_Count','Good%',
                       'Blown_Count','Blown%','Avg_First3','Avg_Last3','Dot_Pct','Boundary_Pct']],
                 width="stretch")

    st.divider()
    # Blown It leaderboard
    st.markdown("### ❌ Who Ends Overs Badly? (Blown It)")
    blown_lb = res.sort_values('Blown%',ascending=False).head(15)
    cb1,cb2 = st.columns([1.2,1])
    with cb1:
        fig_bl = px.bar(blown_lb, x='bowl', y='Blown%',
                        color='Blown%', color_continuous_scale='Reds',
                        text='Blown%', title="Blown It % (sixes or 2+ boundaries)", height=380)
        fig_bl.update_layout(xaxis_tickangle=-40)
        st.plotly_chart(fig_bl, width="stretch")
    with cb2:
        fig_bnd = px.bar(res.sort_values('Boundary_Pct',ascending=False).head(15),
                         x='bowl', y='Boundary_Pct', color='Boundary_Pct',
                         color_continuous_scale='Reds', text='Boundary_Pct',
                         title="Boundary% in Last 3 Balls (higher=worse)", height=380)
        fig_bnd.update_layout(xaxis_tickangle=-40)
        st.plotly_chart(fig_bnd, width="stretch")

    st.divider()
    # Dot ball in last 3
    st.markdown("### 🎯 Dot Ball % in Last 3 Balls (higher=better)")
    fig_dot = px.bar(res.sort_values('Dot_Pct',ascending=False).head(15),
                     x='bowl', y='Dot_Pct', color='Dot_Pct',
                     color_continuous_scale='RdYlGn', text='Dot_Pct',
                     title="Dot% in Last 3 Balls After Bad Start", height=380)
    fig_dot.update_layout(xaxis_tickangle=-40)
    st.plotly_chart(fig_dot, width="stretch")

    st.divider()
    # Phase-wise — only Strict comeback by phase
    st.markdown("### 📊 Phase-wise Resilience")
    ph_res = bad.groupby(['bowl','phase'],observed=True).agg(
        Bad_Starts=('first3_runs','count'),
        Strict=('Strict','sum'), Good=('Good','sum'), Blown=('Blown','sum')
    ).reset_index()
    ph_res['Strict%'] = (ph_res['Strict']/ph_res['Bad_Starts']*100).round(1)
    top10_r = res.head(10)['bowl'].tolist()

    fig_pr1 = px.bar(ph_res[ph_res['bowl'].isin(top10_r)],
                     x='bowl', y='Strict%', color='phase', barmode='group',
                     color_discrete_map={'Powerplay (1–6)':'#636EFA','Middle (7–16)':'#EF553B','Death (17–20)':'#00CC96'},
                     title="Strict Comeback% by Phase – Top 10", height=400)
    fig_pr1.update_layout(xaxis_tickangle=-40)
    st.plotly_chart(fig_pr1, width="stretch")

    st.divider()
    # Team resilience leaderboard
    st.markdown("### 🏆 Team Resilience Leaderboard")
    st.caption("Which team's bowlers bounce back best after a bad start?")
    team_res = bad.groupby('bowl').agg(
        Bad_Starts=('first3_runs','count'),
        Strict=('Strict','sum'), Good=('Good','sum'), Blown=('Blown','sum'),
        Dot_Pct=('Last3_Dot%','mean'), Boundary_Pct=('Last3_Boundary%','mean')
    ).reset_index()
    # Map bowler to team
    bowl_team_map = dff.groupby('bowl')['team_bowl'].agg(lambda x: x.mode()[0]).to_dict()
    team_res['team'] = team_res['bowl'].astype(str).map(bowl_team_map)
    team_res_grp = team_res.groupby('team').agg(
        Bad_Starts=('Bad_Starts','sum'),
        Strict=('Strict','sum'), Good=('Good','sum'), Blown=('Blown','sum'),
        Dot_Pct=('Dot_Pct','mean'), Boundary_Pct=('Boundary_Pct','mean')
    ).reset_index()
    team_res_grp['Strict%']      = (team_res_grp['Strict']/team_res_grp['Bad_Starts']*100).round(1)
    team_res_grp['Good%']        = (team_res_grp['Good']/team_res_grp['Bad_Starts']*100).round(1)
    team_res_grp['Blown%']       = (team_res_grp['Blown']/team_res_grp['Bad_Starts']*100).round(1)
    team_res_grp['Dot_Pct']      = team_res_grp['Dot_Pct'].round(1)
    team_res_grp['Boundary_Pct'] = team_res_grp['Boundary_Pct'].round(1)
    team_res_grp = team_res_grp.sort_values('Strict%',ascending=False).reset_index(drop=True)
    team_res_grp.index += 1

    tr1,tr2 = st.columns([1.2,1])
    with tr1:
        fig_tr = px.bar(team_res_grp, x='team', y='Strict%',
                        color='Strict%', color_continuous_scale='RdYlGn',
                        text='Strict%', title="Team – Strict Comeback% (Bowlers)", height=400)
        fig_tr.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig_tr, width="stretch")
    with tr2:
        st.dataframe(team_res_grp[['team','Bad_Starts','Strict%','Good%','Blown%','Dot_Pct','Boundary_Pct']],
                     width="stretch", height=380)

    st.divider()
    # Individual bowler
    st.markdown("### 🔍 Individual Bowler Resilience Profile")
    sel_rb = st.selectbox("Select Bowler", sorted(bad['bowl'].unique()), key='rb')
    rb = bad[bad['bowl']==sel_rb]

    rg1,rg2,rg3,rg4,rg5,rg6 = st.columns(6)
    rg1.metric("Bad Starts", len(rb))
    rg2.metric("Strict Comebacks", int(rb['Strict'].sum()))
    rg3.metric("Strict%", f"{rb['Strict'].mean()*100:.1f}%")
    rg4.metric("Good Comebacks", int(rb['Good'].sum()))
    rg5.metric("Blown It", int(rb['Blown'].sum()))
    rg6.metric("Avg Last 3 Runs", f"{rb['last3_runs'].mean():.1f}")

    ri1,ri2 = st.columns(2)
    with ri1:
        # Tag breakdown pie
        tag_data = pd.DataFrame({
            'Tag':  ['Strict','Good','Blown'],
            'Count':[int(rb['Strict'].sum()), int(rb['Good'].sum()), int(rb['Blown'].sum())]
        })
        fig_ri1 = px.pie(tag_data, names='Tag', values='Count',
                         color='Tag', color_discrete_map={'Strict':'green','Good':'orange','Blown':'red'},
                         title=f"{sel_rb} – Comeback Tag Breakdown", height=360)
        st.plotly_chart(fig_ri1, width="stretch")
    with ri2:
        ph_rb = rb.groupby('phase',observed=True).agg(
            Bad=('first3_runs','count'), Strict=('Strict','sum'),
            Good=('Good','sum'), Blown=('Blown','sum')
        ).reset_index()
        ph_rb['Strict%'] = (ph_rb['Strict']/ph_rb['Bad']*100).round(1)
        ph_rb['Blown%']  = (ph_rb['Blown']/ph_rb['Bad']*100).round(1)
        fig_ri2 = px.bar(ph_rb, x='phase', y='Strict%', color='phase', text='Strict%',
                         color_discrete_map={'Powerplay (1–6)':'#636EFA','Middle (7–16)':'#EF553B','Death (17–20)':'#00CC96'},
                         title=f"{sel_rb} – Strict Comeback% by Phase", height=360)
        st.plotly_chart(fig_ri2, width="stretch")

    ri3,ri4 = st.columns(2)
    with ri3:
        fig_ri3 = px.scatter(rb, x='first3_runs', y='last3_runs',
                             color=rb.apply(lambda r: 'Strict' if r['Strict'] else ('Good' if r['Good'] else 'Blown'), axis=1),
                             color_discrete_map={'Strict':'green','Good':'orange','Blown':'red'},
                             title=f"{sel_rb} – First 3 vs Last 3 Runs", height=360,
                             labels={'first3_runs':'First 3 Runs','last3_runs':'Last 3 Runs'})
        st.plotly_chart(fig_ri3, width="stretch")
    with ri4:
        fig_ri4 = px.histogram(rb, x='last3_runs', nbins=10,
                               color=rb.apply(lambda r: 'Strict' if r['Strict'] else ('Good' if r['Good'] else 'Blown'), axis=1),
                               color_discrete_map={'Strict':'green','Good':'orange','Blown':'red'},
                               title=f"{sel_rb} – Last 3 Ball Runs Distribution", height=360)
        st.plotly_chart(fig_ri4, width="stretch")

# ── MC TAB 3: Post-Wicket Scoring (rebuilt for DPL) ───────────────────────
with mc3:
    st.subheader("🩹 Post-Wicket Scoring – How Batters React to a Wicket")
    st.caption("Every wicket is an event. **New batter** = the one who walks in. "
               "**Not-dismissed batter** = the one already at the crease. Nothing is counted before "
               "the first wicket of an innings. Balls = balls faced (wides excluded, no-balls counted).")
    pw_n = st.slider("N balls (window, and wickets within N balls = one event)", 4, 12, 6, key='pw')

    recs, mem = post_wicket_engine(df, pw_n)
    R = post_wicket_totals(df, recs, mem, dff.index)
    names = df.drop_duplicates('p_bat').set_index('p_bat')['bat']
    R['bat'] = R['p_bat'].map(names)
    unsure = R[R['uncertain'] == True]
    R = R[R['uncertain'] != True]
    new_r  = R[R['role']=='new']
    ovr_r  = R[R['role']=='nd_overall']
    evt_r  = R[R['role']=='nd_event'].copy()
    evt_r['Event'] = np.where(evt_r['size']>=3, '3+ wickets',
                              np.where(evt_r['size']==2, '2 wickets', '1 wicket'))
    EVENTS = ['1 wicket', '2 wickets', '3+ wickets']

    def sr(r, b):
        return (r / b.where(b > 0) * 100).round(1)

    if not unsure.empty:
        with st.expander(f"⚠️ {unsure['wicket'].count()} records left out: run-outs where it can't be told who walked in"):
            st.caption("The other batter hadn't faced a ball yet when the run-out happened, and the data "
                       "doesn't record the non-striker, so these records are not counted anywhere below.")
            st.dataframe(unsure[['year','team_bat','p_match','inns','wicket','role','bat']].rename(columns={
                'year':'Year','team_bat':'Team','p_match':'Match','inns':'Inns','wicket':'Wicket at',
                'role':'Record','bat':'Batter as recorded'}), hide_index=True, width="stretch")
    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Wickets (new batters)", len(new_r))
    for col, ev in zip([k2,k3,k4], EVENTS):
        col.metric(f"{ev} events", evt_r.loc[evt_r['Event']==ev, 'event'].nunique())
    st.divider()

    # ── New batter ──
    st.markdown(f"### 🚶 New Batter – First {pw_n} Balls")
    nb = new_r.groupby('bat').agg(Times=('runs','count'), Balls=('balls','sum'), Runs=('runs','sum'),
                                   Dots=('dots','sum'), Fours=('fours','sum'), Sixes=('sixes','sum'),
                                   Out_Within_N=('got_out','sum')).reset_index()
    nb = nb[nb['Times'] >= 2]
    nb['SR'] = sr(nb['Runs'], nb['Balls'])
    nb['Dot%'] = (nb['Dots']/nb['Balls']*100).round(1)
    nb['Out%'] = (nb['Out_Within_N']/nb['Times']*100).round(1)
    nb = nb.sort_values('SR', ascending=False).reset_index(drop=True); nb.index += 1
    n1,n2 = st.columns([1.2,1])
    with n1:
        fig = px.bar(nb.head(15), x='bat', y='SR', color='SR', color_continuous_scale='RdYlGn',
                     text='SR', title=f"New Batter SR – First {pw_n} Balls", height=400)
        fig.update_layout(xaxis_tickangle=-40)
        st.plotly_chart(fig, width="stretch")
    with n2:
        st.dataframe(nb[['bat','Times','Balls','Runs','SR','Dot%','Fours','Sixes','Out_Within_N','Out%']],
                     width="stretch", height=380)

    st.divider()
    # ── Not-dismissed: overall ──
    st.markdown("### 🧱 Not-Dismissed Batter – Overall After the Wicket")
    st.caption(f"From the first wicket he survives until he gets out or the innings ends. "
               f"SR Before = his last {pw_n} balls before that wicket.")
    od = ovr_r.groupby('bat').agg(Innings=('runs','count'), Balls=('balls','sum'), Runs=('runs','sum'),
                                   B_Runs=('before_runs','sum'), B_Balls=('before_balls','sum')).reset_index()
    od = od[od['Innings'] >= 2]
    od['SR_After'] = sr(od['Runs'], od['Balls']); od['SR_Before'] = sr(od['B_Runs'], od['B_Balls'])
    od['Change'] = (od['SR_After'] - od['SR_Before']).round(1)
    od = od.sort_values('SR_After', ascending=False).reset_index(drop=True); od.index += 1
    o1,o2 = st.columns([1.2,1])
    with o1:
        fig = px.bar(od.head(15), x='bat', y='SR_After', color='Change', color_continuous_scale='RdYlGn',
                     text='SR_After', title="SR After the Wicket (colour = change vs before)", height=400)
        fig.update_layout(xaxis_tickangle=-40)
        st.plotly_chart(fig, width="stretch")
    with o2:
        st.dataframe(od[['bat','Innings','Balls','Runs','SR_After','SR_Before','Change']],
                     width="stretch", height=380)

    st.divider()
    # ── Not-dismissed: by wicket event ──
    st.markdown("### 💥 Not-Dismissed Batter – By Wicket Event")
    st.caption(f"Wickets falling within {pw_n} balls of each other form one event. Counting runs from the "
               f"wicket he survives until {pw_n} balls pass with no further wicket, he gets out, "
               f"or the innings ends. SR Before = his last {pw_n} balls before the event.")
    ev_all = evt_r.groupby('Event').agg(Events=('runs','count'), Balls=('balls','sum'),
                                         Runs=('runs','sum')).reindex(EVENTS).reset_index()
    ev_all['SR'] = sr(ev_all['Runs'], ev_all['Balls'])
    e1,e2 = st.columns([1,1.2])
    with e1:
        fig = px.bar(ev_all, x='Event', y='SR', text='SR', color='Event',
                     color_discrete_sequence=['#3D7A5F','#D29B3C','#A4243B'],
                     title="All batters – SR by event size", height=340)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width="stretch")
    with e2:
        sel_ev = st.radio("Event size", EVENTS, horizontal=True, key='pw_ev')
        eb = evt_r[evt_r['Event']==sel_ev].groupby('bat').agg(
            Events=('runs','count'), Balls=('balls','sum'), Runs=('runs','sum'),
            B_Runs=('before_runs','sum'), B_Balls=('before_balls','sum')).reset_index()
        eb['SR'] = sr(eb['Runs'], eb['Balls']); eb['SR_Before'] = sr(eb['B_Runs'], eb['B_Balls'])
        eb['Change'] = (eb['SR'] - eb['SR_Before']).round(1)
        eb = eb.sort_values(['Events','SR'], ascending=False).reset_index(drop=True); eb.index += 1
        st.dataframe(eb[['bat','Events','Balls','Runs','SR','SR_Before','Change']],
                     width="stretch", height=300)

    st.divider()
    # ── Team ──
    st.markdown("### 🏆 Team Post-Wicket Leaderboard")
    t_new = new_r.groupby('team_bat').agg(NB_Runs=('runs','sum'), NB_Balls=('balls','sum')).reset_index()
    t_nd  = ovr_r.groupby('team_bat').agg(ND_Runs=('runs','sum'), ND_Balls=('balls','sum')).reset_index()
    tm = t_new.merge(t_nd, on='team_bat', how='outer').fillna(0)
    tm['New_Batter_SR'] = sr(tm['NB_Runs'], tm['NB_Balls'])
    tm['Not_Dismissed_SR'] = sr(tm['ND_Runs'], tm['ND_Balls'])
    tm = tm.sort_values('New_Batter_SR', ascending=False).reset_index(drop=True); tm.index += 1
    t1,t2 = st.columns([1.2,1])
    with t1:
        fig = px.bar(tm, x='team_bat', y=['New_Batter_SR','Not_Dismissed_SR'], barmode='group',
                     color_discrete_map={'New_Batter_SR':'#636EFA','Not_Dismissed_SR':'#00CC96'},
                     title="Team – New Batter vs Not-Dismissed Batter SR", height=400)
        fig.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig, width="stretch")
    with t2:
        st.dataframe(tm[['team_bat','NB_Balls','New_Batter_SR','ND_Balls','Not_Dismissed_SR']],
                     width="stretch", height=380)

    st.divider()
    # ── Individual profile ──
    st.markdown("### 🔍 Individual Post-Wicket Profile")
    sel_pw = st.selectbox("Select Batter", sorted(R['bat'].dropna().unique()), key='pw_bat')
    me_new, me_ovr, me_evt = (x[x['bat']==sel_pw] for x in (new_r, ovr_r, evt_r))
    st.markdown("**As the new batter**")
    a1,a2,a3,a4 = st.columns(4)
    a1.metric("Times Walked In", len(me_new))
    a2.metric(f"SR First {pw_n} Balls", f"{sr(me_new['runs'].sum(), pd.Series([me_new['balls'].sum()])).iat[0]}"
              if me_new['balls'].sum() else "–")
    a3.metric("Out Within N", int(me_new['got_out'].sum()))
    a4.metric("4s / 6s", f"{int(me_new['fours'].sum())} / {int(me_new['sixes'].sum())}")
    st.markdown("**As the not-dismissed batter**")
    b1,b2,b3,b4 = st.columns(4)
    b1.metric("Innings", len(me_ovr))
    rb, bb = me_ovr['runs'].sum(), me_ovr['balls'].sum()
    b2.metric("Runs / Balls After", f"{int(rb)} / {int(bb)}")
    b3.metric("SR After", f"{rb/bb*100:.1f}" if bb else "–")
    brb, bbb = me_ovr['before_runs'].sum(), me_ovr['before_balls'].sum()
    b4.metric("SR Before", f"{brb/bbb*100:.1f}" if bbb else "–")
    if not me_evt.empty:
        pe = me_evt.groupby('Event').agg(Events=('runs','count'), Balls=('balls','sum'),
                                         Runs=('runs','sum')).reindex(EVENTS).dropna().reset_index()
        pe['SR'] = sr(pe['Runs'], pe['Balls'])
        st.dataframe(pe, hide_index=True, width="stretch")
    log = pd.concat([me_new.assign(Role='New batter'), me_evt.assign(Role='Not dismissed')])
    if not log.empty:
        log = log.sort_values(['year','p_match','rec_id'])
        log['Event'] = log['Event'].fillna('')
        st.markdown("**Every post-wicket record**")
        st.dataframe(log[['year','p_match','wicket','Role','Event','balls','runs','got_out']].rename(columns={
            'year':'Year','p_match':'Match','wicket':'Wicket at','balls':'Balls','runs':'Runs','got_out':'Out'}),
            hide_index=True, width="stretch", height=260)

# ── MC TAB 4: Team Momentum ───────────────────────────────────────────────
with mc4:
    st.subheader("📊 Team Momentum – Batting vs Bowling Over Analysis")
    st.caption("Classify every over by runs scored/conceded and see which teams dominate")

    # Over-level data including extras (all deliveries per over)
    over_full = dff.groupby(['p_match','inns','over','team_bat','team_bowl']).agg(
        over_runs=('bowlruns','sum'),
        balls    =('score','count'),
        wickets  =('bowl_wicket','sum')
    ).reset_index()

    # Classification function
    def classify_over(r):
        if r < 7:   return '🔒 Dot Dominant (0-6)'
        if r < 10:  return '💤 Soft Over (7-9)'
        if r < 12:  return '⚡ Impact Over (10-11)'
        if r < 20:  return '🔥 High Capacity (12-19)'
        return '💥 Game Changer (20+)'

    over_full['Classification'] = over_full['over_runs'].apply(classify_over)
    cat_order = ['🔒 Dot Dominant (0-6)','💤 Soft Over (7-9)','⚡ Impact Over (10-11)',
                 '🔥 High Capacity (12-19)','💥 Game Changer (20+)']
    over_full['Classification'] = pd.Categorical(over_full['Classification'],
                                                  categories=cat_order, ordered=True)
    color_map = {
        '🔒 Dot Dominant (0-6)' :'#2ecc71',
        '💤 Soft Over (7-9)'    :'#f1c40f',
        '⚡ Impact Over (10-11)' :'#e67e22',
        '🔥 High Capacity (12-19)':'#e74c3c',
        '💥 Game Changer (20+)'  :'#9b59b6'
    }

    run_thr_tm = st.slider("Threshold – highlight overs above this run total", 0, 36, 10, key='tm_thr')
    st.divider()

    # ── BATTING SIDE ──
    st.markdown("### 🏏 Batting – Which Teams Score Big Overs?")

    bat_overs = over_full.copy()
    bat_overs['Above_Threshold'] = (bat_overs['over_runs'] >= run_thr_tm).astype(int)

    # Team batting classification breakdown
    team_bat_cls = bat_overs.groupby(['team_bat','Classification'], observed=True).agg(
        Count=('over_runs','count')
    ).reset_index()
    team_bat_total = bat_overs.groupby('team_bat').agg(Total_Overs=('over_runs','count')).reset_index()
    team_bat_cls = pd.merge(team_bat_cls, team_bat_total, on='team_bat', how='left')
    team_bat_cls['Freq%'] = (team_bat_cls['Count']/team_bat_cls['Total_Overs']*100).round(1)

    # Above threshold leaderboard
    bat_thr = bat_overs[bat_overs['over_runs'] >= run_thr_tm].groupby('team_bat').agg(
        Above_Count=('over_runs','count'),
        Avg_Runs   =('over_runs','mean'),
        Max_Over   =('over_runs','max')
    ).reset_index()
    bat_thr = pd.merge(bat_thr, team_bat_total, on='team_bat', how='left')
    bat_thr['Freq%'] = (bat_thr['Above_Count']/bat_thr['Total_Overs']*100).round(1)
    bat_thr['Avg_Runs'] = bat_thr['Avg_Runs'].round(1)
    bat_thr = bat_thr.sort_values('Above_Count',ascending=False).reset_index(drop=True)
    bat_thr.index += 1

    cb1,cb2 = st.columns([1.2,1])
    with cb1:
        fig_bt = px.bar(bat_thr, x='team_bat', y='Above_Count',
                        color='Freq%', color_continuous_scale='RdYlGn',
                        text='Above_Count',
                        title=f"Teams with Most {run_thr_tm}+ Run Overs (Batting)", height=380)
        fig_bt.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig_bt, width="stretch")
    with cb2:
        st.dataframe(bat_thr[['team_bat','Above_Count','Total_Overs','Freq%','Avg_Runs','Max_Over']],
                     width="stretch", height=360)

    # Classification stacked bar — batting
    fig_bc = px.bar(team_bat_cls, x='team_bat', y='Count', color='Classification',
                    color_discrete_map=color_map, barmode='stack',
                    title="Batting – Over Classification Breakdown per Team", height=420)
    fig_bc.update_layout(xaxis_tickangle=-30)
    st.plotly_chart(fig_bc, width="stretch")

    # Freq% heatmap batting
    bat_heat = team_bat_cls.pivot_table(index='team_bat', columns='Classification', values='Freq%', fill_value=0)
    fig_bh = px.imshow(bat_heat[cat_order], text_auto=True, color_continuous_scale='RdYlGn',
                       title="Batting – Over Classification Frequency % Heatmap", height=500)
    fig_bh.update_traces(textfont_size=13)
    st.plotly_chart(fig_bh, width="stretch")

    st.divider()

    # ── BOWLING SIDE ──
    st.markdown("### 🎳 Bowling – Which Teams Concede Big Overs?")

    bowl_overs = over_full.copy()
    bowl_overs['Above_Threshold'] = (bowl_overs['over_runs'] >= run_thr_tm).astype(int)

    team_bowl_cls = bowl_overs.groupby(['team_bowl','Classification'], observed=True).agg(
        Count=('over_runs','count')
    ).reset_index()
    team_bowl_total = bowl_overs.groupby('team_bowl').agg(Total_Overs=('over_runs','count')).reset_index()
    team_bowl_cls = pd.merge(team_bowl_cls, team_bowl_total, on='team_bowl', how='left')
    team_bowl_cls['Freq%'] = (team_bowl_cls['Count']/team_bowl_cls['Total_Overs']*100).round(1)

    bowl_thr = bowl_overs[bowl_overs['over_runs'] >= run_thr_tm].groupby('team_bowl').agg(
        Above_Count=('over_runs','count'),
        Avg_Runs   =('over_runs','mean'),
        Max_Over   =('over_runs','max')
    ).reset_index()
    bowl_thr = pd.merge(bowl_thr, team_bowl_total, on='team_bowl', how='left')
    bowl_thr['Freq%'] = (bowl_thr['Above_Count']/bowl_thr['Total_Overs']*100).round(1)
    bowl_thr['Avg_Runs'] = bowl_thr['Avg_Runs'].round(1)
    bowl_thr = bowl_thr.sort_values('Above_Count',ascending=False).reset_index(drop=True)
    bowl_thr.index += 1

    cw1,cw2 = st.columns([1.2,1])
    with cw1:
        fig_bwt = px.bar(bowl_thr, x='team_bowl', y='Above_Count',
                         color='Freq%', color_continuous_scale='RdYlGn_r',
                         text='Above_Count',
                         title=f"Teams Conceding Most {run_thr_tm}+ Run Overs (Bowling)", height=380)
        fig_bwt.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig_bwt, width="stretch")
    with cw2:
        st.dataframe(bowl_thr[['team_bowl','Above_Count','Total_Overs','Freq%','Avg_Runs','Max_Over']],
                     width="stretch", height=360)

    # Classification stacked bar — bowling
    fig_bwc = px.bar(team_bowl_cls, x='team_bowl', y='Count', color='Classification',
                     color_discrete_map=color_map, barmode='stack',
                     title="Bowling – Over Classification Breakdown per Team", height=420)
    fig_bwc.update_layout(xaxis_tickangle=-30)
    st.plotly_chart(fig_bwc, width="stretch")

    st.divider()

    # ── INDIVIDUAL BOWLER BREAKDOWN per team ──
    st.markdown("### 🔍 Which Bowler Leaked the Most Big Overs?")
    sel_tm_team = st.selectbox("Select Team (Bowling)", sorted(over_full['team_bowl'].unique()), key='tm_team')

    # Over-level per bowler
    over_bowl = dff.groupby(['p_match','inns','over','bowl','team_bowl']).agg(
        over_runs=('bowlruns','sum'), balls=('bowlruns','count')
    ).reset_index()
    over_bowl['Classification'] = over_bowl['over_runs'].apply(classify_over)

    team_bowl_df = over_bowl[over_bowl['team_bowl']==sel_tm_team]
    bowl_leak = team_bowl_df[team_bowl_df['over_runs'] >= run_thr_tm].groupby('bowl').agg(
        Big_Overs =('over_runs','count'),
        Avg_Runs  =('over_runs','mean'),
        Max_Over  =('over_runs','max')
    ).reset_index()
    bowl_total_ov = team_bowl_df.groupby('bowl').agg(Total_Overs=('over_runs','count')).reset_index()
    bowl_leak = pd.merge(bowl_leak, bowl_total_ov, on='bowl', how='left')
    bowl_leak['Freq%']    = (bowl_leak['Big_Overs']/bowl_leak['Total_Overs']*100).round(1)
    bowl_leak['Avg_Runs'] = bowl_leak['Avg_Runs'].round(1)
    bowl_leak = bowl_leak.sort_values('Big_Overs',ascending=False).reset_index(drop=True)
    bowl_leak.index += 1

    bl1,bl2 = st.columns([1.2,1])
    with bl1:
        fig_bl = px.bar(bowl_leak, x='bowl', y='Big_Overs',
                        color='Freq%', color_continuous_scale='Reds',
                        text='Big_Overs',
                        title=f"{sel_tm_team} – Bowler Big Over Leakage ({run_thr_tm}+)", height=400)
        fig_bl.update_layout(xaxis_tickangle=-40)
        st.plotly_chart(fig_bl, width="stretch")
    with bl2:
        st.dataframe(bowl_leak[['bowl','Big_Overs','Total_Overs','Freq%','Avg_Runs','Max_Over']],
                     width="stretch", height=380)

    # Classification breakdown per bowler in selected team
    bowl_cls = team_bowl_df.groupby(['bowl','Classification'], observed=True).agg(
        Count=('over_runs','count')).reset_index()
    fig_bcls = px.bar(bowl_cls, x='bowl', y='Count', color='Classification',
                      color_discrete_map=color_map, barmode='stack',
                      title=f"{sel_tm_team} – Bowler Over Classification Breakdown", height=420)
    fig_bcls.update_layout(xaxis_tickangle=-40)
    st.plotly_chart(fig_bcls, width="stretch")

    st.divider()

    # ── BATTING vs BOWLING COMPARISON per team ──
    st.markdown("### ⚔️ Batting vs Bowling – Who Dominates? Who Leaks?")
    compare = pd.merge(
        bat_thr[['team_bat','Above_Count','Freq%']].rename(columns={'team_bat':'team','Above_Count':'Bat_Big_Overs','Freq%':'Bat_Freq%'}),
        bowl_thr[['team_bowl','Above_Count','Freq%']].rename(columns={'team_bowl':'team','Above_Count':'Bowl_Big_Overs','Freq%':'Bowl_Freq%'}),
        on='team', how='outer'
    ).fillna(0)
    compare['Net'] = compare['Bat_Big_Overs'] - compare['Bowl_Big_Overs']
    compare = compare.sort_values('Net',ascending=False).reset_index(drop=True)
    compare.index += 1

    cmp1,cmp2 = st.columns([1.2,1])
    with cmp1:
        fig_cmp = px.bar(compare, x='team', y=['Bat_Big_Overs','Bowl_Big_Overs'],
                         barmode='group',
                         color_discrete_map={'Bat_Big_Overs':'#2ecc71','Bowl_Big_Overs':'#e74c3c'},
                         title=f"Batting vs Bowling Big Overs ({run_thr_tm}+) per Team", height=420)
        fig_cmp.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig_cmp, width="stretch")