"""
Game Changers | Data Premier League 2026 (IPL 2023–24)
Batters who produce big overs: an "Impact Over" is an over in which the batter
scores at least the threshold runs off the bat.
Logic ported from the Game Changers tab in ipl_dashboard.py.

Run: streamlit run game_changers.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Game Changers | DPL 2026", layout="wide", page_icon="🏏")

DATA_FILE = Path(__file__).parent / "Data_Premier_League_Prelims_Dataset.csv"
PHASES = ['Powerplay (1–6)', 'Middle (7–16)', 'Death (17–20)']
USE_COLS = ['p_match', 'inns', 'over_num', 'bat', 'team_bat', 'team_bowl', 'bowl_kind',
            'batruns', 'ballfaced', 'out', 'year', 'bat_hand']

# Over Band Profile: runs per ball the batter scored in one over
BANDS = ['Negative Play', 'Solidify', 'Par Hitting', 'Positive Impact', 'High Ceiling', 'Game Changer']
BAND_RANGE = {'Negative Play': '1.00 or less', 'Solidify': '1.01 – 1.49', 'Par Hitting': '1.50 – 1.99',
              'Positive Impact': '2.00 – 2.49', 'High Ceiling': '2.50 – 2.99', 'Game Changer': '3.00+'}
BAND_COLORS = {'Negative Play': '#9AA3AE', 'Solidify': '#6FA8B8', 'Par Hitting': '#3D7A5F',
               'Positive Impact': '#D29B3C', 'High Ceiling': '#E0662F', 'Game Changer': '#A4243B'}


def band_of(rpb):
    return np.select([rpb <= 1.0, rpb < 1.5, rpb < 2.0, rpb < 2.5, rpb < 3.0],
                     BANDS[:5], default=BANDS[5])


@st.cache_data(show_spinner="Loading deliveries…")
def load_data(source):
    df = pd.read_csv(source, usecols=USE_COLS)
    df = df.rename(columns={'over_num': 'over'})
    df['bat'] = df['bat'].str.strip()
    df['team_bat'] = df['team_bat'].str.strip().str.replace('Royal Challengers Bengaluru', 'RCB')
    df['batruns'] = pd.to_numeric(df['batruns'], errors='coerce').fillna(0).astype(int)
    df['out'] = df['out'].astype(int)
    df['ballfaced'] = pd.to_numeric(df['ballfaced'], errors='coerce').fillna(0).astype(int)
    df['team_bowl'] = df['team_bowl'].str.strip()
    df['bowl_kind'] = df['bowl_kind'].fillna('mixture/unknown')
    df['phase'] = pd.cut(df['over'], bins=[0, 6, 16, 20], labels=PHASES)
    return df


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

st.title("💥 Game Changers")
st.caption(f"IPL {'–'.join(str(y) for y in sel_year) if sel_year else ''}  |  "
           f"**{dff['p_match'].nunique()}** matches  |  **{dff['bat'].nunique()}** batters")
if dff.empty:
    st.warning("No data matches current filters.")
    st.stop()

over_data = dff.groupby(
    ['p_match','inns','over','bat','team_bat','team_bowl','year','bowl_kind','phase'], observed=True
).agg(over_runs=('batruns','sum'), balls_faced=('ballfaced','sum'), outs=('out','sum')).reset_index()

# ── Min overs batted (sidebar) ────────────────────────────────────────────────
# An "over batted" = any over in which the batter faced at least one ball.
bat_overs = over_data.groupby('bat').size()
max_ov = int(bat_overs.max())
if max_ov > 1:
    min_ov = st.sidebar.slider("Min overs batted", 1, max_ov, 1,
                               help="Only show batters who batted in at least this many overs")
else:
    min_ov = 1
keep = bat_overs[bat_overs >= min_ov].index
over_data = over_data[over_data['bat'].isin(keep)]
st.caption(f"**{len(keep)}** batters with **{min_ov}+** overs batted")

band_data = dff.groupby(['p_match','inns','over','bat'], observed=True).agg(
    runs=('batruns','sum'), balls=('ballfaced','sum')).reset_index()
band_data = band_data[band_data['balls'] > 0]          # ignore overs where he faced only wides
ball_opts = list(range(1, int(band_data['balls'].max()) + 1))
sel_balls = st.sidebar.multiselect("Balls faced in over (Over Band Profile only)", ball_opts,
                                   default=ball_opts)

run_thr = st.slider("Impact Over: Min runs in one over", 6, 20, 10)
impact  = over_data[over_data['over_runs'] >= run_thr].copy()

if impact.empty:
    st.warning("No data. Lower the threshold.")
else:
    tot_ov = over_data.groupby('bat').agg(Total_Overs_Batted=('over_runs','count')).reset_index()

    def add_balls(d):
        d['Avg_Balls'] = (d['Total_Balls']/d['Impact_Overs']).round(1)
        d['Runs_per_Ball'] = (d['Total_Runs']/d['Total_Balls']).round(2)
        return d

    def add_freq(d):
        d = add_balls(d)
        d = pd.merge(d, tot_ov, on='bat', how='left')
        d['Impact_Freq%'] = (d['Impact_Overs']/d['Total_Overs_Batted']*100).round(1)
        return d

    st.markdown("### 🏆 Overall Leaderboard")
    overall = impact.groupby('bat').agg(
        Impact_Overs=('over_runs','count'), Total_Runs=('over_runs','sum'),
        Best_Over=('over_runs','max'), Total_Balls=('balls_faced','sum'), Avg_Runs=('over_runs','mean')
    ).reset_index().sort_values('Impact_Overs',ascending=False).reset_index(drop=True)
    overall = add_freq(overall); overall['Avg_Runs'] = overall['Avg_Runs'].round(1); overall.index += 1

    c1,c2 = st.columns([1.2,1])
    with c1:
        fig = px.bar(overall.head(15), x='bat', y='Impact_Overs',
                     color='Impact_Overs', color_continuous_scale='Plasma',
                     text='Impact_Overs', title=f"Most {run_thr}+ Run Overs", height=400)
        fig.update_layout(xaxis_tickangle=-40)
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.dataframe(overall[['bat','Impact_Overs','Total_Overs_Batted','Impact_Freq%','Total_Runs','Best_Over','Avg_Runs','Avg_Balls','Runs_per_Ball']],
                     width="stretch", height=380)

    st.divider()
    freq_df = overall.sort_values('Impact_Freq%',ascending=False).reset_index(drop=True)
    cf1,cf2 = st.columns([1.2,1])
    with cf1:
        fig_f = px.bar(freq_df.head(15), x='bat', y='Impact_Freq%',
                       color='Impact_Freq%', color_continuous_scale='RdYlGn',
                       text='Impact_Freq%', title="Highest Impact Freq%", height=400)
        fig_f.update_layout(xaxis_tickangle=-40)
        st.plotly_chart(fig_f, width="stretch")
    with cf2:
        st.dataframe(freq_df[['bat','Impact_Freq%','Impact_Overs','Total_Overs_Batted','Total_Runs','Best_Over','Avg_Runs','Avg_Balls','Runs_per_Ball']],
                     width="stretch", height=380)

    st.divider()
    st.markdown("### ⚡ Pace vs Spin")
    cp1,cp2 = st.columns(2)
    for col, kind, cscale, title in [
        (cp1,'pace bowler','Reds','vs PACE'),
        (cp2,'spin bowler','Blues','vs SPIN')
    ]:
        with col:
            lb = impact[impact['bowl_kind']==kind].groupby('bat').agg(
                Impact_Overs=('over_runs','count'), Total_Runs=('over_runs','sum'), Best_Over=('over_runs','max'), Total_Balls=('balls_faced','sum')
            ).reset_index().sort_values('Impact_Overs',ascending=False).reset_index(drop=True)
            lb = add_freq(lb); lb.index += 1
            fig_k = px.bar(lb.head(10), x='bat', y='Impact_Overs',
                           color='Impact_Overs', color_continuous_scale=cscale,
                           text='Impact_Overs', title=f"Top 10 {title}", height=360)
            fig_k.update_layout(xaxis_tickangle=-40)
            st.plotly_chart(fig_k, width="stretch")
            st.dataframe(lb[['bat','Impact_Overs','Total_Overs_Batted','Impact_Freq%','Total_Runs','Best_Over','Avg_Balls','Runs_per_Ball']], width="stretch")

    st.divider()
    st.markdown("### 📊 By Phase")
    for ph_n, color in zip(['Powerplay (1–6)','Middle (7–16)','Death (17–20)'],['Teal','Oranges','Purples']):
        st.markdown(f"#### 🏏 {ph_n}")
        ph_tot = over_data[over_data['phase'].astype(str)==ph_n].groupby('bat').agg(
            Total_Overs_Batted=('over_runs','count')).reset_index()
        ph_lb = impact[impact['phase'].astype(str)==ph_n].groupby('bat').agg(
            Impact_Overs=('over_runs','count'), Total_Runs=('over_runs','sum'),
            Best_Over=('over_runs','max'), Total_Balls=('balls_faced','sum'), Avg_Runs=('over_runs','mean')
        ).reset_index().sort_values('Impact_Overs',ascending=False).reset_index(drop=True)
        ph_lb = pd.merge(ph_lb, ph_tot, on='bat', how='left')
        ph_lb['Impact_Freq%'] = (ph_lb['Impact_Overs']/ph_lb['Total_Overs_Batted']*100).round(1)
        ph_lb['Avg_Runs'] = ph_lb['Avg_Runs'].round(1); ph_lb = add_balls(ph_lb)
        ca,cb,cc = st.columns(3)
        with ca:
            fig_a = px.bar(ph_lb.head(15), x='bat', y='Impact_Overs',
                           color='Impact_Overs', color_continuous_scale=color,
                           text='Impact_Overs', title="Impact Overs Count", height=320)
            fig_a.update_layout(xaxis_tickangle=-40)
            st.plotly_chart(fig_a, width="stretch")
        with cb:
            fig_b = px.bar(ph_lb.sort_values('Impact_Freq%',ascending=False).head(15),
                           x='bat', y='Impact_Freq%', color='Impact_Freq%',
                           color_continuous_scale='RdYlGn', text='Impact_Freq%',
                           title="Impact Freq%", height=320)
            fig_b.update_layout(xaxis_tickangle=-40)
            st.plotly_chart(fig_b, width="stretch")
        with cc:
            ph_lb.index = ph_lb.index + 1
            st.dataframe(ph_lb[['bat','Impact_Overs','Total_Overs_Batted','Impact_Freq%','Total_Runs','Best_Over','Avg_Runs','Avg_Balls','Runs_per_Ball']],
                         width="stretch", height=320)

    st.divider()
    st.markdown("### 🎯 Over-wise Leaderboard")
    sel_ov = st.slider("Select Over", 1, 20, 1, key='gc_ov')
    ov_f   = impact[impact['over']==sel_ov]
    if ov_f.empty:
        st.info(f"No {run_thr}+ run overs in Over {sel_ov}.")
    else:
        ov_tot = over_data[over_data['over']==sel_ov].groupby('bat').agg(
            Total_Times_Batted=('over_runs','count')).reset_index()
        ov_lb = ov_f.groupby('bat').agg(
            Impact_Overs=('over_runs','count'), Total_Runs=('over_runs','sum'),
            Best_Over=('over_runs','max'), Total_Balls=('balls_faced','sum'), Avg_Runs=('over_runs','mean')
        ).reset_index()
        ov_lb = pd.merge(ov_lb, ov_tot, on='bat', how='left')
        ov_lb['Impact_Freq%'] = (ov_lb['Impact_Overs']/ov_lb['Total_Times_Batted']*100).round(1)
        ov_lb['Avg_Runs'] = ov_lb['Avg_Runs'].round(1); ov_lb = add_balls(ov_lb)
        ov_lb = ov_lb.sort_values('Impact_Overs',ascending=False).reset_index(drop=True); ov_lb.index += 1
        co1,co2 = st.columns([1.2,1])
        with co1:
            fig_o = px.bar(ov_lb.head(15), x='bat', y='Impact_Overs',
                           color='Impact_Overs', color_continuous_scale='Plasma',
                           text='Impact_Overs', title=f"Over {sel_ov} – Impact Overs", height=380)
            fig_o.update_layout(xaxis_tickangle=-40)
            st.plotly_chart(fig_o, width="stretch")
        with co2:
            fig_o2 = px.bar(ov_lb.sort_values('Impact_Freq%',ascending=False).head(15),
                            x='bat', y='Impact_Freq%', color='Impact_Freq%',
                            color_continuous_scale='RdYlGn', text='Impact_Freq%',
                            title=f"Over {sel_ov} – Impact Freq%", height=380)
            fig_o2.update_layout(xaxis_tickangle=-40)
            st.plotly_chart(fig_o2, width="stretch")
        st.dataframe(ov_lb[['bat','Impact_Overs','Total_Times_Batted','Impact_Freq%','Total_Runs','Best_Over','Avg_Runs','Avg_Balls','Runs_per_Ball']],
                     width="stretch")

    st.divider()
    st.markdown("### 🎚️ Over Band Leaderboard")
    st.caption("Every over a batter batted in, classified by runs per ball in that over. "
               "Uses the 'Balls faced in over' filter in the sidebar. Wides excluded, no-balls counted.")

    lb_src = band_data[band_data['bat'].isin(keep) & band_data['balls'].isin(sel_balls)].copy()
    if lb_src.empty:
        st.info("No overs match the selected balls-faced counts.")
    else:
        lb_src['Band'] = band_of(lb_src['runs']/lb_src['balls'])
        sel_band = st.selectbox("Band", BANDS, index=len(BANDS)-1, key='band_pick',
                                format_func=lambda b: f"{b}  ({BAND_RANGE[b]} runs per ball)")

        per_bat = lb_src.groupby('bat').agg(Overs_Counted=('runs','count')).reset_index()
        in_band = lb_src[lb_src['Band']==sel_band].groupby('bat').agg(
            Band_Overs=('runs','count'), Band_Runs=('runs','sum'), Band_Balls=('balls','sum')).reset_index()
        band_lb = per_bat.merge(in_band, on='bat', how='left').fillna(
            {'Band_Overs': 0, 'Band_Runs': 0, 'Band_Balls': 0})
        band_lb[['Band_Overs','Band_Runs','Band_Balls']] = band_lb[['Band_Overs','Band_Runs','Band_Balls']].astype(int)

        lo, hi = int(band_lb['Overs_Counted'].min()), int(band_lb['Overs_Counted'].max())
        if hi > lo:
            ov_lo, ov_hi = st.slider("Overs range (Overs Counted)", lo, hi, (lo, hi), key='band_range')
        else:
            ov_lo, ov_hi = lo, hi
        band_lb = band_lb[band_lb['Overs_Counted'].between(ov_lo, ov_hi)]

        band_lb['Band_%'] = (band_lb['Band_Overs']/band_lb['Overs_Counted']*100).round(1)
        band_lb['Runs_per_Ball'] = (band_lb['Band_Runs']/band_lb['Band_Balls'].where(band_lb['Band_Balls']>0)).round(2)
        band_lb = band_lb.sort_values(['Band_%','Band_Overs'], ascending=False).reset_index(drop=True)
        band_lb.index += 1

        st.caption(f"**{len(band_lb)}** batters with **{ov_lo}–{ov_hi}** Overs Counted")
        bl1, bl2 = st.columns([1.2, 1])
        with bl1:
            fig_bl = px.bar(band_lb.head(15), x='bat', y='Band_%', text='Band_%',
                            color_discrete_sequence=[BAND_COLORS[sel_band]],
                            hover_data={'Band_Overs': True, 'Overs_Counted': True},
                            title=f"Highest {sel_band} % (top 15)", height=400)
            fig_bl.update_traces(texttemplate='%{text:.1f}%')
            fig_bl.update_layout(xaxis_tickangle=-40)
            st.plotly_chart(fig_bl, width="stretch")
        with bl2:
            st.dataframe(band_lb[['bat','Band_%','Band_Overs','Overs_Counted','Runs_per_Ball']],
                         width="stretch", height=400)

    st.divider()
    st.markdown("### 🔍 Individual Batter Profile")
    sel_gc = st.selectbox("Select Batter", sorted(impact['bat'].unique()), key='gc_bat')
    bi     = impact[impact['bat']==sel_gc]
    bi_tot = over_data[over_data['bat']==sel_gc].shape[0]
    bi_freq= (len(bi)/bi_tot*100) if bi_tot > 0 else 0
    bi_avg = bi['over_runs'].mean()

    g1,g2,g3,g4,g5,g6 = st.columns(6)
    g1.metric("Impact Overs", len(bi)); g2.metric("Total Overs", bi_tot)
    g3.metric("Impact Freq%", f"{bi_freq:.1f}%"); g4.metric("Best Over", int(bi['over_runs'].max()))
    g5.metric("Avg in Impact Over", f"{bi_avg:.1f}"); g6.metric("Total Runs", int(bi['over_runs'].sum()))

    bi_balls = bi['balls_faced'].sum()
    h1,h2,_,_,_,_ = st.columns(6)
    h1.metric("Avg Balls per Impact Over", f"{bi_balls/len(bi):.1f}")
    h2.metric("Runs per Ball in Impact Overs", f"{bi['over_runs'].sum()/bi_balls:.2f}" if bi_balls else "–")

    st.markdown("#### 📋 Every Impact Over")
    ev = bi.sort_values(['year','p_match','over'])[['year','team_bowl','over','over_runs','balls_faced']].copy()
    ev['Runs_per_Ball'] = (ev['over_runs']/ev['balls_faced'].where(ev['balls_faced']>0)).round(2)
    ev.columns = ['Year','vs','Over','Runs','Balls','Runs_per_Ball']
    st.dataframe(ev, hide_index=True, width="stretch", height=260)

    # ── Over Band Profile ────────────────────────────────────────────────────
    st.markdown("#### 🎚️ Over Band Profile")
    bb = band_data[(band_data['bat']==sel_gc) & band_data['balls'].isin(sel_balls)].copy()
    if bb.empty:
        st.info("No overs for this batter with the selected balls-faced counts.")
    else:
        bb['Band'] = pd.Categorical(band_of(bb['runs']/bb['balls']), categories=BANDS, ordered=True)
        bt = bb.groupby('Band', observed=False).agg(
            Overs=('runs','count'), Runs=('runs','sum'), Balls=('balls','sum')).reset_index()
        bt['% of Overs'] = (bt['Overs']/bt['Overs'].sum()*100).round(1)
        bt['Runs_per_Ball'] = (bt['Runs']/bt['Balls'].where(bt['Balls']>0)).round(2)
        bt.insert(1, 'Runs per ball range', bt['Band'].astype(str).map(BAND_RANGE))
        st.caption(f"All {len(bb)} overs {sel_gc} batted in (not just Impact Overs), classified by runs per ball "
                   f"in that over. Balls faced in over: {', '.join(map(str, sel_balls))}. "
                   "Wides excluded, no-balls counted.")
        cb1, cb2 = st.columns([1, 1.3])
        with cb1:
            fig_band = px.bar(bt, x='% of Overs', y='Band', orientation='h', color='Band',
                              color_discrete_map=BAND_COLORS, text='% of Overs',
                              category_orders={'Band': BANDS}, height=320,
                              title=f"{sel_gc} – Share of Overs by Band")
            fig_band.update_traces(texttemplate='%{text:.1f}%')
            fig_band.update_layout(showlegend=False)
            st.plotly_chart(fig_band, width="stretch")
        with cb2:
            st.dataframe(bt[['Band','Runs per ball range','Overs','% of Overs','Runs','Balls','Runs_per_Ball']],
                         hide_index=True, width="stretch", height=320)

    # Consecutive
    st.markdown("#### 🔥 Consecutive Impact Overs")
    c_data = []
    for mid in bi['p_match'].unique():
        mi = bi[bi['p_match']==mid].sort_values(['inns','over'])
        ol = mi['over'].tolist(); mc=1; cc=1
        for i in range(1,len(ol)):
            if ol[i]==ol[i-1]+1: cc+=1; mc=max(mc,cc)
            else: cc=1
        c_data.append({'Match':mid,'Impact_In_Match':len(ol),'Max_Consecutive':mc,'Runs':mi['over_runs'].sum()})
    c_df = pd.DataFrame(c_data).sort_values('Impact_In_Match',ascending=False)
    c_df.index = range(1,len(c_df)+1)
    cg1,cg2,cg3 = st.columns(3)
    cg1.metric("Matches 2+ Impact Overs", int((c_df['Impact_In_Match']>=2).sum()))
    cg2.metric("Max Impact in a Match", int(c_df['Impact_In_Match'].max()))
    cg3.metric("Max Consecutive", int(c_df['Max_Consecutive'].max()))
    st.dataframe(c_df, width="stretch")

    # Consecutive overall leaderboard
    st.markdown("#### 🏆 Who Has Most Consecutive Impact Overs?")
    ca_all = []
    for batter in impact['bat'].unique():
        for mid in impact[impact['bat']==batter]['p_match'].unique():
            mi = impact[(impact['bat']==batter)&(impact['p_match']==mid)].sort_values(['inns','over'])
            ol = mi['over'].tolist(); mc=1; cc=1
            for i in range(1,len(ol)):
                if ol[i]==ol[i-1]+1: cc+=1; mc=max(mc,cc)
                else: cc=1
            if mc >= 2:
                ca_all.append({'bat':batter,'match':mid,'Max_Consecutive':mc})
    if ca_all:
        ca_df  = pd.DataFrame(ca_all)
        ca_sum = ca_df.groupby('bat').agg(Times=('Max_Consecutive','count'), Max=('Max_Consecutive','max')).reset_index()
        ca_sum = ca_sum.sort_values('Times',ascending=False).reset_index(drop=True); ca_sum.index += 1
        cc1,cc2 = st.columns([1.2,1])
        with cc1:
            fig_cc = px.bar(ca_sum.head(15), x='bat', y='Times',
                            color='Times', color_continuous_scale='Plasma', text='Times',
                            title="Most Consecutive Impact Overs", height=360)
            fig_cc.update_layout(xaxis_tickangle=-40)
            st.plotly_chart(fig_cc, width="stretch")
        with cc2:
            st.dataframe(ca_sum, width="stretch", height=360)

    st.markdown("#### ⚡ vs Pace & Spin")
    bk = bi.groupby('bowl_kind').agg(Impact_Overs=('over_runs','count'), Avg_Runs=('over_runs','mean')).reset_index()
    bk['Avg_Runs'] = bk['Avg_Runs'].round(1)
    bk_t = over_data[over_data['bat']==sel_gc].groupby('bowl_kind').agg(Total=('over_runs','count')).reset_index()
    bk = pd.merge(bk, bk_t, on='bowl_kind', how='left')
    bk['Impact_Freq%'] = (bk['Impact_Overs']/bk['Total']*100).round(1)
    ck1,ck2 = st.columns(2)
    with ck1:
        fig_k1 = px.bar(bk, x='bowl_kind', y='Impact_Overs', color='bowl_kind', text='Impact_Overs',
                        title=f"{sel_gc} – Impact Overs vs Pace/Spin", height=300)
        fig_k1.update_layout(showlegend=False)
        st.plotly_chart(fig_k1, width="stretch")
    with ck2:
        fig_k2 = px.bar(bk, x='bowl_kind', y='Impact_Freq%', color='bowl_kind', text='Impact_Freq%',
                        color_discrete_sequence=['#EF553B','#636EFA'],
                        title=f"{sel_gc} – Impact Freq% vs Pace/Spin", height=300)
        fig_k2.update_layout(showlegend=False)
        st.plotly_chart(fig_k2, width="stretch")

    st.markdown("#### 📊 Phase-wise")
    ph_b2 = bi.groupby('phase',observed=True).agg(Impact_Overs=('over_runs','count'), Avg_Runs=('over_runs','mean')).reset_index()
    ph_b2['Avg_Runs'] = ph_b2['Avg_Runs'].round(1)
    ph_t = over_data[over_data['bat']==sel_gc].groupby('phase',observed=True).agg(Total=('over_runs','count')).reset_index()
    ph_b2 = pd.merge(ph_b2, ph_t, on='phase', how='left')
    ph_b2['Impact_Freq%'] = (ph_b2['Impact_Overs']/ph_b2['Total']*100).round(1)
    cp1,cp2 = st.columns(2)
    with cp1:
        fig_p1 = px.bar(ph_b2, x='phase', y='Impact_Overs', color='phase', text='Impact_Overs',
                        color_discrete_map={'Powerplay (1–6)':'#636EFA','Middle (7–16)':'#EF553B','Death (17–20)':'#00CC96'},
                        title=f"{sel_gc} – Impact Overs by Phase", height=300)
        fig_p1.update_layout(showlegend=False)
        st.plotly_chart(fig_p1, width="stretch")
    with cp2:
        fig_p2 = px.bar(ph_b2, x='phase', y='Impact_Freq%', color='phase', text='Impact_Freq%',
                        color_discrete_map={'Powerplay (1–6)':'#636EFA','Middle (7–16)':'#EF553B','Death (17–20)':'#00CC96'},
                        title=f"{sel_gc} – Impact Freq% by Phase", height=300)
        fig_p2.update_layout(showlegend=False)
        st.plotly_chart(fig_p2, width="stretch")

    st.markdown("#### 🎯 Distributions")
    cd1,cd2 = st.columns(2)
    with cd1:
        fig_h = px.histogram(bi, x='over_runs', nbins=15,
                             title=f"{sel_gc} – Runs in Impact Overs",
                             color_discrete_sequence=['#636EFA'], height=300)
        st.plotly_chart(fig_h, width="stretch")
    with cd2:
        ow = bi.groupby('over').agg(Count=('over_runs','count')).reset_index()
        ow['over'] = ow['over'].astype(int)
        fig_ow = px.bar(ow.sort_values('over'), x='over', y='Count',
                        color='Count', color_continuous_scale='Plasma', text='Count',
                        title=f"{sel_gc} – Which Over He Dominates", height=300)
        fig_ow.update_layout(xaxis=dict(tickmode='linear',tick0=1,dtick=1))
        st.plotly_chart(fig_ow, width="stretch")

