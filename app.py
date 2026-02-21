from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from database import (
    build_player_vectors,
    initialize_vector_db,
    find_similar_players,
    recommend_transfer_targets
)

from fatigue_dashboard import render_fatigue_dashboard


# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="CoachLens MVP", layout="wide")

DATA_PATH = Path("fbref_PL_2024-25.csv")
TALENT_PALETTE = ["#0B6623", "#1F8A70", "#F2C14E", "#D7263D", "#2E86AB"]

REQUIRED_COLS = [
    "Player", "Nation", "Pos", "Squad", "Age", "Born", "MP", "Starts",
    "Minutes Played", "90s",
    "Yellow Cards", "Red Cards",
    "Non-penalty xG per 90", "Expected assists per 90 minutes",
    "Non-penalty xG+xAG per 90", "xG+xAG per 90",
    "Non-penalty G+A per 90",
    "Progressive Carries", "Progressive Passes", "Progressive Receptions",
]


# -----------------------------
# Helpers
# -----------------------------
def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace, fix common column variants if needed."""
    df = df.copy()
    # Also strip trailing colons (FBref sometimes exports "xG+xAG per 90:")
    df.columns = [c.strip().rstrip(":") for c in df.columns]

    # Handle a couple of common FBref naming variants (just in case)
    rename_map = {
        "Min": "Minutes Played",
        "Minutes": "Minutes Played",
        "CrdY": "Yellow Cards",
        "CrdR": "Red Cards",
        "PrgC": "Progressive Carries",
        "PrgP": "Progressive Passes",
        "PrgR": "Progressive Receptions",
        "npxG per 90": "Non-penalty xG per 90",
        "npxG+xAG per 90": "Non-penalty xG+xAG per 90",
        "xG per 90": "Expected goals per 90 minutes",
        "xAG per 90": "Expected assists per 90 minutes",
        "G+A per 90": "Goal+Assist per 90",
        "G+A-PK per 90": "Non-penalty G+A per 90",
    }
    for k, v in rename_map.items():
        if k in df.columns and v not in df.columns:
            df = df.rename(columns={k: v})

    return df


def coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Avoid divide-by-zero
    df["90s"] = df["90s"].replace(0, np.nan)

    df["cards_p90"] = (df["Yellow Cards"] + 2 * df["Red Cards"]) / df["90s"]
    df["progression_p90"] = (df["Progressive Carries"] + df["Progressive Passes"]) / df["90s"]

    # Optional: receptions per 90 (can be useful)
    df["prgr_p90"] = df["Progressive Receptions"] / df["90s"]

    return df


def validate_required_cols(df: pd.DataFrame) -> list[str]:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    return missing


def safe_filter_minutes(df: pd.DataFrame, min_90s: float) -> pd.DataFrame:
    out = df.copy()
    out = out[out["90s"].notna()]
    out = out[out["90s"] >= min_90s]
    return out


def filter_by_position(df: pd.DataFrame, pos_choice: str) -> pd.DataFrame:
    if pos_choice == "All":
        return df
    tokens = [t.strip() for t in str(pos_choice).split(",") if t.strip()]

    def matches(pos_value: str) -> bool:
        pos_tokens = [p.strip() for p in str(pos_value).split(",") if p.strip()]
        return any(t in pos_tokens for t in tokens)

    return df[df["Pos"].apply(matches)]


def unique_position_tokens(series: pd.Series) -> list[str]:
    tokens: set[str] = set()
    for val in series.dropna().astype(str):
        for t in val.split(","):
            t = t.strip()
            if t:
                tokens.add(t)
    return sorted(tokens)


def compute_clusters(df: pd.DataFrame, k: int, feature_cols: list[str], random_state: int = 42):
    """
    Returns df with:
      - cluster label
      - PCA x,y for visualization
      - cluster feature means
    """
    work = df.copy()

    # Drop rows with missing needed features
    X = work[feature_cols].replace([np.inf, -np.inf], np.nan).dropna()
    keep_idx = X.index
    work = work.loc[keep_idx].copy()

    scaler = StandardScaler()
    Xs = scaler.fit_transform(work[feature_cols].values)

    model = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
    labels = model.fit_predict(Xs)
    work["role_cluster"] = labels

    pca = PCA(n_components=2, random_state=random_state)
    p2 = pca.fit_transform(Xs)
    work["pca_x"] = p2[:, 0]
    work["pca_y"] = p2[:, 1]

    cluster_means = work.groupby("role_cluster")[feature_cols].mean().reset_index()

    return work, cluster_means


def compute_impact(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple, explainable impact score using your per-90 features + progression and cards.
    """
    work = df.copy()

    # Make sure key fields exist
    needed = ["Non-penalty xG+xAG per 90", "xG+xAG per 90", "progression_p90", "cards_p90"]
    for c in needed:
        if c not in work.columns:
            work[c] = np.nan

    work["impact"] = (
        0.50 * work["Non-penalty xG+xAG per 90"]
        + 0.30 * work["xG+xAG per 90"]
        + 0.20 * work["progression_p90"]
        - 0.10 * work["cards_p90"]
    )

    return work


def suggested_xi(work: pd.DataFrame, squad: str, pos_filter: str | None = None, n: int = 11) -> pd.DataFrame:
    """
    MVP: returns top n players by impact for a squad (optionally within a pos group).
    """
    df = work[work["Squad"] == squad].copy()
    if pos_filter and pos_filter != "All":
        df = df[df["Pos"].astype(str).str.contains(pos_filter, na=False)]
    df = df.sort_values("impact", ascending=False)
    cols = ["Player", "Pos", "Age", "Starts", "90s", "impact",
            "Non-penalty xG+xAG per 90", "Expected assists per 90 minutes",
            "Non-penalty xG per 90", "progression_p90", "cards_p90"]
    cols = [c for c in cols if c in df.columns]
    return df[cols].head(n)


def high_impact_low_starts(work: pd.DataFrame, squad: str, min_90s: float) -> pd.DataFrame:
    """
    Players with good impact but fewer starts (rotation opportunities).
    """
    df = work[(work["Squad"] == squad) & (work["90s"] >= min_90s)].copy()

    # Define "low starts" relative to teammates
    starts_med = df["Starts"].median() if df["Starts"].notna().any() else 0

    # Define "high impact" threshold as top 30% in squad
    imp_thr = df["impact"].quantile(0.70) if df["impact"].notna().any() else np.nan

    out = df[(df["Starts"] <= starts_med) & (df["impact"] >= imp_thr)].copy()
    out = out.sort_values("impact", ascending=False)

    cols = ["Player", "Pos", "Starts", "90s", "impact",
            "Non-penalty xG+xAG per 90", "xG+xAG per 90", "progression_p90"]
    cols = [c for c in cols if c in out.columns]
    return out[cols]


# -----------------------------
# UI
# -----------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Montserrat:wght@400;600;700&display=swap');

    .stApp {
        background: linear-gradient(135deg, #3A0720 0%, #4B0B2B 55%, #5A0E34 100%);
        color: #F6F2F5;
    }

    h1, h2, h3, h4, h5 {
        font-family: 'Bebas Neue', sans-serif !important;
        letter-spacing: 0.5px;
    }

    p, div, span, label, .stTextInput, .stSelectbox, .stSlider {
        font-family: 'Montserrat', sans-serif !important;
    }

    .stTabs [data-baseweb="tab"] {
        font-family: 'Bebas Neue', sans-serif !important;
        font-size: 18px;
        letter-spacing: 0.6px;
        color: #F6F2F5;
    }

    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #F2C14E;
    }

    .title-row {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 6px;
    }

    .title-row h1 {
        margin: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="title-row">
        <span style="font-size: 34px; line-height: 1;">⚽️</span>
        <h1>CoachLens</h1>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption("A coach's assistant for player analysis and team optimization.")

st.caption(f"Using data file: {DATA_PATH.name}")
min_90s = 6.0
k = 5

if not DATA_PATH.exists():
    st.error(f"Local CSV not found: {DATA_PATH.name}")
    st.stop()

df_raw = pd.read_csv(DATA_PATH)
df_raw = normalize_cols(df_raw)

missing = validate_required_cols(df_raw)
if missing:
    st.error("Your CSV is missing required columns:")
    st.write(missing)
    st.stop()

# Coerce numeric columns
numeric_cols = [
    "Age", "Born", "MP", "Starts", "Minutes Played", "90s",
    "Yellow Cards", "Red Cards",
    "Non-penalty xG per 90", "Expected assists per 90 minutes",
    "Non-penalty xG+xAG per 90", "xG+xAG per 90",
    "Non-penalty G+A per 90",
    "Progressive Carries", "Progressive Passes", "Progressive Receptions",
]
df = coerce_numeric(df_raw, numeric_cols)
df = add_engineered_features(df)

# Apply baseline filter
df_f = safe_filter_minutes(df, min_90s=min_90s)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Role Explorer", "Selection Optimizer", "Talent Finder", "Recommender", "Fatigue Monitor"])
# -----------------------------
# Tab 1: Role Clustering
# -----------------------------
with tab1:
    st.subheader("Player Role Clustering")

    cluster_features = [
        "Non-penalty xG per 90",
        "Expected assists per 90 minutes",
        "progression_p90",
        "prgr_p90",
        "Non-penalty G+A per 90",
    ]

    # Extra filters
    colA, colB, colC = st.columns(3)
    with colA:
        pos_choice = st.selectbox("Filter by position", ["All"] + unique_position_tokens(df_f["Pos"]))
    with colB:
        squad_choice = st.selectbox("Filter by squad", ["All"] + sorted(df_f["Squad"].dropna().astype(str).unique().tolist()))
    with colC:
        st.markdown("&nbsp;", unsafe_allow_html=True)

    df_role = df_f.copy()
    df_role = filter_by_position(df_role, pos_choice)
    if squad_choice != "All":
        df_role = df_role[df_role["Squad"].astype(str) == squad_choice]

    if len(df_role) < k + 5:
        st.warning("Not enough players after filtering to cluster. Reduce filters.")
        st.stop()

    clustered, cluster_means = compute_clusters(df_role, k=k, feature_cols=cluster_features)

    left, right = st.columns([2, 1])
    cluster_order = sorted(clustered["role_cluster"].unique().tolist())
    clustered["role_cluster_str"] = clustered["role_cluster"].astype(int).astype(str)

    with left:
        fig = px.scatter(
            clustered,
            x="pca_x",
            y="pca_y",
            color="role_cluster_str",
            hover_data=["Player", "Squad", "Pos", "90s"],
            title="PCA view of player roles",
            category_orders={"role_cluster_str": [str(c) for c in cluster_order]},
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Inspect a cluster")
        chosen_cluster = st.selectbox("Cluster", cluster_order)
        cluster_players = clustered[clustered["role_cluster"] == chosen_cluster].copy()
        cluster_players = cluster_players.sort_values("Non-penalty xG+xAG per 90", ascending=False)

        cols_show = [
            "Player", "Squad", "Pos", "90s",
            "Non-penalty xG per 90",
            "Expected assists per 90 minutes",
            "Non-penalty xG+xAG per 90",
            "progression_p90",
            "cards_p90",
        ]
        cols_show = [c for c in cols_show if c in cluster_players.columns]
        st.dataframe(cluster_players[cols_show].head(25), use_container_width=True, hide_index=True)

    with right:
        st.markdown("### Cluster Profiles Average")
        pretty_means = cluster_means.copy()
        pretty_means["role_cluster"] = pretty_means["role_cluster"].astype(int)
        pretty_means = pretty_means.sort_values("role_cluster").set_index("role_cluster")
        st.dataframe(pretty_means, use_container_width=True, height=380)

# -----------------------------
# Tab 2: Selection Optimizer
# -----------------------------
with tab2:
    st.subheader("Selection Optimizer (Impact-based squad suggestions)")

    work = compute_impact(df_f)
    # Focus selection optimizer on attacking/midfield roles
    work = work[~work["Pos"].astype(str).str.contains("DF|GK", na=False)]

    squad = st.selectbox("Select a squad", sorted(work["Squad"].dropna().astype(str).unique().tolist()))
    pos_group = st.selectbox("Focus on a position", ["All", "FW", "MF", "DF", "GK"])

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Suggested XI (high impact)")
        xi = suggested_xi(work, squad=squad, pos_filter=None if pos_group == "All" else pos_group, n=11)
        st.dataframe(xi, use_container_width=True, height=420, hide_index=True)

        fig = px.bar(
            xi.sort_values("impact", ascending=True),
            x="impact",
            y="Player",
            orientation="h",
            title="Impact score of suggested XI",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Underused Players (high impact, low starts)")
        rot = high_impact_low_starts(work, squad=squad, min_90s=min_90s)
        if len(rot) == 0:
            st.info("No strong mismatches found with current thresholds. Try lowering min 90s or using All positions.")
        else:
            st.dataframe(rot, use_container_width=True, height=420, hide_index=True)

        st.markdown("### Impact vs Starts (squad view)")
        squad_df = work[work["Squad"] == squad].copy()
        fig2 = px.scatter(
            squad_df,
            x="Starts",
            y="impact",
            hover_data=["Player", "Pos", "90s"],
            title="Are starts aligned with impact?",
        )
        st.plotly_chart(fig2, use_container_width=True)


    st.caption(
        "Impact score: "
        "0.50*(Non-pen npxG+xAG/90) + 0.30*(xG+xAG/90) + 0.20*(progression/90) - 0.10*(cards/90)."
    )

# -----------------------------
# Tab 3: Talent Finder
# -----------------------------
with tab3:
    st.subheader("Talent Finder (league-wide leaders)")

    # Define metric groups
    metric_groups = {
        "Attacking": [
            "Goals per 90",
            "Non-Penalty Goals per 90",
            "Goal+Assist per 90",
            "Non-penalty G+A per 90",
            "Expected goals per 90 minutes",
            "Non-penalty xG per 90",
        ],
        "Creating": [
            "Assists per 90",
            "Expected assists per 90 minutes",
        ],
        "Overall Impact": [
            "Non-penalty xG+xAG per 90",
            "xG+xAG per 90",
        ],
        "Progression": [
            "progression_p90",
            "prgr_p90",
            "Progressive Carries",
            "Progressive Passes",
            "Progressive Receptions",
        ],
        "Discipline": [
            "cards_p90",
            "Yellow Cards",
            "Red Cards",
        ],
    }

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        group = st.selectbox("Category", list(metric_groups.keys()))
    with c2:
        metric = st.selectbox("Metric", [m for m in metric_groups[group] if m in df_f.columns])
    with c3:
        pos_f = st.selectbox("Position filter", ["All"] + sorted(df_f["Pos"].dropna().astype(str).unique().tolist()))
    with c4:
        squad_f = st.selectbox("Squad filter", ["All"] + sorted(df_f["Squad"].dropna().astype(str).unique().tolist()))

    top_n = st.slider("Show top N", min_value=5, max_value=50, value=20, step=5)

    view = df_f.copy()
    view = filter_by_position(view, pos_f)
    if squad_f != "All":
        view = view[view["Squad"].astype(str) == squad_f]

    # Ensure numeric
    view[metric] = pd.to_numeric(view[metric], errors="coerce")

    leaderboard = view.dropna(subset=[metric]).sort_values(metric, ascending=False)
    cols = ["Player", "Squad", "Pos", "Age", "90s", metric]
    cols = [c for c in cols if c in leaderboard.columns]

    st.markdown("### Leaderboard")
    st.dataframe(leaderboard[cols].head(top_n), use_container_width=True, hide_index=True)

    fig = px.bar(
        leaderboard.head(top_n).sort_values(metric, ascending=True),
        x=metric,
        y="Player",
        orientation="h",
        title=f"Top {top_n}: {metric}",
    )
    fig.update_traces(marker_color=TALENT_PALETTE[3])
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Distribution")
    fig2 = px.histogram(leaderboard, x=metric, nbins=40, title=f"Distribution of {metric}")
    fig2.update_traces(marker_color=TALENT_PALETTE[4])
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Metric by Position")
    box_df = view.dropna(subset=[metric]).copy()
    fig5 = px.box(
        box_df,
        x="Pos",
        y=metric,
        title=f"{metric} distribution by position",
    )
    fig5.update_traces(marker_color=TALENT_PALETTE[1])
    st.plotly_chart(fig5, use_container_width=True)

    st.markdown("### Squad Averages")
    squad_avg = view.dropna(subset=[metric]).groupby("Squad", as_index=False)[metric].mean().sort_values(metric, ascending=False)
    fig6 = px.bar(
        squad_avg,
        x=metric,
        y="Squad",
        orientation="h",
        title=f"Average {metric} by squad",
    )
    fig6.update_traces(marker_color=TALENT_PALETTE[2])
    st.plotly_chart(fig6, use_container_width=True)
    
with tab4:
    st.subheader("Recommender Engine")
    st.caption("Role similarity + transfer target suggestions based on vector similarity.")

    @st.cache_resource
    def setup_vector_system(df_in: pd.DataFrame) -> pd.DataFrame:
        df_vectors_local = build_player_vectors(df_in)
        initialize_vector_db(df_vectors_local)
        return df_vectors_local

    # Use the already-prepared dataframe (minutes filtered, engineered features, etc.)
    df_vectors = setup_vector_system(df_f)

    st.markdown("### Role Similarity Engine")
    selected_player = st.selectbox(
        "Select Player",
        sorted(df_vectors["Player"].dropna().unique().tolist()),
        key="sim_player"
    )

    exclude_same_team = st.checkbox("Exclude same squad", value=True, key="sim_exclude_team")
    top_k = st.slider("Number of similar players", 5, 20, 10, key="sim_topk")

    if selected_player:
        results = find_similar_players(
            df_vectors,
            selected_player,
            top_k,
            exclude_same_team
        )
        st.dataframe(results, use_container_width=True, hide_index=True)

    st.divider()

    st.markdown("### Transfer Recommender")
    selected_player_transfer = st.selectbox(
        "Select Player to Replace",
        sorted(df_vectors["Player"].dropna().unique().tolist()),
        key="transfer_player"
    )

    max_age = st.slider("Maximum Age Target", 18, 35, 28, key="transfer_max_age")

    preserve_identity = st.checkbox("Preserve Tactical Identity", value=False, key="transfer_preserve")
    similarity_threshold = None
    if preserve_identity:
        similarity_threshold = st.slider(
            "Minimum Similarity Threshold",
            0.5, 0.95, 0.75, step=0.01,
            key="transfer_thresh"
        )

    top_k_transfer = st.slider("Number of Recommendations", 5, 20, 10, key="transfer_topk")

    if selected_player_transfer:
        transfer_results = recommend_transfer_targets(
            df_vectors,
            selected_player_transfer,
            max_age,
            top_k_transfer,
            similarity_threshold
        )
        st.subheader("Recommended Transfer Targets")
        st.dataframe(transfer_results, use_container_width=True, hide_index=True)
  

# -----------------------------
# Tab 5: Fatigue Monitor
# -----------------------------
with tab5:
    render_fatigue_dashboard()