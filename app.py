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


# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="CoachLens MVP", layout="wide")

DATA_PATH = Path("fbref_PL_2024-25.csv")

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
st.title("CoachLens MVP ⚽")
st.caption("Role clustering + selection optimizer + talent finder (FBref-style player season stats).")

with st.sidebar:
    st.header("Data")
    st.caption(f"Using local data file: {DATA_PATH.name}")
    st.divider()
    min_90s = st.slider("Minimum 90s played filter", min_value=0.0, max_value=20.0, value=6.0, step=0.5)
    st.divider()
    st.header("Clustering")
    k = st.slider("Number of clusters (k)", min_value=3, max_value=8, value=5, step=1)

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

tab1, tab2, tab3 = st.tabs(["🧩 Role Explorer", "🧠 Selection Optimizer", "🔍 Talent Finder"])

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
        pos_choice = st.selectbox("Filter by position (optional)", ["All"] + sorted(df_f["Pos"].dropna().astype(str).unique().tolist()))
    with colB:
        squad_choice = st.selectbox("Filter by squad (optional)", ["All"] + sorted(df_f["Squad"].dropna().astype(str).unique().tolist()))
    with colC:
        show_labels = st.checkbox("Show player labels on plot (can be noisy)", value=False)

    df_role = df_f.copy()
    if pos_choice != "All":
        df_role = df_role[df_role["Pos"].astype(str) == pos_choice]
    if squad_choice != "All":
        df_role = df_role[df_role["Squad"].astype(str) == squad_choice]

    if len(df_role) < k + 5:
        st.warning("Not enough players after filtering to cluster. Reduce filters or reduce k.")
        st.stop()

    clustered, cluster_means = compute_clusters(df_role, k=k, feature_cols=cluster_features)

    left, right = st.columns([2, 1])

    with left:
        fig = px.scatter(
            clustered,
            x="pca_x",
            y="pca_y",
            color=clustered["role_cluster"].astype(str),
            hover_data=["Player", "Squad", "Pos", "90s"],
            title="PCA view of player roles (clusters)",
        )
        if show_labels:
            fig.update_traces(text=clustered["Player"], textposition="top center")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("### Cluster Profiles (average features)")
        pretty_means = cluster_means.copy()
        pretty_means["role_cluster"] = pretty_means["role_cluster"].astype(int)
        st.dataframe(pretty_means.sort_values("role_cluster"), use_container_width=True, height=380)

        chosen_cluster = st.selectbox("Inspect a cluster", sorted(clustered["role_cluster"].unique().tolist()))
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
        st.markdown("### Players in selected cluster")
        st.dataframe(cluster_players[cols_show].head(25), use_container_width=True)

# -----------------------------
# Tab 2: Selection Optimizer
# -----------------------------
with tab2:
    st.subheader("Selection Optimizer (Impact-based squad suggestions)")

    work = compute_impact(df_f)

    squad = st.selectbox("Select a squad", sorted(work["Squad"].dropna().astype(str).unique().tolist()))
    pos_group = st.selectbox("Optional: focus on a position label substring", ["All", "FW", "MF", "DF", "GK"])

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Suggested XI (top by impact)")
        xi = suggested_xi(work, squad=squad, pos_filter=None if pos_group == "All" else pos_group, n=11)
        st.dataframe(xi, use_container_width=True, height=420)

        fig = px.bar(
            xi.sort_values("impact", ascending=True),
            x="impact",
            y="Player",
            orientation="h",
            title="Impact score of suggested XI",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Rotation Opportunities (high impact, low starts)")
        rot = high_impact_low_starts(work, squad=squad, min_90s=min_90s)
        if len(rot) == 0:
            st.info("No strong mismatches found with current thresholds. Try lowering min 90s or using All positions.")
        else:
            st.dataframe(rot, use_container_width=True, height=420)

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
        "Impact score (explainable): "
        "0.50*(Non-pen npxG+xAG/90) + 0.30*(xG+xAG/90) + 0.20*(progression/90) - 0.10*(cards/90)."
    )

# -----------------------------
# Tab 3: Talent Finder
# -----------------------------
with tab3:
    st.subheader("Talent Finder (league-wide leaderboard)")

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
    if pos_f != "All":
        view = view[view["Pos"].astype(str) == pos_f]
    if squad_f != "All":
        view = view[view["Squad"].astype(str) == squad_f]

    # Ensure numeric
    view[metric] = pd.to_numeric(view[metric], errors="coerce")

    leaderboard = view.dropna(subset=[metric]).sort_values(metric, ascending=False)
    cols = ["Player", "Squad", "Pos", "Age", "90s", metric]
    cols = [c for c in cols if c in leaderboard.columns]

    st.markdown("### Leaderboard")
    st.dataframe(leaderboard[cols].head(top_n), use_container_width=True)

    fig = px.bar(
        leaderboard.head(top_n).sort_values(metric, ascending=True),
        x=metric,
        y="Player",
        orientation="h",
        title=f"Top {top_n}: {metric}",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Distribution")
    fig2 = px.histogram(leaderboard, x=metric, nbins=40, title=f"Distribution of {metric}")
    st.plotly_chart(fig2, use_container_width=True)

# Recommend similar players
df = pd.read_csv("fbref_PL_2024-25.csv")

@st.cache_resource
def setup_vector_system(df):
    df_vectors = build_player_vectors(df)
    initialize_vector_db(df_vectors)
    return df_vectors

df_vectors = setup_vector_system(df)

df_vectors = build_player_vectors(df)
initialize_vector_db(df_vectors)

st.header("Role Similarity Engine")

selected_player = st.selectbox(
    "Select Player",
    df_vectors["Player"].unique()
)

exclude_same_team = st.checkbox("Exclude same squad", value=True)
top_k = st.slider("Number of similar players", 5, 20, 10)

if selected_player:
    results = find_similar_players(
        df_vectors,
        selected_player,
        top_k,
        exclude_same_team
    )
    st.dataframe(results)

st.header("Transfer Recommender")

selected_player_transfer = st.selectbox(
    "Select Player to Replace",
    df_vectors["Player"].unique(),
    key="transfer"
)

max_age = st.slider("Maximum Age Target", 18, 35, 28)

preserve_identity = st.checkbox(
    "Preserve Tactical Identity",
    value=False
)

similarity_threshold = None

if preserve_identity:
    similarity_threshold = st.slider(
        "Minimum Similarity Threshold",
        0.5,
        0.95,
        0.75,
        step=0.01
    )

top_k_transfer = st.slider("Number of Recommendations", 5, 20, 10)

if selected_player_transfer:

    transfer_results = recommend_transfer_targets(
    df_vectors,
    selected_player_transfer,
    max_age,
    top_k_transfer,
    similarity_threshold
    )

    st.subheader("Recommended Transfer Targets")
    st.dataframe(transfer_results)

# Footer
st.caption("MVP note: This uses season aggregates (not match-by-match), so it’s a decision-support dashboard, not a tactics simulator.")
