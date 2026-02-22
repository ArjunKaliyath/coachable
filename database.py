from cortex import CortexClient, DistanceMetric
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

VECTOR_COLLECTION = "player_roles"

SIMILARITY_FEATURES = [
    "Non-penalty xG per 90",
    "Expected assists per 90 minutes",
    "Non-penalty xG+xAG per 90",
    "Non-penalty G+A per 90",
    "Progressive Carries",
    "Progressive Passes",
    "Progressive Receptions"
]

df = pd.read_csv("fbref_PL_2024-25.csv")
# Clean the CSV data before passing it to build_player_vectors
df = df.drop_duplicates(subset=['Player'], keep='first')
df['Age'] = df['Age'].fillna(df['Age'].median())
# Ensure all feature columns are numeric
for col in SIMILARITY_FEATURES:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

def build_player_vectors(df):

    feature_cols = [
        "Non-penalty xG per 90",
        "Expected assists per 90 minutes",
        "Non-penalty xG+xAG per 90",
        "Non-penalty G+A per 90",
        "Progressive Carries",
        "Progressive Passes",
        "Progressive Receptions"]

    feature_df = df[feature_cols].fillna(0)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(feature_df)
    scaled = np.nan_to_num(scaled)  # remove NaN / inf
    vectors = [list(map(float, row)) for row in scaled]

    df_vectors = df.copy()
    df_vectors["vector"] = vectors

    return df_vectors

def initialize_vector_db(df_vectors):
    dimension = len(df_vectors["vector"].iloc[0])
    with CortexClient("localhost:50051") as client:
        try:
            client.delete_collection(VECTOR_COLLECTION)
        except:
            pass

        client.create_collection(
            name=VECTOR_COLLECTION,
            dimension=dimension,
            distance_metric=DistanceMetric.COSINE,)

        # Ensure IDs are pure Python integers
        ids = [int(i) for i in range(len(df_vectors))]
        vectors = [[float(v) for v in vec] for vec in df_vectors["vector"].tolist()]

        payloads = [
            {
                "player": str(row["Player"]),
                "squad": str(row["Squad"]),
                "position": str(row["Pos"]),
                "age": int(row["Age"]) if pd.notna(row["Age"]) else 25,
            }
            for _, row in df_vectors.iterrows()]

        client.batch_upsert(
            VECTOR_COLLECTION,
            ids=ids,
            vectors=vectors,
            payloads=payloads, # ✅ Corrected keyword
        )
        client.flush(VECTOR_COLLECTION)

def find_similar_players(df_vectors, player_name, top_k=10, exclude_same_team=True):
    player_row = df_vectors[df_vectors["Player"] == player_name].iloc[0]
    query_vector = player_row["vector"]
    player_team = player_row["Squad"]

    with CortexClient("localhost:50051") as client:
        results = client.search(
            VECTOR_COLLECTION,
            query=query_vector,
            top_k=top_k + 5,
            with_payload=True # ✅ Added this to fetch the data
        )

    similar = []
    for r in results:
        # Check if payload exists
        if not r.payload:
            continue
            
        if r.payload.get("player") == player_name:
            continue

        if exclude_same_team and r.payload.get("squad") == player_team:
            continue

        # Use distance if score is missing (Actian returns 'distance')
        score = getattr(r, 'score', getattr(r, 'distance', 0))

        similar.append({
            "Player": r.payload.get("player"),
            "Squad": r.payload.get("squad"),
            "Position": r.payload.get("position"),
            "Age": r.payload.get("age"),
            "Similarity": round(score, 4),
        })

        if len(similar) == top_k:
            break
    return similar

def recommend_transfer_targets(df_vectors, player_name, max_age=28, top_k=10, similarity_threshold=None):
    player_row = df_vectors[df_vectors["Player"] == player_name].iloc[0]
    query_vector = player_row["vector"]
    original_team = player_row["Squad"]
    original_age = player_row["Age"]

    with CortexClient("localhost:50051") as client:
        results = client.search(
            VECTOR_COLLECTION,
            query=query_vector,
            top_k=top_k * 3,
            with_payload=True # ✅ Added this
        )

    recommendations = []
    for r in results:
        if not r.payload or r.payload.get("player") == player_name:
            continue

        if r.payload.get("squad") == original_team:
            continue

        p_age = r.payload.get("age")
        if p_age is None or p_age > max_age:
            continue
        
        score = getattr(r, 'score', getattr(r, 'distance', 0))
        if similarity_threshold is not None and score < similarity_threshold:
            continue

        age_gap = abs(p_age - original_age)
        age_factor = max(0, 1 - (age_gap / 10))
        replacement_score = score * age_factor

        recommendations.append({
            "Player": r.payload["player"],
            "Squad": r.payload["squad"],
            "Position": r.payload["position"],
            "Age": p_age,
            "Similarity": round(score, 4),
            "Replacement Score": round(replacement_score, 4)
        })

    return sorted(recommendations, key=lambda x: x["Replacement Score"], reverse=True)[:top_k]