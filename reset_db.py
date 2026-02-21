import pandas as pd
import numpy as np
import time
from cortex import CortexClient, DistanceMetric
from database import build_player_vectors, VECTOR_COLLECTION, SIMILARITY_FEATURES

def hard_reset_database():
    print("🚀 Starting Actian Cortex Reset...")

    # 1. Load and Clean
    df = pd.read_csv('fbref_PL_2024-25.csv')
    df = df[df['Player'] != 'Player'].copy()
    df = df.drop_duplicates(subset=['Player'], keep='first')
    
    # 2. Build Vectors
    df_vectors = build_player_vectors(df)
    
    # 3. Convert to Native Python Types
    ids = [int(i) for i in range(len(df_vectors))]
    vectors = [[float(x) for x in vec] for vec in df_vectors["vector"].tolist()]

    payloads = []
    for _, row in df_vectors.iterrows():
        payloads.append({
            "player": str(row["Player"]).strip(),
            "squad": str(row["Squad"]).strip(),
            "position": str(row["Pos"]).strip(),
            "age": int(row["Age"]) if pd.notna(row["Age"]) else 25
        })

    dimension = len(vectors[0])

    with CortexClient("localhost:50051") as client:
        print(f"🗑️ Deleting collection...")
        try: client.delete_collection(VECTOR_COLLECTION)
        except: pass

        print(f"🆕 Creating collection (Dim: {dimension})...")
        client.create_collection(name=VECTOR_COLLECTION, dimension=dimension, distance_metric=DistanceMetric.COSINE)

        print(f"📤 Upserting {len(payloads)} players...")
        client.batch_upsert(
            VECTOR_COLLECTION,
            ids=ids,
            vectors=vectors,
            payloads=payloads # ✅ Reverted to 'payloads'
        )
        
        client.flush(VECTOR_COLLECTION)
        time.sleep(2) 

        print("🔍 Verifying Data Retrieval...")
        results = client.search(
            VECTOR_COLLECTION, 
            query=vectors[0], 
            top_k=1, 
            with_payload=True # ✅ Added to fetch the data
        )
        
        if results and results[0].payload:
            print(f"✅ SUCCESS! Found: {results[0].payload.get('player')}")
        else:
            print("❌ ERROR: Payload is missing. Ensure search uses with_payload=True.")

if __name__ == "__main__":
    hard_reset_database()