"""
Fatigue Monitoring Dashboard
Displays player fatigue predictions, injury history, and match performance indicators
"""

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add injury_data_model to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'injury_data_model'))
from predict_fatigue import predict_player_fatigue


# Available players
AVAILABLE_PLAYERS = [
    'Ousmane Dembélé',
    'Pedri',
    'Cole Palmer',
    'Martin Ødegaard',
    'Désiré Doué'
]

INJURY_DATA_PATH = Path(__file__).parent / 'injury_data_model' / 'soccer_injury_dataset.csv'


def load_injury_history(player_name):
    """Load injury history for a specific player"""
    if not INJURY_DATA_PATH.exists():
        return pd.DataFrame()
    
    df = pd.read_csv(INJURY_DATA_PATH)
    player_injuries = df[df['player'] == player_name].copy()
    
    # Sort by most recent first (injury_number descending)
    player_injuries = player_injuries.sort_values('injury_number', ascending=False)
    
    return player_injuries


def create_fatigue_gauge(fatigue_score):
    """Create a gauge chart for fatigue score with color gradient"""
    
    # Determine color based on score
    if fatigue_score < 33:
        color = "#22c55e"  # Green
        risk_level = "Low Risk"
    elif fatigue_score < 67:
        color = "#eab308"  # Yellow
        risk_level = "Medium Risk"
    else:
        color = "#ef4444"  # Red
        risk_level = "High Risk"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=fatigue_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Fatigue Score<br><span style='font-size:0.6em;color:gray'>{risk_level}</span>", 
               'font': {'size': 24}},
        number={'font': {'size': 60}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkgray"},
            'bar': {'color': color, 'thickness': 0.75},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 33], 'color': '#dcfce7'},  # Light green
                {'range': [33, 67], 'color': '#fef9c3'},  # Light yellow
                {'range': [67, 100], 'color': '#fee2e2'}  # Light red
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': fatigue_score
            }
        }
    ))
    
    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=80, b=20),
        paper_bgcolor="white",
        font={'color': "darkgray", 'family': "Arial"}
    )
    
    return fig


def get_color_for_minutes(minutes):
    """Return color based on minutes played"""
    if minutes >= 80:
        return '#fee2e2'  # Light red
    elif minutes >= 60:
        return '#fef9c3'  # Light yellow
    else:
        return '#dcfce7'  # Light green


def get_color_intensity(value, max_value, reverse=False):
    """
    Get color intensity for a value
    reverse=False: higher values = more red (bad)
    reverse=True: higher values = more green (good)
    """
    if max_value == 0 or pd.isna(value):
        return '#f9fafb'  # Very light gray
    
    intensity = min(value / max_value, 1.0)
    
    if reverse:
        # Higher is better (green)
        if intensity > 0.7:
            return '#dcfce7'  # Light green
        elif intensity > 0.3:
            return '#fef9c3'  # Light yellow
        else:
            return '#fee2e2'  # Light red
    else:
        # Higher is worse (red)
        if intensity > 0.7:
            return '#fee2e2'  # Light red
        elif intensity > 0.3:
            return '#fef9c3'  # Light yellow
        else:
            return '#dcfce7'  # Light green


def style_match_dataframe(df):
    """Apply color styling to match history dataframe"""
    
    # Create a copy for display
    display_df = df.copy()
    
    # Calculate max values for normalization
    max_fld = df['Fld'].max() if 'Fld' in df.columns and df['Fld'].max() > 0 else 1
    max_crdy = df['CrdY'].max() if 'CrdY' in df.columns and df['CrdY'].max() > 0 else 1
    max_sh = df['Sh'].max() if 'Sh' in df.columns and df['Sh'].max() > 0 else 1
    
    def highlight_row(row):
        colors = [''] * len(row)
        
        for idx, col in enumerate(display_df.columns):
            if col == 'Min':
                colors[idx] = f'background-color: {get_color_for_minutes(row[col])}'
            elif col == 'Fld':
                colors[idx] = f'background-color: {get_color_intensity(row[col], max_fld)}'
            elif col == 'CrdY':
                colors[idx] = f'background-color: {get_color_intensity(row[col], max_crdy)}'
            elif col == 'Sh':
                colors[idx] = f'background-color: {get_color_intensity(row[col], max_sh)}'
        
        return colors
    
    return display_df.style.apply(highlight_row, axis=1)


def load_player_recent_matches(player_name):
    """Load the last 5 matches for a player from FBref data"""
    fbref_files = {
        'Ousmane Dembélé': 'injury_data_model/fbref_data/dembele_fbref.csv',
        'Pedri': 'injury_data_model/fbref_data/pedri_fbref.csv',
        'Cole Palmer': 'injury_data_model/fbref_data/palmer_fbref.csv',
        'Martin Ødegaard': 'injury_data_model/fbref_data/odegaard_fbref.csv',
        'Désiré Doué': 'injury_data_model/fbref_data/doue_fbref.csv'
    }
    
    if player_name not in fbref_files:
        return pd.DataFrame()
    
    file_path = Path(__file__).parent / fbref_files[player_name]
    if not file_path.exists():
        return pd.DataFrame()
    
    df = pd.read_csv(file_path)
    
    # Fix column alignment (Player header is shifted)
    if 'Player' in df.columns and df.columns[0] == 'Player':
        new_columns = df.columns[1:].tolist()
        df = df.iloc[:, :-1]
        df.columns = new_columns
    
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date', ascending=False)
    
    # Get last 5 matches
    recent = df.head(5).copy()
    
    # Select relevant columns
    display_cols = ['Date', 'Comp', 'Round', 'Venue', 'Result', 'Min', 'Fld', 'CrdY', 'Sh']
    display_cols = [c for c in display_cols if c in recent.columns]
    recent = recent[display_cols]
    
    # Ensure numeric columns
    numeric_cols = ['Min', 'Fld', 'CrdY', 'Sh']
    for col in numeric_cols:
        if col in recent.columns:
            recent[col] = pd.to_numeric(recent[col], errors='coerce').fillna(0).astype(int)
    
    # Format date
    if 'Date' in recent.columns:
        recent['Date'] = recent['Date'].dt.strftime('%Y-%m-%d')
    
    return recent


def render_fatigue_dashboard():
    """Main function to render the fatigue monitoring dashboard"""
    
    st.subheader("⚕️ Player Fatigue Monitor")
    st.caption("AI-powered fatigue prediction based on recent match performance and workload")
    
    # Player selection
    st.markdown("---")
    selected_player = st.selectbox(
        "Select a player to analyze",
        AVAILABLE_PLAYERS,
        index=0,
        key="fatigue_player_selector"
    )
    
    if not selected_player:
        st.info("Please select a player to view their fatigue analysis")
        return
    
    try:
        # Get fatigue prediction
        with st.spinner(f"Analyzing {selected_player}'s fatigue level..."):
            result = predict_player_fatigue(selected_player)
        
        st.markdown("---")
        
        # Display fatigue gauge
        col1, col2 = st.columns([2, 3])
        
        with col1:
            fig = create_fatigue_gauge(result['fatigue_score'])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 📊 Analysis Summary")
            st.markdown(f"**Analysis Period:** {result['date_range']}")
            st.markdown(f"**Matches Analyzed:** {result['n_matches']}")
            
            # Metrics in columns
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            
            with metric_col1:
                st.metric("Total Minutes", result['total_minutes'])
                st.metric("Avg Minutes", f"{result['avg_minutes']} min")
            
            with metric_col2:
                st.metric("High Intensity Games", result['high_intensity_games'])
                st.metric("Times Fouled", result['total_fouled'])
            
            with metric_col3:
                st.metric("Yellow Cards", result['yellow_cards'])
                
                # Risk indicator
                score = result['fatigue_score']
                if score < 33:
                    st.success("✅ Low Risk")
                elif score < 67:
                    st.warning("⚠️ Medium Risk")
                else:
                    st.error("🚨 High Risk")
        
        st.markdown("---")
        
        # Injury History Section
        st.markdown("### 🏥 Injury History")
        injury_df = load_injury_history(selected_player)
        
        if not injury_df.empty:
            # Prepare display dataframe
            display_injury = injury_df[['injury_type', 'injury_start_date', 'injury_end_date', 'days_missed']].copy()
            display_injury.columns = ['Injury Type', 'Start Date', 'End Date', 'Days Missed']
            
            # Highlight active injuries
            def highlight_active(row):
                if pd.isna(row['End Date']) or row['End Date'] == '':
                    return ['background-color: #fee2e2'] * len(row)
                return [''] * len(row)
            
            styled_injury = display_injury.style.apply(highlight_active, axis=1)
            st.dataframe(styled_injury, use_container_width=True, hide_index=True)
            
            # Show statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Injuries", len(injury_df))
            with col2:
                avg_days = injury_df['days_missed'].mean()
                st.metric("Avg Days Missed", f"{avg_days:.0f}" if not pd.isna(avg_days) else "N/A")
            with col3:
                active_injuries = injury_df[injury_df['injury_end_date'].isna()].shape[0]
                if active_injuries > 0:
                    st.error(f"⚠️ {active_injuries} Active")
                else:
                    st.success("✅ No Active")
        else:
            st.info("No injury history found for this player")
        
        st.markdown("---")
        
        # Last 5 Matches Section
        st.markdown("### 📋 Last 5 Matches Performance")
        st.caption("Color coding: 🟢 Low | 🟡 Medium | 🔴 High")
        
        matches_df = load_player_recent_matches(selected_player)
        
        if not matches_df.empty:
            styled_matches = style_match_dataframe(matches_df)
            st.dataframe(styled_matches, use_container_width=True, hide_index=True)
            
            # Legend
            st.markdown("""
            **Indicator Legend:**
            - **Min (Minutes):** High load if ≥80 min, Medium 60-80 min, Low <60 min
            - **Fld (Fouled):** Higher values indicate more physical contact (potential injury risk)
            - **CrdY (Yellow Cards):** Disciplinary issues or aggressive play
            - **Sh (Shots):** Offensive workload indicator
            """)
        else:
            st.info("No recent match data available for this player")
            
    except FileNotFoundError as e:
        st.error(f"Data files not found: {e}")
        st.info("Please ensure all player model files and data are in the correct location")
    except Exception as e:
        st.error(f"An error occurred while analyzing {selected_player}: {e}")
        st.exception(e)
