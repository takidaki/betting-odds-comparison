import requests
import os
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np
from collections import defaultdict

# Load environment variables
load_dotenv()

# --- Function to Load CSS ---
def load_css(file_path):
    try:
        # Explicitly specify encoding='utf-8'
        with open(file_path, 'r', encoding='utf-8') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
            print("DEBUG: CSS loaded successfully.") # Add a success message
    except FileNotFoundError:
        st.error(f"CSS file not found at {file_path}")
    except Exception as e: # Catch other potential errors during read/markdown
        st.error(f"Error loading CSS file: {e}")
# --- Configuration ---
API_KEY = os.getenv("api_key")
if not API_KEY:
    st.error("API key not found. Please set the 'api_key' environment variable.", icon="🚨")
    st.stop()

SPORTS_API_URL = "https://api.the-odds-api.com/v4/sports/"
ODDS_API_URL_TEMPLATE = "https://api.the-odds-api.com/v4/sports/{sport_key}/odds/"

# Constants for API parameters
REGIONS_TO_REQUEST = ["eu","us","uk","au"] # Example: Focus on EU bookies maybe? Or use 'us', 'uk', 'au'
MARKETS_TO_REQUEST = ['h2h', 'spreads', 'totals'] # Request markets needed for detailed view
ODDS_FORMAT = 'decimal'
DATE_FORMAT = 'iso'
DEFAULT_REFERENCE_BOOKMAKER_KEY = "pinnacle" # Default bookmaker for reference odds

# --- API Fetching Functions ---
@st.cache_data(ttl=3600) # Cache sports list for 1 hour
def fetch_available_sports(api_key: str) -> Optional[List[Dict[str, Any]]]:
    """Fetches the list of available sports."""
    params = {"apiKey": api_key}
    try:
        response = requests.get(SPORTS_API_URL, params=params, timeout=15)
        response.raise_for_status()
        sports_data = response.json()
        # Filter for active sports with necessary details
        return [s for s in sports_data if s.get('group') and (s.get('title') or s.get('description')) and s.get('active')]
    except requests.exceptions.RequestException as e:
        st.error(f"API Error (Sports List): {e}")
        return None
    except Exception as e:
        st.error(f"Error (Sports List): {e}")
        return None

@st.cache_data(ttl=300) # Cache league odds for 5 minutes
def fetch_league_odds_all_bookies(api_key: str, sport_key: str, regions: List[str], markets: List[str]) -> Optional[List[Dict[str, Any]]]:
    """Fetches odds for ALL bookmakers for a specific sport/league."""
    params = {
        "apiKey": api_key,
        "regions": ",".join(regions),
        "markets": ",".join(markets),
        "oddsFormat": ODDS_FORMAT,
        "dateFormat": DATE_FORMAT,
    }
    odds_url = ODDS_API_URL_TEMPLATE.format(sport_key=sport_key)
    print(f"DEBUG: Fetching League Odds URL: {odds_url}") # Keep for debugging API calls
    try:
        response = requests.get(odds_url, params=params, timeout=25)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
             print(f"DEBUG: 404 - No events found for sport '{sport_key}' matching parameters.")
             return [] # Return empty list for not found
        st.error(f"HTTP Error (League Odds): {e.response.status_code} - {e.response.text}")
        print(f"DEBUG: Error Response: {e.response.text}")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Network Error (League Odds): {e}")
        return None
    except Exception as e:
        st.error(f"Error (League Odds): {e}")
        return None

# --- Helper Functions ---

def format_datetime(iso_string, fmt='%Y-%m-%d %H:%M %Z'):
    """Converts ISO date string to a specified format."""
    try:
        # Handle potential 'Z' for UTC
        dt = datetime.fromisoformat(iso_string.replace('Z', '+00:00'))
        return dt.strftime(fmt)
    except (ValueError, TypeError, AttributeError):
        return iso_string # Fallback

def get_bookmaker_market_odds(bookmaker_data: Dict[str, Any], market_key: str, outcome_names: List[str]) -> Dict[str, Any]:
    """Extracts odds for specific outcomes (by name) from a specific market for a single bookmaker."""
    odds = {name: None for name in outcome_names} # Initialize with None
    if not bookmaker_data or not bookmaker_data.get('markets'):
        return odds # Return defaults if no data

    # Find the specific market (e.g., 'h2h')
    market = next((m for m in bookmaker_data['markets'] if m.get('key') == market_key), None)
    if not market or not market.get('outcomes'):
        return odds # Return defaults if market/outcomes not found

    # Extract prices for the requested outcome names
    for outcome in market['outcomes']:
        name = outcome.get('name')
        if name in odds: # Check if this outcome name is one we're looking for
            try:
                odds[name] = float(outcome.get('price'))
            except (ValueError, TypeError):
                odds[name] = None # Assign None if price is invalid
    return odds

def calculate_payout(odds_list: List[Optional[float]]) -> Optional[float]:
    """Calculates payout percentage from a list of decimal odds."""
    valid_odds = [o for o in odds_list if o is not None and o > 0]
    if not valid_odds:
        return None # Cannot calculate if no valid odds
    try:
        # Calculate implied probability sum (margin)
        margin = sum(1 / o for o in valid_odds)
        if margin == 0: return None # Avoid division by zero
        # Calculate payout (1 / margin)
        payout = (1 / margin) * 100
        return round(payout, 1) # Round to one decimal place
    except ZeroDivisionError:
        return None # Should be caught by margin check, but good practice

def find_event_by_id(event_list: List[Dict[str, Any]], event_id: str) -> Optional[Dict[str, Any]]:
    """Finds an event dictionary in a list by its ID."""
    if not event_list or not event_id: return None
    return next((event for event in event_list if event.get('id') == event_id), None)

def find_market_data(bookmaker_data: Dict[str, Any], market_key: str) -> Optional[Dict[str, Any]]:
    """Finds a specific market (e.g., 'totals') within a bookmaker's market list."""
    if not bookmaker_data or 'markets' not in bookmaker_data:
        return None
    # Use .get() for safer access and return None if key not found or list empty
    return next((m for m in bookmaker_data.get('markets', []) if m.get('key') == market_key), None)

def aggregate_totals_lines_with_details(event_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """ Aggregates best Over/Under odds and includes raw bookie odds for each line. """
    # Structure: lines_data[point] = {'over_odds': [(bookie, price)], 'under_odds': [(bookie, price)]}
    lines_data = defaultdict(lambda: {'over_odds': [], 'under_odds': []})

    if not event_data or not event_data.get('bookmakers'): return [] # Return empty list if no bookies

    for bookie in event_data['bookmakers']:
        totals_market = find_market_data(bookie, 'totals')
        if totals_market and totals_market.get('outcomes'):
            bookie_title = bookie.get('title', 'Unknown') # Get bookie title once
            for outcome in totals_market['outcomes']:
                try:
                    point = float(outcome.get('point'))
                    price = float(outcome.get('price'))
                    name = outcome.get('name') # 'Over' or 'Under'

                    # Store raw odds with bookmaker name
                    if name == 'Over': lines_data[point]['over_odds'].append((bookie_title, price))
                    elif name == 'Under': lines_data[point]['under_odds'].append((bookie_title, price))
                except (ValueError, TypeError, AttributeError):
                    continue # Skip if point/price is invalid or missing

    processed_lines = []
    # Sort by point value for consistent display order
    for point, data in sorted(lines_data.items()):
        over_prices = [p for _, p in data['over_odds']]
        under_prices = [p for _, p in data['under_odds']]
        # Find the best (highest) odds for Over and Under
        best_over = max(over_prices) if over_prices else None
        best_under = max(under_prices) if under_prices else None
        # Calculate payout based on the BEST odds available for this line
        payout = calculate_payout([best_over, best_under])

        processed_lines.append({
            "Line": f"Over/Under {point:+.2f}", # Format point consistently (e.g., +2.50)
            "Bookies": len(set(b for b, _ in data['over_odds'] + data['under_odds'])), # Count unique bookies offering this line
            "BestOver": best_over,
            "BestUnder": best_under,
            "Payout": payout,
            "RawOver": sorted(data['over_odds'], key=lambda x: x[1], reverse=True), # Sort raw odds (best first)
            "RawUnder": sorted(data['under_odds'], key=lambda x: x[1], reverse=True)
        })
    return processed_lines

def aggregate_spread_lines_with_details(event_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """ Aggregates best spread odds and includes raw bookie odds for each line. """
    # Structure: lines_data[abs_point] = {'team1_odds': [(bookie, point, price)], 'team2_odds': [(bookie, point, price)]}
    lines_data = defaultdict(lambda: {'team1_odds': [], 'team2_odds': []})
    team1_name = event_data.get('home_team')
    team2_name = event_data.get('away_team')
    # Check if essential data is present
    if not team1_name or not team2_name or not event_data.get('bookmakers'): return []

    for bookie in event_data['bookmakers']:
        spread_market = find_market_data(bookie, 'spreads')
        if spread_market and spread_market.get('outcomes'):
            bookie_title = bookie.get('title', 'Unknown')
            for outcome in spread_market['outcomes']:
                 try:
                    point = float(outcome.get('point'))
                    price = float(outcome.get('price'))
                    name = outcome.get('name') # Team name

                    # Group by the absolute point value to handle +/- pairs
                    point_key = abs(point)

                    # Store raw odds with bookie, actual point, and price
                    if name == team1_name: lines_data[point_key]['team1_odds'].append((bookie_title, point, price))
                    elif name == team2_name: lines_data[point_key]['team2_odds'].append((bookie_title, point, price))
                 except (ValueError, TypeError, AttributeError):
                     continue # Skip invalid data

    processed_lines = []
    # Sort by absolute point value for display
    for point_key, data in sorted(lines_data.items()):
        # Find the odds entry with the best (highest) price for each team
        best_team1_data = max(data['team1_odds'], key=lambda x: x[2]) if data['team1_odds'] else None
        best_team2_data = max(data['team2_odds'], key=lambda x: x[2]) if data['team2_odds'] else None
        # Extract the best prices
        best_team1_price = best_team1_data[2] if best_team1_data else None
        best_team2_price = best_team2_data[2] if best_team2_data else None

        # Extract the actual points associated with the best prices
        best_team1_point = best_team1_data[1] if best_team1_data else -point_key # Default assumption
        best_team2_point = best_team2_data[1] if best_team2_data else point_key # Default assumption

        # Calculate payout based on the BEST prices
        payout = calculate_payout([best_team1_price, best_team2_price])
        processed_lines.append({
            "Line": f"Spread {point_key}", # Basic line identifier
            "Bookies": len(set(b for b, _, _ in data['team1_odds'] + data['team2_odds'])), # Unique bookies count
            "Team1Name": team1_name,
            "Team1Point": f"{best_team1_point:+.1f}", # Display actual point from best odd
            "BestTeam1Price": best_team1_price,
            "Team2Name": team2_name,
            "Team2Point": f"{best_team2_point:+.1f}", # Display actual point from best odd
            "BestTeam2Price": best_team2_price,
            "Payout": payout,
            "RawTeam1": sorted(data['team1_odds'], key=lambda x: x[2], reverse=True), # Sort raw odds best first
            "RawTeam2": sorted(data['team2_odds'], key=lambda x: x[2], reverse=True)
        })
    return processed_lines

# --- Initialize Session State ---
if 'selected_event_id' not in st.session_state: st.session_state.selected_event_id = None
if 'league_odds_data' not in st.session_state: st.session_state.league_odds_data = None
if 'current_league_key' not in st.session_state: st.session_state.current_league_key = None
if 'favorite_bookmaker' not in st.session_state:
    st.session_state.favorite_bookmaker = DEFAULT_REFERENCE_BOOKMAKER_KEY
if 'selected_bookmakers' not in st.session_state:
    # Initialize with a default list, including the favorite
    st.session_state.selected_bookmakers = list(set([DEFAULT_REFERENCE_BOOKMAKER_KEY, 'bet365', 'unibet'])) # Use set for uniqueness


# --- Streamlit App UI ---
st.set_page_config(layout="wide")
load_css("style.css")
st.title("Betting Odds Comparator")

# --- Sidebar ---
st.sidebar.header("Sport & League Selection")
all_sports_data = fetch_available_sports(API_KEY)
selected_sport_details = None
league_changed = False # Flag to track if league selection changed

if all_sports_data:
    # Group sports by category
    sports_by_group = defaultdict(list)
    for sport in all_sports_data:
        group = sport.get('group')
        if group: sports_by_group[group].append(sport)
    unique_sorted_groups = sorted(sports_by_group.keys())

    if not unique_sorted_groups:
        st.sidebar.warning("No sport groups found.")
        st.stop() # Stop if no groups

    # 1. Select Sport Category
    selected_group = st.sidebar.selectbox(
        "1. Select Sport Category:",
        options=unique_sorted_groups,
        index=unique_sorted_groups.index("Soccer") if "Soccer" in unique_sorted_groups else 0, # Default to Soccer if available
        key="group_select_sidebar"
    )

    # 2. Select League (dynamically updated)
    if selected_group and selected_group in sports_by_group:
        leagues_in_group = sorted(sports_by_group[selected_group], key=lambda x: x.get('title', '')) # Sort leagues by title
        if leagues_in_group:
            # Store previous league key to detect changes
            prev_league_selection = st.session_state.get("current_league_key", None)

            selected_sport_details = st.sidebar.radio(
                "2. Select League:",
                options=leagues_in_group,
                format_func=lambda sd: sd.get('title') or sd.get('description') or sd.get('key'), # Display title or key
                key="league_select_sidebar",
            )
            current_league_key = selected_sport_details.get('key') if selected_sport_details else None

            # Check if league selection changed
            if current_league_key != prev_league_selection:
                league_changed = True
                st.session_state.selected_event_id = None # Reset event selection
                st.session_state.league_odds_data = None # Clear old odds data
                st.session_state.current_league_key = current_league_key # Store new key
                print(f"DEBUG: League changed to {current_league_key}. Resetting event view and odds data.") # Debug log
        else:
            st.sidebar.info(f"No active leagues found for {selected_group}.")
            selected_sport_details = None # Ensure no league is treated as selected
            # Reset dependent states if no leagues are available
            if st.session_state.current_league_key is not None:
                 st.session_state.selected_event_id = None
                 st.session_state.league_odds_data = None
                 st.session_state.current_league_key = None
else:
    st.sidebar.error("Could not load initial sports list.")
    st.stop() # Stop if we can't get the sports list


# --- Bookmaker Settings Section (in Sidebar) ---
st.sidebar.divider()
st.sidebar.header("Bookmaker Settings")

# Extract available bookmakers dynamically from fetched data
available_bookies_dict = {} # Use dict {key: title} for easier lookup
bookie_keys_in_data = set()
if st.session_state.league_odds_data:
    # Iterate through all events to capture all bookies present in the league data
    for event in st.session_state.league_odds_data:
        if event.get('bookmakers'):
            for bookie in event['bookmakers']:
                key = bookie.get('key')
                title = bookie.get('title', key) # Use title, fallback to key
                if key and key not in bookie_keys_in_data:
                     available_bookies_dict[key] = title
                     bookie_keys_in_data.add(key)
    # Sort available bookies by title for display
    available_bookies_list = sorted(available_bookies_dict.items(), key=lambda item: item[1])
else:
    available_bookies_list = []

if not available_bookies_list:
    st.sidebar.caption("Select a league to load bookmaker options.")
else:
    # Ensure current favorite is in the list, add if missing
    if st.session_state.favorite_bookmaker not in bookie_keys_in_data:
        fav_title = st.session_state.favorite_bookmaker # Use key as title if not found
        # Prepend to list to ensure it's selectable
        available_bookies_list.insert(0, (st.session_state.favorite_bookmaker, fav_title))
        bookie_keys_in_data.add(st.session_state.favorite_bookmaker) # Add to set too

    # Find index of current favorite for selectbox default
    fav_options_keys = [key for key, _ in available_bookies_list]
    fav_index = fav_options_keys.index(st.session_state.favorite_bookmaker) if st.session_state.favorite_bookmaker in fav_options_keys else 0

    # Setting 1: Favorite Bookmaker (for List View)
    selected_fav_key = st.sidebar.selectbox(
        "Reference Bookmaker (List View):",
        options=fav_options_keys, # Use keys as options
        format_func=lambda key: available_bookies_dict.get(key, key), # Show title using dict lookup
        index=fav_index,
        key="favorite_bookie_selector",
        help="Select the bookmaker whose odds are shown in the main event list."
    )
    # Update state if changed (no rerun needed immediately)
    if selected_fav_key != st.session_state.favorite_bookmaker:
        st.session_state.favorite_bookmaker = selected_fav_key

    # Setting 2: Bookmakers to Show (for Detail View)
    # Ensure current selections are valid options based on available data
    valid_selected_bookies = [key for key in st.session_state.selected_bookmakers if key in bookie_keys_in_data]
    # If user hasn't selected any valid ones, default to favorite or all? Let's default to favorite.
    if not valid_selected_bookies and st.session_state.favorite_bookmaker in bookie_keys_in_data:
         valid_selected_bookies = [st.session_state.favorite_bookmaker]
    elif not valid_selected_bookies: # If even favorite isn't available
         valid_selected_bookies = list(bookie_keys_in_data) # Fallback to all available


    selected_multi_keys = st.sidebar.multiselect(
        "Bookmakers to Show (Detail View):",
        options=fav_options_keys, # Use keys as options
        format_func=lambda key: available_bookies_dict.get(key, key), # Show title
        default=valid_selected_bookies, # Use validated list as default
        key="multi_bookie_selector",
        help="Select the bookmakers whose odds you want to compare in the detailed event view."
    )
    # Update state if changed
    if set(selected_multi_keys) != set(st.session_state.selected_bookmakers):
        st.session_state.selected_bookmakers = list(selected_multi_keys)


# --- Main Area Logic ---

# 1. Fetch League Odds if a league is selected and data isn't loaded/cached
# This runs automatically if needed when selected_sport_details is set and data is None
if selected_sport_details and st.session_state.league_odds_data is None:
    sport_key = selected_sport_details.get('key')
    league_title = selected_sport_details.get('title', 'Selected League')
    if sport_key:
        with st.spinner(f"Fetching all bookmaker odds for {league_title}..."):
            st.session_state.league_odds_data = fetch_league_odds_all_bookies(
                API_KEY, sport_key, REGIONS_TO_REQUEST, MARKETS_TO_REQUEST
            )
            # After fetching, re-run to update sidebar bookie lists if needed
            st.rerun()
    else:
        st.error("Selected league is missing 'key'.")
        st.session_state.league_odds_data = [] # Set to empty list on error


# 2. Display Logic: Detailed Event View or League List View
if st.session_state.selected_event_id and st.session_state.league_odds_data is not None:
    # --- DETAILED EVENT VIEW ---
    selected_event = find_event_by_id(st.session_state.league_odds_data, st.session_state.selected_event_id)

    if selected_event:
        # --- Event Header Area + Selector ---
        hdr_cols = st.columns([1, 5]) # Columns for back button and event selector
        with hdr_cols[0]:
             if st.button("⬅️ Back"): # Simpler back button text
                 st.session_state.selected_event_id = None
                 st.rerun() # Force rerun to show list view

        with hdr_cols[1]:
            # Prepare event list for dropdown
            event_options = sorted(st.session_state.league_odds_data, key=lambda x: x.get('commence_time', ''))
            event_display_names = {e.get('id'): f"{e.get('home_team','?')} vs {e.get('away_team','?')} ({format_datetime(e.get('commence_time'), '%H:%M')})" for e in event_options}
            event_option_ids = [e.get('id') for e in event_options]
            current_event_index = event_option_ids.index(st.session_state.selected_event_id) if st.session_state.selected_event_id in event_option_ids else 0

            newly_selected_event_id = st.selectbox(
                "Change Event:",
                options=event_option_ids, # Use IDs as options
                format_func=lambda event_id: event_display_names.get(event_id, "Unknown Event"),
                index=current_event_index,
                key=f"event_selector_{st.session_state.current_league_key}", # Unique key per league
                label_visibility="collapsed" # Hide the label
            )
            # If selection changed in dropdown, update state and rerun
            if newly_selected_event_id != st.session_state.selected_event_id:
                 st.session_state.selected_event_id = newly_selected_event_id
                 st.rerun()

        # Display Event Info
        home_team = selected_event.get('home_team', 'N/A')
        away_team = selected_event.get('away_team', 'N/A')
        event_time_str = format_datetime(selected_event.get('commence_time', ''), fmt='%a, %d %b %Y, %H:%M %Z')
        st.header(f"{home_team} vs {away_team}")
        st.caption(f"📅 {event_time_str}")
        st.markdown("---")

        # Filter bookmakers based on user selection in sidebar
        bookies_to_show_keys = st.session_state.selected_bookmakers
        if not bookies_to_show_keys: # Handle case where user deselects all
             st.warning("No bookmakers selected in the sidebar settings. Please select at least one.")
             filtered_bookies = []
        else:
             filtered_bookies = [
                 bookie for bookie in selected_event.get('bookmakers', [])
                 if bookie.get('key') in bookies_to_show_keys
             ]
             if not filtered_bookies and selected_event.get('bookmakers'): # Check if event had bookies originally
                  st.warning(f"Selected bookmakers ({', '.join(bookies_to_show_keys)}) not found for this event.")


        # Create Tabs for Markets
        tab_titles = ["1X2 (Match Odds)", "Over/Under", "Spread"]
        tabs = st.tabs(tab_titles)

        # --- 1X2 Tab (Uses filtered_bookies) ---
        with tabs[0]:
            if not filtered_bookies:
                st.info("Select bookmakers in the sidebar to see 1X2 odds.")
            else:
                h2h_outcomes = [home_team, 'Draw', away_team]
                bookmaker_data_h2h = []
                # Loop over the FILTERED list of bookies
                for bookie in sorted(filtered_bookies, key=lambda b: b.get('title', '')):
                    bookie_title = bookie.get('title', bookie.get('key')) # Prefer title
                    h2h_odds = get_bookmaker_market_odds(bookie, 'h2h', h2h_outcomes)
                    payout = calculate_payout(list(h2h_odds.values()))
                    # Add row only if at least one odd exists for this bookie
                    if any(v is not None for v in h2h_odds.values()):
                        bookmaker_data_h2h.append({
                            "Bookmaker": bookie_title,
                            "1": h2h_odds.get(home_team),
                            "X": h2h_odds.get('Draw'),
                            "2": h2h_odds.get(away_team),
                            "Payout": f"{payout:.1f}%" if payout is not None else "-" # Format payout
                        })

                # Calculate Average and Highest based on filtered data
                if bookmaker_data_h2h:
                    df_h2h_numeric = pd.DataFrame(bookmaker_data_h2h)[["1", "X", "2"]].apply(pd.to_numeric, errors='coerce')
                    avg_odds = df_h2h_numeric.mean(skipna=True); avg_payout = calculate_payout(avg_odds.tolist())
                    high_odds = df_h2h_numeric.max(skipna=True); high_payout = calculate_payout(high_odds.tolist())
                    # Append rows for Average and Highest
                    bookmaker_data_h2h.append({"Bookmaker": "**Average**", "1": avg_odds['1'], "X": avg_odds['X'], "2": avg_odds['2'], "Payout": f"{avg_payout:.1f}%" if avg_payout else "-"})
                    bookmaker_data_h2h.append({"Bookmaker": "**Highest**", "1": high_odds['1'], "X": high_odds['X'], "2": high_odds['2'], "Payout": f"{high_payout:.1f}%" if high_payout else "-"})

                    # Create and format DataFrame for display
                    df_display = pd.DataFrame(bookmaker_data_h2h)
                    for col in ["1", "X", "2"]: # Format numeric columns nicely
                        df_display[col] = df_display[col].map(lambda x: f"{x:.2f}" if pd.notna(x) else '-')

                    st.dataframe(df_display, hide_index=True, use_container_width=True)
                else:
                    st.info(f"No H2H odds found from the selected bookmakers.")


        # --- Over/Under Tab (Uses filtered_bookies passed to helper) ---
        with tabs[1]:
            if not filtered_bookies:
                 st.info("Select bookmakers in the sidebar to see Over/Under odds.")
            else:
                st.subheader("Best Odds Comparison: Over/Under Lines")
                # Pass filtered bookies to the aggregation helper
                temp_event_data_totals = selected_event.copy()
                temp_event_data_totals['bookmakers'] = filtered_bookies
                processed_totals_data = aggregate_totals_lines_with_details(temp_event_data_totals)

                if processed_totals_data:
                    # Header Row for aggregated data
                    cols = st.columns([3, 1, 2, 2, 2]) # Weights: Line | Bookies | Over | Under | Payout
                    cols[0].markdown("**Line**"); cols[1].markdown("**Bookies**"); cols[2].markdown("**Best Over**"); cols[3].markdown("**Best Under**"); cols[4].markdown("**Payout**")
                    st.divider()

                    # Display each aggregated line
                    for line_data in processed_totals_data:
                        cols = st.columns([3, 1, 2, 2, 2])
                        cols[0].markdown(f"{line_data['Line']}")
                        cols[1].markdown(f"{line_data['Bookies']}")
                        cols[2].markdown(f"**{line_data['BestOver']:.2f}**" if line_data['BestOver'] else "-")
                        cols[3].markdown(f"**{line_data['BestUnder']:.2f}**" if line_data['BestUnder'] else "-")
                        cols[4].markdown(f"{line_data['Payout']:.1f}%" if line_data['Payout'] else "-")

                        # --- Expander with DataFrame for Bookmaker Details ---
                        with st.expander(f"Show Bookmaker Odds for {line_data['Line']}"):
                            # Prepare data: Map bookies to their Over/Under odds for this specific line
                            expander_rows = []
                            bookie_line_data = defaultdict(lambda: {'Over': None, 'Under': None})
                            for bookie, price in line_data['RawOver']: bookie_line_data[bookie]['Over'] = price
                            for bookie, price in line_data['RawUnder']: bookie_line_data[bookie]['Under'] = price

                            # Create rows for the DataFrame inside the expander
                            for bookie, odds in sorted(bookie_line_data.items()):
                                over_price = odds.get('Over'); under_price = odds.get('Under')
                                payout = calculate_payout([over_price, under_price]) # Payout for this bookie on this line
                                expander_rows.append({
                                    "Bookmaker": bookie,
                                    "Over": f"{over_price:.2f}" if over_price else "-",
                                    "Under": f"{under_price:.2f}" if under_price else "-",
                                    "Payout": f"{payout:.1f}%" if payout else "-"
                                })

                            if expander_rows:
                                st.dataframe(pd.DataFrame(expander_rows), hide_index=True, use_container_width=True)
                            else:
                                st.caption("No individual bookmaker odds found for this line.") # Should not happen if aggregated row exists
                        st.divider() # Divider between aggregated lines
                else:
                    st.info(f"No Over/Under market data found from selected bookmakers.")


        # --- Spread Tab (Uses filtered_bookies passed to helper) ---
        with tabs[2]:
            if not filtered_bookies:
                 st.info("Select bookmakers in the sidebar to see Spread odds.")
            else:
                st.subheader("Best Odds Comparison: Spread Lines")
                # Pass filtered bookies to the aggregation helper
                temp_event_data_spreads = selected_event.copy()
                temp_event_data_spreads['bookmakers'] = filtered_bookies
                processed_spreads_data = aggregate_spread_lines_with_details(temp_event_data_spreads)

                if processed_spreads_data:
                    # Header Row for aggregated data (dynamic team names)
                    team1_name = processed_spreads_data[0]['Team1Name']
                    team2_name = processed_spreads_data[0]['Team2Name']
                    cols = st.columns([3, 1, 3, 3, 2]) # Weights: Line | Bookies | Team1 | Team2 | Payout
                    cols[0].markdown("**Line**"); cols[1].markdown("**Bookies**"); cols[2].markdown(f"**Best {team1_name}**"); cols[3].markdown(f"**Best {team2_name}**"); cols[4].markdown("**Payout**")
                    st.divider()

                    # Display each aggregated line
                    for line_data in processed_spreads_data:
                        cols = st.columns([3, 1, 3, 3, 2])
                        # Display the line with points associated with the BEST odds
                        cols[0].markdown(f"{line_data['Team1Name']} {line_data['Team1Point']} / {line_data['Team2Name']} {line_data['Team2Point']}")
                        cols[1].markdown(f"{line_data['Bookies']}")
                        cols[2].markdown(f"**{line_data['BestTeam1Price']:.2f}**" if line_data['BestTeam1Price'] else "-")
                        cols[3].markdown(f"**{line_data['BestTeam2Price']:.2f}**" if line_data['BestTeam2Price'] else "-")
                        cols[4].markdown(f"{line_data['Payout']:.1f}%" if line_data['Payout'] else "-")

                        # --- Expander with DataFrame for Bookmaker Details ---
                        with st.expander(f"Show Bookmaker Odds for Spread {line_data['Line'].split(' ')[-1]}"):
                            # Prepare data: Map bookies to their odds (point & price) for this line
                            expander_rows = []
                            bookie_line_data = defaultdict(lambda: {team1_name: None, team2_name: None})
                            # RawTeam data structure: (bookie, point, price)
                            for bookie, point, price in line_data['RawTeam1']: bookie_line_data[bookie][team1_name] = (point, price)
                            for bookie, point, price in line_data['RawTeam2']: bookie_line_data[bookie][team2_name] = (point, price)

                            # Create rows for the DataFrame
                            for bookie, odds in sorted(bookie_line_data.items()):
                                team1_data = odds.get(team1_name); team2_data = odds.get(team2_name)
                                team1_point = team1_data[0] if team1_data else None; team1_price = team1_data[1] if team1_data else None
                                team2_point = team2_data[0] if team2_data else None; team2_price = team2_data[1] if team2_data else None
                                payout = calculate_payout([team1_price, team2_price]) # Payout for this bookie on this line
                                expander_rows.append({
                                    "Bookmaker": bookie,
                                    f"{team1_name}": f"{team1_price:.2f} ({team1_point:+.1f})" if team1_price and team1_point is not None else "-",
                                    f"{team2_name}": f"{team2_price:.2f} ({team2_point:+.1f})" if team2_price and team2_point is not None else "-",
                                    "Payout": f"{payout:.1f}%" if payout else "-"
                                })

                            if expander_rows:
                                df_expander = pd.DataFrame(expander_rows)
                                # Optionally rename columns if team names are very long
                                rename_cols = {}
                                if len(team1_name) > 12: rename_cols[team1_name] = f"{team1_name[:10]}.."
                                if len(team2_name) > 12: rename_cols[team2_name] = f"{team2_name[:10]}.."
                                if rename_cols: df_expander = df_expander.rename(columns=rename_cols)
                                st.dataframe(df_expander, hide_index=True, use_container_width=True)
                            else:
                                st.caption("No individual bookmaker odds found for this line.")
                        st.divider()
                else:
                    st.info(f"No Spread market data found from selected bookmakers.")

    else: # Event not found (should be rare if data is consistent)
        st.error("Selected event data could not be found. Returning to league view.")
        st.session_state.selected_event_id = None
        st.session_state.league_odds_data = None # Clear potentially inconsistent data
        st.rerun()


elif selected_sport_details: # LEAGUE LIST VIEW
    league_title = selected_sport_details.get('title', 'Selected League')
    st.header(f"Upcoming Events: {league_title}")
    odds_data = st.session_state.league_odds_data

    if odds_data is None:
        # This state occurs briefly while fetching after league change
        st.warning("Fetching or processing odds data...")
    elif not odds_data:
        st.info(f"No upcoming events found for {league_title} with the selected API parameters.")
    else:
        st.success(f"Found {len(odds_data)} events. Click event title for details.")
        st.markdown("---")
        events_by_date = defaultdict(list)
        for event in odds_data:
             events_by_date[format_datetime(event.get('commence_time'), fmt='%a, %d %b %Y')].append(event)
        sorted_dates = sorted(events_by_date.keys(), key=lambda d: datetime.strptime(d, '%a, %d %b %Y'))

        # Use selected favorite bookmaker from session state
        favorite_bookie_key = st.session_state.favorite_bookmaker
        # Get the title for the favorite bookie (fallback to key)
        favorite_bookie_title = available_bookies_dict.get(favorite_bookie_key, favorite_bookie_key)

        for event_date in sorted_dates:
            st.subheader(event_date)
            date_events = sorted(events_by_date[event_date], key=lambda x: x.get('commence_time', ''))

            # Display Header Row using favorite bookie title
            cols_h = st.columns([1, 4, 1, 1, 1]) # Adjust weights as needed
            cols_h[0].markdown("**Time**")
            cols_h[1].markdown("**Event**")
            cols_h[2].markdown(f"**1 ({favorite_bookie_title[:4]})**") # Abbreviate title
            cols_h[3].markdown(f"**X ({favorite_bookie_title[:4]})**")
            cols_h[4].markdown(f"**2 ({favorite_bookie_title[:4]})**")

            # Display Event Rows
            for event in date_events:
                 event_id = event.get('id')
                 home_team = event.get('home_team', 'N/A')
                 away_team = event.get('away_team', 'N/A')
                 event_title = f"{home_team} vs {away_team}"
                 event_time = format_datetime(event.get('commence_time'), fmt='%H:%M')

                 # Find the favorite bookmaker's data for this event
                 ref_bookie_data = next((b for b in event.get('bookmakers', []) if b.get('key') == favorite_bookie_key), None)
                 # Get H2H odds from the favorite bookie
                 ref_odds = get_bookmaker_market_odds(ref_bookie_data, 'h2h', [home_team, 'Draw', away_team])

                 # Display row data in columns
                 cols = st.columns([1, 4, 1, 1, 1])
                 cols[0].markdown(f"{event_time}")
                 # Use button for event title to make it clickable
                 if cols[1].button(event_title, key=f"event_{event_id}", help="Click to see detailed odds comparison"):
                     st.session_state.selected_event_id = event_id
                     print(f"DEBUG: Event button clicked: {event_id}")
                     st.rerun() # Trigger rerun to show detail view
                 # Display reference odds
                 cols[2].markdown(f"{ref_odds.get(home_team):.2f}" if ref_odds.get(home_team) else "-")
                 cols[3].markdown(f"{ref_odds.get('Draw'):.2f}" if ref_odds.get('Draw') else "-")
                 cols[4].markdown(f"{ref_odds.get(away_team):.2f}" if ref_odds.get(away_team) else "-")
            st.markdown("---") # Separator between dates


elif not selected_sport_details and all_sports_data:
    # This case handles when a category is selected but has no leagues
    st.info("Select a league from the sidebar to view events.")


# Initial state if no category/league selected yet
elif not selected_sport_details:
    st.info("👈 Select a sport category and then a league from the sidebar to view odds.")