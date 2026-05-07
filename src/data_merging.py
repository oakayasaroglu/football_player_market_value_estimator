import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

class DataMerger:
    def __init__(self, raw_data_path: str = "../dataset/raw"):
        self.raw_path = Path(raw_data_path)

    def load_data(self):
        """Load all CSV files"""
        print("Loading data...")

        # Main player data
        self.profiles = pd.read_csv(self.raw_path / "player_profiles.csv")
        self.latest_value = pd.read_csv(self.raw_path / "player_latest_market_value.csv")

        # Performance data
        self.performances = pd.read_csv(self.raw_path / "player_performances.csv")
        self.national_performances = pd.read_csv(self.raw_path / "player_national_performances.csv")

        # History data
        self.injuries = pd.read_csv(self.raw_path / "player_injuries.csv")
        self.transfers = pd.read_csv(self.raw_path / "transfer_history.csv")
        self.market_value_history = pd.read_csv(self.raw_path / "player_market_value.csv")

        # Team data
        self.team_details = pd.read_csv(self.raw_path / "team_details.csv")

        print(f"Loaded {len(self.profiles)} player profiles")

    def aggregate_performances(self):
        """Aggregate club performances per player"""
        print("Aggregating club performances...")

        perf_agg = self.performances.groupby('player_id').agg({
            # Career totals
            'goals': 'sum',
            'assists': 'sum',
            'minutes_played': 'sum',
            'yellow_cards': 'sum',
            'second_yellow_cards': 'sum',
            'direct_red_cards': 'sum',
            'penalty_goals': 'sum',
            'own_goals': 'sum',
            'nb_in_group': 'sum',  # Total games
            'nb_on_pitch': 'sum',  # Games started
            'subed_in': 'sum',
            'subed_out': 'sum',
            'goals_conceded': 'sum',
            'clean_sheets': 'sum'
        }).reset_index()

        # Rename columns
        perf_agg.columns = ['player_id'] + [f'career_{col}' if col != 'player_id' else col
                                             for col in perf_agg.columns[1:]]

        return perf_agg

    def aggregate_recent_performances(self, last_n_seasons=3):
        """Get recent season performance trends"""
        print(f"Aggregating last {last_n_seasons} seasons performance...")

        # Sort by season and get latest seasons
        def parse_season(val):
            val = str(val).strip()
            if '/' in val:
                start_year = val.split('/')[0]
                if len(start_year) == 2:
                    y = int(start_year)
                    return 1900 + y if y > 50 else 2000 + y
                return int(start_year)
            elif len(val) >= 4:
                return int(val[:4])
            else:
                return 0
                
        self.performances['season_year'] = self.performances['season_name'].apply(parse_season)
        
        # Filter strictly by the last N seasons globally
        max_year = self.performances['season_year'].max()
        recent_perf = self.performances[self.performances['season_year'] >= max_year - last_n_seasons + 1]

        recent_agg = recent_perf.groupby('player_id').agg({
            'goals': 'sum',
            'assists': 'sum',
            'minutes_played': 'sum',
            'nb_in_group': 'sum',
            'yellow_cards': 'sum'
        }).reset_index()

        recent_agg.columns = ['player_id'] + [f'recent_{col}' if col != 'player_id' else col
                                               for col in recent_agg.columns[1:]]

        return recent_agg

    def aggregate_national_performances(self):
        """Aggregate national team performances"""
        print("Aggregating national team performances...")

        nat_agg = self.national_performances.groupby('player_id').agg({
            'matches': 'sum',
            'goals': 'sum'
        }).reset_index()

        nat_agg.columns = ['player_id', 'national_matches', 'national_goals']

        return nat_agg

    def aggregate_injuries(self):
        """Aggregate injury history"""
        print("Aggregating injury history...")

        injury_agg = self.injuries.groupby('player_id').agg({
            'days_missed': 'sum',
            'games_missed': 'sum',
            'injury_reason': 'count'
        }).reset_index()

        injury_agg.columns = ['player_id', 'total_days_injured', 'total_games_missed', 'injury_count']

        # Get most recent injury date
        last_injury = self.injuries.sort_values('end_date', ascending=False).groupby('player_id').first()[['end_date']].reset_index()
        last_injury.columns = ['player_id', 'last_injury_date']

        injury_agg = injury_agg.merge(last_injury, on='player_id', how='left')

        return injury_agg

    def aggregate_transfers(self):
        """Aggregate transfer history"""
        print("Aggregating transfer history...")

        transfer_agg = self.transfers.groupby('player_id').agg({
            'transfer_fee': ['sum', 'max', 'mean', 'count'],
            'value_at_transfer': 'max'
        }).reset_index()

        transfer_agg.columns = ['player_id', 'total_transfer_fees', 'max_transfer_fee',
                                'avg_transfer_fee', 'transfer_count', 'max_value_at_transfer']

        # Get last transfer date
        last_transfer = self.transfers.sort_values('transfer_date', ascending=False).groupby('player_id').first()[
            ['transfer_date', 'transfer_fee']
        ].reset_index()
        last_transfer.columns = ['player_id', 'last_transfer_date', 'last_transfer_fee']

        transfer_agg = transfer_agg.merge(last_transfer, on='player_id', how='left')

        return transfer_agg

    def process_profiles(self):
        """Process player profile data"""
        print("Processing player profiles...")

        profiles = self.profiles.copy()

        # Select relevant columns, preserving raw dates for feature engineering
        profile_cols = [
            'player_id', 'player_name', 'date_of_birth', 'height', 'position', 'main_position', 'foot',
            'citizenship', 'is_eu', 'current_club_id', 'current_club_name',
            'contract_expires', 'joined', 'country_of_birth'
        ]

        # Ensure we only pick columns that exist to prevent KeyErrors
        existing_cols = [col for col in profile_cols if col in profiles.columns]

        return profiles[existing_cols]

    def merge_all(self):
        """Merge all data sources"""
        print("\n" + "="*50)
        print("Starting data merging process...")
        print("="*50 + "\n")

        # Load data
        self.load_data()

        # Process each data source
        profiles = self.process_profiles()
        career_perf = self.aggregate_performances()
        recent_perf = self.aggregate_recent_performances()
        national_perf = self.aggregate_national_performances()
        injuries = self.aggregate_injuries()
        transfers = self.aggregate_transfers()

        # Start with profiles as base
        merged = profiles.copy()
        print(f"\nBase (profiles): {len(merged)} players")

        # Merge latest market value (LEFT JOIN - keep all players)
        merged = merged.merge(self.latest_value, on='player_id', how='left')
        print(f"After adding market value: {len(merged)} players")

        # Merge all aggregated data (LEFT JOIN)
        merged = merged.merge(career_perf, on='player_id', how='left')
        print(f"After adding career performance: {len(merged)} players")

        merged = merged.merge(recent_perf, on='player_id', how='left')
        print(f"After adding recent performance: {len(merged)} players")

        merged = merged.merge(national_perf, on='player_id', how='left')
        print(f"After adding national performance: {len(merged)} players")

        merged = merged.merge(injuries, on='player_id', how='left')
        print(f"After adding injuries: {len(merged)} players")

        merged = merged.merge(transfers, on='player_id', how='left')
        print(f"After adding transfers: {len(merged)} players")

        # Fill NaN values for players with no performance/injury/transfer data
        merged = merged.fillna({
            # Career
            'career_goals': 0,
            'career_assists': 0,
            'career_minutes_played': 0,
            'career_yellow_cards': 0,
            'career_second_yellow_cards': 0,
            'career_direct_red_cards': 0,
            'career_penalty_goals': 0,
            'career_own_goals': 0,
            'career_nb_in_group': 0,
            'career_nb_on_pitch': 0,
            'career_subed_in': 0,
            'career_subed_out': 0,
            'career_goals_conceded': 0,
            'career_clean_sheets': 0,
            
            # Recent
            'recent_goals': 0,
            'recent_assists': 0,
            'recent_minutes_played': 0,
            'recent_nb_in_group': 0,
            'recent_yellow_cards': 0,

            # National
            'national_matches': 0,
            'national_goals': 0,

            # Injuries
            'injury_count': 0,
            'total_days_injured': 0,
            'total_games_missed': 0,

            # Transfers
            'transfer_count': 0,
            'total_transfer_fees': 0,
            'max_transfer_fee': 0,
            'avg_transfer_fee': 0,
            'max_value_at_transfer': 0,
            'last_transfer_fee': 0
        })

        print(f"\n{'='*50}")
        print(f"Final merged dataset: {len(merged)} players x {len(merged.columns)} features")
        print(f"{'='*50}\n")

        return merged

    def save_merged_data(self, output_path: str = "../dataset/processed/merged_player_data.csv"):
        """Merge and save final dataset"""
        merged = self.merge_all()

        # Create output directory if not exists
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Save
        merged.to_csv(output_file, index=False)
        print(f"\nSaved merged data to: {output_path}")
        print(f"  Shape: {merged.shape}")
        print(f"  Columns: {list(merged.columns)}")

        # Print data summary
        print(f"\n{'='*50}")
        print("Data Summary:")
        print(f"{'='*50}")
        print(merged.info())
        print(f"\n{'='*50}")
        print("Market Value Statistics:")
        print(f"{'='*50}")
        print(merged['value'].describe())

        return merged


if __name__ == "__main__":
    merger = DataMerger()
    df = merger.save_merged_data()

    # Show sample
    print("\n" + "="*50)
    print("Sample of merged data (first 5 rows):")
    print("="*50)
    print(df.head())
