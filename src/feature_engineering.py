import pandas as pd
import numpy as np
from pathlib import Path

class FeatureEngineer:
    def __init__(self, input_path: str = "../dataset/processed/cleaned_player_data.csv"):
        self.input_path = Path(input_path)
        
    def load_data(self):
        print(f"Loading merged data from {self.input_path}...")
        self.df = pd.read_csv(self.input_path)
        return self.df

    def engineer_performance_features(self, df):
        print("Engineering performance features...")
        # Career
        df['career_goals_per_game'] = df['career_goals'] / df['career_nb_in_group'].replace(0, np.nan)
        df['career_assists_per_game'] = df['career_assists'] / df['career_nb_in_group'].replace(0, np.nan)
        df['career_minutes_per_game'] = df['career_minutes_played'] / df['career_nb_in_group'].replace(0, np.nan)
        df['career_started_ratio'] = df['career_nb_on_pitch'] / df['career_nb_in_group'].replace(0, np.nan)
        df['clean_sheet_ratio'] = df['career_clean_sheets'] / df['career_nb_on_pitch'].replace(0, np.nan)
        df['goal_contribution_ratio'] = (df['career_goals'] + df['career_assists']) / df['career_nb_in_group'].replace(0, np.nan)
        df['subbed_in_ratio'] = df['career_subed_in'] / df['career_nb_in_group'].replace(0, np.nan)
        df['subbed_out_ratio'] = df['career_subed_out'] / df['career_nb_on_pitch'].replace(0, np.nan)

        # Recent
        df['recent_goals_per_game'] = df['recent_goals'] / df['recent_nb_in_group'].replace(0, np.nan)
        df['recent_assists_per_game'] = df['recent_assists'] / df['recent_nb_in_group'].replace(0, np.nan)
        df['recent_minutes_per_game'] = df['recent_minutes_played'] / df['recent_nb_in_group'].replace(0, np.nan)

        # Form
        df['playing_time_form'] = df['recent_minutes_per_game'] / df['career_minutes_per_game'].replace(0, np.nan)
        df['goal_form'] = df['recent_goals_per_game'] / df['career_goals_per_game'].replace(0, np.nan)

        # National
        df['national_goals_per_match'] = df['national_goals'] / df['national_matches'].replace(0, np.nan)
        
        return df

    def engineer_injury_features(self, df):
        print("Engineering injury features...")
        df['avg_injury_days'] = df['total_days_injured'] / df['injury_count'].replace(0, np.nan)
        df['games_missed_per_injury'] = df['total_games_missed'] / df['injury_count'].replace(0, np.nan)
        return df

    def engineer_discipline_features(self, df):
        print("Engineering discipline features (cards)...")
        df['yellow_cards_per_game'] = df['career_yellow_cards'] / df['career_nb_in_group'].replace(0, np.nan)
        # Combines direct red cards and second yellow cards for a total red card count
        df['red_cards_per_game'] = (df['career_direct_red_cards'] + df['career_second_yellow_cards']) / df['career_nb_in_group'].replace(0, np.nan)
        return df

    def engineer_transfer_features(self, df):
        print("Engineering transfer features...")
        if 'last_transfer_fee' in df.columns and 'max_value_at_transfer' in df.columns:
            df['fee_to_max_value_ratio'] = df['last_transfer_fee'] / df['max_value_at_transfer'].replace(0, np.nan)
        return df

    def engineer_temporal_features(self, df):
        print("Engineering temporal features (age, contract, etc.)...")
        # Determine reference date dynamically from dataset (e.g. from latest market value date)
        if 'date_unix' in df.columns:
            reference_date = pd.to_datetime(df['date_unix'], errors='coerce')
        elif 'date' in df.columns:
            reference_date = pd.to_datetime(df['date'], errors='coerce')
        else:
            reference_date = pd.Timestamp.now()

        # Calculate age
        if 'date_of_birth' in df.columns:
            df['date_of_birth'] = pd.to_datetime(df['date_of_birth'], errors='coerce')
            df['age'] = (reference_date - df['date_of_birth']).dt.days / 365.25

        # Calculate contract remaining days
        if 'contract_expires' in df.columns:
            df['contract_expires'] = pd.to_datetime(df['contract_expires'], errors='coerce')
            df['contract_days_remaining'] = (df['contract_expires'] - reference_date).dt.days

        # Parse joined date
        if 'joined' in df.columns:
            df['joined'] = pd.to_datetime(df['joined'], errors='coerce')
            df['days_at_current_club'] = (reference_date - df['joined']).dt.days

        # Calculate days since last injury
        if 'last_injury_date' in df.columns:
            df['last_injury_date'] = pd.to_datetime(df['last_injury_date'], errors='coerce')
            df['days_since_last_injury'] = (reference_date - df['last_injury_date']).dt.days
            # Players with no injuries will have NaN. We fill this with a large number (e.g. 3650 days = 10 years)
            df['days_since_last_injury'] = df['days_since_last_injury'].fillna(3650)

        return df

    def run_pipeline(self, output_path: str = "../dataset/processed/engineered_player_data.csv"):
        print("\n" + "="*50)
        print("Starting feature engineering process...")
        print("="*50 + "\n")
        
        df = self.load_data()
        
        # Apply feature engineering steps
        df = self.engineer_performance_features(df)
        df = self.engineer_injury_features(df)
        df = self.engineer_temporal_features(df)
        df = self.engineer_discipline_features(df)
        df = self.engineer_transfer_features(df)
        
        # Resolve any NaNs generated by division by zero for specific ratio features
        # since 0 matches played naturally means 0 goals per game, not NaN
        derived_ratios = [
            'career_goals_per_game', 'career_assists_per_game', 'career_minutes_per_game', 'career_started_ratio',
            'recent_goals_per_game', 'recent_assists_per_game', 'national_goals_per_match', 'avg_injury_days',
            'yellow_cards_per_game', 'red_cards_per_game', 'games_missed_per_injury',
            'recent_minutes_per_game', 'playing_time_form', 'goal_form', 'clean_sheet_ratio',
            'goal_contribution_ratio', 'subbed_in_ratio', 'subbed_out_ratio', 'fee_to_max_value_ratio'
        ]
        
        # Only fillna for the derived columns that actually exist to avoid KeyError
        existing_ratios = [col for col in derived_ratios if col in df.columns]
        df[existing_ratios] = df[existing_ratios].fillna(0)
        
        # Create output directory if not exists
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save
        df.to_csv(output_file, index=False)
        print(f"\nSaved engineered data to: {output_path}")
        print(f"  Shape: {df.shape}")
        
        return df

if __name__ == "__main__":
    fe = FeatureEngineer()
    df = fe.run_pipeline()
    
    print("\n" + "="*50)
    print("Sample of engineered data (first 5 rows):")
    print("="*50)
    
    # Display a sample of some of the engineered columns
    cols_to_show = ['player_name', 'age', 'career_goals_per_game', 'recent_goals_per_game', 'avg_injury_days']
    existing_cols = [col for col in cols_to_show if col in df.columns]
    
    if existing_cols:
        print(df[existing_cols].head())
    else:
        print(df.head())
