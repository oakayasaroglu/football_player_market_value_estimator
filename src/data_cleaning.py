import pandas as pd
from pathlib import Path

class DataCleaner:
    def __init__(self, input_path: str = "../dataset/processed/merged_player_data.csv"):
        self.input_path = Path(input_path)

    def load_data(self):
        print(f"Loading merged data from {self.input_path}...")
        self.df = pd.read_csv(self.input_path)
        return self.df

    def clean_target_variable(self, df):
        """Drop rows where the target variable (market value) is missing."""
        print("Cleaning target variable (value)...")
        initial_len = len(df)
        
        # Sadece piyasa değeri NaN olan satırları düşür
        df = df.dropna(subset=['value'])
        
        dropped = initial_len - len(df)
        print(f"Dropped {dropped} players with missing market value.")
        return df

    def run_pipeline(self, output_path: str = "../dataset/processed/cleaned_player_data.csv"):
        print("\n" + "="*50)
        print("Starting data cleaning process...")
        print("="*50 + "\n")
        
        df = self.load_data()
        
        # Temizleme adımları
        df = self.clean_target_variable(df)
        
        # Çıktı klasörünü oluştur
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Temizlenmiş veriyi kaydet
        df.to_csv(output_file, index=False)
        print(f"\nSaved cleaned data to: {output_path}")
        print(f"  Shape: {df.shape}")
        
        return df

if __name__ == "__main__":
    cleaner = DataCleaner()
    df = cleaner.run_pipeline()
