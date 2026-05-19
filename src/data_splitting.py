import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

class DataSplitter:
    def __init__(self, input_path: str = "../dataset/processed/engineered_player_data.csv"):
        self.input_path = Path(input_path)
        
    def drop_unnecessary_columns(self, df):
        """Modele hiçbir faydası olmayan veya ezbere (overfitting) yol açacak ID kolonlarını siler."""
        print("Dropping unnecessary columns (IDs etc.)...")
        
        # Sileceğimiz kolonların listesi
        # player_id: Sadece kimlik numarası, tahmin gücü yok.
        # current_club_id: Biz zaten kulüpleri frequency_encoding ile sayısallaştırdık, bu ID'nin kalmasına gerek yok.
        cols_to_drop = ['player_id', 'current_club_id']
        
        # Sadece var olanları düşür ki kod hata vermesin
        existing_to_drop = [col for col in cols_to_drop if col in df.columns]
        
        if existing_to_drop:
            df = df.drop(columns=existing_to_drop)
            print(f"Dropped columns: {existing_to_drop}")
            
        return df

    def split_data(self, output_dir: str = "../dataset/model_input"):
        print(f"\nLoading engineered data from {self.input_path}...")
        df = pd.read_csv(self.input_path)
        
        # 1. Gereksiz kolonları at
        df = self.drop_unnecessary_columns(df)
        
        # 2. X ve y (Öznitelikler ve Hedef) olarak ayır
        print("\nSeparating Features (X) and Target (y)...")
        y = df['value']
        X = df.drop(columns=['value'])
        
        # 3. Train-Test Split (%80 Eğitim, %20 Test)
        print("Splitting data into Train and Test sets (80% / 20%)...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print(f"  X_train shape: {X_train.shape}")
        print(f"  X_test shape:  {X_test.shape}")
        print(f"  y_train shape: {y_train.shape}")
        print(f"  y_test shape:  {y_test.shape}")
        
        # 4. Çıktıları yeni bir klasöre kaydet
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        X_train.to_csv(out_path / "X_train.csv", index=False)
        X_test.to_csv(out_path / "X_test.csv", index=False)
        y_train.to_csv(out_path / "y_train.csv", index=False)
        y_test.to_csv(out_path / "y_test.csv", index=False)
        
        print(f"\nSuccessfully saved split datasets to: {out_path}/")
        
        return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    print("="*50)
    print("Starting Data Splitting Process...")
    print("="*50)
    
    splitter = DataSplitter()
    X_train, X_test, y_train, y_test = splitter.split_data()
