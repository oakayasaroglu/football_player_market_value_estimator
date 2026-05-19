import pandas as pd
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import joblib

class DataScaler:
    def __init__(self, input_dir: str = "../dataset/model_input"):
        self.input_dir = Path(input_dir)
        
    def scale_data(self):
        print(f"Loading split data from {self.input_dir}...")
        
        # Sadece X (Öznitelik) matrislerini yüklüyoruz. y (Hedef) değerleri ölçeklendirilmez.
        X_train = pd.read_csv(self.input_dir / "X_train.csv")
        X_test = pd.read_csv(self.input_dir / "X_test.csv")
        
        print("\nInitializing StandardScaler...")
        scaler = StandardScaler()
        
        # 1. Eğitim setini (Train) hem öğrenir (fit) hem de ölçeklendirir (transform)
        print("Fitting and transforming X_train...")
        X_train_scaled_array = scaler.fit_transform(X_train)
        
        # 2. Test setini sadece dönüştürür (transform) - Eğitim setinden öğrendiği metriklerle
        print("Transforming X_test...")
        X_test_scaled_array = scaler.transform(X_test)
        
        # NumPy array'lerini kolon isimlerini kaybetmemek için tekrar DataFrame'e çeviriyoruz
        X_train_scaled = pd.DataFrame(X_train_scaled_array, columns=X_train.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled_array, columns=X_test.columns)
        
        # Ölçeklendirilmiş verileri kaydet
        print("\nSaving scaled datasets...")
        X_train_scaled.to_csv(self.input_dir / "X_train_scaled.csv", index=False)
        X_test_scaled.to_csv(self.input_dir / "X_test_scaled.csv", index=False)
        
        # İleride sisteme yeni bir futbolcu eklendiğinde aynı ölçeklendirmeyi yapabilmek için 
        # Scaler objesini (öğrenilmiş metrikleriyle birlikte) kaydediyoruz
        scaler_path = self.input_dir / "standard_scaler.pkl"
        joblib.dump(scaler, scaler_path)
        
        print(f"\nSuccess! Scaled data and scaler object saved to {self.input_dir}/")
        print(f"  - X_train_scaled.csv")
        print(f"  - X_test_scaled.csv")
        print(f"  - standard_scaler.pkl")

if __name__ == "__main__":
    print("="*50)
    print("Starting Data Scaling Process...")
    print("="*50)
    
    scaler = DataScaler()
    scaler.scale_data()
