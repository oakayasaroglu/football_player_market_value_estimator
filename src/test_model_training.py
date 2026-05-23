import pandas as pd
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Metrics
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Models
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor



class ModelBenchmarkerTest:
    def __init__(self, input_dir: str = None, output_dir: str = None):
        base_dir = Path(__file__).parent.parent
        self.input_dir = Path(input_dir) if input_dir else base_dir / "dataset" / "model_input"
        self.output_dir = Path(output_dir) if output_dir else base_dir / "results"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
        
    def load_data_sample(self, n_samples: int = 500):
        print(f"Loading a small sample ({n_samples} rows) from {self.input_dir} for testing...")
        # Read only a few rows for testing
        X_train_df = pd.read_csv(self.input_dir / "X_train_scaled.csv", nrows=n_samples)
        X_test_df = pd.read_csv(self.input_dir / "X_test_scaled.csv", nrows=n_samples // 5)
        
        # Handle NaNs
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='median')
        self.X_train = pd.DataFrame(imputer.fit_transform(X_train_df), columns=X_train_df.columns)
        self.X_test = pd.DataFrame(imputer.transform(X_test_df), columns=X_test_df.columns)
        
        self.y_train = pd.read_csv(self.input_dir / "y_train.csv", nrows=n_samples).values.ravel()
        self.y_test = pd.read_csv(self.input_dir / "y_test.csv", nrows=n_samples // 5).values.ravel()
        
        print(f"Sample X_train shape: {self.X_train.shape}, y_train shape: {self.y_train.shape}")
        
    def evaluate_model(self, model_name: str, y_true, y_pred, training_time: float):
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        self.results.append({
            "Model": model_name,
            "R2_Score": r2,
            "RMSE": rmse,
            "MAE": mae,
            "Training_Time_s": training_time
        })
        
        print(f"{model_name} - R2: {r2:.4f} | RMSE: {rmse:.4f} | Time: {training_time:.2f}s")
        
    def train_and_evaluate_fast(self):
        # 1. Ridge Regression
        print("\n--- Testing Ridge Regression ---")
        ridge = Ridge(alpha=1.0)
        start_time = time.time()
        ridge.fit(self.X_train, self.y_train)
        self.evaluate_model("Ridge Regression", self.y_test, ridge.predict(self.X_test), time.time() - start_time)
        
        # 2. Random Forest (Only 2 trees)
        print("\n--- Testing Random Forest ---")
        rf = RandomForestRegressor(n_estimators=2, random_state=42, n_jobs=-1)
        start_time = time.time()
        rf.fit(self.X_train, self.y_train)
        self.evaluate_model("Random Forest", self.y_test, rf.predict(self.X_test), time.time() - start_time)
        
        # 3. XGBoost (Only 2 iterations)
        print("\n--- Testing XGBoost ---")
        xgb = XGBRegressor(n_estimators=2, learning_rate=0.05, random_state=42, n_jobs=-1)
        start_time = time.time()
        xgb.fit(self.X_train, self.y_train)
        self.evaluate_model("XGBoost", self.y_test, xgb.predict(self.X_test), time.time() - start_time)
        
        # 4. LightGBM (Only 2 iterations)
        print("\n--- Testing LightGBM ---")
        lgb = LGBMRegressor(n_estimators=2, learning_rate=0.05, random_state=42, n_jobs=-1)
        start_time = time.time()
        lgb.fit(self.X_train, self.y_train)
        self.evaluate_model("LightGBM", self.y_test, lgb.predict(self.X_test), time.time() - start_time)
        
        # 5. CatBoost (Only 2 iterations)
        print("\n--- Testing CatBoost ---")
        cat = CatBoostRegressor(iterations=2, learning_rate=0.05, random_seed=42, verbose=0)
        start_time = time.time()
        cat.fit(self.X_train, self.y_train)
        self.evaluate_model("CatBoost", self.y_test, cat.predict(self.X_test), time.time() - start_time)
        

    def save_and_plot_results(self):
        df_results = pd.DataFrame(self.results)
        csv_path = self.output_dir / "test_benchmark_results.csv"
        df_results.to_csv(csv_path, index=False)
        print(f"\nTest results saved to {csv_path}")
        print(df_results)
        
        # Simple plot to verify it works
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(8, 4))
        sns.barplot(x="R2_Score", y="Model", data=df_results, palette="viridis")
        plt.title("Test R2 Score Comparison")
        plt.tight_layout()
        plot_path = self.output_dir / "test_benchmark_plot.png"
        plt.savefig(plot_path)
        print(f"Test plot saved to {plot_path}")

    def run_test(self):
        print("="*50)
        print("Starting Quick Test for Benchmark Pipeline")
        print("="*50)
        self.load_data_sample(n_samples=500)
        self.train_and_evaluate_fast()
        self.save_and_plot_results()
        print("="*50)
        print("Test Pipeline Completed Successfully")
        print("="*50)

if __name__ == "__main__":
    tester = ModelBenchmarkerTest()
    tester.run_test()
