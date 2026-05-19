import pandas as pd
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib

# Metrics
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Models
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# TabNet
from pytorch_tabnet.tab_model import TabNetRegressor
import torch

class ModelBenchmarker:
    def __init__(self, input_dir: str = None, output_dir: str = None):
        base_dir = Path(__file__).parent.parent
        self.input_dir = Path(input_dir) if input_dir else base_dir / "dataset" / "model_input"
        self.output_dir = Path(output_dir) if output_dir else base_dir / "results"
        self.models_dir = self.output_dir / "saved_models"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
        
    def load_data(self):
        print(f"Loading data from {self.input_dir}...")
        X_train_df = pd.read_csv(self.input_dir / "X_train_scaled.csv")
        X_test_df = pd.read_csv(self.input_dir / "X_test_scaled.csv")
        
        # Handle NaNs
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='median')
        self.X_train = pd.DataFrame(imputer.fit_transform(X_train_df), columns=X_train_df.columns)
        self.X_test = pd.DataFrame(imputer.transform(X_test_df), columns=X_test_df.columns)
        
        self.y_train = pd.read_csv(self.input_dir / "y_train.csv").values.ravel()
        self.y_test = pd.read_csv(self.input_dir / "y_test.csv").values.ravel()
        
        print(f"X_train shape: {self.X_train.shape}, y_train shape: {self.y_train.shape}")
        print(f"X_test shape: {self.X_test.shape}, y_test shape: {self.y_test.shape}")
        
    def save_trained_model(self, model, model_name: str):
        safe_name = model_name.replace(" ", "_").lower()
        if model_name == "TabNet":
            # TabNet automatically appends .zip
            save_path = self.models_dir / safe_name
            model.save_model(str(save_path))
            print(f"Model saved to {save_path}.zip")
        elif model_name == "CatBoost":
            save_path = self.models_dir / f"{safe_name}.cbm"
            model.save_model(str(save_path))
            print(f"Model saved to {save_path}")
        elif model_name == "XGBoost":
            save_path = self.models_dir / f"{safe_name}.json"
            model.save_model(str(save_path))
            print(f"Model saved to {save_path}")
        else:
            save_path = self.models_dir / f"{safe_name}.pkl"
            joblib.dump(model, save_path)
            print(f"Model saved to {save_path}")

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
        
        print(f"{model_name} - R2: {r2:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f} | Time: {training_time:.2f}s")
        
    def train_and_evaluate(self):
        # 1. Ridge Regression (Baseline Linear)
        print("\n--- Training Ridge Regression ---")
        ridge = Ridge(alpha=1.0)
        start_time = time.time()
        ridge.fit(self.X_train, self.y_train)
        ridge_time = time.time() - start_time
        self.save_trained_model(ridge, "Ridge Regression")
        self.evaluate_model("Ridge Regression", self.y_test, ridge.predict(self.X_test), ridge_time)
        
        # 2. Random Forest Regressor (Baseline Tree)
        print("\n--- Training Random Forest ---")
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        start_time = time.time()
        rf.fit(self.X_train, self.y_train)
        rf_time = time.time() - start_time
        self.save_trained_model(rf, "Random Forest")
        self.evaluate_model("Random Forest", self.y_test, rf.predict(self.X_test), rf_time)
        
        # 3. XGBoost
        print("\n--- Training XGBoost ---")
        xgb = XGBRegressor(n_estimators=500, learning_rate=0.05, random_state=42, tree_method='hist', device='cuda')
        start_time = time.time()
        xgb.fit(self.X_train, self.y_train)
        xgb_time = time.time() - start_time
        self.save_trained_model(xgb, "XGBoost")
        self.evaluate_model("XGBoost", self.y_test, xgb.predict(self.X_test), xgb_time)
        
        # 4. LightGBM
        print("\n--- Training LightGBM ---")
        lgb = LGBMRegressor(n_estimators=500, learning_rate=0.05, random_state=42, n_jobs=-1)
        start_time = time.time()
        lgb.fit(self.X_train, self.y_train)
        lgb_time = time.time() - start_time
        self.save_trained_model(lgb, "LightGBM")
        self.evaluate_model("LightGBM", self.y_test, lgb.predict(self.X_test), lgb_time)
        
        # 5. CatBoost
        print("\n--- Training CatBoost ---")
        cat = CatBoostRegressor(iterations=500, learning_rate=0.05, random_seed=42, verbose=0, task_type='GPU')
        start_time = time.time()
        cat.fit(self.X_train, self.y_train)
        cat_time = time.time() - start_time
        self.save_trained_model(cat, "CatBoost")
        self.evaluate_model("CatBoost", self.y_test, cat.predict(self.X_test), cat_time)
        
        # 6. TabNet
        print("\n--- Training TabNet ---")
        # TabNet requires 2D arrays for target and numpy arrays for features
        X_train_np = self.X_train.values
        X_test_np = self.X_test.values
        y_train_np = self.y_train.reshape(-1, 1)
        y_test_np = self.y_test.reshape(-1, 1)
        
        tabnet = TabNetRegressor(
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-2),
            scheduler_params={"step_size":10, "gamma":0.9},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            mask_type='entmax',
            verbose=0
        )
        
        start_time = time.time()
        tabnet.fit(
            X_train=X_train_np, y_train=y_train_np,
            eval_set=[(X_test_np, y_test_np)],
            eval_metric=['rmse'],
            max_epochs=100,
            patience=10,
            batch_size=256,
            virtual_batch_size=128
        )
        tabnet_time = time.time() - start_time
        self.save_trained_model(tabnet, "TabNet")
        
        preds_tabnet = tabnet.predict(X_test_np).ravel()
        self.evaluate_model("TabNet", self.y_test, preds_tabnet, tabnet_time)

    def save_and_plot_results(self):
        df_results = pd.DataFrame(self.results)
        
        # Save to CSV
        csv_path = self.output_dir / "benchmark_results.csv"
        df_results.to_csv(csv_path, index=False)
        print(f"\nResults saved to {csv_path}")
        print(df_results.to_string(index=False))
        
        # Plotting
        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # R2 Plot
        sns.barplot(x="R2_Score", y="Model", data=df_results.sort_values("R2_Score", ascending=False), ax=axes[0], palette="viridis")
        axes[0].set_title("R2 Score Comparison (Higher is Better)")
        axes[0].set_xlabel("R2 Score")
        axes[0].set_ylabel("")
        
        # RMSE Plot
        sns.barplot(x="RMSE", y="Model", data=df_results.sort_values("RMSE", ascending=True), ax=axes[1], palette="magma")
        axes[1].set_title("RMSE Comparison (Lower is Better)")
        axes[1].set_xlabel("RMSE")
        axes[1].set_ylabel("")
        
        # Time Plot
        sns.barplot(x="Training_Time_s", y="Model", data=df_results.sort_values("Training_Time_s", ascending=True), ax=axes[2], palette="crest")
        axes[2].set_title("Training Time Comparison (Lower is Better)")
        axes[2].set_xlabel("Time (seconds)")
        axes[2].set_ylabel("")
        
        plt.tight_layout()
        plot_path = self.output_dir / "benchmark_plot.png"
        plt.savefig(plot_path, dpi=300)
        print(f"Plot saved to {plot_path}")

    def run_pipeline(self):
        print("="*50)
        print("Starting Algorithm Benchmarking Pipeline")
        print("="*50)
        self.load_data()
        self.train_and_evaluate()
        self.save_and_plot_results()
        print("="*50)
        print("Pipeline Completed Successfully")
        print("="*50)

if __name__ == "__main__":
    benchmarker = ModelBenchmarker()
    benchmarker.run_pipeline()
