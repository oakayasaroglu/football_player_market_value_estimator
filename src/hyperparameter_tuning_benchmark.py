import pandas as pd
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import optuna

# Metrics
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.impute import SimpleImputer

# Models
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

class HPOTunerBenchmark:
    def __init__(self, input_dir: str = None, output_dir: str = None, n_trials: int = 10, test_mode: bool = False):
        base_dir = Path(__file__).parent.parent
        self.input_dir = Path(input_dir) if input_dir else base_dir / "dataset" / "model_input"
        self.output_dir = Path(output_dir) if output_dir else base_dir / "results"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
        self.test_mode = test_mode
        self.n_trials = 1 if test_mode else n_trials
        self.min_iters = 2 if test_mode else 100
        self.max_iters = 5 if test_mode else 500
        
        # Suppress Optuna logs unless it's an error to keep console clean
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    def load_data(self):
        print(f"Loading data from {self.input_dir}...")
        X_train_df = pd.read_csv(self.input_dir / "X_train_scaled.csv")
        X_test_df = pd.read_csv(self.input_dir / "X_test_scaled.csv")
        
        imputer = SimpleImputer(strategy='median')
        self.X_train = pd.DataFrame(imputer.fit_transform(X_train_df), columns=X_train_df.columns)
        self.X_test = pd.DataFrame(imputer.transform(X_test_df), columns=X_test_df.columns)
        
        self.y_train = pd.read_csv(self.input_dir / "y_train.csv").values.ravel()
        self.y_test = pd.read_csv(self.input_dir / "y_test.csv").values.ravel()
        
        print(f"X_train shape: {self.X_train.shape}, y_train shape: {self.y_train.shape}")
        
    def evaluate_model(self, model_name: str, phase: str, y_true, y_pred, exec_time: float):
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        self.results.append({
            "Model": model_name,
            "Phase": phase,  # Baseline or Tuned
            "R2_Score": r2,
            "RMSE": rmse,
            "MAE": mae,
            "Time_s": exec_time
        })
        
        print(f"{model_name} ({phase}) - R2: {r2:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f} | Time: {exec_time:.2f}s")

    def save_trained_model(self, model, model_name: str):
        safe_name = model_name.replace(" ", "_").lower()
        models_dir = self.output_dir / "saved_models"
        models_dir.mkdir(parents=True, exist_ok=True)
        if model_name == "CatBoost":
            save_path = models_dir / f"{safe_name}_tuned.cbm"
            model.save_model(str(save_path))
            print(f"Tuned model saved to {save_path}")
        elif model_name == "XGBoost":
            save_path = models_dir / f"{safe_name}_tuned.json"
            model.save_model(str(save_path))
            print(f"Tuned model saved to {save_path}")
        else:
            save_path = models_dir / f"{safe_name}_tuned.pkl"
            joblib.dump(model, save_path)
            print(f"Tuned model saved to {save_path}")

    def _cross_validate_rmse(self, model):
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        scores = cross_val_score(model, self.X_train, self.y_train, 
                                 scoring='neg_root_mean_squared_error', cv=kf, n_jobs=1)
        return -scores.mean()

    # ==========================================
    # RIDGE REGRESSION
    # ==========================================
    def tune_ridge(self):
        print("\n--- Tuning Ridge Regression ---")
        
        # Baseline
        start = time.time()
        base_model = Ridge()
        base_model.fit(self.X_train, self.y_train)
        self.evaluate_model("Ridge", "Baseline", self.y_test, base_model.predict(self.X_test), time.time() - start)
        
        # Optuna Objective
        def objective(trial):
            alpha = trial.suggest_float('alpha', 1e-3, 1000.0, log=True)
            model = Ridge(alpha=alpha)
            return self._cross_validate_rmse(model)
            
        study = optuna.create_study(direction='minimize')
        print(f"Running {self.n_trials} trials for Ridge...")
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
        
        # Tuned
        start = time.time()
        tuned_model = Ridge(**study.best_params)
        tuned_model.fit(self.X_train, self.y_train)
        self.evaluate_model("Ridge", "Tuned", self.y_test, tuned_model.predict(self.X_test), time.time() - start)
        self.save_trained_model(tuned_model, "Ridge")
        print(f"Best Ridge Params: {study.best_params}")

    # ==========================================
    # RANDOM FOREST
    # ==========================================
    def tune_rf(self):
        print("\n--- Tuning Random Forest ---")
        
        # Baseline
        start = time.time()
        base_model = RandomForestRegressor(n_estimators=self.min_iters, random_state=42, n_jobs=-1)
        base_model.fit(self.X_train, self.y_train)
        self.evaluate_model("Random Forest", "Baseline", self.y_test, base_model.predict(self.X_test), time.time() - start)
        
        # Optuna Objective
        def objective(trial):
            rf_min = 2 if self.test_mode else 50
            rf_max = 5 if self.test_mode else 100
            param = {
                'n_estimators': trial.suggest_int('n_estimators', rf_min, rf_max, step=1 if self.test_mode else 25),
                'max_depth': trial.suggest_int('max_depth', 5, 15),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 5),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 3),
                'random_state': 42,
                'n_jobs': -1
            }
            model = RandomForestRegressor(**param)
            return self._cross_validate_rmse(model)
            
        study = optuna.create_study(direction='minimize')
        print(f"Running {self.n_trials} trials for Random Forest...")
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
        
        # Tuned
        start = time.time()
        best_params = study.best_params.copy()
        best_params['random_state'] = 42
        best_params['n_jobs'] = -1
        tuned_model = RandomForestRegressor(**best_params)
        tuned_model.fit(self.X_train, self.y_train)
        self.evaluate_model("Random Forest", "Tuned", self.y_test, tuned_model.predict(self.X_test), time.time() - start)
        self.save_trained_model(tuned_model, "Random Forest")
        print(f"Best RF Params: {study.best_params}")

    # ==========================================
    # XGBOOST
    # ==========================================
    def tune_xgb(self):
        print("\n--- Tuning XGBoost ---")
        
        # Baseline
        start = time.time()
        base_model = XGBRegressor(n_estimators=self.min_iters, random_state=42, tree_method='hist', device='cuda')
        base_model.fit(self.X_train, self.y_train)
        self.evaluate_model("XGBoost", "Baseline", self.y_test, base_model.predict(self.X_test), time.time() - start)
        
        # Optuna Objective
        def objective(trial):
            param = {
                'n_estimators': trial.suggest_int('n_estimators', self.min_iters, self.max_iters, step=1 if self.test_mode else 100),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'tree_method': 'hist',
                'device': 'cuda',
                'random_state': 42
            }
            model = XGBRegressor(**param)
            return self._cross_validate_rmse(model)
            
        study = optuna.create_study(direction='minimize')
        print(f"Running {self.n_trials} trials for XGBoost...")
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
        
        # Tuned
        start = time.time()
        best_params = study.best_params.copy()
        best_params['tree_method'] = 'hist'
        best_params['device'] = 'cuda'
        best_params['random_state'] = 42
        tuned_model = XGBRegressor(**best_params)
        tuned_model.fit(self.X_train, self.y_train)
        self.evaluate_model("XGBoost", "Tuned", self.y_test, tuned_model.predict(self.X_test), time.time() - start)
        self.save_trained_model(tuned_model, "XGBoost")
        print(f"Best XGB Params: {study.best_params}")

    # ==========================================
    # LIGHTGBM
    # ==========================================
    def tune_lgbm(self):
        print("\n--- Tuning LightGBM ---")
        
        # Baseline
        start = time.time()
        base_model = LGBMRegressor(n_estimators=self.min_iters, random_state=42, n_jobs=-1, verbose=-1)
        base_model.fit(self.X_train, self.y_train)
        self.evaluate_model("LightGBM", "Baseline", self.y_test, base_model.predict(self.X_test), time.time() - start)
        
        # Optuna Objective
        def objective(trial):
            param = {
                'n_estimators': trial.suggest_int('n_estimators', self.min_iters, self.max_iters, step=1 if self.test_mode else 100),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'random_state': 42,
                'n_jobs': -1,
                'verbose': -1
            }
            model = LGBMRegressor(**param)
            return self._cross_validate_rmse(model)
            
        study = optuna.create_study(direction='minimize')
        print(f"Running {self.n_trials} trials for LightGBM...")
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
        
        # Tuned
        start = time.time()
        best_params = study.best_params.copy()
        best_params['random_state'] = 42
        best_params['n_jobs'] = -1
        best_params['verbose'] = -1
        tuned_model = LGBMRegressor(**best_params)
        tuned_model.fit(self.X_train, self.y_train)
        self.evaluate_model("LightGBM", "Tuned", self.y_test, tuned_model.predict(self.X_test), time.time() - start)
        self.save_trained_model(tuned_model, "LightGBM")
        print(f"Best LGBM Params: {study.best_params}")

    # ==========================================
    # CATBOOST
    # ==========================================
    def tune_catboost(self):
        print("\n--- Tuning CatBoost ---")
        
        # Baseline
        start = time.time()
        base_model = CatBoostRegressor(iterations=self.min_iters, random_seed=42, verbose=0, task_type='GPU')
        base_model.fit(self.X_train, self.y_train)
        self.evaluate_model("CatBoost", "Baseline", self.y_test, base_model.predict(self.X_test), time.time() - start)
        
        # Optuna Objective
        def objective(trial):
            param = {
                'iterations': trial.suggest_int('iterations', self.min_iters, self.max_iters, step=1 if self.test_mode else 100),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'depth': trial.suggest_int('depth', 4, 10),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
                'random_seed': 42,
                'verbose': 0,
                'task_type': 'GPU'
            }
            model = CatBoostRegressor(**param)
            return self._cross_validate_rmse(model)
            
        study = optuna.create_study(direction='minimize')
        print(f"Running {self.n_trials} trials for CatBoost...")
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
        
        # Tuned
        start = time.time()
        best_params = study.best_params.copy()
        best_params['random_seed'] = 42
        best_params['verbose'] = 0
        best_params['task_type'] = 'GPU'
        tuned_model = CatBoostRegressor(**best_params)
        tuned_model.fit(self.X_train, self.y_train)
        self.evaluate_model("CatBoost", "Tuned", self.y_test, tuned_model.predict(self.X_test), time.time() - start)
        self.save_trained_model(tuned_model, "CatBoost")
        print(f"Best CatBoost Params: {study.best_params}")

    def save_and_plot_results(self):
        df_results = pd.DataFrame(self.results)
        
        # Save to CSV
        csv_path = self.output_dir / "hpo_comparison_results.csv"
        df_results.to_csv(csv_path, index=False)
        print(f"\nResults saved to {csv_path}")
        print("\nSummary of Results:")
        print(df_results.to_string(index=False))
        
        # Plotting
        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # R2 Plot (Higher is better)
        sns.barplot(x="Model", y="R2_Score", hue="Phase", data=df_results, ax=axes[0], palette="Set2")
        axes[0].set_title("R2 Score: Baseline vs Tuned (Higher is Better)")
        axes[0].set_ylim(0, 1.0) # Assuming R2 is between 0 and 1
        
        # RMSE Plot (Lower is better)
        sns.barplot(x="Model", y="RMSE", hue="Phase", data=df_results, ax=axes[1], palette="Set1")
        axes[1].set_title("RMSE: Baseline vs Tuned (Lower is Better)")
        
        plt.tight_layout()
        plot_path = self.output_dir / "hpo_comparison_plot.png"
        plt.savefig(plot_path, dpi=300)
        print(f"Plot saved to {plot_path}")

    def run_all(self):
        print("="*60)
        print(f"Starting Hyperparameter Optimization Benchmark ({self.n_trials} Trials)")
        print("="*60)
        self.load_data()
        
        # Run tuning for all models
        self.tune_ridge()
        self.tune_rf()
        self.tune_xgb()
        self.tune_lgbm()
        self.tune_catboost()
        
        # Save and plot
        self.save_and_plot_results()
        print("="*60)
        print("HPO Pipeline Completed Successfully")
        print("="*60)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run HPO benchmark")
    parser.add_argument("--test", action="store_true", help="Run in fast one-shot test mode")
    args = parser.parse_args()
    
    tuner = HPOTunerBenchmark(n_trials=10, test_mode=args.test)
    tuner.run_all()
