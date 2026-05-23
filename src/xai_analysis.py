import pandas as pd
import numpy as np
import os
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path

class XAIAnalyzer:
    def __init__(self, model_path: str = None, data_dir: str = None, output_dir: str = None):
        base_dir = Path(__file__).parent.parent
        self.model_path = Path(model_path) if model_path else base_dir / "results" / "saved_models" / "lightgbm.pkl"
        self.data_dir = Path(data_dir) if data_dir else base_dir / "dataset" / "model_input"
        self.output_dir = Path(output_dir) if output_dir else base_dir / "results" / "xai_plots"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_data_and_model(self):
        print(f"Loading model from {self.model_path}...")
        self.model = joblib.load(self.model_path)
        
        print(f"Loading test data from {self.data_dir}...")
        self.X_test = pd.read_csv(self.data_dir / "X_test_scaled.csv")
        
        # In case of missing values, impute them since SHAP might complain about NaNs depending on the model, 
        # though LightGBM handles them natively. It's safer to use the exact same data seen during eval.
        from sklearn.impute import SimpleImputer
        X_train_df = pd.read_csv(self.data_dir / "X_train_scaled.csv")
        imputer = SimpleImputer(strategy='median')
        imputer.fit(X_train_df)
        self.X_test_imputed = pd.DataFrame(imputer.transform(self.X_test), columns=self.X_test.columns)
        
        print(f"Data loaded. X_test shape: {self.X_test_imputed.shape}")

    def generate_explanations(self):
        print("Calculating SHAP values with TreeExplainer...")
        self.explainer = shap.TreeExplainer(self.model)
        self.shap_values = self.explainer(self.X_test_imputed)
        print("SHAP values calculated successfully.")

    def save_plots(self):
        print("Generating and saving SHAP plots...")
        
        # 1. Summary Plot (Beeswarm)
        plt.figure(figsize=(10, 6))
        shap.summary_plot(self.shap_values, self.X_test_imputed, show=False)
        plt.tight_layout()
        summary_path = self.output_dir / "shap_summary_plot.png"
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Summary plot saved to {summary_path}")

        # 2. Feature Importance (Bar Plot)
        plt.figure(figsize=(10, 6))
        shap.plots.bar(self.shap_values, show=False)
        plt.tight_layout()
        bar_path = self.output_dir / "shap_feature_importance.png"
        plt.savefig(bar_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Feature importance plot saved to {bar_path}")

        # 3. Local Explanation (Waterfall Plot for the first instance)
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(self.shap_values[0], show=False)
        plt.tight_layout()
        waterfall_path = self.output_dir / "shap_waterfall_plot_instance_0.png"
        plt.savefig(waterfall_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Waterfall plot saved to {waterfall_path}")

    def run_pipeline(self):
        print("="*50)
        print("Starting XAI Analysis Pipeline")
        print("="*50)
        self.load_data_and_model()
        self.generate_explanations()
        self.save_plots()
        print("="*50)
        print("XAI Pipeline Completed Successfully")
        print("="*50)

if __name__ == "__main__":
    analyzer = XAIAnalyzer()
    analyzer.run_pipeline()
