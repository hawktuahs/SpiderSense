"""
Backtesting engine for SmartSense forecasting models.

Implements sliding window backtesting with comprehensive metrics
and visualization for model comparison and validation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any, Tuple
import logging
from pathlib import Path
import json
from datetime import datetime
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Backtesting engine for time series forecasting models."""
    
    def __init__(self, output_dir: str = "data/exports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set matplotlib backend for server environments
        plt.switch_backend('Agg')
        
        # Configure plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    async def run_backtest(
        self,
        series_data: Dict[str, Any],
        models: List[str],
        train_size: float = 0.8,
        horizon: int = 24,
        step_size: int = 12
    ) -> Dict[str, Any]:
        """
        Run sliding window backtesting across multiple models.
        
        Args:
            series_data: Series information with DataFrame
            models: List of model names to test
            train_size: Initial training data fraction
            horizon: Forecast horizon for each test
            step_size: Step size for sliding window
            
        Returns:
            Dictionary with backtest results
        """
        df = series_data["dataframe"]
        freq = series_data["frequency"]
        
        # Validate data size
        min_required = int(len(df) * train_size) + horizon * 3
        if len(df) < min_required:
            raise ValueError(f"Insufficient data for backtesting. Need at least {min_required} points.")
        
        # Initialize results storage
        all_results = []
        model_forecasts = {model: [] for model in models}
        
        # Calculate test windows
        initial_train_size = int(len(df) * train_size)
        test_starts = range(initial_train_size, len(df) - horizon, step_size)
        
        logger.info(f"Running backtest with {len(test_starts)} windows")
        
        # Import model classes
        from forecasting import NaiveSeasonalForecaster, ETSForecaster, ARIMAForecaster
        model_classes = {
            "naive_seasonal": NaiveSeasonalForecaster,
            "ets": ETSForecaster,
            "arima": ARIMAForecaster
        }
        
        # Run sliding window backtest
        for i, test_start in enumerate(test_starts):
            train_data = df.iloc[:test_start]
            test_data = df.iloc[test_start:test_start + horizon]
            
            if len(test_data) < horizon:
                continue
            
            window_results = {
                "window": i,
                "train_start": train_data.index[0],
                "train_end": train_data.index[-1],
                "test_start": test_data.index[0],
                "test_end": test_data.index[-1],
                "train_size": len(train_data),
                "test_size": len(test_data)
            }
            
            # Test each model
            for model_name in models:
                try:
                    # Initialize and fit model
                    model_class = model_classes[model_name]
                    model = model_class(random_seed=42)
                    
                    # Fit on training data
                    model.fit(train_data, freq=freq, target_col="value")
                    
                    # Generate forecast
                    forecast_result = model.predict(h=horizon)
                    forecasts = forecast_result.yhat
                    
                    # Calculate metrics
                    y_true = test_data["value"].values
                    y_pred = forecasts[:len(y_true)]
                    
                    metrics = self._calculate_metrics(y_true, y_pred)
                    
                    # Store results
                    window_results[f"{model_name}_metrics"] = metrics
                    window_results[f"{model_name}_forecasts"] = y_pred.tolist()
                    
                    # Store for aggregation
                    model_forecasts[model_name].append({
                        "y_true": y_true,
                        "y_pred": y_pred,
                        "timestamps": test_data.index.tolist()
                    })
                    
                except Exception as e:
                    logger.warning(f"Model {model_name} failed on window {i}: {e}")
                    window_results[f"{model_name}_metrics"] = None
                    window_results[f"{model_name}_forecasts"] = None
            
            all_results.append(window_results)
        
        # Aggregate results
        aggregated_metrics = self._aggregate_metrics(all_results, models)
        leaderboard = self._create_leaderboard(aggregated_metrics)
        best_model = leaderboard[0]["model"] if leaderboard else models[0]
        
        # Generate metadata
        metadata = {
            "series_id": series_data.get("series_id", "unknown"),
            "total_windows": len(all_results),
            "train_size": train_size,
            "horizon": horizon,
            "step_size": step_size,
            "frequency": freq,
            "backtest_timestamp": datetime.utcnow().isoformat()
        }
        
        return {
            "metrics": aggregated_metrics,
            "leaderboard": leaderboard,
            "best_model": best_model,
            "window_results": all_results,
            "model_forecasts": model_forecasts,
            "metadata": metadata
        }
    
    async def export_plots(
        self,
        backtest_results: Dict[str, Any],
        series_id: str
    ) -> List[str]:
        """Export visualization plots for backtest results."""
        plot_urls = []
        
        try:
            # 1. Model comparison plot
            comparison_path = await self._plot_model_comparison(
                backtest_results, series_id
            )
            plot_urls.append(str(comparison_path))
            
            # 2. Metrics heatmap
            heatmap_path = await self._plot_metrics_heatmap(
                backtest_results, series_id
            )
            plot_urls.append(str(heatmap_path))
            
            # 3. Forecast accuracy over time
            accuracy_path = await self._plot_accuracy_over_time(
                backtest_results, series_id
            )
            plot_urls.append(str(accuracy_path))
            
            # 4. Residual analysis
            residual_path = await self._plot_residual_analysis(
                backtest_results, series_id
            )
            plot_urls.append(str(residual_path))
            
        except Exception as e:
            logger.error(f"Plot export failed: {e}")
        
        return plot_urls
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive forecasting metrics."""
        # Handle edge cases
        if len(y_true) == 0 or len(y_pred) == 0:
            return {"mape": np.inf, "smape": np.inf, "mae": np.inf, "rmse": np.inf, "mase": np.inf}
        
        # Ensure same length
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        
        # Basic metrics
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        
        # MAPE (handle zero values)
        mape_mask = y_true != 0
        if np.sum(mape_mask) > 0:
            mape = np.mean(np.abs((y_true[mape_mask] - y_pred[mape_mask]) / y_true[mape_mask])) * 100
        else:
            mape = np.inf
        
        # sMAPE (symmetric MAPE)
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        smape_mask = denominator != 0
        if np.sum(smape_mask) > 0:
            smape = np.mean(np.abs(y_true[smape_mask] - y_pred[smape_mask]) / denominator[smape_mask]) * 100
        else:
            smape = np.inf
        
        # MASE (Mean Absolute Scaled Error)
        try:
            # Use naive seasonal forecast as baseline
            if len(y_true) > 24:  # Assume daily seasonality
                seasonal_lag = 24
                naive_forecast = y_true[:-seasonal_lag]
                naive_mae = np.mean(np.abs(y_true[seasonal_lag:] - naive_forecast))
                mase = mae / naive_mae if naive_mae > 0 else np.inf
            else:
                # Use naive forecast (previous value)
                naive_mae = np.mean(np.abs(np.diff(y_true)))
                mase = mae / naive_mae if naive_mae > 0 else np.inf
        except:
            mase = np.inf
        
        return {
            "mape": float(mape),
            "smape": float(smape),
            "mae": float(mae),
            "rmse": float(rmse),
            "mase": float(mase)
        }
    
    def _aggregate_metrics(
        self,
        window_results: List[Dict],
        models: List[str]
    ) -> List[Dict[str, Any]]:
        """Aggregate metrics across all backtest windows."""
        aggregated = []
        
        for model in models:
            model_metrics = []
            
            # Collect metrics from all windows
            for window in window_results:
                metrics = window.get(f"{model}_metrics")
                if metrics is not None:
                    model_metrics.append(metrics)
            
            if not model_metrics:
                # No valid results for this model
                aggregated.append({
                    "model": model,
                    "mape": np.inf,
                    "smape": np.inf,
                    "mae": np.inf,
                    "rmse": np.inf,
                    "mase": np.inf,
                    "n_forecasts": 0
                })
                continue
            
            # Calculate mean and std for each metric
            metrics_df = pd.DataFrame(model_metrics)
            
            # Handle infinite values
            for col in metrics_df.columns:
                metrics_df[col] = metrics_df[col].replace([np.inf, -np.inf], np.nan)
            
            agg_metrics = {
                "model": model,
                "mape": float(metrics_df["mape"].mean()),
                "smape": float(metrics_df["smape"].mean()),
                "mae": float(metrics_df["mae"].mean()),
                "rmse": float(metrics_df["rmse"].mean()),
                "mase": float(metrics_df["mase"].mean()),
                "n_forecasts": len(model_metrics)
            }
            
            # Add standard deviations
            agg_metrics.update({
                "mape_std": float(metrics_df["mape"].std()),
                "smape_std": float(metrics_df["smape"].std()),
                "mae_std": float(metrics_df["mae"].std()),
                "rmse_std": float(metrics_df["rmse"].std()),
                "mase_std": float(metrics_df["mase"].std())
            })
            
            aggregated.append(agg_metrics)
        
        return aggregated
    
    def _create_leaderboard(self, aggregated_metrics: List[Dict]) -> List[Dict[str, Any]]:
        """Create ranked leaderboard based on multiple metrics."""
        if not aggregated_metrics:
            return []
        
        # Create DataFrame for easier manipulation
        df = pd.DataFrame(aggregated_metrics)
        
        # Handle missing/infinite values
        for col in ["mape", "smape", "mae", "rmse", "mase"]:
            if col in df.columns:
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        
        # Calculate composite score (lower is better)
        # Normalize metrics to 0-1 scale and average
        score_components = []
        
        for metric in ["mape", "smape", "mae", "rmse"]:
            if metric in df.columns and not df[metric].isna().all():
                # Min-max normalization
                min_val = df[metric].min()
                max_val = df[metric].max()
                if max_val > min_val:
                    normalized = (df[metric] - min_val) / (max_val - min_val)
                    score_components.append(normalized)
        
        if score_components:
            df["composite_score"] = np.mean(score_components, axis=0)
        else:
            df["composite_score"] = 1.0  # Default score
        
        # Sort by composite score (lower is better)
        df = df.sort_values("composite_score", na_position="last")
        
        # Create leaderboard
        leaderboard = []
        for i, (_, row) in enumerate(df.iterrows()):
            entry = {
                "rank": i + 1,
                "model": row["model"],
                "composite_score": float(row["composite_score"]) if not pd.isna(row["composite_score"]) else 1.0,
                "mape": float(row["mape"]) if not pd.isna(row["mape"]) else np.inf,
                "smape": float(row["smape"]) if not pd.isna(row["smape"]) else np.inf,
                "mae": float(row["mae"]) if not pd.isna(row["mae"]) else np.inf,
                "rmse": float(row["rmse"]) if not pd.isna(row["rmse"]) else np.inf,
                "n_forecasts": int(row["n_forecasts"])
            }
            leaderboard.append(entry)
        
        return leaderboard
    
    async def _plot_model_comparison(
        self,
        results: Dict[str, Any],
        series_id: str
    ) -> Path:
        """Create model comparison visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"Model Comparison - Series {series_id}", fontsize=16)
        
        metrics_df = pd.DataFrame(results["metrics"])
        
        # 1. MAPE comparison
        axes[0, 0].bar(metrics_df["model"], metrics_df["mape"])
        axes[0, 0].set_title("MAPE by Model")
        axes[0, 0].set_ylabel("MAPE (%)")
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. RMSE comparison
        axes[0, 1].bar(metrics_df["model"], metrics_df["rmse"])
        axes[0, 1].set_title("RMSE by Model")
        axes[0, 1].set_ylabel("RMSE")
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Multiple metrics radar chart (simplified as line plot)
        metrics_to_plot = ["mape", "smape", "mae", "rmse"]
        for _, row in metrics_df.iterrows():
            values = [row[m] for m in metrics_to_plot if not pd.isna(row[m])]
            if values:
                axes[1, 0].plot(metrics_to_plot[:len(values)], values, marker='o', label=row["model"])
        axes[1, 0].set_title("Multi-Metric Comparison")
        axes[1, 0].legend()
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Leaderboard visualization
        leaderboard_df = pd.DataFrame(results["leaderboard"])
        axes[1, 1].barh(leaderboard_df["model"], leaderboard_df["composite_score"])
        axes[1, 1].set_title("Composite Score Ranking")
        axes[1, 1].set_xlabel("Composite Score (lower is better)")
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / f"model_comparison_{series_id}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    async def _plot_metrics_heatmap(
        self,
        results: Dict[str, Any],
        series_id: str
    ) -> Path:
        """Create metrics heatmap."""
        metrics_df = pd.DataFrame(results["metrics"])
        
        # Select numeric metrics for heatmap
        metric_cols = ["mape", "smape", "mae", "rmse", "mase"]
        heatmap_data = metrics_df.set_index("model")[metric_cols]
        
        # Normalize for better visualization
        heatmap_normalized = heatmap_data.div(heatmap_data.max(), axis=1)
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(
            heatmap_normalized.T,
            annot=True,
            cmap="RdYlBu_r",
            fmt=".3f",
            cbar_kws={"label": "Normalized Metric Value"}
        )
        plt.title(f"Metrics Heatmap - Series {series_id}")
        plt.ylabel("Metrics")
        plt.xlabel("Models")
        
        plot_path = self.output_dir / f"metrics_heatmap_{series_id}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    async def _plot_accuracy_over_time(
        self,
        results: Dict[str, Any],
        series_id: str
    ) -> Path:
        """Plot forecast accuracy over time."""
        window_results = results["window_results"]
        models = [m["model"] for m in results["metrics"]]
        
        plt.figure(figsize=(12, 8))
        
        for model in models:
            mape_values = []
            window_dates = []
            
            for window in window_results:
                metrics = window.get(f"{model}_metrics")
                if metrics:
                    mape_values.append(metrics["mape"])
                    window_dates.append(window["test_start"])
            
            if mape_values:
                plt.plot(window_dates, mape_values, marker='o', label=model, alpha=0.7)
        
        plt.title(f"Forecast Accuracy Over Time - Series {series_id}")
        plt.xlabel("Test Period Start")
        plt.ylabel("MAPE (%)")
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        plot_path = self.output_dir / f"accuracy_over_time_{series_id}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    async def _plot_residual_analysis(
        self,
        results: Dict[str, Any],
        series_id: str
    ) -> Path:
        """Create residual analysis plots."""
        model_forecasts = results["model_forecasts"]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"Residual Analysis - Series {series_id}", fontsize=16)
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(model_forecasts)))
        
        for i, (model, forecasts) in enumerate(model_forecasts.items()):
            if not forecasts:
                continue
            
            # Aggregate all residuals for this model
            all_residuals = []
            for forecast in forecasts:
                residuals = forecast["y_true"] - forecast["y_pred"]
                all_residuals.extend(residuals)
            
            if not all_residuals:
                continue
            
            color = colors[i % len(colors)]
            
            # Residual histogram
            axes[0, 0].hist(all_residuals, bins=20, alpha=0.6, label=model, color=color)
            
            # Q-Q plot (simplified as residual vs predicted)
            all_predicted = []
            for forecast in forecasts:
                all_predicted.extend(forecast["y_pred"])
            
            if len(all_residuals) == len(all_predicted):
                axes[0, 1].scatter(all_predicted, all_residuals, alpha=0.5, label=model, color=color)
        
        axes[0, 0].set_title("Residual Distribution")
        axes[0, 0].set_xlabel("Residuals")
        axes[0, 0].set_ylabel("Frequency")
        axes[0, 0].legend()
        
        axes[0, 1].set_title("Residuals vs Predicted")
        axes[0, 1].set_xlabel("Predicted Values")
        axes[0, 1].set_ylabel("Residuals")
        axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        axes[0, 1].legend()
        
        # Residual autocorrelation (simplified)
        for i, (model, forecasts) in enumerate(model_forecasts.items()):
            if forecasts:
                all_residuals = []
                for forecast in forecasts:
                    residuals = forecast["y_true"] - forecast["y_pred"]
                    all_residuals.extend(residuals)
                
                if len(all_residuals) > 10:
                    # Simple lag-1 autocorrelation
                    lag1_corr = np.corrcoef(all_residuals[:-1], all_residuals[1:])[0, 1]
                    axes[1, 0].bar(model, lag1_corr, color=colors[i % len(colors)])
        
        axes[1, 0].set_title("Lag-1 Autocorrelation of Residuals")
        axes[1, 0].set_ylabel("Correlation")
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Model performance summary
        metrics_summary = pd.DataFrame(results["metrics"])
        axes[1, 1].barh(metrics_summary["model"], metrics_summary["mape"])
        axes[1, 1].set_title("MAPE Summary")
        axes[1, 1].set_xlabel("MAPE (%)")
        
        plt.tight_layout()
        
        plot_path = self.output_dir / f"residual_analysis_{series_id}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
