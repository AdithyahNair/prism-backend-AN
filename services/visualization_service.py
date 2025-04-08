import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime

class VisualizationService:
    def __init__(self):
        self.output_dir = Path("outputs/visualizations")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def plot_performance_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None,
        plots_dir: Optional[Path] = None
    ) -> Dict[str, str]:
        """Generate performance visualization plots"""
        if plots_dir is None:
            plots_dir = self.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
        plots_dir.mkdir(parents=True, exist_ok=True)

        plots = {}

        # Confusion Matrix
        plt.figure(figsize=(8, 6))
        cm = np.zeros((2, 2))
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1
        sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plots['confusion_matrix'] = str(plots_dir / 'confusion_matrix.png')
        plt.savefig(plots['confusion_matrix'])
        plt.close()

        # ROC Curve
        if y_pred_proba is not None:
            plt.figure(figsize=(8, 6))
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
            plots['roc_curve'] = str(plots_dir / 'roc_curve.png')
            plt.savefig(plots['roc_curve'])
            plt.close()

        return plots

    async def plot_regression_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        plots_dir: Optional[Path] = None
    ) -> Dict[str, str]:
        """Generate regression visualization plots"""
        if plots_dir is None:
            plots_dir = self.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
        plots_dir.mkdir(parents=True, exist_ok=True)

        plots = {}

        # Scatter Plot
        plt.figure(figsize=(8, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title('Predicted vs True Values')
        plots['scatter_plot'] = str(plots_dir / 'scatter_plot.png')
        plt.savefig(plots['scatter_plot'])
        plt.close()

        # Residual Plot
        plt.figure(figsize=(8, 6))
        residuals = y_true - y_pred
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        plots['residual_plot'] = str(plots_dir / 'residual_plot.png')
        plt.savefig(plots['residual_plot'])
        plt.close()

        return plots

    async def plot_fairness_metrics(
        self,
        metrics: Dict[str, Any],
        plots_dir: Optional[Path] = None
    ) -> Dict[str, str]:
        """Generate fairness visualization plots"""
        if plots_dir is None:
            plots_dir = self.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
        plots_dir.mkdir(parents=True, exist_ok=True)

        plots = {}

        # Group Performance Comparison
        plt.figure(figsize=(10, 6))
        groups = []
        accuracies = []
        for key, value in metrics.items():
            if key.endswith('_accuracy'):
                groups.append(key.replace('_accuracy', ''))
                accuracies.append(value)
        
        plt.bar(groups, accuracies)
        plt.xlabel('Groups')
        plt.ylabel('Accuracy')
        plt.title('Accuracy by Group')
        plt.xticks(rotation=45)
        plots['group_performance'] = str(plots_dir / 'group_performance.png')
        plt.savefig(plots['group_performance'])
        plt.close()

        return plots

    async def plot_drift_metrics(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        plots_dir: Optional[Path] = None
    ) -> Dict[str, str]:
        """Generate drift visualization plots"""
        if plots_dir is None:
            plots_dir = self.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
        plots_dir.mkdir(parents=True, exist_ok=True)

        plots = {}

        # Feature Distribution Comparison
        for i in range(X_train.shape[1]):
            plt.figure(figsize=(8, 6))
            plt.hist(X_train[:, i], alpha=0.5, label='Training', bins=30)
            plt.hist(X_test[:, i], alpha=0.5, label='Test', bins=30)
            plt.xlabel(f'Feature {i}')
            plt.ylabel('Count')
            plt.title(f'Distribution Comparison - Feature {i}')
            plt.legend()
            plots[f'feature_{i}_distribution'] = str(plots_dir / f'feature_{i}_distribution.png')
            plt.savefig(plots[f'feature_{i}_distribution'])
            plt.close()

        return plots

    async def plot_explainability_metrics(
        self,
        metrics: Dict[str, Any],
        plots_dir: Optional[Path] = None
    ) -> Dict[str, str]:
        """Generate explainability visualization plots"""
        if plots_dir is None:
            plots_dir = self.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
        plots_dir.mkdir(parents=True, exist_ok=True)

        plots = {}

        # Feature Importance Plot
        if 'shap_values' in metrics:
            plt.figure(figsize=(10, 6))
            shap_values = np.array(metrics['shap_values'])
            feature_importance = np.abs(shap_values).mean(axis=0)
            plt.bar(range(len(feature_importance)), feature_importance)
            plt.xlabel('Features')
            plt.ylabel('Mean |SHAP Value|')
            plt.title('Feature Importance (SHAP)')
            plots['feature_importance'] = str(plots_dir / 'feature_importance.png')
            plt.savefig(plots['feature_importance'])
            plt.close()

        return plots 