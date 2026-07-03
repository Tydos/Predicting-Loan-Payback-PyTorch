"""
Feature importance analysis for credit risk scoring.
Analyzes correlations, missing values, variance, and distributions.

Run from project root:
    python notebooks/feature_analysis.py
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_feature_importance(df: pd.DataFrame, target_col: str = "loan_status") -> dict:
    results = {}

    logger.info("Performing correlation analysis...")
    numeric_df = df.select_dtypes(include=["int64", "float64"]).copy()

    if target_col in numeric_df.columns:
        correlations = numeric_df.corr()[target_col].sort_values(ascending=False)
        results["correlations"] = correlations
        logger.info("\nTop 15 features by correlation with %s:\n%s", target_col, correlations.head(15))
    else:
        logger.warning("Target column '%s' not found or not numeric", target_col)

    logger.info("\nPerforming missing value analysis...")
    missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    missing_pct_filtered = missing_pct[missing_pct > 0]
    results["missing_values"] = missing_pct_filtered
    logger.info("\nFeatures with missing values (%d):\n%s", len(missing_pct_filtered), missing_pct_filtered.head(15))

    logger.info("\nPerforming variance analysis...")
    variances = numeric_df.var().sort_values(ascending=False)
    zero_var_cols = variances[variances == 0]
    results["variances"] = variances
    results["zero_variance_cols"] = zero_var_cols
    logger.info("Zero variance columns: %s", list(zero_var_cols.index))
    logger.info("\nTop 15 features by variance:\n%s", variances.head(15))

    logger.info("\nPerforming categorical feature analysis...")
    categorical_df = df.select_dtypes(include=["object"]).copy()
    results["categorical_features"] = {}
    logger.info("Found %d categorical features", len(categorical_df.columns))
    for col in categorical_df.columns[:5]:
        unique_count = df[col].nunique()
        results["categorical_features"][col] = unique_count
        logger.info("\n%s: %d unique values\n%s", col, unique_count, df[col].value_counts().head())

    logger.info("\nPerforming skewness analysis...")
    skewness = numeric_df.skew().sort_values(ascending=False)
    results["skewness"] = skewness
    highly_skewed = skewness[abs(skewness) > 2]
    results["highly_skewed_cols"] = highly_skewed
    logger.info("Highly skewed features (|skewness| > 2): %d\n%s", len(highly_skewed), highly_skewed.head(10))

    logger.info("\nChecking for multicollinearity...")
    correlation_matrix = numeric_df.corr()
    high_corr_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i + 1, len(correlation_matrix.columns)):
            if abs(correlation_matrix.iloc[i, j]) > 0.9:
                high_corr_pairs.append((
                    correlation_matrix.columns[i],
                    correlation_matrix.columns[j],
                    correlation_matrix.iloc[i, j],
                ))
    results["multicollinearity"] = high_corr_pairs
    logger.info("Highly correlated pairs (|r| > 0.9): %d", len(high_corr_pairs))
    for col1, col2, corr in high_corr_pairs[:10]:
        logger.info("  %s <-> %s: %.3f", col1, col2, corr)

    logger.info("\nSummary statistics:\n%s", numeric_df.describe())
    return results


def get_feature_recommendations(results: dict) -> dict:
    recommendations = {
        "drop_features": [],
        "transform_features": [],
        "investigate_features": [],
    }

    if "zero_variance_cols" in results:
        recommendations["drop_features"].extend(results["zero_variance_cols"].index.tolist())

    if "missing_values" in results:
        high_missing = results["missing_values"][results["missing_values"] > 50]
        recommendations["drop_features"].extend(high_missing.index.tolist())

    if "highly_skewed_cols" in results:
        recommendations["transform_features"].extend(results["highly_skewed_cols"].index.tolist())

    if "multicollinearity" in results:
        investigated = set()
        for col1, col2, _ in results["multicollinearity"]:
            for col in (col1, col2):
                if col not in investigated:
                    recommendations["investigate_features"].append(col)
                    investigated.add(col)

    return recommendations


def print_recommendations(recommendations: dict) -> None:
    logger.info("\n%s\nFEATURE RECOMMENDATIONS\n%s", "=" * 60, "=" * 60)

    if recommendations["drop_features"]:
        logger.info("\n1. DROP FEATURES (%d):", len(recommendations["drop_features"]))
        for f in recommendations["drop_features"]:
            logger.info("   - %s", f)

    if recommendations["transform_features"]:
        logger.info("\n2. TRANSFORM FEATURES (%d):", len(recommendations["transform_features"]))
        for f in recommendations["transform_features"]:
            logger.info("   - %s", f)

    if recommendations["investigate_features"]:
        logger.info("\n3. INVESTIGATE FOR MULTICOLLINEARITY (%d):", len(recommendations["investigate_features"]))
        for f in recommendations["investigate_features"][:10]:
            logger.info("   - %s", f)


if __name__ == "__main__":
    from pipeline.etl import read_chunks
    from core.config import load_config

    config = load_config("core/config.yaml")
    logger.info("Loading dataset...")
    df = read_chunks(config.raw_dataset)
    logger.info("Dataset shape: %s  |  Columns: %d", df.shape, len(df.columns))

    results = analyze_feature_importance(df, target_col="loan_status")
    recommendations = get_feature_recommendations(results)
    print_recommendations(recommendations)
