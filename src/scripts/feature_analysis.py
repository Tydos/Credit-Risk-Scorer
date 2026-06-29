"""
Feature importance analysis for credit risk scoring.
Analyzes correlations, missing values, variance, and distributions.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import logging
from scipy.stats import spearmanr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_feature_importance(df: pd.DataFrame, target_col: str = 'loan_status') -> dict:
    """
    Comprehensive feature importance analysis for credit risk scoring.

    Args:
        df: DataFrame to analyze
        target_col: Target column name for correlation analysis

    Returns:
        Dictionary containing analysis results
    """
    results = {}

    # 1. CORRELATION ANALYSIS (for numeric features)
    logger.info("Performing correlation analysis...")
    numeric_df = df.select_dtypes(include=['int64', 'float64']).copy()

    if target_col in numeric_df.columns:
        correlations = numeric_df.corr()[target_col].sort_values(ascending=False)
        results['correlations'] = correlations
        logger.info(f"\nTop 15 features by correlation with {target_col}:")
        logger.info(correlations.head(15))
    else:
        logger.warning(f"Target column '{target_col}' not found or not numeric")

    # 2. MISSING VALUE ANALYSIS
    logger.info("\nPerforming missing value analysis...")
    missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    missing_pct_filtered = missing_pct[missing_pct > 0]
    results['missing_values'] = missing_pct_filtered
    logger.info(f"\nFeatures with missing values ({len(missing_pct_filtered)} features):")
    logger.info(missing_pct_filtered.head(15))

    # 3. VARIANCE ANALYSIS (low variance = low predictive power)
    logger.info("\nPerforming variance analysis...")
    variances = numeric_df.var().sort_values(ascending=False)
    zero_var_cols = variances[variances == 0]
    results['variances'] = variances
    results['zero_variance_cols'] = zero_var_cols
    logger.info(f"\nZero variance columns: {list(zero_var_cols.index)}")
    logger.info(f"\nTop 15 features by variance:")
    logger.info(variances.head(15))

    # 4. CATEGORICAL FEATURE ANALYSIS
    logger.info("\nPerforming categorical feature analysis...")
    categorical_df = df.select_dtypes(include=['object']).copy()
    results['categorical_features'] = {}

    logger.info(f"Found {len(categorical_df.columns)} categorical features")
    for col in categorical_df.columns[:5]:
        unique_count = df[col].nunique()
        results['categorical_features'][col] = unique_count
        logger.info(f"\n{col}: {unique_count} unique values")
        logger.info(df[col].value_counts().head())

    # 5. FEATURE DISTRIBUTION (skewness)
    logger.info("\nPerforming skewness analysis...")
    skewness = numeric_df.skew().sort_values(ascending=False)
    results['skewness'] = skewness
    highly_skewed = skewness[abs(skewness) > 2]
    results['highly_skewed_cols'] = highly_skewed
    logger.info(f"\nHighly skewed features (|skewness| > 2): {len(highly_skewed)}")
    logger.info(highly_skewed.head(10))

    # 6. MULTICOLLINEARITY CHECK
    logger.info("\nChecking for multicollinearity...")
    correlation_matrix = numeric_df.corr()
    high_corr_pairs = []

    for i in range(len(correlation_matrix.columns)):
        for j in range(i + 1, len(correlation_matrix.columns)):
            if abs(correlation_matrix.iloc[i, j]) > 0.9:
                high_corr_pairs.append(
                    (
                        correlation_matrix.columns[i],
                        correlation_matrix.columns[j],
                        correlation_matrix.iloc[i, j],
                    )
                )

    results['multicollinearity'] = high_corr_pairs
    logger.info(f"\nHighly correlated feature pairs (|r| > 0.9): {len(high_corr_pairs)}")
    for col1, col2, corr in high_corr_pairs[:10]:
        logger.info(f"  {col1} <-> {col2}: {corr:.3f}")

    # 7. SUMMARY STATISTICS
    logger.info("\nSummary statistics for numeric features:")
    logger.info(numeric_df.describe())

    return results


def get_feature_recommendations(results: dict) -> dict:
    """
    Generate recommendations based on feature analysis.

    Args:
        results: Dictionary from analyze_feature_importance

    Returns:
        Recommendations dictionary
    """
    recommendations = {
        'drop_features': [],
        'transform_features': [],
        'investigate_features': [],
    }

    # Drop zero variance features
    if 'zero_variance_cols' in results:
        recommendations['drop_features'].extend(results['zero_variance_cols'].index.tolist())

    # Drop features with >50% missing
    if 'missing_values' in results:
        high_missing = results['missing_values'][results['missing_values'] > 50]
        recommendations['drop_features'].extend(high_missing.index.tolist())

    # Features to transform (highly skewed)
    if 'highly_skewed_cols' in results:
        recommendations['transform_features'].extend(
            results['highly_skewed_cols'].index.tolist()
        )

    # Features to investigate (high multicollinearity)
    if 'multicollinearity' in results:
        investigated = set()
        for col1, col2, _ in results['multicollinearity']:
            if col1 not in investigated:
                recommendations['investigate_features'].append(col1)
                investigated.add(col1)
            if col2 not in investigated:
                recommendations['investigate_features'].append(col2)
                investigated.add(col2)

    return recommendations


def print_recommendations(recommendations: dict) -> None:
    """Print feature recommendations in readable format."""
    logger.info("\n" + "=" * 60)
    logger.info("FEATURE RECOMMENDATIONS")
    logger.info("=" * 60)

    if recommendations['drop_features']:
        logger.info(f"\n1. DROP FEATURES ({len(recommendations['drop_features'])}):")
        for feature in recommendations['drop_features']:
            logger.info(f"   - {feature}")

    if recommendations['transform_features']:
        logger.info(f"\n2. TRANSFORM FEATURES ({len(recommendations['transform_features'])}):")
        logger.info("   (Apply log/sqrt transformation for skewed features)")
        for feature in recommendations['transform_features']:
            logger.info(f"   - {feature}")

    if recommendations['investigate_features']:
        logger.info(f"\n3. INVESTIGATE FOR MULTICOLLINEARITY ({len(recommendations['investigate_features'])}):")
        logger.info("   (Consider removing one from highly correlated pairs)")
        for feature in recommendations['investigate_features'][:10]:
            logger.info(f"   - {feature}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from lending_club_etl import read_chunks

    file_path = Path("/home/prsdj/projects/Credit-Risk-Scorer/dataset") / "accepted_2007_to_2018Q4.csv"

    logger.info("Loading dataset...")
    df = read_chunks(file_path, nrows=100000)

    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Columns: {len(df.columns)}")

    results = analyze_feature_importance(df, target_col='loan_status')
    recommendations = get_feature_recommendations(results)
    print_recommendations(recommendations)
