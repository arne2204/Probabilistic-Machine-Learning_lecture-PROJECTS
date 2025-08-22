import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
import logging
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.feature_selection import mutual_info_classif, SelectKBest, f_classif
import scipy.stats as stats
import networkx as nx
from dateutil.parser import parse
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import LabelEncoder
import sklearn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

class EnhancedDataPipeline:
    """
    Enhanced data preprocessing pipeline specifically designed for ENSIN health data
    with integrated Bayesian network preparation
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._get_default_config()
        self.df_original = None
        self.df_processed = None
        self.encoders = {}
        self.feature_importance_scores = {}
        self.processing_log = []
        
        # Define mandatory columns for health analysis
        self.mandatory_cols = [
        'edad', 'sexo', 'estrato', 'x_region',
        'a0104', 'a0107', 'a0108', 'a0109',
        'a0202', 'a0203', 'a0204', 'a0205', 'a0206', 'a0207',
        'a0301', 'a0401', 'a0604',
        'a0303num', 'a0306e', 'a0406e', 'a0410a', 'a0410b', 'a0410c',
        'a0701p', 'a0702p', 'a0703p',
        'a1503', 'a1210'
        ]
    
    def _get_default_config(self) -> Dict:
        return {
        'outlier_method': 'iqr',
        'outlier_threshold': 1.5,
        'missing_threshold': 0.5,
        'cardinality_threshold': 0.9,
        'entropy_threshold': 0.05,
        'feature_importance_threshold': 0.01,
        'similarity_threshold': 0.6,
        'rare_category_threshold': 0.01,
        'normalization_method': 'minmax',
        'discretization_bins': 3,
        'random_state': 42,
        'data_folder': "./data"
        }
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load and perform initial data inspection"""
        logger.info(f"Loading data from {filepath}")
        
        try:
            self.df_original = pd.read_csv(filepath, sep=";")
            self.df_original = self.df_original.replace(r'^\s*$', np.nan, regex=True)
            logger.info(f"Data loaded successfully. Shape: {self.df_original.shape}")
            
            # Initial data quality report
            self._generate_data_quality_report()
            
            return self.df_original
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def _generate_data_quality_report(self) -> Dict:
        """Generate comprehensive data quality report"""
        if self.df_original is None:
            return {}
        
        df = self.df_original
        
        report = {
            'shape': df.shape,
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # MB
            'missing_values': df.isnull().sum().sum(),
            'missing_percentage': (df.isnull().sum().sum() / df.size) * 100,
            'duplicate_rows': df.duplicated().sum(),
            'data_types': df.dtypes.value_counts().to_dict(),
            'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(df.select_dtypes(include=['object', 'category']).columns),
            'high_cardinality_cols': [],
            'constant_cols': [],
            'mandatory_cols_present': []
        }
        
        # Identify problematic columns
        for col in df.columns:
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio > self.config['cardinality_threshold']:
                report['high_cardinality_cols'].append(col)
            if df[col].nunique() <= 1:
                report['constant_cols'].append(col)
        
        # Check mandatory columns
        report['mandatory_cols_present'] = [col for col in self.mandatory_cols if col in df.columns]
        report['mandatory_cols_missing'] = [col for col in self.mandatory_cols if col not in df.columns]
        
        logger.info("Data Quality Report:")
        logger.info(f"  Shape: {report['shape']}")
        logger.info(f"  Missing values: {report['missing_percentage']:.2f}%")
        logger.info(f"  Duplicate rows: {report['duplicate_rows']}")
        logger.info(f"  High cardinality columns: {len(report['high_cardinality_cols'])}")
        logger.info(f"  Constant columns: {len(report['constant_cols'])}")
        logger.info(f"  Mandatory columns present: {len(report['mandatory_cols_present'])}/{len(self.mandatory_cols)}")
        
        self.data_quality_report = report
        return report
    
    def handle_missing_values(self, df: pd.DataFrame, method: str = 'smart') -> pd.DataFrame:
        """
        Enhanced missing value handling with multiple strategies
        """
        logger.info("Handling missing values...")
        df = df.copy()
        
        if method == 'smart':
            # Different strategies for different variable types
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            
            for col in numeric_cols:
                if df[col].isnull().sum() > 0:
                    if col in self.mandatory_cols:
                        # Use median for mandatory numeric columns
                        df[col].fillna(df[col].median(), inplace=True)
                    else:
                        # Use mode for non-mandatory or create 'missing' category
                        if df[col].isnull().sum() / len(df) > 0.3:
                            df[f'{col}_missing'] = df[col].isnull().astype(int)
                        df[col].fillna(df[col].median(), inplace=True)
            
            for col in categorical_cols:
                if df[col].isnull().sum() > 0 or (df[col] == '').any():
                    # Convert to string
                    df[col] = df[col].astype(str)
                    
                    # Replace string 'nan' with 'missing'
                    df[col].replace('nan', 'missing', inplace=True)
                    
                    # Replace empty strings '' with 'NA'
                    df[col].replace(' ', 'missing', inplace=True)
                    df[col].replace('', 'missing', inplace=True)
                    
                    # Fill remaining NaNs with 'missing' as fallback
                    df[col].fillna('missing', inplace=True)
        
        elif method == 'drop_columns':
            # Drop columns with too many missing values
            threshold = self.config['missing_threshold']
            cols_to_drop = []
            for col in df.columns:
                missing_ratio = df[col].isnull().sum() / len(df)
                if missing_ratio > threshold and col not in self.mandatory_cols:
                    cols_to_drop.append(col)
            
            df = df.drop(columns=cols_to_drop)
            logger.info(f"Dropped {len(cols_to_drop)} columns with >{threshold*100}% missing values")
        
        elif method == 'simple':
            # Simple fill: numeric with mean, categorical with mode
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            
            for col in categorical_cols:
                mode_val = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else 'unknown'
                df[col].fillna(mode_val, inplace=True)
        
        logger.info(f"Missing values after handling: {df.isnull().sum().sum()}")
        self.processing_log.append(f"Missing value handling: {method}")
        
        return df
    
    def detect_and_handle_outliers(self, df: pd.DataFrame, method: str = None) -> pd.DataFrame:
        method = method or self.config['outlier_method']
        logger.info(f"Detecting and handling outliers using {method} method...")
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outliers_capped = 0
        
        
        if method == 'iqr':
            for col in numeric_cols:
                if col not in self.mandatory_cols:
                    Q1, Q3 = df[col].quantile([0.25, 0.75])
                    IQR = Q3 - Q1
                    lower, upper = Q1 - self.config['outlier_threshold']*IQR, Q3 + self.config['outlier_threshold']*IQR
                    mask = (df[col] < lower) | (df[col] > upper)
                    outliers_capped += mask.sum()
                    df[col] = df[col].clip(lower, upper)
        
        
        elif method == 'zscore':
            from scipy import stats
            for col in numeric_cols:
                if col not in self.mandatory_cols:
                    z = np.abs(stats.zscore(df[col].fillna(df[col].median())))
                    mask = z > 3
                    outliers_capped += mask.sum()
                    df[col] = np.where(mask, df[col].median(), df[col])
    
        
        logger.info(f"Capped {outliers_capped} outliers")
        self.processing_log.append(f"Outlier handling: {method}, capped: {outliers_capped}")
        return df

    def infer_and_convert_df(self, df: pd.DataFrame, min_samples=5) -> pd.DataFrame:
        def infer_column_type(series):
            s = series.dropna().replace('', np.nan).dropna()
            if len(s) < min_samples:
                return 'string'
            try:
                pd.to_numeric(s)
                return 'numeric'
            except:
                pass
                date_count = 0
            for val in s:
                try:
                    parse(str(val), fuzzy=False)
                    date_count += 1
                except:
                    continue
            if date_count / len(s) >= 0.8:
                return 'datetime_or_time'
            if s.nunique() / len(s) < 0.5:
                return 'categorical'
            return 'string'
        
        
        for col in df.columns:
            true_type = infer_column_type(df[col])
            if true_type == 'numeric':
                df[col] = pd.to_numeric(df[col], errors='coerce')
            elif true_type == 'datetime_or_time':
                # Instead of dropping, extract useful parts
                df[f'{col}_year'] = pd.to_datetime(df[col], errors='coerce').dt.year
                df[f'{col}_month'] = pd.to_datetime(df[col], errors='coerce').dt.month
                df = df.drop(columns=[col])
            elif true_type in ['categorical', 'string']:
                df[col] = df[col].astype(str).replace(['nan', ''], 'NA')
        return df
       
    
    def normalize_and_discretize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize numeric features and discretize for Bayesian networks
        """
        logger.info("Normalizing and discretizing features...")
        df = df.copy()
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Normalize numeric columns
        if self.config['normalization_method'] == 'minmax':
            for col in numeric_cols:
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val > min_val:  # Avoid division by zero
                    df[col] = (df[col] - min_val) / (max_val - min_val)
                    self.encoders[f'{col}_minmax'] = (min_val, max_val)
        
        elif self.config['normalization_method'] == 'standard':
            for col in numeric_cols:
                mean_val = df[col].mean()
                std_val = df[col].std()
                if std_val > 0:
                    df[col] = (df[col] - mean_val) / std_val
                    self.encoders[f'{col}_standard'] = (mean_val, std_val)
        
        # Discretize for Bayesian network compatibility
        n_bins = self.config['discretization_bins']
        for col in numeric_cols:
            try:
                # Use quantile-based discretization
                df[f'{col}_discrete'], bins = pd.qcut(
                    df[col], 
                    q=n_bins, 
                    labels=[f'low', f'medium', f'high'][:n_bins], 
                    retbins=True, 
                    duplicates='drop'
                )
                self.encoders[f'{col}_bins'] = bins
                
                # Keep both continuous and discrete versions
                # df = df.drop(columns=[col])  # Remove if you only want discrete
                
            except Exception as e:
                logger.warning(f"Could not discretize {col}: {e}")
        
        self.processing_log.append(f"Normalization: {self.config['normalization_method']}")
        self.processing_log.append(f"Discretization: {n_bins} bins")
        
        return df
    
    def advanced_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create domain-specific features for health data
        """
        logger.info("Performing advanced feature engineering...")
        df = df.copy()
        
        # Health risk composite scores
        if all(col in df.columns for col in ['a0301', 'a0401', 'a0604']):
            df['chronic_disease_count'] = (
                df['a0301'].astype(bool).astype(int) +  # Diabetes
                df['a0401'].astype(bool).astype(int) +  # Hypertension
                df['a0604'].astype(bool).astype(int)    # High cholesterol
            )
        
        # Mental health composite score
        mental_health_cols = [f'a020{i}' for i in range(2, 8) if f'a020{i}' in df.columns]
        if mental_health_cols:
            df['mental_health_score'] = df[mental_health_cols].sum(axis=1)
        
        # Weight management behavior composite
        weight_cols = ['a0107', 'a0108', 'a0109']
        weight_cols_available = [col for col in weight_cols if col in df.columns]
        if weight_cols_available:
            df['weight_management_activity'] = df[weight_cols_available].sum(axis=1)
        
        # Dietary quality score
        diet_cols = ['a0701p', 'a0702p', 'a0703p']
        diet_cols_available = [col for col in diet_cols if col in df.columns]
        if diet_cols_available:
            df['dietary_quality_score'] = df[diet_cols_available].mean(axis=1)
        
        # Age-related risk factors
        if 'edad' in df.columns:
            df['age_group'] = pd.cut(df['edad'], 
                                   bins=[0, 18, 35, 50, 65, 100], 
                                   labels=['child', 'young_adult', 'adult', 'middle_aged', 'elderly'])
        
        # Socioeconomic risk
        if 'estrato' in df.columns:
            df['ses_risk'] = df['estrato'].apply(lambda x: 'high_risk' if x <= 2 else 'low_risk')
        
        logger.info(f"Added {len([col for col in df.columns if col not in self.df_original.columns])} engineered features")
        self.processing_log.append("Advanced feature engineering completed")
        
        return df
    
    def intelligent_feature_selection(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Performing intelligent feature selection...")
        df = df.copy()
    
        # Remove constant/near-constant features
        constant_cols = [col for col in df.columns if col not in self.mandatory_cols and df[col].nunique()/len(df) < 0.01]
        df = df.drop(columns=constant_cols)
    
        # Remove high-cardinality columns
        high_card_cols = []
        for col in df.select_dtypes(include=['object', 'category']).columns:
            if col not in self.mandatory_cols:
                if df[col].nunique() > 50 and df[col].nunique()/len(df) > self.config['cardinality_threshold']:
                    high_card_cols.append(col)
        df = df.drop(columns=high_card_cols)
    
        # Remove highly correlated features
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr = df[numeric_cols].corr().abs()
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
            df = df.drop(columns=to_drop)
    
        # Mutual information analysis
        for target in [c for c in ['a1503', 'a1210', 'a0301'] if c in df.columns]:
            features = [c for c in df.columns if c != target]
            X, y = df[features].copy(), df[target].copy()
    
            # Convert all non-numeric cols to numeric using LabelEncoder
            for col in X.columns:
                if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
    
            try:
                if y.dtype.kind in 'biu':
                    scores = mutual_info_classif(X, y, random_state=self.config['random_state'])
                else:
                    scores = mutual_info_regression(X, y, random_state=self.config['random_state'])
                self.feature_importance_scores[target] = pd.Series(scores, index=features).sort_values(ascending=False)
            except Exception as e:
                logger.warning(f"MI failed for target {target}: {e}")
    
        return df
        
        
    
    def _perform_mutual_info_selection(self, df: pd.DataFrame, targets: List[str]) -> None:
        """
        Perform mutual information-based feature selection
        """
        logger.info("Computing mutual information scores...")
        
        # Prepare encoded dataset
        df_encoded = df.copy()
        categorical_cols = df_encoded.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            if col not in targets:
                le = LabelEncoder()
                try:
                    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                    self.encoders[f'{col}_label'] = le
                except:
                    # If encoding fails, drop the column
                    df_encoded = df_encoded.drop(columns=[col])
        
        # Calculate mutual information for each target
        for target in targets:
            if target in df_encoded.columns:
                try:
                    features = [col for col in df_encoded.columns if col != target]
                    X = df_encoded[features]
                    y = df_encoded[target]
                    
                    # Handle target encoding
                    if y.dtype == 'object' or y.dtype.name == 'category':
                        le_target = LabelEncoder()
                        y = le_target.fit_transform(y.astype(str))
                        self.encoders[f'{target}_label'] = le_target
                    
                    mi_scores = mutual_info_classif(X, y, random_state=self.config['random_state'])
                    mi_scores = pd.Series(mi_scores, index=features).sort_values(ascending=False)
                    
                    self.feature_importance_scores[target] = mi_scores
                    logger.info(f"Top 5 features for {target}:")
                    for feature, score in mi_scores.head().items():
                        logger.info(f"  {feature}: {score:.4f}")
                        
                except Exception as e:
                    logger.warning(f"Could not compute mutual information for {target}: {e}")
    
    def prepare_for_bayesian_network(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Final preprocessing specifically for Bayesian network compatibility
        """
        logger.info("Preparing data for Bayesian network...")
        df = df.copy()
        
        # Ensure all variables are categorical or discrete
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                if df[col].nunique() > 10:  # Too many unique values
                    try:
                        # Discretize into meaningful categories
                        if col.endswith('_discrete'):
                            continue  # Already discretized
                        else:
                            df[col] = pd.qcut(df[col], q=3, labels=['low', 'medium', 'high'], duplicates='drop')
                    except:
                        # If qcut fails, use cut
                        try:
                            df[col] = pd.cut(df[col], bins=3, labels=['low', 'medium', 'high'])
                        except:
                            logger.warning(f"Could not discretize {col}")
                
                # Convert to categorical
                df[col] = df[col].astype('category')
            
            elif df[col].dtype == 'object':
                df[col] = df[col].astype('category')
        
        # Ensure no missing values
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if df[col].dtype.name == 'category':
                    df[col] = df[col].cat.add_categories(['missing'])
                    df[col] = df[col].fillna('missing')
                else:
                    df[col] = df[col].fillna('missing')
        
        # Final validation
        logger.info(f"Final dataset shape: {df.shape}")
        logger.info(f"Data types: {df.dtypes.value_counts().to_dict()}")
        logger.info(f"Missing values: {df.isnull().sum().sum()}")
        
        return df
    
    def run_complete_pipeline(self, filepath: str, save_intermediate: bool = True) -> pd.DataFrame:
        """
        Run the complete data preprocessing pipeline
        """
        data_folder = "~/projects/Probabilistic-Machine-Learning_lecture-PROJECTS/projects/05-2F_XXXX_health_surveys_mx/project/data"
        
        logger.info("=" * 60)
        logger.info("STARTING ENHANCED ENSIN DATA PREPROCESSING PIPELINE")
        logger.info("=" * 60)
        
        # Step 1: Load data
        df = self.load_data(filepath)
        df = self.infer_and_convert_df(df)
        if save_intermediate:
            df.to_csv(f'{data_folder}/01_raw_data.csv', index=False)
        
        # Step 2: Handle missing values
        df = self.handle_missing_values(df, method='smart')
        if save_intermediate:
            df.to_csv(f'{data_folder}/02_missing_handled.csv', index=False)
        
        # Step 3: Detect and handle outliers
        df = self.detect_and_handle_outliers(df)
        if save_intermediate:
            df.to_csv(f'{data_folder}/03_outliers_handled.csv', index=False)
        
        # Step 4: Advanced feature engineering
        df = self.advanced_feature_engineering(df)
        if save_intermediate:
            df.to_csv(f'{data_folder}/04_features_engineered.csv', index=False)
        
        # Step 5: Normalize and discretize
        df = self.normalize_and_discretize(df)
        if save_intermediate:
            df.to_csv(f'{data_folder}/05_normalized_discretized.csv', index=False)
        
        # Step 6: Feature selection
        df = self.intelligent_feature_selection(df)
        if save_intermediate:
            df.to_csv(f'{data_folder}/06_features_selected.csv', index=False)
        
        # Step 7: Prepare for Bayesian network
        df = self.prepare_for_bayesian_network(df)
        
        # Final save
        df.to_csv(f'{data_folder}/cleaned_dataset.csv', index=False)
        self.df_processed = df
        
        # Generate final report
        self._generate_processing_report()
        
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        
        return df
    
    def _generate_processing_report(self) -> Dict:
        """
        Generate comprehensive processing report
        """
        if self.df_processed is None or self.df_original is None:
            return {}
        
        report = {
            'original_shape': self.df_original.shape,
            'final_shape': self.df_processed.shape,
            'features_retained': self.df_processed.shape[1] / self.df_original.shape[1],
            'rows_retained': self.df_processed.shape[0] / self.df_original.shape[0],
            'processing_steps': self.processing_log,
            'encoders_created': len(self.encoders),
            'mandatory_cols_in_final': [col for col in self.mandatory_cols if col in self.df_processed.columns],
            'data_quality_improvement': {
                'missing_values_before': self.data_quality_report.get('missing_values', 0),
                'missing_values_after': self.df_processed.isnull().sum().sum(),
                'duplicates_before': self.data_quality_report.get('duplicate_rows', 0),
                'duplicates_after': self.df_processed.duplicated().sum()
            }
        }
        
        logger.info("\nFINAL PROCESSING REPORT:")
        logger.info(f"Shape change: {report['original_shape']} → {report['final_shape']}")
        logger.info(f"Features retained: {report['features_retained']:.2%}")
        logger.info(f"Rows retained: {report['rows_retained']:.2%}")
        logger.info(f"Encoders created: {report['encoders_created']}")
        logger.info(f"Missing values eliminated: {report['data_quality_improvement']['missing_values_before']} → {report['data_quality_improvement']['missing_values_after']}")
        
        # Save report
        import json
        with open('processing_report.json', 'w') as f:
            json.dump(report, f, indent=4, default=str)
        
        self.processing_report = report
        return report

    def get_bayesian_network_structure(self) -> Dict[str, List[str]]:
        """
        Suggest Bayesian network structure based on processed data
        """
        if self.df_processed is None:
            logger.error("Data not processed yet. Run pipeline first.")
            return {}
        
        # Enhanced layer structure based on available columns
        structure = {
            'demographics': [],
            'socioeconomic': [],
            'body_perception': [],
            'mental_health': [],
            'chronic_diseases': [],
            'dietary_patterns': [],
            'nutritional_status': [],
            'health_outcomes': [],
            'engineered_features': []
        }
        
        # Categorize available columns
        for col in self.df_processed.columns:
            if 'edad' in col or 'sexo' in col:
                structure['demographics'].append(col)
            elif 'estrato' in col or 'region' in col:
                structure['socioeconomic'].append(col)
            elif col.startswith('a010'):
                structure['body_perception'].append(col)
            elif col.startswith('a020'):
                structure['mental_health'].append(col)
            elif col in ['a0301', 'a0401', 'a0604']:
                structure['chronic_diseases'].append(col)
            elif col.startswith('a070'):
                structure['dietary_patterns'].append(col)
            elif col == 'a1503':
                structure['nutritional_status'].append(col)
            elif col in ['a1210', 'a1301']:
                structure['health_outcomes'].append(col)
            elif any(x in col for x in ['score', 'count', 'group', 'risk']):
                structure['engineered_features'].append(col)
        
        return structure