import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from datetime import datetime
import os
import warnings

# Suppress all warnings for cleaner console output.
warnings.filterwarnings("ignore")

class RFMTransformer(BaseEstimator, TransformerMixin):
    """Calculates RFM and temporal features."""
    def __init__(self, snapshot_date=None):
        self.snapshot_date = pd.to_datetime(snapshot_date) if snapshot_date else datetime.now()
        # Ensure snapshot date is timezone-naive for consistent calculations.
        if hasattr(self.snapshot_date, 'tz') and self.snapshot_date.tz is not None:
            self.snapshot_date = self.snapshot_date.tz_localize(None)
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        
        X_transformed['TransactionStartTime'] = pd.to_datetime(X_transformed['TransactionStartTime'])
        if hasattr(X_transformed['TransactionStartTime'].dtype, 'tz'):
            X_transformed['TransactionStartTime'] = X_transformed['TransactionStartTime'].dt.tz_localize(None)
        
        rfm = X_transformed.groupby('CustomerId').agg(
            Recency=('TransactionStartTime', lambda x: (self.snapshot_date - x.max()).days),
            Frequency=('TransactionId', 'count'),
            MonetarySum=('Value', 'sum'),
            MonetaryMean=('Value', 'mean'),
            MonetaryStd=('Value', 'std')
        )
        rfm['MonetaryStd'] = rfm['MonetaryStd'].fillna(0)
        
        X_transformed['TransactionHour'] = X_transformed['TransactionStartTime'].dt.hour
        X_transformed['TransactionDay'] = X_transformed['TransactionStartTime'].dt.day
        X_transformed['TransactionMonth'] = X_transformed['TransactionStartTime'].dt.month
        
        # Merge RFM features back using a left merge to keep all original transactions.
        return pd.merge(X_transformed, rfm, on='CustomerId', how='left')

class HighRiskLabelGenerator(BaseEstimator, TransformerMixin):
    """Generates a binary 'is_high_risk' label using KMeans clustering."""
    def __init__(self, n_clusters=2, random_state=42):
        self.n_clusters = n_clusters # Set to 2 (e.g., high vs. low risk)
        self.random_state = random_state
        self.kmeans = None
        self.scaler = StandardScaler()
        self.high_risk_cluster_ = None
        
    def fit(self, X, y=None):
        customer_rfm_for_clustering = X.groupby('CustomerId')[['Recency', 'Frequency', 'MonetarySum']].first().reset_index()
        X_scaled = self.scaler.fit_transform(customer_rfm_for_clustering[['Recency', 'Frequency', 'MonetarySum']])
        
        # Use n_init='auto' for KMeans to select the best initialization.
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init='auto')
        self.kmeans.fit(X_scaled)
        
        # Identify the high-risk cluster based on mean RFM values (high Recency, low Frequency/Monetary).
        cluster_centers_scaled = self.kmeans.cluster_centers_
        cluster_centers_original_scale = self.scaler.inverse_transform(cluster_centers_scaled)
        risk_order_df = pd.DataFrame(cluster_centers_original_scale, columns=['Recency', 'Frequency', 'MonetarySum'])
        
        self.high_risk_cluster_ = risk_order_df.sort_values(
            ['Recency', 'Frequency', 'MonetarySum'],
            ascending=[False, True, True]
        ).index[0]
        
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        
        customer_rfm_for_predict = X_transformed.groupby('CustomerId')[['Recency', 'Frequency', 'MonetarySum']].first().reset_index()
        X_scaled = self.scaler.transform(customer_rfm_for_predict[['Recency', 'Frequency', 'MonetarySum']])
        clusters_per_customer = self.kmeans.predict(X_scaled)
        
        customer_id_to_risk_label = dict(zip(
            customer_rfm_for_predict['CustomerId'],
            (clusters_per_customer == self.high_risk_cluster_).astype(int)
        ))
        
        X_transformed['is_high_risk'] = X_transformed['CustomerId'].map(customer_id_to_risk_label)
        
        return X_transformed

def build_feature_pipeline():
    """Builds the complete feature engineering pipeline."""
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    numerical_features = ['Recency', 'Frequency', 'MonetarySum', 'MonetaryMean', 'MonetaryStd']
    categorical_features = ['ProductCategory', 'ChannelId', 'PricingStrategy']
    temporal_features = ['TransactionHour', 'TransactionDay', 'TransactionMonth']
    
    # 'is_high_risk' is generated by HighRiskLabelGenerator and passed through for the final output.
    target_as_feature_column = ['is_high_risk'] 
    
    final_preprocessor = ColumnTransformer(
        [
            ('num', numerical_pipeline, numerical_features),
            ('cat', categorical_pipeline, categorical_features),
            ('temp', 'passthrough', temporal_features),
            ('target', 'passthrough', target_as_feature_column) 
        ],
        remainder='drop' # Explicitly drops columns not specified in transformers.
    )
    
    return Pipeline([
        ('rfm_features', RFMTransformer()),
        ('risk_labels', HighRiskLabelGenerator(n_clusters=2, random_state=42)),
        ('final_preprocessing', final_preprocessor)
    ])

def main():
    """Main execution function to process raw data and save model-ready data."""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        processed_dir = os.path.normpath(os.path.join(script_dir, '../data/processed'))
        os.makedirs(processed_dir, exist_ok=True)
        
        input_path = os.path.normpath(os.path.join(script_dir, '../data/raw/data.csv'))
        output_path = os.path.normpath(os.path.join(processed_dir, 'model_ready_data.csv'))
        
        if not os.path.exists(input_path):



            raise FileNotFoundError(f"Input file not found at {input_path}")
        data = pd.read_csv(input_path)
        
        print(f"Loaded raw data with {data.shape[0]} rows and {data.shape[1]} columns.")
        
        pipeline = build_feature_pipeline()
        processed_data_array = pipeline.fit_transform(data) 
        
        # Get column names from the final ColumnTransformer step to match the array output!!
        all_feature_names = pipeline.named_steps['final_preprocessing'].get_feature_names_out()
        
        processed_df = pd.DataFrame(processed_data_array, columns=all_feature_names)
        processed_df.to_csv(output_path, index=False)
        
        print(f"\nData successfully processed and saved to:\n{output_path}")
        print(f"Shape of processed data: {processed_df.shape}")



        print(f"Columns in output: {processed_df.columns.tolist()}")
        
        # Check and print distribution for the 'is_high_risk' column (prefixed by ColumnTransformer).
        actual_high_risk_col_name = 'target__is_high_risk' 
        if actual_high_risk_col_name in processed_df.columns:
            print(f"\nTarget distribution ({actual_high_risk_col_name}):")
            print(processed_df[actual_high_risk_col_name].value_counts())
        else:
            print(f"\nWarning: '{actual_high_risk_col_name}' column not found. Check pipeline config.")
            
    except FileNotFoundError as fnf_error:
        print(f"File Error: {fnf_error}")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    # Disable parallel processing warnings from joblib/sklearn.
    os.environ['LOKY_MAX_CPU_COUNT'] = '1'
    main()