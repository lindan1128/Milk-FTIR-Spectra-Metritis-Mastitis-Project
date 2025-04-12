import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score
from scipy.signal import savgol_filter
import os
from tqdm import tqdm
import warnings
import scipy.stats as stats
warnings.filterwarnings('ignore')

def process_data(df):
    """Preprocess data, including NA interpolation by dim"""
    print(f"Initial data shape: {df.shape}")
    
    # Check NA situation
    columns_to_check = ['milkweightlbs', 'cells', 'parity']
    na_counts = df[columns_to_check].isna().sum()
    print("\nNA counts in relevant columns:")
    print(na_counts[na_counts > 0])
    
    # Interpolate milkweightlbs and cells by dim
    for col in ['milkweightlbs', 'cells']:
        # Check if the column has NA values
        if df[col].isna().any():
            print(f"\nInterpolating {col} by dim...")
            # Interpolate by dim group
            df[col] = df.groupby('dim')[col].transform(lambda x: x.interpolate(method='linear'))
            
            # If there are still NAs (e.g., at the beginning or end of a dim), fill with the mean of that dim
            if df[col].isna().any():
                df[col] = df.groupby('dim')[col].transform(lambda x: x.fillna(x.mean()))
                
            # Check if there are still NAs
            remaining_na = df[col].isna().sum()
            if remaining_na > 0:
                print(f"Warning: {remaining_na} NA values remain in {col} after interpolation")
                # If there are still NAs, fill with the overall mean
                df[col] = df[col].fillna(df[col].mean())
    
    # Process parity column as before (remove NAs)
    df = df.dropna(subset=['parity'])
    print(f"\nShape after handling NA: {df.shape}")
    
    # Recategorize parity
    df['parity'] = df['parity'].apply(lambda x: '2+' if x > 2 else str(x))
    
    # Check final data
    print(f"\nFinal data shape: {df.shape}")
    print("\nFinal NA counts:")
    print(df[columns_to_check].isna().sum())
    
    return df

def get_spectral_data(df, type='original'):
    """Get spectral data
    type: 'original', 'derivative', or 'rmR4'
    """
    # Get spectral columns (assuming non-spectral columns are known)
    non_spectral_cols = ['disease_in', 'disease', 'day_group', 'milkweightlbs', 
                        'cells', 'parity', 'Unnamed: 0', 'index']  # Add additional non-spectral columns
    
    # Get all numeric column names (spectral wavelengths)
    spectral_cols = [col for col in df.columns if col not in non_spectral_cols]
    
    # Ensure all column names can be converted to float
    spectral_cols = [col for col in spectral_cols if col.replace('.', '').isdigit()]
    
    # Convert column names to numeric values for range selection
    wavelengths = [float(col) for col in spectral_cols]
    
    # Select columns in the 1000-3000 range, excluding 1580-1700 and 1800-2800
    valid_cols = [col for col, wave in zip(spectral_cols, wavelengths)
                 if 1000 <= wave <= 3000 and not (1580 <= wave <= 1700) and not (1800 <= wave <= 2800)]
    
    if type == 'original':
        return df[valid_cols]
    elif type == 'derivative':
        # Calculate first derivative
        spectra = df[valid_cols].values
        derivatives = savgol_filter(spectra, window_length=7, polyorder=2, deriv=1, axis=1)
        return pd.DataFrame(derivatives, columns=valid_cols, index=df.index)
    elif type == 'rmR4':
        # Remove 1800-2800 region from original data
        rmR4_cols = [col for col, wave in zip(valid_cols, map(float, valid_cols))
                    if wave < 1800 or wave > 2800]
        return df[rmR4_cols]
    else:
        raise ValueError(f"Unknown spectral type: {type}")

def calculate_derivatives(spectra):
    """Calculate first derivative"""
    return pd.DataFrame(
        savgol_filter(spectra, window_length=7, polyorder=2, deriv=1),
        columns=spectra.columns,
        index=spectra.index
    )

def prepare_features(df, spectral_data, feature_type='spc'):
    """Prepare feature data"""
    
    # Convert to numpy array
    spectral_array = spectral_data.values
    
    # Prepare data based on feature_type
    if feature_type == 'spc':
        X = spectral_array
    else:
        # Prepare individual variables
        scaler = StandardScaler()
        encoder = OneHotEncoder(sparse=False)
        
        features = []
        if 'my' in feature_type:
            milk_weight = scaler.fit_transform(df[['milkweightlbs']])
            features.append(milk_weight)
        if 'scc' in feature_type:
            scc = scaler.fit_transform(df[['cells']])
            features.append(scc)
        if 'parity' in feature_type:
            parity = encoder.fit_transform(df[['parity']])
            features.append(parity)
            
        # Combine features
        X = np.hstack([spectral_array] + features)
    
    return X

def calculate_metrics(y_true, y_pred_proba):
    """Calculate performance metrics"""
    # Check if input is empty
    if len(y_true) == 0 or len(y_pred_proba) == 0:
        print("Warning: Empty input for metrics calculation")
        return {
            'auc': np.nan,
            'acc': np.nan,
            'sen': np.nan,
            'spc': np.nan
        }
    
    try:
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Check if there is only one class
        unique_classes = np.unique(y_true)
        if len(unique_classes) == 1:
            print(f"Warning: Only one class present ({unique_classes[0]}). Skipping metrics calculation.")
            return {
                'auc': np.nan,
                'acc': np.nan,
                'sen': np.nan,
                'spc': np.nan
            }
        
        return {
            'auc': roc_auc_score(y_true, y_pred_proba),
            'acc': accuracy_score(y_true, y_pred),
            'sen': recall_score(y_true, y_pred),
            'spc': recall_score(y_true, y_pred, pos_label=0)
        }
    except Exception as e:
        print(f"Warning: Error calculating metrics: {str(e)}")
        return {
            'auc': np.nan,
            'acc': np.nan,
            'sen': np.nan,
            'spc': np.nan
        }

def calculate_ci(values, confidence=0.95):
    """Calculate confidence interval"""
    mean = np.mean(values)
    se = stats.sem(values)
    ci = stats.t.interval(confidence, len(values)-1, loc=mean, scale=se)
    return mean, ci[0], ci[1]

def calculate_feature_importance(model, X):
    """Calculate Random Forest feature importance (MDI)"""
    return model.feature_importances_

def sensitivity_score(y_true, y_pred):
    """Calculate sensitivity (same as recall)"""
    return recall_score(y_true, y_pred)

def specificity_score(y_true, y_pred):
    """Calculate specificity"""
    # Specificity = TN / (TN + FP) = recall_score for negative class
    return recall_score(y_true, y_pred, pos_label=0)

def double_cv(X, y, n_estimators_range, cv1=5, cv2=5, n_repeats_cv2=5):
    """Perform double cross-validation"""
    if len(n_estimators_range) == 0:
        raise ValueError("n_estimators_range cannot be empty")
    
    # Store results
    test_predictions = []
    test_true = []
    best_n_estimators_list = []
    feature_importance = []
    
    # Store inner CV training and validation performance
    inner_cv_performances = {
        'train': {'auc': [], 'acc': [], 'sen': [], 'spc': []},
        'val': {'auc': [], 'acc': [], 'sen': [], 'spc': []}
    }
    
    try:
        # Repeat outer CV n_repeats_cv2 times
        for _ in range(n_repeats_cv2):
            outer_cv = KFold(n_splits=cv2, shuffle=True)
            
            # Outer CV
            for train_idx, test_idx in outer_cv.split(X):
                X_rest, X_test = X[train_idx], X[test_idx]
                y_rest, y_test = y[train_idx], y[test_idx]
                
                # Check if there are enough samples and classes
                if len(np.unique(y_rest)) < 2 or len(np.unique(y_test)) < 2:
                    print("Warning: Insufficient classes in train or test set")
                    continue
                
                # Inner CV
                best_n_estimators = n_estimators_range[0]  # Default use first value
                best_score = -np.inf
                
                inner_cv = KFold(n_splits=cv1, shuffle=True)
                for n_est in n_estimators_range:
                    val_scores = []
                    train_scores = []
                    for train_inner_idx, val_idx in inner_cv.split(X_rest):
                        X_train, X_val = X_rest[train_inner_idx], X_rest[val_idx]
                        y_train, y_val = y_rest[train_inner_idx], y_rest[val_idx]
                        
                        # Check training and validation set classes
                        if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
                            continue
                        
                        # Train model
                        model = RandomForestClassifier(n_estimators=n_est, random_state=42)
                        model.fit(X_train, y_train)
                        
                        # Calculate training set performance
                        try:
                            y_train_pred_proba = model.predict_proba(X_train)[:, 1]
                            train_metrics = calculate_metrics(y_train, y_train_pred_proba)
                            train_scores.append(train_metrics)
                        except Exception as e:
                            print(f"Warning: Error in training metrics calculation: {str(e)}")
                        
                        # Calculate validation set performance
                        try:
                            y_val_pred_proba = model.predict_proba(X_val)[:, 1]
                            val_metrics = calculate_metrics(y_val, y_val_pred_proba)
                            val_scores.append(val_metrics)
                        except Exception as e:
                            print(f"Warning: Error in validation metrics calculation: {str(e)}")
                    
                    # Only calculate average if there are valid scores
                    if train_scores and val_scores:
                        # Calculate average performance
                        mean_train_metrics = {k: np.nanmean([s[k] for s in train_scores]) for k in ['auc', 'acc', 'sen', 'spc']}
                        mean_val_metrics = {k: np.nanmean([s[k] for s in val_scores]) for k in ['auc', 'acc', 'sen', 'spc']}
                        
                        # Store inner CV performance
                        for metric in ['auc', 'acc', 'sen', 'spc']:
                            inner_cv_performances['train'][metric].append(mean_train_metrics[metric])
                            inner_cv_performances['val'][metric].append(mean_val_metrics[metric])
                        
                        # Use validation set AUC to select best parameters
                        mean_val_auc = mean_val_metrics['auc']
                        if not np.isnan(mean_val_auc) and mean_val_auc > best_score:
                            best_score = mean_val_auc
                            best_n_estimators = n_est
            
                # Train final model with best parameters
                final_model = RandomForestClassifier(n_estimators=best_n_estimators, random_state=42)
                final_model.fit(X_rest, y_rest)
                
                # Calculate feature importance
                try:
                    importance_scores = calculate_feature_importance(final_model, X_rest)
                    feature_importance.append(importance_scores)
                except Exception as e:
                    print(f"Warning: Error calculating feature importance: {str(e)}")
                    feature_importance.append(None)
                
                # Predict test set
                try:
                    y_test_pred_proba = final_model.predict_proba(X_test)[:, 1]
                    test_predictions.extend(y_test_pred_proba)
                    test_true.extend(y_test)
                    best_n_estimators_list.append(best_n_estimators)
                except Exception as e:
                    print(f"Warning: Error in test set prediction: {str(e)}")
        
        # Calculate average performance of inner CV (using nanmean to handle possible NA values)
        avg_inner_cv_performance = {
            'train': {k: np.nanmean(v) if v else np.nan for k, v in inner_cv_performances['train'].items()},
            'val': {k: np.nanmean(v) if v else np.nan for k, v in inner_cv_performances['val'].items()}
        }
        
        return np.array(test_predictions), np.array(test_true), feature_importance, avg_inner_cv_performance
    except Exception as e:
        print(f"Error in double_cv: {str(e)}")
        return np.array([]), np.array([]), [], {
            'train': {'auc': np.nan, 'acc': np.nan, 'sen': np.nan, 'spc': np.nan},
            'val': {'auc': np.nan, 'acc': np.nan, 'sen': np.nan, 'spc': np.nan}
        }

def permutation_test(orig_metrics, X, y, n_estimators_range, n_permutations=1000, **kwargs):
    """Perform permutation test"""
    # First downsample the original data
    class1_idx = np.where(y == 0)[0]
    class2_idx = np.where(y == 1)[0]
    min_samples = min(len(class1_idx), len(class2_idx))
    
    # Downsample to the size of the minority class
    if len(class1_idx) > min_samples:
        class1_idx = np.random.choice(class1_idx, size=min_samples, replace=False)
    if len(class2_idx) > min_samples:
        class2_idx = np.random.choice(class2_idx, size=min_samples, replace=False)
    
    # Merge selected samples
    selected_idx = np.concatenate([class1_idx, class2_idx])
    X_balanced = X[selected_idx]
    y_balanced = y[selected_idx]
    
    # Permutation test
    null_metrics = []
    for _ in tqdm(range(n_permutations), desc="Permutation test"):
        # Permute labels
        y_perm = np.random.permutation(y_balanced)
        
        # Execute double_cv
        print(f"samples in X_balanced: {len(X_balanced)}")
        perm_pred, perm_true, _, _ = double_cv(
            X_balanced, 
            y_perm,
            n_estimators_range=n_estimators_range,
            **kwargs
        )
        
        # Only calculate metrics when prediction is not empty
        if len(perm_pred) > 0:
            metrics = calculate_metrics(perm_true, perm_pred)
            null_metrics.append(metrics)
    
    # If no valid permutation results, return empty result
    if not null_metrics:
        print("Warning: No valid permutation results")
        return pd.DataFrame(), {
            'auc': np.nan,
            'acc': np.nan,
            'sen': np.nan,
            'spc': np.nan
        }
    
    # Calculate p-values
    null_metrics = pd.DataFrame(null_metrics)
    p_values = {
        metric: (np.sum(null_metrics[metric] >= orig_metrics[metric]) + 1) / (n_permutations + 1)
        for metric in ['auc', 'acc', 'sen', 'spc']
    }
    
    return null_metrics, p_values

def main(input_file, output_dir, repeat=50, cv2=5, cv1=5, n_repeats_cv2=5):
    """Main function"""
    print(f"Starting analysis with input file: {input_file}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory created: {output_dir}")
    
    # Read data
    print("Reading input data...")
    df = pd.read_csv(input_file)
    df = process_data(df)
    print(f"Data loaded and processed. Shape: {df.shape}")
    
    # Prepare result storage
    results = []
    importance_results = []
    permutation_results = []
    all_p_values = []
    
    # Feature combinations
    feature_types = ['spc', 'spc+my', 'spc+scc', 'spc+parity', 'spc+my+scc+parity'] # , 'spc_de', 'spc_rmR4'
    
    # Classification pairs
    class_pairs = [
        ('health', 'mast', 'health_vs_mast'),
        ('health', 'met', 'health_vs_met'),
        ('mast', 'met', 'mast_vs_met')
    ]
    
    # Define parameters to test
    n_estimators_range = [10, 20, 30, 40, 50]
    
    # Store skipped cases
    skipped_cases = []
    
    for class1, class2, pair_name in class_pairs:
        print(f"\nProcessing classification: {pair_name}")
        
        # Get samples for both classes
        samples1 = df[df['group'] == class1]
        samples2 = df[df['group'] == class2]
        
        print(f"Number of {class1} samples: {len(samples1)}")
        print(f"Number of {class2} samples: {len(samples2)}")
        
        # Analyze each dim
        for dim in range(8):  # 0-7
            print(f"\nProcessing dim: {dim}")
            
            # Get samples for this dim
            dim_samples1 = samples1[samples1['dim'] == dim]
            dim_samples2 = samples2[samples2['dim'] == dim]
            
            print(f"{class1} samples in dim {dim}: {len(dim_samples1)}")
            print(f"{class2} samples in dim {dim}: {len(dim_samples2)}")
            
            # Check if each class has enough samples (at least 10)
            if len(dim_samples1) < 10 or len(dim_samples2) < 10:
                msg = f"Skipping {pair_name}, dim {dim}: insufficient samples (class1: {len(dim_samples1)}, class2: {len(dim_samples2)}, minimum required: 10)"
                print(msg)
                skipped_cases.append({
                    'type': pair_name,
                    'dim': dim,
                    'reason': msg
                })
                
                # Add NA results for all feature types
                for feature_type in feature_types:
                    # Add NA results
                    results.append({
                        'type': pair_name,
                        'dim': dim,
                        'feature': feature_type,
                        'outer_auc_mean': np.nan,
                        'outer_auc_ci_low': np.nan,
                        'outer_auc_ci_up': np.nan,
                        'outer_acc_mean': np.nan,
                        'outer_acc_ci_low': np.nan,
                        'outer_acc_ci_up': np.nan,
                        'outer_sen_mean': np.nan,
                        'outer_sen_ci_low': np.nan,
                        'outer_sen_ci_up': np.nan,
                        'outer_spc_mean': np.nan,
                        'outer_spc_ci_low': np.nan,
                        'outer_spc_ci_up': np.nan,
                        'inner_train_auc_mean': np.nan,
                        'inner_train_auc_ci_low': np.nan,
                        'inner_train_auc_ci_up': np.nan,
                        'inner_train_acc_mean': np.nan,
                        'inner_train_acc_ci_low': np.nan,
                        'inner_train_acc_ci_up': np.nan,
                        'inner_train_sen_mean': np.nan,
                        'inner_train_sen_ci_low': np.nan,
                        'inner_train_sen_ci_up': np.nan,
                        'inner_train_spc_mean': np.nan,
                        'inner_train_spc_ci_low': np.nan,
                        'inner_train_spc_ci_up': np.nan,
                        'inner_val_auc_mean': np.nan,
                        'inner_val_auc_ci_low': np.nan,
                        'inner_val_auc_ci_up': np.nan,
                        'inner_val_acc_mean': np.nan,
                        'inner_val_acc_ci_low': np.nan,
                        'inner_val_acc_ci_up': np.nan,
                        'inner_val_sen_mean': np.nan,
                        'inner_val_sen_ci_low': np.nan,
                        'inner_val_sen_ci_up': np.nan,
                        'inner_val_spc_mean': np.nan,
                        'inner_val_spc_ci_low': np.nan,
                        'inner_val_spc_ci_up': np.nan
                    })
                    
                    # Add NA permutation results
                    permutation_results.append({
                        'type': pair_name,
                        'dim': dim,
                        'feature': feature_type,
                        'null_auc': np.nan,
                        'null_acc': np.nan,
                        'null_sen': np.nan,
                        'null_spc': np.nan
                    })
                    
                    # Add NA p-values
                    all_p_values.append({
                        'type': pair_name,
                        'dim': dim,
                        'feature': feature_type,
                        'auc': np.nan,
                        'acc': np.nan,
                        'sen': np.nan,
                        'spc': np.nan
                    })
                
                continue
            
            for feature_type in feature_types:
                print(f"\nProcessing feature type: {feature_type}")
                
                # Store metrics for each repeat
                repeat_metrics = {
                    'outer_auc': [], 'outer_acc': [], 'outer_sen': [], 'outer_spc': [],
                    'inner_train_auc': [], 'inner_train_acc': [], 'inner_train_sen': [], 'inner_train_spc': [],
                    'inner_val_auc': [], 'inner_val_acc': [], 'inner_val_sen': [], 'inner_val_spc': []
                }
                
                # Prepare complete dataset for permutation test
                balanced_data_full = pd.concat([dim_samples1, dim_samples2]).reset_index(drop=True)
                balanced_data_full['disease'] = (balanced_data_full['group'] == class2).astype(int)
                
                if feature_type == 'spc_de':
                    spectral_type = 'derivative'
                    base_feature = 'spc'
                elif feature_type == 'spc_rmR4':
                    spectral_type = 'rmR4'
                    base_feature = 'spc'
                else:
                    spectral_type = 'original'
                    base_feature = feature_type
                
                balanced_spectral_full = get_spectral_data(balanced_data_full, type=spectral_type)
                X_full = prepare_features(balanced_data_full, balanced_spectral_full, feature_type=base_feature)
                y_full = balanced_data_full['disease'].values
                
                for i in range(repeat):
                    print(f"\nRepeat {i+1}/{repeat}")
                    
                    # Downsample to the size of the smaller class
                    min_samples = min(len(dim_samples1), len(dim_samples2))
                    if len(dim_samples1) > min_samples:
                        sampled_indices1 = np.random.choice(dim_samples1.index, size=min_samples, replace=False)
                        balanced_samples1 = dim_samples1.loc[sampled_indices1]
                    else:
                        balanced_samples1 = dim_samples1
                    
                    if len(dim_samples2) > min_samples:
                        sampled_indices2 = np.random.choice(dim_samples2.index, size=min_samples, replace=False)
                        balanced_samples2 = dim_samples2.loc[sampled_indices2]
                    else:
                        balanced_samples2 = dim_samples2
                    
                    # Merge data and prepare labels
                    balanced_data = pd.concat([balanced_samples1, balanced_samples2]).reset_index(drop=True)
                    balanced_data['disease'] = (balanced_data['group'] == class2).astype(int)
                    
                    # Prepare features
                    if feature_type == 'spc_de':
                        spectral_type = 'derivative'
                        base_feature = 'spc'
                    elif feature_type == 'spc_rmR4':
                        spectral_type = 'rmR4'
                        base_feature = 'spc'
                    else:
                        spectral_type = 'original'
                        base_feature = feature_type
                    
                    balanced_spectral = get_spectral_data(balanced_data, type=spectral_type)
                    
                    X = prepare_features(balanced_data, balanced_spectral, feature_type=base_feature)
                    y = balanced_data['disease'].values
                    
                    # Execute double cross-validation
                    pred, true, importance, inner_cv_perf = double_cv(
                        X, y, 
                        n_estimators_range=n_estimators_range,
                        cv1=cv1, 
                        cv2=cv2,
                        n_repeats_cv2=n_repeats_cv2
                    )
                    
                    metrics = calculate_metrics(true, pred)
                    
                    # Store results for this repeat
                    for key in repeat_metrics:
                        if key.startswith('outer_'):
                            metric_name = key[6:]
                            repeat_metrics[key].append(metrics[metric_name])
                        elif key.startswith('inner_'):
                            phase, metric_name = key[6:].split('_', 1)
                            repeat_metrics[key].append(inner_cv_perf[phase][metric_name])
                
                # Calculate mean and CI for each metric
                final_metrics = {}
                for key in repeat_metrics:
                    values = np.array(repeat_metrics[key])
                    mean, ci_low, ci_up = calculate_ci(values)
                    final_metrics[f'{key}_mean'] = mean
                    final_metrics[f'{key}_ci_low'] = ci_low
                    final_metrics[f'{key}_ci_up'] = ci_up
                
                # Store results
                results.append({
                    'type': pair_name,
                    'dim': dim,
                    'feature': feature_type,
                    **final_metrics
                })
                
                # Calculate average metrics for permutation test
                avg_metrics = {
                    metric: np.mean(repeat_metrics[f'outer_{metric}'])
                    for metric in ['auc', 'acc', 'sen', 'spc']
                }
                
                # Execute permutation test
                print(f"\nPerforming permutation test for {feature_type}...")
                null_dist, p_values = permutation_test(
                    avg_metrics,
                    X_full, y_full,
                    n_estimators_range=n_estimators_range,
                    n_permutations=1000,
                    cv1=cv1,
                    cv2=cv2
                )
                
                # Store permutation results
                p_values.update({
                    'type': pair_name,
                    'dim': dim,
                    'feature': feature_type
                })
                all_p_values.append(p_values)
                
                for _, row in null_dist.iterrows():
                    permutation_results.append({
                        'type': pair_name,
                        'dim': dim,
                        'feature': feature_type,
                        'null_auc': row['auc'],
                        'null_acc': row['acc'],
                        'null_sen': row['sen'],
                        'null_spc': row['spc']
                    })
                
                # Store feature importance (only for original spectral features)
                if feature_type == 'spc':
                    wavelengths = [float(col) for col in balanced_spectral.columns]
                    for fold_idx, fold_importance in enumerate(importance):
                        if fold_importance is not None:  # Check if feature importance was successfully calculated
                            if len(fold_importance) == len(wavelengths):  # Ensure dimensions match
                                for wave_idx, wave in enumerate(wavelengths):
                                    importance_results.append({
                                        'type': pair_name,
                                        'dim': dim,
                                        'repeat': i+1,
                                        'fold': fold_idx + 1,
                                        'wavenumber': wave,
                                        'vip': fold_importance[wave_idx]
                                    })
                            else:
                                print(f"Warning: Feature importance dimension mismatch. Expected {len(wavelengths)}, got {len(fold_importance)}")
    
    # Save results
    print("\nSaving results...")
    results_df = pd.DataFrame(results)
    
    # Check if importance_results is empty
    if importance_results:
        importance_df = pd.DataFrame(importance_results)
    else:
        importance_df = pd.DataFrame(columns=['type', 'dim', 'repeat', 'fold', 'wavenumber', 'vip'])
    
    permutation_df = pd.DataFrame(permutation_results)
    p_values_df = pd.DataFrame(all_p_values)
    skipped_df = pd.DataFrame(skipped_cases)
    
    # Create separate directories for each classification type
    for pair_name in ['health_vs_mast', 'health_vs_met', 'mast_vs_met']:
        pair_dir = os.path.join(output_dir, pair_name)
        os.makedirs(pair_dir, exist_ok=True)
        
        # Filter and save results for this classification
        pair_results = results_df[results_df['type'] == pair_name]
        pair_importance = importance_df[importance_df['type'] == pair_name] if not importance_df.empty else pd.DataFrame()
        pair_permutation = permutation_df[permutation_df['type'] == pair_name]
        pair_p_values = p_values_df[p_values_df['type'] == pair_name]
        pair_skipped = skipped_df[skipped_df['type'] == pair_name]
        
        if not pair_results.empty:
            pair_results.to_csv(os.path.join(pair_dir, 'pre.csv'), index=False)
        if not pair_importance.empty:
            pair_importance.to_csv(os.path.join(pair_dir, 'imp.csv'), index=False)
        if not pair_permutation.empty:
            pair_permutation.to_csv(os.path.join(pair_dir, 'permu.csv'), index=False)
        if not pair_p_values.empty:
            pair_p_values.to_csv(os.path.join(pair_dir, 'p_values.csv'), index=False)
        if not pair_skipped.empty:
            pair_skipped.to_csv(os.path.join(pair_dir, 'skipped.csv'), index=False)
    
    print("Results saved successfully!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', required=True, help='Input CSV file path')
    parser.add_argument('--output_dir', required=True, help='Output directory path')
    parser.add_argument('--repeat', type=int, default=50, help='Number of downsample repeats')
    parser.add_argument('--cv2', type=int, default=5, help='Number of outer CV folds')
    parser.add_argument('--cv1', type=int, default=5, help='Number of inner CV folds')
    parser.add_argument('--n_repeats_cv2', type=int, default=5, help='Number of outer CV repeats')
    
    args = parser.parse_args()
    main(**vars(args))