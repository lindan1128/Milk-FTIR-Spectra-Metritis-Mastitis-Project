import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score
from scipy.signal import savgol_filter
import os
from tqdm import tqdm
import warnings
import scipy.stats as stats
from mbpls.mbpls import MBPLS
warnings.filterwarnings('ignore')

def process_data(df):
    print(f"Initial data shape: {df.shape}")
    columns_to_check = ['milkweightlbs', 'cells', 'parity']
    na_counts = df[columns_to_check].isna().sum()
    print("\nNA counts in relevant columns:")
    print(na_counts[na_counts > 0])
    for col in ['milkweightlbs', 'cells']:
        if df[col].isna().any():
            print(f"\nInterpolating {col} by dim...")
            df[col] = df.groupby('dim')[col].transform(lambda x: x.interpolate(method='linear'))
            if df[col].isna().any():
                df[col] = df.groupby('dim')[col].transform(lambda x: x.fillna(x.mean()))
            remaining_na = df[col].isna().sum()
            if remaining_na > 0:
                print(f"Warning: {remaining_na} NA values remain in {col} after interpolation")
                df[col] = df[col].fillna(df[col].mean())
    df = df.dropna(subset=['parity'])
    print(f"\nShape after handling NA: {df.shape}")
    df['parity'] = df['parity'].apply(lambda x: '2+' if x > 2 else str(x))
    print(f"\nFinal data shape: {df.shape}")
    print("\nFinal NA counts:")
    print(df[columns_to_check].isna().sum())
    return df

def get_spectral_data(df, type='original'):
    non_spectral_cols = ['disease_in', 'disease', 'day_group', 'milkweightlbs', 'cells', 'parity', 'Unnamed: 0', 'index']
    spectral_cols = [col for col in df.columns if col not in non_spectral_cols]
    spectral_cols = [col for col in spectral_cols if col.replace('.', '').isdigit()]
    wavelengths = [float(col) for col in spectral_cols]
    valid_cols = [col for col, wave in zip(spectral_cols, wavelengths) if 1000 <= wave <= 3000 and not (1580 <= wave <= 1700) and not (1800 <= wave <= 2800)]
    if type == 'original':
        return df[valid_cols]
    elif type == 'derivative':
        spectra = df[valid_cols].values
        derivatives = savgol_filter(spectra, window_length=7, polyorder=2, deriv=1, axis=1)
        return pd.DataFrame(derivatives, columns=valid_cols, index=df.index)
    elif type == 'rmR4':
        rmR4_cols = [col for col, wave in zip(valid_cols, map(float, valid_cols)) if wave < 1800 or wave > 2800]
        return df[rmR4_cols]
    else:
        raise ValueError(f"Unknown spectral type: {type}")

def calculate_derivatives(spectra):
    return pd.DataFrame(savgol_filter(spectra, window_length=7, polyorder=2, deriv=1), columns=spectra.columns, index=spectra.index)

def prepare_multiblock_features(df, spectral_data, feature_type='spc', scaler_my=None, scaler_scc=None, dim_encoder=None, parity_encoder=None):
    cow_ids = df['cow_id'].unique()
    blocks = []
    valid_cow_ids = []
    feature_dim = 0
    if 'my' in feature_type:
        feature_dim += 1
    if 'scc' in feature_type:
        feature_dim += 1
    if 'dim' in feature_type:
        feature_dim += 5
    if 'parity' in feature_type:
        feature_dim += len(parity_encoder.categories_[0])
    if 'spc' in feature_type:
        spectral_size = spectral_data.shape[1]
        block_size = spectral_size * 5
    else:
        block_size = feature_dim * 5
    for cow_id in cow_ids:
        cow_data = df[df['cow_id'] == cow_id]
        cow_block = []
        valid_days = 0
        for dim in range(1, 6):
            day_data = cow_data[cow_data['dim'] == dim]
            if len(day_data) == 0:
                if 'spc' in feature_type:
                    day_features = np.zeros(spectral_size)
                else:
                    day_features = np.zeros(feature_dim)
            else:
                valid_days += 1
                features = []
                if 'my' in feature_type:
                    features.append(scaler_my.transform(day_data[['milkweightlbs']])[0])
                if 'scc' in feature_type:
                    features.append(scaler_scc.transform(day_data[['cells']])[0])
                if 'dim' in feature_type:
                    dim_onehot = dim_encoder.transform([[dim]])[0]
                    features.append(dim_onehot)
                if 'parity' in feature_type:
                    parity_onehot = parity_encoder.transform(day_data[['parity']])[0]
                    features.append(parity_onehot)
                if 'spc' in feature_type:
                    spectral_array = spectral_data.loc[day_data.index].values
                    sample_mins = np.min(spectral_array, axis=1, keepdims=True)
                    sample_maxs = np.max(spectral_array, axis=1, keepdims=True)
                    sample_ranges = sample_maxs - sample_mins
                    sample_ranges[sample_ranges == 0] = 1
                    spectral_array = (spectral_array - sample_mins) / sample_ranges
                    day_features = spectral_array[0]
                else:
                    day_features = np.concatenate(features)
            cow_block.append(day_features)
        if valid_days < 3:
            continue
        cow_block = np.concatenate(cow_block)
        blocks.append(cow_block)
        valid_cow_ids.append(cow_id)
    X = np.vstack(blocks)
    print(f"[DEBUG] Number of cows kept: {len(valid_cow_ids)}; First 10 cow_ids: {valid_cow_ids[:10]}")
    return X, valid_cow_ids

def calculate_metrics(y_true, y_pred_proba):
    y_pred = (y_pred_proba > 0.5).astype(int)
    unique_classes = np.unique(y_true)
    if len(unique_classes) == 1:
        print(f"Warning: Only one class present ({unique_classes[0]}). Skipping metrics calculation.")
        return {
            'auc': np.nan,
            'acc': np.nan,
            'sen': np.nan,
            'spc': np.nan
        }
    try:
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
    mean = np.mean(values)
    se = stats.sem(values)
    ci = stats.t.interval(confidence, len(values)-1, loc=mean, scale=se)
    return mean, ci[0], ci[1]

def sensitivity_score(y_true, y_pred):
    return recall_score(y_true, y_pred)

def specificity_score(y_true, y_pred):
    return recall_score(y_true, y_pred, pos_label=0)

def double_cv(X, y, n_components_range, cv1=5, cv2=5, n_repeats_cv1=1):
    if len(n_components_range) == 0:
        raise ValueError("n_components_range cannot be empty")
    outer_cv = KFold(n_splits=cv2, shuffle=True)
    test_predictions = []
    test_true = []
    best_n_components_list = []
    inner_cv_performances = {
        'train': {'auc': [], 'acc': [], 'sen': [], 'spc': []},
        'val': {'auc': [], 'acc': [], 'sen': [], 'spc': []}
    }
    try:
        for train_idx, test_idx in outer_cv.split(X):
            X_rest, X_test = X[train_idx], X[test_idx]
            y_rest, y_test = y[train_idx], y[test_idx]
            if len(np.unique(y_rest)) < 2 or len(np.unique(y_test)) < 2:
                print("Warning: Insufficient classes in train or test set")
                continue
            best_n_components = n_components_range[0]
            best_score = -np.inf
            for _ in range(n_repeats_cv1):
                inner_cv = KFold(n_splits=cv1, shuffle=True)
                for n_comp in n_components_range:
                    val_scores = []
                    train_scores = []
                    for train_inner_idx, val_idx in inner_cv.split(X_rest):
                        X_train, X_val = X_rest[train_inner_idx], X_rest[val_idx]
                        y_train, y_val = y_rest[train_inner_idx], y_rest[val_idx]
                        model = PLSRegression(n_components=n_comp)
                        model.fit(X_train, y_train)
                        y_train_pred = model.predict(X_train)
                        train_metrics = calculate_metrics(y_train, y_train_pred)
                        train_scores.append(train_metrics)
                        y_val_pred = model.predict(X_val)
                        val_metrics = calculate_metrics(y_val, y_val_pred)
                        val_scores.append(val_metrics)
                    mean_train_metrics = {k: np.mean([s[k] for s in train_scores]) for k in train_metrics}
                    mean_val_metrics = {k: np.mean([s[k] for s in val_scores]) for k in val_metrics}
                    for metric in ['auc', 'acc', 'sen', 'spc']:
                        inner_cv_performances['train'][metric].append(mean_train_metrics[metric])
                        inner_cv_performances['val'][metric].append(mean_val_metrics[metric])
                    mean_val_auc = mean_val_metrics['auc']
                    if mean_val_auc > best_score:
                        best_score = mean_val_auc
                        best_n_components = n_comp
            final_model = PLSRegression(n_components=best_n_components)
            final_model.fit(X_rest, y_rest)
            y_test_pred = final_model.predict(X_test)
            test_predictions.extend(y_test_pred)
            test_true.extend(y_test)
            best_n_components_list.append(best_n_components)
        avg_inner_cv_performance = {
            'train': {k: np.nanmean(v) if v else np.nan for k, v in inner_cv_performances['train'].items()},
            'val': {k: np.nanmean(v) if v else np.nan for k, v in inner_cv_performances['val'].items()}
        }
        return np.array(test_predictions), np.array(test_true), None, avg_inner_cv_performance
    except Exception as e:
        print(f"Error in double_cv: {str(e)}")
        return np.array([]), np.array([]), [], {'train': {}, 'val': {}}

def permutation_test(orig_metrics, X, y, n_components_range, n_permutations=2, **kwargs):
    try:
        class1_idx = np.where(y == 0)[0]
        class2_idx = np.where(y == 1)[0]
        min_samples = min(len(class1_idx), len(class2_idx))
        
        print(f"Original class sizes - Class 0: {len(class1_idx)}, Class 1: {len(class2_idx)}")
        print(f"Downsampling to {min_samples} samples per class")
        
        if len(class1_idx) > min_samples:
            class1_idx = np.random.choice(class1_idx, size=min_samples, replace=False)
        if len(class2_idx) > min_samples:
            class2_idx = np.random.choice(class2_idx, size=min_samples, replace=False)
        
        selected_idx = np.concatenate([class1_idx, class2_idx])
        
        if np.max(selected_idx) >= len(X):
            print(f"Warning: Index {np.max(selected_idx)} out of bounds for X with size {len(X)}")
            return pd.DataFrame(), {
                'auc': np.nan,
                'acc': np.nan,
                'sen': np.nan,
                'spc': np.nan
            }
        
        X_balanced = X[selected_idx]
        y_balanced = y[selected_idx]
        
        print(f"Balanced data shape - X: {X_balanced.shape}, y: {y_balanced.shape}")
        
        null_metrics = []
        for i in tqdm(range(n_permutations), desc="Permutation test"):
            y_perm = np.random.permutation(y_balanced)
            
            perm_pred, perm_true, _, _ = double_cv(
                X_balanced, 
                y_perm,
                n_components_range=n_components_range,
                **kwargs
            )
            
            if len(perm_pred) > 0:
                metrics = calculate_metrics(perm_true, perm_pred)
                null_metrics.append(metrics)
                if i % 10 == 0:
                    print(f"Permutation {i}: metrics = {metrics}")
        
        if not null_metrics:
            print("Warning: No valid permutation results")
            return pd.DataFrame(), {
                'auc': np.nan,
                'acc': np.nan,
                'sen': np.nan,
                'spc': np.nan
            }
        
        null_metrics_df = pd.DataFrame(null_metrics)
        print("\nNull metrics summary:")
        print(null_metrics_df.describe())
        
        print("\nOriginal metrics:")
        print(orig_metrics)
        
        p_values = {}
        for metric in ['auc', 'acc', 'sen', 'spc']:
            if metric in null_metrics_df.columns and metric in orig_metrics:
                p_value = (np.sum(null_metrics_df[metric] >= orig_metrics[metric]) + 1) / (len(null_metrics_df) + 1)
                p_values[metric] = p_value
                print(f"{metric} p-value: {p_value}")
            else:
                print(f"Warning: {metric} not found in metrics")
                p_values[metric] = np.nan
        
        return null_metrics_df, p_values
        
    except Exception as e:
        print(f"Error in permutation_test: {str(e)}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame(), {
            'auc': np.nan,
            'acc': np.nan,
            'sen': np.nan,
            'spc': np.nan
        }

def main(input_file, output_dir, repeat=50, run_permutation=False):
    print(f"Starting analysis with input file: {input_file}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory created: {output_dir}")
    df = pd.read_csv(input_file)
    df = process_data(df)
    print(f"Data loaded and processed. Shape: {df.shape}")

    feature_types = [
        'my+scc+parity',
        'spc',
        'my+scc+parity+totalfa+lactose+protein',
        'my+scc+parity+totalfa+lactose+protein+spc'
    ]

    class_pairs = [
        ('health', 'mast', 'health_vs_mast'),
        ('health', 'met', 'health_vs_met'),
        ('mast', 'met', 'mast_vs_met')
    ]

    all_detailed_metrics = []
    skipped_cases = []

    for class1, class2, pair_name in class_pairs:
        print(f"\nProcessing classification: {pair_name}")
        samples1 = df[df['group'] == class1]
        samples2 = df[df['group'] == class2]
        print(f"Number of {class1} samples: {len(samples1)}")
        print(f"Number of {class2} samples: {len(samples2)}")
        
        dim_samples1 = samples1[samples1['dim'].between(1, 5)]
        dim_samples2 = samples2[samples2['dim'].between(1, 5)]
        print(f"{class1} samples in dim 1-5: {len(dim_samples1)}")
        print(f"{class2} samples in dim 1-5: {len(dim_samples2)}")
        
        if len(dim_samples1) < 10 or len(dim_samples2) < 10:
            msg = f"Skipping {pair_name}: insufficient samples (class1: {len(dim_samples1)}, class2: {len(dim_samples2)}, minimum required: 10)"
            print(msg)
            skipped_cases.append({'type': pair_name, 'dim': 'all', 'reason': msg})
            continue
            
        for feature_type in feature_types:
            print(f"\nProcessing feature type: {feature_type}")
            for i in range(repeat):
                cows1 = dim_samples1['cow_id'].unique()
                cows2 = dim_samples2['cow_id'].unique()
                min_cows = min(len(cows1), len(cows2))
                if len(cows1) > min_cows:
                    sampled_cows1 = np.random.choice(cows1, size=min_cows, replace=False)
                    balanced_samples1 = dim_samples1[dim_samples1['cow_id'].isin(sampled_cows1)]
                else:
                    balanced_samples1 = dim_samples1
                if len(cows2) > min_cows:
                    sampled_cows2 = np.random.choice(cows2, size=min_cows, replace=False)
                    balanced_samples2 = dim_samples2[dim_samples2['cow_id'].isin(sampled_cows2)]
                else:
                    balanced_samples2 = dim_samples2
                balanced_data = pd.concat([balanced_samples1, balanced_samples2]).reset_index(drop=True)
                balanced_data['disease'] = (balanced_data['group'] == class2).astype(int)
                
                balanced_spectral = get_spectral_data(balanced_data, type='original')
                
                scaler_my = MinMaxScaler().fit(balanced_data[['milkweightlbs']])
                scaler_scc = MinMaxScaler().fit(balanced_data[['cells']])
                dim_encoder = OneHotEncoder(categories=[[1,2,3,4,5]], sparse=False)
                dim_encoder.fit(np.array([1,2,3,4,5]).reshape(-1,1))
                parity_encoder = OneHotEncoder(sparse=False)
                parity_encoder.fit(balanced_data[['parity']])

                X, valid_cow_ids = prepare_multiblock_features(balanced_data, balanced_spectral, feature_type=feature_type, scaler_my=scaler_my, scaler_scc=scaler_scc, dim_encoder=dim_encoder, parity_encoder=parity_encoder)
                y = balanced_data.groupby('cow_id')['disease'].first().reindex(valid_cow_ids).values
                
                if 'spc' in feature_type:
                    n_components = 15
                else:
                    n_components = X.shape[1] - 1
                
                n_samples = X.shape[0]
                y_true_all = []
                y_pred_all = []
                cv_metrics_results = []
                n_days = 5
                n_features_per_day = X.shape[1] // n_days

                for loo_idx in range(n_samples):
                    train_idx = np.array([i for i in range(n_samples) if i != loo_idx])
                    test_idx = np.array([loo_idx])
                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]

                    X_train_reshaped = X_train.reshape(X_train.shape[0], n_days, n_features_per_day)
                    X_test_reshaped = X_test.reshape(X_test.shape[0], n_days, n_features_per_day)
                    X_train_blocks = [X_train_reshaped[:, day, :] for day in range(n_days)]
                    X_test_blocks = [X_test_reshaped[:, day, :] for day in range(n_days)]

                    model = MBPLS(n_components=n_components, standardize=True, sparse_data=True)
                    model.fit(X_train_blocks, y_train.reshape(-1, 1))
                    y_pred = model.predict(X_test_blocks).ravel()

                    y_true = y_test[0]
                    y_pred_score = y_pred[0]
                    y_pred_label = int(y_pred_score > 0.5)
                    acc = int(y_pred_label == y_true)
                    sen = int((y_true == 1) and (y_pred_label == 1))
                    spc = int((y_true == 0) and (y_pred_label == 0))
                    auc = float('nan')
                    cv_metrics_results.append({
                        'repeat': i+1,
                        'cv': loo_idx+1,
                        'true': y_true,
                        'pred': y_pred_score,
                        'acc': acc,
                        'sen': sen,
                        'spc': spc,
                        'auc': auc,
                        'cow_id': valid_cow_ids[loo_idx] if 'valid_cow_ids' in locals() else None
                    })
                    y_true_all.append(y_true)
                    y_pred_all.append(y_pred_score)
                metrics = calculate_metrics(np.array(y_true_all), np.array(y_pred_all))
                if 'all_cv_metrics_results' not in locals():
                    all_cv_metrics_results = []
                all_cv_metrics_results.extend(cv_metrics_results)
                
                if run_permutation:
                    print(f"[PERMUTATION] Running permutation test for {pair_name}, feature {feature_type}, repeat {i+1}")
                    n_components_range = [n_components] if 'spc' in feature_type else [X.shape[1] - 1]
                    null_metrics, p_values = permutation_test(
                        metrics,
                        X,
                        y,
                        n_components_range=n_components_range,
                        n_permutations=100
                    )
                    metrics_dict = {
                        'type': pair_name,
                        'dim': 'all',
                        'feature': feature_type,
                        'repeat': i+1,
                        'auc': metrics['auc'],
                        'acc': metrics['acc'],
                        'sen': metrics['sen'],
                        'spc': metrics['spc'],
                        'auc_p_value': p_values['auc'],
                        'acc_p_value': p_values['acc'],
                        'sen_p_value': p_values['sen'],
                        'spc_p_value': p_values['spc']
                    }
                else:
                    metrics_dict = {
                        'type': pair_name,
                        'dim': 'all',
                        'feature': feature_type,
                        'repeat': i+1,
                        'auc': metrics['auc'],
                        'acc': metrics['acc'],
                        'sen': metrics['sen'],
                        'spc': metrics['spc']
                    }
                
                all_detailed_metrics.append(metrics_dict)
    
    detailed_df = pd.DataFrame(all_detailed_metrics)
    detailed_df.to_csv(os.path.join(output_dir, 'detailed_metrics.csv'), index=False)
    print('Detailed metrics saved!')

    if 'all_cv_metrics_results' in locals() and len(all_cv_metrics_results) > 0:
        cv_metrics_df = pd.DataFrame(all_cv_metrics_results)
        cv_metrics_df.to_csv(os.path.join(output_dir, 'cv_metrics_per_fold.csv'), index=False)
        print('CV metrics per fold saved!')

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', required=True, help='Input CSV file path')
    parser.add_argument('--output_dir', required=True, help='Output directory path')
    parser.add_argument('--repeat', type=int, default=50, help='Number of downsample repeats')
    parser.add_argument('--run_permutation', action='store_true', help='Whether to run permutation test')
    
    args = parser.parse_args()
    main(**vars(args))