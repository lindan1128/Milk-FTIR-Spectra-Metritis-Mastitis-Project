# Milk-FTIR-Spectra-Metritis-Mastitis-Prediction-Project

This study evaluated the use of milk Fourier-transform infrared (FTIR) spectroscopy to predict and classify early-lactation diseases—metritis and clinical mastitis—in Holstein cows within the first 7 days in milk (DIM). To ensure robust and unbiased performance evaluation, we employed a repeated down-sampled double cross-validation framework that balanced class distributions and integrated hyperparameter tuning and model assessment through nested 5-fold cross-validation.

![Workflow diagram](https://github.com/lindan1128/Milk-FTIR-Spectra-Metritis-Mastitis-Project/blob/main/workflow.png){width=500 height=400}

### Project structure
	Main/
	├── Codes/                        # Folder for codes for analysis and visualization
	│   ├── Figure 1B-1D.r            # codes for generating results for Figure 1B-1D
	│   ├── Figure 2A-2B.r            # codes for generating results for Figure 2A-2B
	│   ├── Figure 2C.r               # codes for generating results for Figure 2C
	│   ├── Figure 2D.r               # codes for generating results for Figure 2D
  	│   ├── Figure 3.r                # codes for generating results for Figure 3
	├── Supplemental_Table/           # Folder for supplemental tables
	│   ├── Supplemental Table 1      
	│   ├── Supplemental Table 2
	│   ├── Supplemental Table 3
	│   ├── Supplemental Table 4
	│   └── Supplemental Table 5
	│── pls-da.py                     # Main function for PLS-DA
	│── rf.py                         # Main function for random forest
	│── lstm.py                       # Main function for LSTM
	├── README.md                     # Readme file
	└── requirements.txt              # Dependencies
	
### Pseudocode for modeling -- PLS-DA as example
	
	Input Parameters
	input_file = "path to input CSV file"
	output_dir = "path to output directory"
	repeat = 50  # Number of downsample repeats
	cv2 = 5      # Number of outer CV folds
	cv1 = 5      # Number of inner CV folds
	n_repeats_cv2 = 5  # Number of outer CV repeats

	Initialize
	Create output directory
	Read and process input data
	Initialize result storage arrays for:
    - Performance metrics
    - Feature importance
    - Permutation test results
    - P-values
    - Skipped cases

	Define Analysis Parameters
	feature_types = ['spc', 'spc+my', 'spc+scc', 'spc+parity', 'spc+my+scc+parity']
	class_pairs = [
    ('health', 'mast', 'health_vs_mast'),
    ('health', 'met', 'health_vs_met'),
    ('mast', 'met', 'mast_vs_met')
	]
	n_components_range = [2, 3, 4, 5, 6, 7, 8, 9, 10]

	Main Analysis Loop
	For each class_pair in class_pairs:
    class1, class2, pair_name = class_pair
    Get samples for both classes
    
    For each dim in range(8):
        Get samples for this dimension
        Check if each class has enough samples (minimum 10)
        
        If insufficient samples:
            Add NA results for all feature types
            Skip to next dimension
            
        For each feature_type in feature_types:
            Prepare complete dataset for permutation test
            
            For each repeat in range(repeat):
                Downsample to size of smaller class
                Prepare balanced dataset
                Prepare features based on feature type
                
                For each outer CV repeat (from 1 to n_repeats_cv2):
                    Execute double cross-validation:
                        - Outer CV (cv2 folds)
                        - Inner CV for each fold (cv1 folds)
                        - Train models with different component numbers
                        - Select best component number
                        - Calculate VIP scores
                        - Store predictions and metrics
                
                Store results for this repeat
            
            Calculate mean and confidence intervals for metrics
            
            Execute permutation test:
                - Perform n_permutations
                - Calculate p-values
                - Store null distribution
            
            Store feature importance (for original spectral features)

	Save Results
	For each classification pair:
    Create separate directory
    Save results to CSV files:
        - Performance metrics (pre.csv)
        - Feature importance (imp.csv)
        - Permutation results (permu.csv)
        - P-values (p_values.csv)
        - Skipped cases (skipped.csv)

	Print Statistics
	Print summary of analysis results
	
### Modeling

	## PLS-DA
	python pls-da.py --input_file INPUT_FILE --output_dir OUTPUT_DIR [options]
	## random forest
	python pls-da.py --input_file INPUT_FILE --output_dir OUTPUT_DIR [options]
	## LSTM
	python pls-da.py --input_file INPUT_FILE --output_dir OUTPUT_DIR [options]

    The key hyperparameters for the model are:
    --input_file INPUT_FILE     Path to the input CSV file containing spectral data
    --output_dir OUTPUT_DIR     Directory to save output files
    --repeat REPEAT             Number of downsample repeats (default: 50)
    --cv2 CV2                   Number of outer CV folds (default: 5)
    --cv1 CV1                   Number of inner CV folds (default: 5)
    --n_repeats_cv2 N_REPEATS   Number of repeats for outer CV (default: 5)

    The output for the model are:
    * pre.csv: Performance metrics for each classification pair, dimension, and feature type
    * imp.csv: VIP scores for spectral features (only for original spectral features)
    * permu.csv: Permutation test results
    * p_values.csv: P-values from permutation tests
    * skipped.csv: Information about skipped cases due to insufficient samples

    The results are organized in separate directories for each classification pair:
    * health_vs_mast/
    * health_vs_met/
    * mast_vs_met/

    Each directory contains the following files:
    * pre.csv: Contains performance metrics including:
        - Outer CV metrics (AUC, accuracy, sensitivity, specificity)
        - Inner CV metrics (training and validation)
        - Confidence intervals for all metrics
    * imp.csv: Contains VIP scores for each wavenumber
    * permu.csv: Contains null distribution metrics from permutation tests
    * p_values.csv: Contains p-values for all performance metrics
    * skipped.csv: Contains information about cases skipped due to insufficient samples
