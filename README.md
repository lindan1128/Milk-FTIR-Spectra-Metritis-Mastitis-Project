# Milk-FTIR-Spectra-Metritis-Mastitis-Prediction-Project

This study evaluated the use of milk Fourier-transform infrared (FTIR) spectroscopy to predict and classify early-lactation diseases—metritis and clinical mastitis—in Holstein cows within the first 7 days in milk (DIM). To ensure robust and unbiased performance evaluation, we employed a repeated down-sampled LOOCV framework coupled with three PLS-DA-based modeling strategies: Pooled PLS-DA, Multiblock PLS-DA and single-day (DIM) PLS-DA.


<img src="https://github.com/lindan1128/Milk-FTIR-Spectra-Metritis-Mastitis-Project/blob/main/workflow.png" alt="Workflow diagram">


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
	├── Supplemental_Figure/           # Folder for supplemental figures
	│   ├── Supplemental Figure 1      
	│   ├── Supplemental Figure 2
	│   ├── Supplemental Figure 3
	│   ├── Supplemental Figure 4
	│   ├── Supplemental Figure 5
	│   ├── Supplemental Figure 6
	│   ├── Supplemental Figure 7
	│   ├── Supplemental Figure 8
	│   ├── Supplemental Figure 9
	│── pooled_plsda.py               # Main function for Pooled PLS-DA
	│── multiblock_plsda              # Main function for Multiblock PLS-DA
	│── singleday_plsda.py            # Main function for Single-day/DIM PLS-DA
	├── README.md                     # Readme file
	└── requirements.txt              # Dependencies
	
### Pseudocode for modeling 
#### Pooled PLS-DA
	
	**Input Parameters**
	- input_file: path to input CSV file
	- output_dir: path to output directory
	- repeat: number of downsample repeats (default 50)
	- run_permutation: whether to run permutation test

	**Initialize**
	- Create output directory
	- Read and process input data
	- Initialize result storage for:
    	- all_detailed_metrics
    	- skipped_cases

	**Define Analysis Parameters**
	- feature_types = [
    'my+scc+dim+parity',
    'spc',
    'spc+dim',
    'my+scc+dim+parity+totalfa+lactose+protein',
    'my+scc+dim+parity+totalfa+lactose+protein+spc'
  	]
	- class_pairs = [
    ('health', 'mast', 'health_vs_mast'),
    ('health', 'met', 'health_vs_met'),
    ('mast', 'met', 'mast_vs_met')
  	]

	**Main Analysis Loop**
	For each class_pair in class_pairs:
    	- class1, class2, pair_name = class_pair
    	- Get samples for both classes

    	- Select samples with dim in 1-7 for both classes
    	- Check if each class has at least 10 samples
    	- If insufficient samples:
        	- Record skip reason
        	- Continue to next class_pair

    	For each feature_type in feature_types:
        	For each repeat in range(repeat):
            	- Downsample both classes to same size and DIM
            	- Concatenate and shuffle, assign binary label
            	- Select spectral type and prepare features
            	- Set n_components (15 if 'spc' in feature_type, else X.shape[1]-1)
            	- Leave-one-out cross-validation (by cow):
                	For each sample:
                    	- Train on all except one, test on the left-out
                    	- Store predictions
            	- Calculate metrics (AUC, ACC, SEN, SPC)
            	- If run_permutation:
                - Permute labels, repeat CV, calculate null metrics and p-values
            	- Store metrics for this repeat

	**Save Results**
	- Save all_detailed_metrics to CSV

	**Print Statistics**
	- Print summary and completion message
 
#### Multi-block PLS-DA

	**Input Parameters**
	- input_file: path to input CSV file
	- output_dir: path to output directory
	- repeat: number of downsample repeats (default 50)
	- run_permutation: whether to run permutation test

	**Initialize**
	- Create output directory
	- Read and process input data
	- Initialize result storage for:
    	- all_detailed_metrics
    	- skipped_cases
    	- all_cv_metrics_results

	**Define Analysis Parameters**
	- feature_types = [
    'my+scc+parity',
    'spc',
    'my+scc+parity+totalfa+lactose+protein',
    'my+scc+parity+totalfa+lactose+protein+spc'
  	]
	- class_pairs = [
    ('health', 'mast', 'health_vs_mast'),
    ('health', 'met', 'health_vs_met'),
    ('mast', 'met', 'mast_vs_met')
  	]

	**Main Analysis Loop**
	For each class_pair in class_pairs:
    	- class1, class2, pair_name = class_pair
    	- Get samples for both classes

    	- Select samples with dim in 1-5 for both classes
    	- Check if each class has at least 10 samples
    	- If insufficient samples:
        	- Record skip reason
        	- Continue to next class_pair

    	For each feature_type in feature_types:
        	For each repeat in range(repeat):
            	- Downsample both classes by cow_id to same number of cows
            	- Concatenate and shuffle, assign binary label
            	- Prepare spectral data and fit scalers/encoders
            	- Prepare multiblock features (each cow as a block, 5 days)
            	- Set n_components (15 if 'spc' in feature_type, else X.shape[1]-1)
            	- Leave-one-out cross-validation (by cow):
                	For each cow:
                    	- Train on all except one, test on the left-out
                    	- Store predictions and per-fold metrics
            	- Calculate metrics (AUC, ACC, SEN, SPC)
            	- Store per-fold metrics
            	- If run_permutation:
                	- Permute labels, repeat CV, calculate null metrics and p-values
            	- Store metrics for this repeat

	**Save Results**
	- Save all_detailed_metrics to CSV
	- Save all_cv_metrics_results to CSV

	**Print Statistics**
	- Print summary and completion message

#### Single-day PLS-DA
	**Input Parameters**
	- input_file: path to input CSV file
	- output_dir: path to output directory
	- repeat: number of downsample repeats (default 50)
	- run_permutation: whether to run permutation test

	**Initialize**
	- Create output directory
	- Read and process input data
	- Initialize result storage for:
    	- all_detailed_metrics
    	- skipped_cases

	**Define Analysis Parameters**
	- feature_types = [
    'my+scc+dim+parity',
    'spc',
    'my+scc+dim+parity+totalfa+lactose+protein',
    'my+scc+dim+parity+totalfa+lactose+protein+spc'
  	]
	- class_pairs = [
    ('health', 'mast', 'health_vs_mast'),
    ('health', 'met', 'health_vs_met'),
    ('mast', 'met', 'mast_vs_met')
  	]

	**Main Analysis Loop**
	For each class_pair in class_pairs:
    	- class1, class2, pair_name = class_pair
    	- Get samples for both classes

    	For each dim in range(8):
        	- Get samples for this dimension for both classes
        	- Check if each class has at least 10 samples
        	- If insufficient samples:
            	- Record skip reason
            	- Continue to next dim

        	For each feature_type in feature_types:
            	For each repeat in range(repeat):
                	- Downsample both classes to same size
                	- Concatenate and shuffle, assign binary label
                	- Select spectral type and prepare features
                	- Set n_components (15 if 'spc' in feature_type, else X.shape[1]-1)
                	- Leave-one-out cross-validation (by cow):
                    	For each sample:
                        	- Train on all except one, test on the left-out
                        	- Store predictions
                	- Calculate metrics (AUC, ACC, SEN, SPC)
                	- If run_permutation:
                    	- Permute labels, repeat CV, calculate null metrics and p-values
                	- Store metrics for this repeat

	**Save Results**
	- Save all_detailed_metrics to CSV

	**Print Statistics**
	- Print summary and completion message

### Modeling

	## Pooled PLS-DA
	python pooled_plsda.py --input_file INPUT_FILE --output_dir OUTPUT_DIR [options]
	## Multi-block PLS-DA
	python multiblock_plsda.py --input_file INPUT_FILE --output_dir OUTPUT_DIR [options]
	## Single-day PLS-DA
	python singleday_plsda.py --input_file INPUT_FILE --output_dir OUTPUT_DIR [options]

    The key hyperparameters for the model are:
    --input_file INPUT_FILE     Path to the input CSV file containing spectral data
    --output_dir OUTPUT_DIR     Directory to save output files
    --repeat REPEAT             Number of downsample repeats (default: 50)
    --run_permutation           Run permutation test or not (default: True)

    The output for the model are:
    * detailed_metrics.csv: Performance metrics for each classification pair, time, feature type and corresponding ermutation test results

