## 3. Combined Analysis
The combined analysis script does three things
1. Copies the face csv file to the pose output folder
1. Generates a combined json file in the pose output folder
1. Runs cross-correlation model 1.0 by calling file_input_run_plots and file_input_run_parameters
	- **This could be adapted to use newer models**

To run the combined analysis, open a new anaconda prompt and type:
- `conda activate synchrony_detection` <br><br>
- `cd {your path}\Combined_Analysis` <br><br>
- `python combined_analysis.py --folder {your analysis folder}`

Results are saved to cross_corr_output\combined_results.csv
Information about whether each step of analysis has run can be found in analysis_info\analysis_info.csv
