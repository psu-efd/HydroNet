#This script compiles the results of the DeepONet and PI-DeepONet models for all test cases into a Latex table.

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as tick
from matplotlib.collections import PolyCollection

import meshio

plt.rc('text', usetex=True)  #allow the use of Latex for math expressions and equations
plt.rc('font', family='serif') #specify the default font family to be "serif"
    

if __name__ == "__main__":

    # Test case result files
    deeponet_result_file = '../../DeepONet/data/DeepONet/test/test_results_summary.json'
    pi_deeponet_1_result_file = '../../PI_DeepONet_1/data/DeepONet/test/test_results_summary.json'
    pi_deeponet_2_result_file = '../../PI_DeepONet_2/data/DeepONet/test/test_results_summary.json'
    pi_deeponet_3_result_file = '../../PI_DeepONet_3/data/DeepONet/test/test_results_summary.json'
    
    # read the result files
    deeponet_results = json.load(open(deeponet_result_file))
    pi_deeponet_1_results = json.load(open(pi_deeponet_1_result_file))
    pi_deeponet_2_results = json.load(open(pi_deeponet_2_result_file))
    pi_deeponet_3_results = json.load(open(pi_deeponet_3_result_file))

    print("deeponet_results: ", deeponet_results)
    print("pi_deeponet_1_results: ", pi_deeponet_1_results)
    print("pi_deeponet_2_results: ", pi_deeponet_2_results)
    print("pi_deeponet_3_results: ", pi_deeponet_3_results)

    # compile the results into a Latex table and print the text to screen (to be copied and pasted into the manuscript)
    
    # Extract data
    # rmse_per_dim is [h, u, v] based on output_dim: 3
    deeponet_total_loss = deeponet_results['test_loss_rmse']
    deeponet_norm_h = deeponet_results['normalized']['rmse_per_dim'][0]
    deeponet_norm_u = deeponet_results['normalized']['rmse_per_dim'][1]
    deeponet_norm_v = deeponet_results['normalized']['rmse_per_dim'][2]
    deeponet_denorm_h = deeponet_results['denormalized']['rmse_per_dim'][0]
    deeponet_denorm_u = deeponet_results['denormalized']['rmse_per_dim'][1]
    deeponet_denorm_v = deeponet_results['denormalized']['rmse_per_dim'][2]
    
    pi_deeponet_1_total_loss = pi_deeponet_1_results['test_loss_rmse']
    pi_deeponet_1_norm_h = pi_deeponet_1_results['normalized']['rmse_per_dim'][0]
    pi_deeponet_1_norm_u = pi_deeponet_1_results['normalized']['rmse_per_dim'][1]
    pi_deeponet_1_norm_v = pi_deeponet_1_results['normalized']['rmse_per_dim'][2]
    pi_deeponet_1_denorm_h = pi_deeponet_1_results['denormalized']['rmse_per_dim'][0]
    pi_deeponet_1_denorm_u = pi_deeponet_1_results['denormalized']['rmse_per_dim'][1]
    pi_deeponet_1_denorm_v = pi_deeponet_1_results['denormalized']['rmse_per_dim'][2]
    
    pi_deeponet_2_total_loss = pi_deeponet_2_results['test_loss_rmse']
    pi_deeponet_2_norm_h = pi_deeponet_2_results['normalized']['rmse_per_dim'][0]
    pi_deeponet_2_norm_u = pi_deeponet_2_results['normalized']['rmse_per_dim'][1]
    pi_deeponet_2_norm_v = pi_deeponet_2_results['normalized']['rmse_per_dim'][2]
    pi_deeponet_2_denorm_h = pi_deeponet_2_results['denormalized']['rmse_per_dim'][0]
    pi_deeponet_2_denorm_u = pi_deeponet_2_results['denormalized']['rmse_per_dim'][1]
    pi_deeponet_2_denorm_v = pi_deeponet_2_results['denormalized']['rmse_per_dim'][2]
    
    pi_deeponet_3_total_loss = pi_deeponet_3_results['test_loss_rmse']
    pi_deeponet_3_norm_h = pi_deeponet_3_results['normalized']['rmse_per_dim'][0]
    pi_deeponet_3_norm_u = pi_deeponet_3_results['normalized']['rmse_per_dim'][1]
    pi_deeponet_3_norm_v = pi_deeponet_3_results['normalized']['rmse_per_dim'][2]
    pi_deeponet_3_denorm_h = pi_deeponet_3_results['denormalized']['rmse_per_dim'][0]
    pi_deeponet_3_denorm_u = pi_deeponet_3_results['denormalized']['rmse_per_dim'][1]
    pi_deeponet_3_denorm_v = pi_deeponet_3_results['denormalized']['rmse_per_dim'][2]
    
    # Format numbers for LaTeX table
    def format_num(x, decimals=4):
        """Format number with scientific notation if needed"""
        if x < 0.001:
            return f"{x:.2e}"
        else:
            return f"{x:.{decimals}f}"
    
    # Generate LaTeX table
    print("\n" + "="*80)
    print("LaTeX Table for Test Case Results")
    print("="*80 + "\n")
    
    latex_table = """\\begin{table}[htp]
    \\centering
    \\caption{Test case performance comparison between SWE-DeepONet and PI-SWE-DeepONet models.}
    \\label{tab:test-case-results}
    \\begin{tabular}{lccccccc}
        \\toprule
        Model & Total RMSE & \\multicolumn{3}{c}{Normalized RMSE} & \\multicolumn{3}{c}{Denormalized RMSE} \\\\
        \\cmidrule(lr){3-5} \\cmidrule(lr){6-8}
        & & $h$ & $u$ & $v$ & $h$ (m) & $u$ (m/s) & $v$ (m/s) \\\\
        \\midrule
        SWE-DeepONet & """ + format_num(deeponet_total_loss, 3) + """ & """ + \
        format_num(deeponet_norm_h, 3) + """ & """ + format_num(deeponet_norm_u, 3) + """ & """ + format_num(deeponet_norm_v, 3) + """ & """ + \
        format_num(deeponet_denorm_h, 3) + """ & """ + format_num(deeponet_denorm_u, 3) + """ & """ + format_num(deeponet_denorm_v, 3) + """ \\\\
        PI-SWE-DeepONet-1 & """ + format_num(pi_deeponet_1_total_loss, 3) + """ & """ + \
        format_num(pi_deeponet_1_norm_h, 3) + """ & """ + format_num(pi_deeponet_1_norm_u, 3) + """ & """ + format_num(pi_deeponet_1_norm_v, 3) + """ & """ + \
        format_num(pi_deeponet_1_denorm_h, 3) + """ & """ + format_num(pi_deeponet_1_denorm_u, 3) + """ & """ + format_num(pi_deeponet_1_denorm_v, 3) + """ \\\\
        PI-SWE-DeepONet-2 & """ + format_num(pi_deeponet_2_total_loss, 3) + """ & """ + \
        format_num(pi_deeponet_2_norm_h, 3) + """ & """ + format_num(pi_deeponet_2_norm_u, 3) + """ & """ + format_num(pi_deeponet_2_norm_v, 3) + """ & """ + \
        format_num(pi_deeponet_2_denorm_h, 3) + """ & """ + format_num(pi_deeponet_2_denorm_u, 3) + """ & """ + format_num(pi_deeponet_2_denorm_v, 3) + """ \\\\
        PI-SWE-DeepONet-3 & """ + format_num(pi_deeponet_3_total_loss, 3) + """ & """ + \
        format_num(pi_deeponet_3_norm_h, 3) + """ & """ + format_num(pi_deeponet_3_norm_u, 3) + """ & """ + format_num(pi_deeponet_3_norm_v, 3) + """ & """ + \
        format_num(pi_deeponet_3_denorm_h, 3) + """ & """ + format_num(pi_deeponet_3_denorm_u, 3) + """ & """ + format_num(pi_deeponet_3_denorm_v, 3) + """ \\\\
        \\bottomrule
    \\end{tabular}
\\end{table}"""
    
    print(latex_table)
    print("\n" + "="*80)

    #generate markdown table (same content as the latex table)
    markdown_table = """| Model | Total RMSE | Normalized RMSE | Denormalized RMSE |
|-------|------------|----------------|------------------|
| SWE-DeepONet | """ + format_num(deeponet_total_loss, 3) + """ | """ + \
format_num(deeponet_norm_h, 3) + """ | """ + format_num(deeponet_norm_u, 3) + """ | """ + format_num(deeponet_norm_v, 3) + """ | """ + \
format_num(deeponet_denorm_h, 3) + """ | """ + format_num(deeponet_denorm_u, 3) + """ | """ + format_num(deeponet_denorm_v, 3) + """ |
| PI-SWE-DeepONet-1 | """ + format_num(pi_deeponet_1_total_loss, 3) + """ | """ + \
format_num(pi_deeponet_1_norm_h, 3) + """ | """ + format_num(pi_deeponet_1_norm_u, 3) + """ | """ + format_num(pi_deeponet_1_norm_v, 3) + """ | """ + \
format_num(pi_deeponet_1_denorm_h, 3) + """ | """ + format_num(pi_deeponet_1_denorm_u, 3) + """ | """ + format_num(pi_deeponet_1_denorm_v, 3) + """ |
| PI-SWE-DeepONet-2 | """ + format_num(pi_deeponet_2_total_loss, 3) + """ | """ + \
format_num(pi_deeponet_2_norm_h, 3) + """ | """ + format_num(pi_deeponet_2_norm_u, 3) + """ | """ + format_num(pi_deeponet_2_norm_v, 3) + """ | """ + \
format_num(pi_deeponet_2_denorm_h, 3) + """ | """ + format_num(pi_deeponet_2_denorm_u, 3) + """ | """ + format_num(pi_deeponet_2_denorm_v, 3) + """ |
| PI-SWE-DeepONet-3 | """ + format_num(pi_deeponet_3_total_loss, 3) + """ | """ + \
format_num(pi_deeponet_3_norm_h, 3) + """ | """ + format_num(pi_deeponet_3_norm_u, 3) + """ | """ + format_num(pi_deeponet_3_norm_v, 3) + """ | """ + \
format_num(pi_deeponet_3_denorm_h, 3) + """ | """ + format_num(pi_deeponet_3_denorm_u, 3) + """ | """ + format_num(pi_deeponet_3_denorm_v, 3) + """ |
"""
    print(markdown_table)
    print("\n" + "="*80)
    
    print("\nPlotting comparison of test case results completed.")