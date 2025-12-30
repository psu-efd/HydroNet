#This script creates a table of the comparison metrics for the SWE-DeepONet and PI-SWE-DeepONet models. The metrics results are in files DeepONet_PI-DeepONet_performance_h.json, DeepONet_PI-DeepONet_performance_u.json, DeepONet_PI-DeepONet_performance_v.json, and DeepONet_PI-DeepONet_performance_Umag.json, for h, u, v, and |u|, respectively. In the table, we include the slope, model breakdown distance, and ratio fraction. Output in the format of nicely formatted latex table and print out to the screen (so to be copy-pasted into a latex document). 

import json

def format_number(x, decimals=3):
    """Format number with specified decimal places."""
    if x is None:  #if the number is None, return positive infinity
        return "$\\infty$"
    if abs(x) < 0.001 and x != 0:
        return f"{x:.2e}"
    else:
        return f"{x:.{decimals}f}"

def main():
    model_1_name = 'DeepONet'
    model_2_name = 'PI_DeepONet_3'
    # Get the metrics results from the json files
    with open(model_1_name+"_"+model_2_name+"_performance_h.json", "r") as f:
        metrics_h = json.load(f)
    with open(model_1_name+"_"+model_2_name+"_performance_u.json", "r") as f:
        metrics_u = json.load(f)
    with open(model_1_name+"_"+model_2_name+"_performance_v.json", "r") as f:
        metrics_v = json.load(f)
    with open(model_1_name+"_"+model_2_name+"_performance_Umag.json", "r") as f:
        metrics_Umag = json.load(f)
    
    # Organize data by variable
    variables = [
        ("h", metrics_h),
        ("u", metrics_u),
        ("v", metrics_v),
        ("|\\mathbf{u}|", metrics_Umag)
    ]
    
    # Build LaTeX table
    latex_table = """\\begin{table}[htp]
    \\centering
    \\caption{Comparison of out-of-distribution performance metrics for SWE-DeepONet and PI-SWE-DeepONet models.}
    \\label{tab:ood-metrics-comparison}
    \\renewcommand{\\arraystretch}{1.3}
    \\begin{tabular}{lccccc}
        \\toprule
        \\multirow{2}{*}{Variable} & \\multicolumn{2}{c}{SWE-DeepONet} & \\multicolumn{2}{c}{PI-SWE-DeepONet} & \\multirow{2}{*}{Ratio Fraction} \\\\
        \\cmidrule(lr){2-3} \\cmidrule(lr){4-5}
        & Slope & Breakdown & Slope & Breakdown & \\\\
        & & Distance & & Distance & \\\\
        \\midrule
"""
    
    for var_name, metrics in variables:
        slope_1 = metrics["slope_model_1"]
        breakdown_1 = metrics["breakdown_distance_model_1"]
        slope_2 = metrics["slope_model_2"]
        breakdown_2 = metrics["breakdown_distance_model_2"]
        ratio_frac = metrics["ratio_frac_less_than_one"]
        
        latex_table += f"        ${var_name}$ & {format_number(slope_1)} & {format_number(breakdown_1)} & {format_number(slope_2)} & {format_number(breakdown_2)} & {format_number(ratio_frac, decimals=2)} \\\\\n"
    
    latex_table += """        \\bottomrule
    \\end{tabular}
    \\renewcommand{\\arraystretch}{1.0}
\\end{table}"""
    
    print(latex_table)


    #generate markdown table (same content as the latex table)
    markdown_table = """| Variable | DeepONet Slope | DeepONet Breakdown Distance | PI-DeepONet Slope | PI-DeepONet Breakdown Distance | Ratio Fraction |
|----------|:------------------:|:--------------------------------:|:---------------------:|:----------------------------------:|:--------------:|
"""
    
    for var_name, metrics in variables:
        slope_1 = metrics["slope_model_1"]
        breakdown_1 = metrics["breakdown_distance_model_1"]
        slope_2 = metrics["slope_model_2"]
        breakdown_2 = metrics["breakdown_distance_model_2"]
        ratio_frac = metrics["ratio_frac_less_than_one"]
        
        # Format variable name for markdown (handle |\mathbf{u}|)
        if var_name == "|\\mathbf{u}|":
            var_name_md = "U_{mag}"
        else:
            var_name_md = var_name
        
        markdown_table += f"| ${var_name_md}$ | {format_number(slope_1)} | {format_number(breakdown_1)} | {format_number(slope_2)} | {format_number(breakdown_2)} | {format_number(ratio_frac, decimals=2)} |\n"

    print(markdown_table)
    print("\n" + "="*80)

if __name__ == "__main__":
    main()