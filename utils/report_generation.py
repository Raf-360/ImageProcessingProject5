"""
HTML and PDF report generation utilities.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import base64
from io import BytesIO


def fig_to_base64(fig) -> str:
    """Convert matplotlib figure to base64 string for embedding in HTML."""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return img_str


def generate_html_report(results_df: pd.DataFrame, output_path: str,
                         include_plots: bool = True, 
                         project_name: str = "Image Denoising Benchmark"):
    """
    Generate comprehensive HTML report from evaluation results.
    
    Args:
        results_df: DataFrame with evaluation results
        output_path: Path to save HTML report
        include_plots: Whether to include visualization plots
        project_name: Title for the report
    """
    from utils.dataset_plots import (plot_metrics_distribution, plot_method_comparison,
                                     plot_psnr_ssim_scatter, plot_summary_statistics)
    
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{project_name} - Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .header h1 {{
            margin: 0 0 10px 0;
        }}
        .header p {{
            margin: 5px 0;
            opacity: 0.9;
        }}
        .section {{
            background: white;
            padding: 25px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h2 {{
            color: #667eea;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
            margin-top: 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th {{
            background-color: #667eea;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }}
        td {{
            padding: 10px;
            border-bottom: 1px solid #ddd;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .metric-card {{
            display: inline-block;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            margin: 10px;
            border-radius: 8px;
            min-width: 200px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .metric-card h3 {{
            margin: 0 0 10px 0;
            font-size: 16px;
            opacity: 0.9;
        }}
        .metric-card .value {{
            font-size: 32px;
            font-weight: bold;
            margin: 0;
        }}
        .plot-container {{
            margin: 20px 0;
            text-align: center;
        }}
        .plot-container img {{
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .footer {{
            text-align: center;
            color: #666;
            margin-top: 40px;
            padding: 20px;
            border-top: 1px solid #ddd;
        }}
        .best-method {{
            background-color: #d4edda;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{project_name}</h1>
        <p>Comprehensive Evaluation Report</p>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Total Images: {len(results_df['Image'].unique()) if 'Image' in results_df.columns else len(results_df) // len(results_df['Method'].unique())}</p>
        <p>Methods Evaluated: {', '.join(results_df['Method'].unique())}</p>
    </div>
"""
    
    # Summary Statistics
    stats = results_df.groupby('Method').agg({
        'PSNR': ['mean', 'std', 'min', 'max'],
        'SSIM': ['mean', 'std', 'min', 'max']
    }).round(3)
    
    if 'Time' in results_df.columns:
        time_stats = results_df.groupby('Method')['Time'].mean().round(3)
    
    html_content += """
    <div class="section">
        <h2>📊 Summary Statistics</h2>
        <div style="text-align: center;">
"""
    
    # Best method cards
    best_psnr_method = stats['PSNR']['mean'].idxmax()
    best_ssim_method = stats['SSIM']['mean'].idxmax()
    
    html_content += f"""
            <div class="metric-card">
                <h3>Best PSNR</h3>
                <p class="value">{stats.loc[best_psnr_method, ('PSNR', 'mean')]:.2f} dB</p>
                <p>{best_psnr_method.upper()}</p>
            </div>
            <div class="metric-card">
                <h3>Best SSIM</h3>
                <p class="value">{stats.loc[best_ssim_method, ('SSIM', 'mean')]:.4f}</p>
                <p>{best_ssim_method.upper()}</p>
            </div>
"""
    
    if 'Time' in results_df.columns:
        fastest_method = time_stats.idxmin()
        html_content += f"""
            <div class="metric-card">
                <h3>Fastest Method</h3>
                <p class="value">{time_stats[fastest_method]:.3f}s</p>
                <p>{fastest_method.upper()}</p>
            </div>
"""
    
    html_content += """
        </div>
    </div>
"""
    
    # Detailed Results Table
    html_content += """
    <div class="section">
        <h2>📋 Detailed Results</h2>
        <table>
            <thead>
                <tr>
                    <th>Method</th>
                    <th>Mean PSNR (dB)</th>
                    <th>Std PSNR</th>
                    <th>Mean SSIM</th>
                    <th>Std SSIM</th>
"""
    
    if 'Time' in results_df.columns:
        html_content += "                    <th>Avg Time (s)</th>\n"
    
    html_content += """
                </tr>
            </thead>
            <tbody>
"""
    
    for method in stats.index:
        is_best_psnr = (method == best_psnr_method)
        is_best_ssim = (method == best_ssim_method)
        row_class = 'best-method' if (is_best_psnr or is_best_ssim) else ''
        
        html_content += f"""
                <tr class="{row_class}">
                    <td><strong>{method.upper()}</strong></td>
                    <td>{stats.loc[method, ('PSNR', 'mean')]:.2f}</td>
                    <td>{stats.loc[method, ('PSNR', 'std')]:.2f}</td>
                    <td>{stats.loc[method, ('SSIM', 'mean')]:.4f}</td>
                    <td>{stats.loc[method, ('SSIM', 'std')]:.4f}</td>
"""
        
        if 'Time' in results_df.columns:
            html_content += f"                    <td>{time_stats[method]:.3f}</td>\n"
        
        html_content += "                </tr>\n"
    
    html_content += """
            </tbody>
        </table>
    </div>
"""
    
    # Include plots if requested
    if include_plots:
        html_content += """
    <div class="section">
        <h2>📈 Visualizations</h2>
"""
        
        # Generate and embed plots
        plot_functions = [
            (plot_method_comparison, "Method Comparison"),
            (plot_metrics_distribution, "Metrics Distribution"),
            (plot_psnr_ssim_scatter, "PSNR vs SSIM"),
            (plot_summary_statistics, "Summary Statistics")
        ]
        
        for plot_func, title in plot_functions:
            try:
                fig = plt.figure(figsize=(12, 6))
                plot_func(results_df, save_path=None)
                img_str = fig_to_base64(fig)
                plt.close(fig)
                
                html_content += f"""
        <div class="plot-container">
            <h3>{title}</h3>
            <img src="data:image/png;base64,{img_str}" alt="{title}">
        </div>
"""
            except Exception as e:
                html_content += f"""
        <div class="plot-container">
            <p style="color: red;">Error generating {title}: {str(e)}</p>
        </div>
"""
        
        html_content += """
    </div>
"""
    
    # Footer
    html_content += """
    <div class="footer">
        <p>Generated by Image Denoising Benchmarking Framework</p>
        <p>Traditional methods: Gaussian, Median, Bilateral, NLM, Wiener</p>
    </div>
</body>
</html>
"""
    
    # Save HTML file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ HTML report generated: {output_path}")
    return output_path


def generate_pdf_report(results_df: pd.DataFrame, output_path: str,
                       project_name: str = "Image Denoising Benchmark"):
    """
    Generate PDF report from evaluation results.
    
    Args:
        results_df: DataFrame with evaluation results
        output_path: Path to save PDF report
        project_name: Title for the report
    """
    try:
        from matplotlib.backends.backend_pdf import PdfPages
        from utils.dataset_plots import (plot_method_comparison, plot_metrics_distribution,
                                         plot_psnr_ssim_scatter, plot_summary_statistics)
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with PdfPages(output_path) as pdf:
            # Title page
            fig = plt.figure(figsize=(11, 8.5))
            fig.text(0.5, 0.7, project_name, ha='center', fontsize=24, fontweight='bold')
            fig.text(0.5, 0.6, 'Comprehensive Evaluation Report', ha='center', fontsize=16)
            fig.text(0.5, 0.5, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                    ha='center', fontsize=12)
            fig.text(0.5, 0.45, f"Methods: {', '.join(results_df['Method'].unique())}", 
                    ha='center', fontsize=11)
            fig.text(0.5, 0.4, f"Total Images: {len(results_df['Image'].unique()) if 'Image' in results_df.columns else 'N/A'}", 
                    ha='center', fontsize=11)
            plt.axis('off')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Statistics summary page
            stats = results_df.groupby('Method').agg({
                'PSNR': ['mean', 'std', 'min', 'max'],
                'SSIM': ['mean', 'std', 'min', 'max']
            }).round(3)
            
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.axis('tight')
            ax.axis('off')
            
            table_data = []
            table_data.append(['Method', 'Mean PSNR', 'Std PSNR', 'Mean SSIM', 'Std SSIM'])
            
            for method in stats.index:
                row = [
                    method.upper(),
                    f"{stats.loc[method, ('PSNR', 'mean')]:.2f}",
                    f"{stats.loc[method, ('PSNR', 'std')]:.2f}",
                    f"{stats.loc[method, ('SSIM', 'mean')]:.4f}",
                    f"{stats.loc[method, ('SSIM', 'std')]:.4f}"
                ]
                table_data.append(row)
            
            table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                           colWidths=[0.2, 0.2, 0.2, 0.2, 0.2])
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
            
            # Style header row
            for i in range(5):
                table[(0, i)].set_facecolor('#667eea')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            ax.set_title('Summary Statistics', fontsize=16, fontweight='bold', pad=20)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Add plots with proper error handling
            plot_functions = [
                (plot_metrics_distribution, "Metrics Distribution"),
                (plot_method_comparison, "Method Comparison"),
                (plot_psnr_ssim_scatter, "PSNR vs SSIM Trade-offs"),
                (plot_summary_statistics, "Detailed Statistics")
            ]
            
            for plot_func, title in plot_functions:
                try:
                    print(f"  Generating {title}...")
                    fig = plot_func(results_df, save_path=None)
                    if fig is not None:
                        pdf.savefig(fig, bbox_inches='tight')
                        plt.close(fig)
                        print(f"  ✓ {title} added")
                    else:
                        print(f"  ⚠️  {title} returned no figure")
                except Exception as e:
                    print(f"  ⚠️  Could not generate {title}: {e}")
                    # Create placeholder page
                    fig = plt.figure(figsize=(11, 8.5))
                    plt.text(0.5, 0.5, f"Error generating {title}\n{str(e)}",
                            ha='center', va='center', fontsize=12, color='red')
                    plt.axis('off')
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)
            
            # Metadata
            d = pdf.infodict()
            d['Title'] = project_name
            d['Author'] = 'Image Denoising Framework'
            d['Subject'] = 'Denoising Method Evaluation'
            d['Keywords'] = 'Image Denoising, PSNR, SSIM, Benchmark'
            d['CreationDate'] = datetime.now()
        
        print(f"✅ PDF report generated: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"❌ Error generating PDF report: {e}")
        print("   Note: PDF generation requires matplotlib with PDF backend support")
        return None
