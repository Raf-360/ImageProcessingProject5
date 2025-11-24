"""
Dataset-wide plotting and analysis utilities.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Optional
import seaborn as sns


def plot_metrics_distribution(results_df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Plot distribution of metrics (PSNR, SSIM) across all images.
    
    Args:
        results_df: DataFrame with columns ['Method', 'PSNR', 'SSIM', ...]
        save_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # PSNR distribution
    methods = results_df['Method'].unique()
    for method in methods:
        method_data = results_df[results_df['Method'] == method]['PSNR']
        axes[0].hist(method_data, alpha=0.6, label=method, bins=15)
    
    axes[0].set_xlabel('PSNR (dB)', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title('PSNR Distribution Across Dataset', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # SSIM distribution
    for method in methods:
        method_data = results_df[results_df['Method'] == method]['SSIM']
        axes[1].hist(method_data, alpha=0.6, label=method, bins=15)
    
    axes[1].set_xlabel('SSIM', fontsize=11)
    axes[1].set_ylabel('Frequency', fontsize=11)
    axes[1].set_title('SSIM Distribution Across Dataset', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved metrics distribution to {save_path}")
    
    return fig


def plot_method_comparison(results_df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Create box plots comparing methods across all images.
    
    Args:
        results_df: DataFrame with columns ['Method', 'PSNR', 'SSIM', 'Time']
        save_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # PSNR comparison
    sns.boxplot(data=results_df, x='Method', y='PSNR', hue='Method', ax=axes[0], palette='Set2', legend=False)
    axes[0].set_title('PSNR Comparison', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('PSNR (dB)', fontsize=11)
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # SSIM comparison
    sns.boxplot(data=results_df, x='Method', y='SSIM', hue='Method', ax=axes[1], palette='Set2', legend=False)
    axes[1].set_title('SSIM Comparison', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('SSIM', fontsize=11)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Time comparison
    if 'Time' in results_df.columns:
        sns.boxplot(data=results_df, x='Method', y='Time', hue='Method', ax=axes[2], palette='Set2', legend=False)
        axes[2].set_title('Processing Time Comparison', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('Time (seconds)', fontsize=11)
        axes[2].tick_params(axis='x', rotation=45)
        axes[2].grid(True, alpha=0.3, axis='y')
        axes[2].set_yscale('log')  # Log scale for time
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved method comparison to {save_path}")
    
    return fig


def plot_psnr_ssim_scatter(results_df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Plot PSNR vs SSIM scatter plot for all methods.
    
    Args:
        results_df: DataFrame with columns ['Method', 'PSNR', 'SSIM']
        save_path: Optional path to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    methods = results_df['Method'].unique()
    colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(methods)))
    
    for method, color in zip(methods, colors):
        method_data = results_df[results_df['Method'] == method]
        ax.scatter(method_data['PSNR'], method_data['SSIM'], 
                  label=method, alpha=0.7, s=100, c=[color], edgecolors='black')
    
    ax.set_xlabel('PSNR (dB)', fontsize=12, fontweight='bold')
    ax.set_ylabel('SSIM', fontsize=12, fontweight='bold')
    ax.set_title('PSNR vs SSIM: Quality Trade-offs', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved PSNR vs SSIM scatter to {save_path}")
    
    return fig


def plot_per_image_performance(results_df: pd.DataFrame, metric: str = 'PSNR',
                               save_path: Optional[str] = None):
    """
    Plot per-image performance for all methods.
    
    Args:
        results_df: DataFrame with columns ['Method', 'Image', 'PSNR', 'SSIM']
        metric: Metric to plot ('PSNR' or 'SSIM')
        save_path: Optional path to save the plot
    """
    if 'Image' not in results_df.columns:
        print("Warning: 'Image' column not found in results_df")
        return
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    methods = results_df['Method'].unique()
    images = sorted(results_df['Image'].unique())
    
    for method in methods:
        method_data = results_df[results_df['Method'] == method]
        values = [method_data[method_data['Image'] == img][metric].values[0] 
                 if len(method_data[method_data['Image'] == img]) > 0 else 0
                 for img in images]
        ax.plot(images, values, marker='o', label=method, linewidth=2, markersize=8)
    
    ax.set_xlabel('Image Index', fontsize=11, fontweight='bold')
    ax.set_ylabel(f'{metric} {"(dB)" if metric == "PSNR" else ""}', fontsize=11, fontweight='bold')
    ax.set_title(f'{metric} Performance Across Dataset', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved per-image {metric} performance to {save_path}")
    
    return fig


def plot_summary_statistics(results_df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Create a comprehensive summary with mean, std, min, max for each method.
    
    Args:
        results_df: DataFrame with columns ['Method', 'PSNR', 'SSIM', 'Time']
        save_path: Optional path to save the plot
    """
    # Calculate statistics
    stats = results_df.groupby('Method').agg({
        'PSNR': ['mean', 'std', 'min', 'max'],
        'SSIM': ['mean', 'std', 'min', 'max']
    }).round(3)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    methods = stats.index
    psnr_means = stats['PSNR']['mean'].values
    psnr_stds = stats['PSNR']['std'].values
    ssim_means = stats['SSIM']['mean'].values
    ssim_stds = stats['SSIM']['std'].values
    
    x = np.arange(len(methods))
    width = 0.6
    
    # PSNR mean with error bars
    axes[0, 0].bar(x, psnr_means, width, yerr=psnr_stds, capsize=5, 
                   alpha=0.8, color='skyblue', edgecolor='black')
    axes[0, 0].set_ylabel('PSNR (dB)', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('Mean PSNR with Std Dev', fontsize=12, fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(methods, rotation=45, ha='right')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # SSIM mean with error bars
    axes[0, 1].bar(x, ssim_means, width, yerr=ssim_stds, capsize=5,
                   alpha=0.8, color='lightcoral', edgecolor='black')
    axes[0, 1].set_ylabel('SSIM', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('Mean SSIM with Std Dev', fontsize=12, fontweight='bold')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(methods, rotation=45, ha='right')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # PSNR range (min-max)
    psnr_mins = stats['PSNR']['min'].values
    psnr_maxs = stats['PSNR']['max'].values
    for i, method in enumerate(methods):
        axes[1, 0].plot([i, i], [psnr_mins[i], psnr_maxs[i]], 'o-', linewidth=3, markersize=8)
    axes[1, 0].set_ylabel('PSNR (dB)', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('PSNR Range (Min-Max)', fontsize=12, fontweight='bold')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(methods, rotation=45, ha='right')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # SSIM range (min-max)
    ssim_mins = stats['SSIM']['min'].values
    ssim_maxs = stats['SSIM']['max'].values
    for i, method in enumerate(methods):
        axes[1, 1].plot([i, i], [ssim_mins[i], ssim_maxs[i]], 'o-', linewidth=3, markersize=8)
    axes[1, 1].set_ylabel('SSIM', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('SSIM Range (Min-Max)', fontsize=12, fontweight='bold')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(methods, rotation=45, ha='right')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved summary statistics to {save_path}")
    
    return fig
    
    # Print statistics table
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(stats.to_string())
    print("=" * 80 + "\n")
