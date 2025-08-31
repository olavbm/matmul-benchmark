#!/usr/bin/env python3
"""
Statistical Analysis of Matrix Multiplication Benchmarks
Usage: python3 analyze_stats.py [scaling_stats.txt]
Input: CSV with columns: size,algorithm,trial,time_ns,gflops
"""

import sys
import csv
import statistics
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

def load_statistical_data(filename):
    """Load multi-trial benchmark data"""
    data = defaultdict(lambda: defaultdict(list))
    
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            size = int(row['size'])
            algo = row['algorithm']
            gflops = float(row['gflops'])
            time_ns = int(row['time_ns'])
            
            data[size][algo + '_gflops'].append(gflops)
            data[size][algo + '_time'].append(time_ns)
    
    return data

def calculate_statistics(values):
    """Calculate comprehensive statistics"""
    if not values:
        return {}
    
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    
    return {
        'count': n,
        'mean': statistics.mean(values),
        'median': statistics.median(values),
        'std': statistics.stdev(values) if n > 1 else 0,
        'min': min(values),
        'max': max(values),
        'p25': sorted_vals[n//4],
        'p75': sorted_vals[3*n//4],
        'cv': statistics.stdev(values) / statistics.mean(values) if n > 1 and statistics.mean(values) != 0 else 0
    }

def print_statistical_summary(data):
    """Print detailed statistical analysis"""
    print("Matrix Multiplication Statistical Analysis")
    print("=" * 60)
    print()
    
    sizes = sorted(data.keys())
    
    # Performance summary table
    print("Performance Statistics (GFLOP/s)")
    print("-" * 80)
    print(f"{'Size':<6} {'Algorithm':<10} {'Mean':<8} {'Std':<8} {'CV%':<6} {'Min':<8} {'Max':<8} {'Median':<8}")
    print("-" * 80)
    
    for size in sizes:
        for algo in ['naive', 'optimized']:
            key = f'{algo}_gflops'
            if key in data[size]:
                stats = calculate_statistics(data[size][key])
                print(f"{size:<6} {algo:<10} {stats['mean']:<8.3f} {stats['std']:<8.3f} "
                      f"{stats['cv']*100:<6.1f} {stats['min']:<8.3f} {stats['max']:<8.3f} {stats['median']:<8.3f}")
    
    print()
    
    # Speedup analysis with confidence
    print("Speedup Analysis (with statistical significance)")
    print("-" * 70)
    print(f"{'Size':<6} {'Mean Speedup':<12} {'Std':<8} {'95% CI':<15} {'Significant?':<12}")
    print("-" * 70)
    
    for size in sizes:
        naive_key = 'naive_gflops'
        opt_key = 'optimized_gflops'
        
        if naive_key in data[size] and opt_key in data[size]:
            naive_vals = data[size][naive_key]
            opt_vals = data[size][opt_key]
            
            # Calculate speedup for each trial pair
            speedups = [opt_vals[i] / naive_vals[i] for i in range(min(len(opt_vals), len(naive_vals)))]
            speedup_stats = calculate_statistics(speedups)
            
            # 95% confidence interval (approximate)
            margin = 1.96 * speedup_stats['std'] / (len(speedups) ** 0.5) if speedup_stats['std'] > 0 else 0
            ci_low = speedup_stats['mean'] - margin
            ci_high = speedup_stats['mean'] + margin
            
            # Statistical significance test (is speedup consistently different from 1.0?)
            significant = "Yes" if ci_high < 1.0 or ci_low > 1.0 else "No"
            
            print(f"{size:<6} {speedup_stats['mean']:<12.3f} {speedup_stats['std']:<8.3f} "
                  f"[{ci_low:.3f}, {ci_high:.3f}] {significant:<12}")

def create_distribution_plots(data, output_prefix='distribution'):
    """Create comprehensive distribution plots"""
    sizes = sorted(data.keys())
    
    # 1. Performance distribution plots (box plots)
    fig, axes = plt.subplots(2, 1, figsize=(15, 12))
    
    # Prepare data for box plots
    naive_data = []
    opt_data = []
    size_labels = []
    
    for size in sizes:
        if 'naive_gflops' in data[size] and 'optimized_gflops' in data[size]:
            naive_data.append(data[size]['naive_gflops'])
            opt_data.append(data[size]['optimized_gflops'])
            size_labels.append(f'{size}×{size}')
    
    # Box plot for performance distributions
    axes[0].boxplot(naive_data, positions=range(len(size_labels)), 
                   patch_artist=True, boxprops=dict(facecolor='lightcoral'))
    axes[0].boxplot(opt_data, positions=[x + 0.4 for x in range(len(size_labels))], 
                   patch_artist=True, boxprops=dict(facecolor='lightblue'))
    
    axes[0].set_xticks([x + 0.2 for x in range(len(size_labels))])
    axes[0].set_xticklabels(size_labels, rotation=45)
    axes[0].set_ylabel('Performance (GFLOP/s)')
    axes[0].set_title('Performance Distribution by Matrix Size')
    axes[0].legend(['Naive', 'Optimized'], loc='upper left')
    axes[0].grid(True, alpha=0.3)
    
    # Mean performance with error bars
    naive_means = [statistics.mean(vals) for vals in naive_data]
    naive_stds = [statistics.stdev(vals) if len(vals) > 1 else 0 for vals in naive_data]
    opt_means = [statistics.mean(vals) for vals in opt_data]
    opt_stds = [statistics.stdev(vals) if len(vals) > 1 else 0 for vals in opt_data]
    
    x_pos = range(len(size_labels))
    axes[1].errorbar(x_pos, naive_means, yerr=naive_stds, 
                    marker='o', capsize=5, capthick=2, color='red', label='Naive')
    axes[1].errorbar([x + 0.1 for x in x_pos], opt_means, yerr=opt_stds, 
                    marker='s', capsize=5, capthick=2, color='blue', label='Optimized')
    
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(size_labels, rotation=45)
    axes[1].set_ylabel('Performance (GFLOP/s)')
    axes[1].set_title('Mean Performance with Standard Deviation')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_performance.png', dpi=150, bbox_inches='tight')
    print(f"Generated: {output_prefix}_performance.png")
    
    # 2. Speedup distribution analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Speedup over time (trial-by-trial)
    for i, size in enumerate(sizes[::2]):  # Show every other size to avoid clutter
        if 'naive_gflops' in data[size] and 'optimized_gflops' in data[size]:
            naive_vals = data[size]['naive_gflops']
            opt_vals = data[size]['optimized_gflops']
            speedups = [opt_vals[j] / naive_vals[j] for j in range(min(len(opt_vals), len(naive_vals)))]
            
            axes[0, 0].plot(speedups, alpha=0.7, label=f'{size}×{size}')
    
    axes[0, 0].axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='No speedup')
    axes[0, 0].set_xlabel('Trial Number')
    axes[0, 0].set_ylabel('Speedup Factor')
    axes[0, 0].set_title('Speedup Stability Across Trials')
    axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Speedup histogram for largest size
    largest_size = max(sizes)
    if 'naive_gflops' in data[largest_size] and 'optimized_gflops' in data[largest_size]:
        naive_vals = data[largest_size]['naive_gflops']
        opt_vals = data[largest_size]['optimized_gflops']
        speedups = [opt_vals[j] / naive_vals[j] for j in range(min(len(opt_vals), len(naive_vals)))]
        
        axes[0, 1].hist(speedups, bins=20, alpha=0.7, color='green', edgecolor='black')
        axes[0, 1].axvline(statistics.mean(speedups), color='red', linestyle='--', 
                          label=f'Mean: {statistics.mean(speedups):.3f}')
        axes[0, 1].axvline(1.0, color='gray', linestyle='--', alpha=0.7, label='No speedup')
        axes[0, 1].set_xlabel('Speedup Factor')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title(f'Speedup Distribution ({largest_size}×{largest_size})')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Coefficient of Variation analysis
    cv_naive = []
    cv_opt = []
    for size in sizes:
        if 'naive_gflops' in data[size]:
            stats_naive = calculate_statistics(data[size]['naive_gflops'])
            cv_naive.append(stats_naive['cv'] * 100)
        if 'optimized_gflops' in data[size]:
            stats_opt = calculate_statistics(data[size]['optimized_gflops'])
            cv_opt.append(stats_opt['cv'] * 100)
    
    x_pos = range(len(size_labels))
    axes[1, 0].plot(x_pos, cv_naive, 'ro-', label='Naive', markersize=6)
    axes[1, 0].plot(x_pos, cv_opt, 'bs-', label='Optimized', markersize=6)
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(size_labels, rotation=45)
    axes[1, 0].set_ylabel('Coefficient of Variation (%)')
    axes[1, 0].set_title('Performance Variability')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Summary speedup with confidence intervals
    speedup_means = []
    speedup_cis = []
    
    for size in sizes:
        if 'naive_gflops' in data[size] and 'optimized_gflops' in data[size]:
            naive_vals = data[size]['naive_gflops']
            opt_vals = data[size]['optimized_gflops']
            speedups = [opt_vals[j] / naive_vals[j] for j in range(min(len(opt_vals), len(naive_vals)))]
            
            mean_speedup = statistics.mean(speedups)
            std_speedup = statistics.stdev(speedups) if len(speedups) > 1 else 0
            ci = 1.96 * std_speedup / (len(speedups) ** 0.5) if std_speedup > 0 else 0
            
            speedup_means.append(mean_speedup)
            speedup_cis.append(ci)
    
    axes[1, 1].errorbar(x_pos, speedup_means, yerr=speedup_cis, 
                       marker='o', capsize=5, capthick=2, color='green')
    axes[1, 1].axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='No speedup')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(size_labels, rotation=45)
    axes[1, 1].set_ylabel('Speedup Factor')
    axes[1, 1].set_title('Mean Speedup with 95% Confidence Intervals')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_analysis.png', dpi=150, bbox_inches='tight')
    print(f"Generated: {output_prefix}_analysis.png")

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_stats.py <scaling_stats.txt>")
        print("Generate data with: cargo run --release -- --scaling 50 > scaling_stats.txt")
        sys.exit(1)
    
    filename = sys.argv[1]
    
    try:
        data = load_statistical_data(filename)
        print_statistical_summary(data)
        
        print("\nGenerating distribution plots...")
        create_distribution_plots(data)
        
        print(f"\nAnalysis complete! Processed {sum(len(data[s].get('naive_gflops', [])) for s in data)} trials")
        
    except FileNotFoundError:
        print(f"Error: {filename} not found!")
        print("Generate data with: cargo run --release -- --scaling 50 > scaling_stats.txt")
        sys.exit(1)
    except ImportError as e:
        print(f"Error: Missing required Python package. Install with: pip3 install matplotlib numpy")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()