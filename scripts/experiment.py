"""
Full experiment runner with baselines, cross-validation, and statistics.

Usage:
    python scripts/experiment.py --config configs/default.yaml --n-jobs 8
"""

import argparse
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import time
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

from ga_trees.ga.engine import GAEngine, GAConfig, TreeInitializer, Mutation
from ga_trees.fitness.calculator import FitnessCalculator, TreePredictor
from ga_trees.baselines.baseline_models import (
    CARTBaseline, PrunedCARTBaseline, RandomForestBaseline, XGBoostBaseline
)


def load_dataset(name: str):
    """Load dataset by name."""
    datasets = {
        'iris': load_iris,
        'wine': load_wine,
        'breast_cancer': load_breast_cancer,
    }
    
    if name not in datasets:
        raise ValueError(f"Unknown dataset: {name}")
    
    return datasets[name](return_X_y=True)


def run_ga_cv(X, y, config, n_folds=5, random_state=42):
    """Run GA with cross-validation."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    results = {
        'train_acc': [],
        'test_acc': [],
        'train_f1': [],
        'test_f1': [],
        'depth': [],
        'nodes': [],
        'leaves': [],
        'features_used': [],
        'interpretability': [],
        'training_time': []
    }
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f"  Fold {fold}/{n_folds}...")
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Standardize
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Setup GA
        n_features = X_train.shape[1]
        n_classes = len(np.unique(y))
        feature_ranges = {i: (X_train[:, i].min(), X_train[:, i].max()) 
                         for i in range(n_features)}
        
        ga_config = GAConfig(**config['ga'])
        initializer = TreeInitializer(n_features=n_features, n_classes=n_classes,
                                     **config['tree'])
        fitness_calc = FitnessCalculator(**config['fitness'])
        mutation = Mutation(n_features=n_features, feature_ranges=feature_ranges)
        
        # Train
        start_time = time.time()
        ga_engine = GAEngine(ga_config, initializer, 
                           fitness_calc.calculate_fitness, mutation)
        best_tree = ga_engine.evolve(X_train, y_train, verbose=False)
        training_time = time.time() - start_time
        
        # Evaluate
        predictor = TreePredictor()
        y_train_pred = predictor.predict(best_tree, X_train)
        y_test_pred = predictor.predict(best_tree, X_test)
        
        results['train_acc'].append(accuracy_score(y_train, y_train_pred))
        results['test_acc'].append(accuracy_score(y_test, y_test_pred))
        results['train_f1'].append(f1_score(y_train, y_train_pred, average='weighted'))
        results['test_f1'].append(f1_score(y_test, y_test_pred, average='weighted'))
        results['depth'].append(best_tree.get_depth())
        results['nodes'].append(best_tree.get_num_nodes())
        results['leaves'].append(best_tree.get_num_leaves())
        results['features_used'].append(best_tree.get_num_features_used())
        results['interpretability'].append(best_tree.interpretability_)
        results['training_time'].append(training_time)
    
    return results


def run_baseline_cv(model_class, X, y, model_kwargs, n_folds=5, random_state=42):
    """Run baseline with cross-validation."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    results = {
        'train_acc': [],
        'test_acc': [],
        'train_f1': [],
        'test_f1': [],
        'depth': [],
        'nodes': [],
        'leaves': [],
        'features_used': [],
        'training_time': []
    }
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Train
        start_time = time.time()
        model = model_class(**model_kwargs)
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Evaluate
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        results['train_acc'].append(accuracy_score(y_train, y_train_pred))
        results['test_acc'].append(accuracy_score(y_test, y_test_pred))
        results['train_f1'].append(f1_score(y_train, y_train_pred, average='weighted'))
        results['test_f1'].append(f1_score(y_test, y_test_pred, average='weighted'))
        
        metrics = model.get_metrics()
        results['depth'].append(metrics['depth'])
        results['nodes'].append(metrics['num_nodes'])
        results['leaves'].append(metrics['num_leaves'])
        results['features_used'].append(metrics['features_used'])
        results['training_time'].append(training_time)
    
    return results


def compute_statistics(results_dict):
    """Compute mean and std for all metrics."""
    stats_dict = {}
    for model_name, results in results_dict.items():
        stats_dict[model_name] = {}
        for metric, values in results.items():
            if values and values[0] is not None:
                stats_dict[model_name][f'{metric}_mean'] = np.mean(values)
                stats_dict[model_name][f'{metric}_std'] = np.std(values)
    return stats_dict


def paired_ttest(results1, results2, metric='test_acc'):
    """Perform paired t-test."""
    vals1 = results1[metric]
    vals2 = results2[metric]
    statistic, pvalue = stats.ttest_rel(vals1, vals2)
    return statistic, pvalue


def cohens_d(results1, results2, metric='test_acc'):
    """Calculate Cohen's d effect size."""
    vals1 = np.array(results1[metric])
    vals2 = np.array(results2[metric])
    
    pooled_std = np.sqrt(((len(vals1)-1)*np.var(vals1) + 
                          (len(vals2)-1)*np.var(vals2)) / 
                         (len(vals1) + len(vals2) - 2))
    
    return (np.mean(vals1) - np.mean(vals2)) / pooled_std


def main():
    parser = argparse.ArgumentParser(description='Run full experiments')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Config file')
    parser.add_argument('--output', type=str, default='results/',
                       help='Output directory')
    parser.add_argument('--n-jobs', type=int, default=8,
                       help='Number of parallel jobs')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Experiment metadata
    experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n{'='*70}")
    print(f"Starting Experiment: {experiment_id}")
    print(f"{'='*70}\n")
    
    all_results = {}
    
    # Run experiments on each dataset
    for dataset_name in config['experiment']['datasets']:
        print(f"\n{'='*70}")
        print(f"Dataset: {dataset_name.upper()}")
        print(f"{'='*70}")
        
        X, y = load_dataset(dataset_name)
        print(f"Shape: {X.shape}, Classes: {len(np.unique(y))}")
        
        dataset_results = {}
        
        # Run GA
        print("\n1. Running GA-Optimized Tree...")
        ga_results = run_ga_cv(X, y, config, 
                              n_folds=config['experiment']['cv_folds'],
                              random_state=config['experiment']['random_state'])
        dataset_results['GA-Optimized'] = ga_results
        
        # Run baselines
        baselines = {
            'CART': (CARTBaseline, {'max_depth': config['tree']['max_depth']}),
            'Pruned CART': (PrunedCARTBaseline, {'max_depth': config['tree']['max_depth']}),
            'Random Forest': (RandomForestBaseline, {'max_depth': config['tree']['max_depth']}),
        }
        
        if 'xgboost' in config['experiment']['baselines']:
            baselines['XGBoost'] = (XGBoostBaseline, {'max_depth': config['tree']['max_depth']})
        
        for name, (model_class, kwargs) in baselines.items():
            print(f"\n2. Running {name}...")
            results = run_baseline_cv(model_class, X, y, kwargs,
                                     n_folds=config['experiment']['cv_folds'],
                                     random_state=config['experiment']['random_state'])
            dataset_results[name] = results
        
        all_results[dataset_name] = dataset_results
        
        # Print summary for this dataset
        print(f"\n{'-'*70}")
        print(f"Results Summary for {dataset_name}")
        print(f"{'-'*70}")
        stats_dict = compute_statistics(dataset_results)
        for model_name, stats in stats_dict.items():
            acc_mean = stats.get('test_acc_mean', 0)
            acc_std = stats.get('test_acc_std', 0)
            nodes_mean = stats.get('nodes_mean', 0)
            print(f"{model_name:20s}: Acc={acc_mean:.4f}±{acc_std:.4f}, Nodes={nodes_mean:.1f}")
    
    # Aggregate results across datasets
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}\n")
    
    # Create summary table
    summary_data = []
    for dataset_name, dataset_results in all_results.items():
        for model_name, results in dataset_results.items():
            summary_data.append({
                'Dataset': dataset_name,
                'Model': model_name,
                'Test Accuracy': f"{np.mean(results['test_acc']):.4f} ± {np.std(results['test_acc']):.4f}",
                'Test F1': f"{np.mean(results['test_f1']):.4f} ± {np.std(results['test_f1']):.4f}",
                'Depth': f"{np.mean(results['depth']):.1f} ± {np.std(results['depth']):.1f}",
                'Nodes': f"{np.mean(results['nodes']):.1f} ± {np.std(results['nodes']):.1f}",
                'Training Time (s)': f"{np.mean(results['training_time']):.2f}",
            })
    
    df = pd.DataFrame(summary_data)
    print(df.to_string(index=False))
    
    # Save results
    df.to_csv(output_dir / f'summary_{experiment_id}.csv', index=False)
    
    with open(output_dir / f'raw_results_{experiment_id}.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Statistical tests
    print(f"\n{'='*70}")
    print("Statistical Significance Tests")
    print(f"{'='*70}\n")
    
    for dataset_name, dataset_results in all_results.items():
        print(f"\n{dataset_name}:")
        ga_results = dataset_results['GA-Optimized']
        
        for baseline_name in ['CART', 'Pruned CART', 'Random Forest']:
            if baseline_name in dataset_results:
                baseline_results = dataset_results[baseline_name]
                t_stat, p_value = paired_ttest(ga_results, baseline_results)
                effect_size = cohens_d(ga_results, baseline_results)
                
                print(f"  GA vs {baseline_name:15s}: t={t_stat:6.3f}, "
                      f"p={p_value:.4f}, d={effect_size:.3f}")
    
    # Create visualizations
    print(f"\n{'='*70}")
    print("Creating Visualizations...")
    print(f"{'='*70}\n")
    
    # Accuracy comparison plot
    fig, axes = plt.subplots(1, len(all_results), figsize=(5*len(all_results), 5))
    if len(all_results) == 1:
        axes = [axes]
    
    for idx, (dataset_name, dataset_results) in enumerate(all_results.items()):
        ax = axes[idx]
        
        models = list(dataset_results.keys())
        accuracies = [np.mean(dataset_results[m]['test_acc']) for m in models]
        stds = [np.std(dataset_results[m]['test_acc']) for m in models]
        
        bars = ax.bar(range(len(models)), accuracies, yerr=stds, capsize=5,
                      color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylabel('Test Accuracy')
        ax.set_title(f'{dataset_name.title()}')
        ax.set_ylim([0.7, 1.0])
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'accuracy_comparison_{experiment_id}.png', 
                dpi=300, bbox_inches='tight')
    print(f"✓ Saved: accuracy_comparison_{experiment_id}.png")
    
    print(f"\n{'='*70}")
    print(f"Experiment Complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()