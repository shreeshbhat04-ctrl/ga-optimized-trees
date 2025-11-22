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

# Add src to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from ga_trees.ga.engine import GAEngine, GAConfig, TreeInitializer, Mutation
    from ga_trees.fitness.calculator import FitnessCalculator, TreePredictor
    from ga_trees.baselines.baseline_models import (
        CARTBaseline, PrunedCARTBaseline, RandomForestBaseline, XGBoostBaseline
    )
    GA_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import GA modules: {e}")
    GA_AVAILABLE = False


def load_dataset(name: str):
    """Load dataset by name."""
    datasets = {
        'iris': load_iris,
        'wine': load_wine,
        'breast_cancer': load_breast_cancer,
    }
    
    if name not in datasets:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(datasets.keys())}")
    
    data = datasets[name]()
    return data.data, data.target


def run_ga_cv(X, y, config, n_folds=5, random_state=42):
    """Run GA with cross-validation."""
    if not GA_AVAILABLE:
        raise RuntimeError("GA modules not available. Cannot run GA experiments.")
    
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
        
        # Create feature ranges from training data only
        feature_ranges = {i: (float(X_train[:, i].min()), float(X_train[:, i].max())) 
                         for i in range(n_features)}
        
        # Get config with defaults for missing values
        ga_config_dict = config.get('ga', {})
        tree_config_dict = config.get('tree', {})
        fitness_config_dict = config.get('fitness', {})
        
        # Filter tree config to only include valid parameters for TreeInitializer
        valid_tree_params = {'max_depth', 'min_samples_split', 'min_samples_leaf'}
        filtered_tree_config = {k: v for k, v in tree_config_dict.items() if k in valid_tree_params}
        
        ga_config = GAConfig(**ga_config_dict)
        initializer = TreeInitializer(n_features=n_features, n_classes=n_classes,
                                     **filtered_tree_config)
        fitness_calc = FitnessCalculator(**fitness_config_dict)
        mutation = Mutation(n_features=n_features, feature_ranges=feature_ranges)
        
        # Train
        start_time = time.time()
        try:
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
            
            # Handle interpretability safely
            interpretability = getattr(best_tree, 'interpretability_', 0.0)
            results['interpretability'].append(interpretability)
            results['training_time'].append(training_time)
            
        except Exception as e:
            print(f"    Error in fold {fold}: {e}")
            # Append default values for failed fold
            for key in results:
                if key != 'training_time':
                    results[key].append(0.0)
                else:
                    results[key].append(training_time)  # Use the time that passed
    
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
        print(f"  Fold {fold}/{n_folds}...")
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Standardize for consistency with GA
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Train
        start_time = time.time()
        try:
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
            
            # Get metrics safely
            try:
                metrics = model.get_metrics()
                results['depth'].append(metrics.get('depth', 0))
                results['nodes'].append(metrics.get('num_nodes', 0))
                results['leaves'].append(metrics.get('num_leaves', 0))
                results['features_used'].append(metrics.get('features_used', 0))
            except (AttributeError, KeyError):
                # If get_metrics fails, use defaults
                results['depth'].append(0)
                results['nodes'].append(0)
                results['leaves'].append(0)
                results['features_used'].append(0)
                
            results['training_time'].append(training_time)
            
        except Exception as e:
            print(f"    Error in fold {fold}: {e}")
            # Append default values for failed fold
            for key in results:
                if key != 'training_time':
                    results[key].append(0.0)
                else:
                    results[key].append(time.time() - start_time)
    
    return results


def compute_statistics(results_dict):
    """Compute mean and std for all metrics."""
    stats_dict = {}
    for model_name, results in results_dict.items():
        stats_dict[model_name] = {}
        for metric, values in results.items():
            if values and len(values) > 0 and values[0] is not None:
                # Filter out None values and ensure we have numeric data
                clean_values = [v for v in values if v is not None and np.isfinite(v)]
                if clean_values:
                    stats_dict[model_name][f'{metric}_mean'] = np.mean(clean_values)
                    stats_dict[model_name][f'{metric}_std'] = np.std(clean_values)
                else:
                    stats_dict[model_name][f'{metric}_mean'] = 0.0
                    stats_dict[model_name][f'{metric}_std'] = 0.0
    return stats_dict


def paired_ttest(results1, results2, metric='test_acc'):
    """Perform paired t-test."""
    if metric not in results1 or metric not in results2:
        return 0.0, 1.0  # Default if metric missing
    
    vals1 = [v for v in results1[metric] if v is not None and np.isfinite(v)]
    vals2 = [v for v in results2[metric] if v is not None and np.isfinite(v)]
    
    if len(vals1) != len(vals2) or len(vals1) < 2:
        return 0.0, 1.0
    
    try:
        statistic, pvalue = stats.ttest_rel(vals1, vals2)
        return statistic, pvalue
    except:
        return 0.0, 1.0


def cohens_d(results1, results2, metric='test_acc'):
    """Calculate Cohen's d effect size."""
    if metric not in results1 or metric not in results2:
        return 0.0
    
    vals1 = np.array([v for v in results1[metric] if v is not None and np.isfinite(v)])
    vals2 = np.array([v for v in results2[metric] if v is not None and np.isfinite(v)])
    
    if len(vals1) == 0 or len(vals2) == 0:
        return 0.0
    
    pooled_std = np.sqrt(((len(vals1)-1)*np.var(vals1) + 
                          (len(vals2)-1)*np.var(vals2)) / 
                         (len(vals1) + len(vals2) - 2))
    
    if pooled_std == 0:
        return 0.0
    
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
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file {args.config} not found.")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing config file: {e}")
        sys.exit(1)
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Experiment metadata
    experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n{'='*70}")
    print(f"Starting Experiment: {experiment_id}")
    print(f"{'='*70}\n")
    
    all_results = {}
    
    # Get datasets from config with fallback
    datasets = config.get('experiment', {}).get('datasets', ['iris', 'wine', 'breast_cancer'])
    
    # Run experiments on each dataset
    for dataset_name in datasets:
        print(f"\n{'='*70}")
        print(f"Dataset: {dataset_name.upper()}")
        print(f"{'='*70}")
        
        try:
            X, y = load_dataset(dataset_name)
            print(f"Shape: {X.shape}, Classes: {len(np.unique(y))}")
            
            dataset_results = {}
            
            # Run GA if available
            if GA_AVAILABLE:
                print("\n1. Running GA-Optimized Tree...")
                try:
                    ga_results = run_ga_cv(X, y, config, 
                                          n_folds=config.get('experiment', {}).get('cv_folds', 5),
                                          random_state=config.get('experiment', {}).get('random_state', 42))
                    dataset_results['GA-Optimized'] = ga_results
                except Exception as e:
                    print(f"Error running GA: {e}")
                    # Create empty results for GA
                    dataset_results['GA-Optimized'] = {
                        'train_acc': [0.0], 'test_acc': [0.0], 'train_f1': [0.0], 'test_f1': [0.0],
                        'depth': [0], 'nodes': [0], 'leaves': [0], 'features_used': [0],
                        'interpretability': [0.0], 'training_time': [0.0]
                    }
            else:
                print("\nSkipping GA (modules not available)")
            
            # Run baselines - use only parameters that the baseline classes actually accept
            baseline_configs = {
                'CART': (CARTBaseline, {'max_depth': config.get('tree', {}).get('max_depth', 5)}),
                'Pruned CART': (PrunedCARTBaseline, {'max_depth': config.get('tree', {}).get('max_depth', 5)}),
                'Random Forest': (RandomForestBaseline, {
                    'max_depth': config.get('tree', {}).get('max_depth', 5),
                    # Remove n_jobs if baseline doesn't accept it
                }),
            }
            
            # Add XGBoost if specified in config
            if config.get('experiment', {}).get('baselines', []):
                if 'xgboost' in config['experiment']['baselines']:
                    baseline_configs['XGBoost'] = (XGBoostBaseline, {
                        'max_depth': config.get('tree', {}).get('max_depth', 5),
                        # Remove n_jobs if baseline doesn't accept it
                    })
            
            for name, (model_class, kwargs) in baseline_configs.items():
                print(f"\n2. Running {name}...")
                try:
                    results = run_baseline_cv(model_class, X, y, kwargs,
                                             n_folds=config.get('experiment', {}).get('cv_folds', 5),
                                             random_state=config.get('experiment', {}).get('random_state', 42))
                    dataset_results[name] = results
                except Exception as e:
                    print(f"Error running {name}: {e}")
                    # Try with minimal parameters
                    try:
                        print(f"  Retrying {name} with minimal parameters...")
                        minimal_kwargs = {'max_depth': config.get('tree', {}).get('max_depth', 5)}
                        results = run_baseline_cv(model_class, X, y, minimal_kwargs,
                                                 n_folds=config.get('experiment', {}).get('cv_folds', 5),
                                                 random_state=config.get('experiment', {}).get('random_state', 42))
                        dataset_results[name] = results
                    except Exception as e2:
                        print(f"  Failed again: {e2}")
                        # Create empty results for failed baseline
                        dataset_results[name] = {
                            'train_acc': [0.0], 'test_acc': [0.0], 'train_f1': [0.0], 'test_f1': [0.0],
                            'depth': [0], 'nodes': [0], 'leaves': [0], 'features_used': [0],
                            'training_time': [0.0]
                        }
            
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
                
        except Exception as e:
            print(f"Error processing dataset {dataset_name}: {e}")
            continue
    
    if not all_results:
        print("No results collected. Exiting.")
        return
    
    # Aggregate results across datasets
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}\n")
    
    # Create summary table
    summary_data = []
    for dataset_name, dataset_results in all_results.items():
        for model_name, results in dataset_results.items():
            # Handle case where all results are zeros (failed experiment)
            test_acc = results['test_acc']
            if any(acc > 0 for acc in test_acc):
                mean_acc = np.mean([acc for acc in test_acc if acc > 0])
                std_acc = np.std([acc for acc in test_acc if acc > 0])
            else:
                mean_acc, std_acc = 0.0, 0.0
                
            summary_data.append({
                'Dataset': dataset_name,
                'Model': model_name,
                'Test Accuracy': f"{mean_acc:.4f} ± {std_acc:.4f}",
                'Test F1': f"{np.mean(results['test_f1']):.4f} ± {np.std(results['test_f1']):.4f}",
                'Depth': f"{np.mean(results['depth']):.1f} ± {np.std(results['depth']):.1f}",
                'Nodes': f"{np.mean(results['nodes']):.1f} ± {np.std(results['nodes']):.1f}",
                'Training Time (s)': f"{np.mean(results['training_time']):.2f}",
            })
    
    df = pd.DataFrame(summary_data)
    print(df.to_string(index=False))
    
    # Save results
    df.to_csv(output_dir / f'summary_{experiment_id}.csv', index=False)
    
    # Save raw results with proper serialization
    serializable_results = {}
    for dataset, models in all_results.items():
        serializable_results[dataset] = {}
        for model, metrics in models.items():
            serializable_results[dataset][model] = {}
            for metric, values in metrics.items():
                serializable_results[dataset][model][metric] = [float(v) if np.isfinite(v) else 0.0 for v in values]
    
    with open(output_dir / f'raw_results_{experiment_id}.json', 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    # Statistical tests
    print(f"\n{'='*70}")
    print("Statistical Significance Tests")
    print(f"{'='*70}\n")
    
    for dataset_name, dataset_results in all_results.items():
        print(f"\n{dataset_name}:")
        if 'GA-Optimized' in dataset_results:
            ga_results = dataset_results['GA-Optimized']
            
            for baseline_name in ['CART', 'Pruned CART', 'Random Forest', 'XGBoost']:
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
    
    try:
        # Accuracy comparison plot
        n_datasets = len(all_results)
        if n_datasets > 0:
            fig, axes = plt.subplots(1, n_datasets, figsize=(5*n_datasets, 5))
            if n_datasets == 1:
                axes = [axes]
            
            for idx, (dataset_name, dataset_results) in enumerate(all_results.items()):
                ax = axes[idx]
                
                models = list(dataset_results.keys())
                accuracies = []
                stds = []
                
                for model in models:
                    test_acc = dataset_results[model]['test_acc']
                    # Filter out failed runs
                    valid_acc = [acc for acc in test_acc if acc > 0]
                    if valid_acc:
                        accuracies.append(np.mean(valid_acc))
                        stds.append(np.std(valid_acc))
                    else:
                        accuracies.append(0.0)
                        stds.append(0.0)
                
                colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'][:len(models)]
                bars = ax.bar(range(len(models)), accuracies, yerr=stds, capsize=5, color=colors)
                ax.set_xticks(range(len(models)))
                ax.set_xticklabels(models, rotation=45, ha='right')
                ax.set_ylabel('Test Accuracy')
                ax.set_title(f'{dataset_name.replace("_", " ").title()}')
                ax.set_ylim([0.0, 1.0])
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / f'accuracy_comparison_{experiment_id}.png', 
                        dpi=300, bbox_inches='tight')
            print(f"✓ Saved: accuracy_comparison_{experiment_id}.png")
        else:
            print("No data for visualization")
            
    except Exception as e:
        print(f"Error creating visualization: {e}")
    
    print(f"\n{'='*70}")
    print(f"Experiment Complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()