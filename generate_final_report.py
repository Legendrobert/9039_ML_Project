import os
import json
import matplotlib.pyplot as plt
from datetime import datetime

def generate_final_report(base_dir):
    # 确保summary目录存在
    summary_dir = os.path.join(base_dir, 'summary')
    os.makedirs(summary_dir, exist_ok=True)
    
    # 收集所有数据集的结果
    all_results = {}
    
    # 获取所有子目录（每个子目录对应一个数据集）
    dataset_dirs = [d for d in os.listdir(base_dir) 
                    if os.path.isdir(os.path.join(base_dir, d)) and d != 'summary']
    
    print(f"Found {len(dataset_dirs)} dataset directories")
    
    for dataset_dir in dataset_dirs:
        # 将目录名转回数据集名称
        dataset_name = dataset_dir.replace('_', ' ').title()
        print(f"\nProcessing dataset: {dataset_name}")
        
        # 尝试读取该数据集的metrics文件
        metrics_path = os.path.join(base_dir, dataset_dir, 'all_metrics.json')
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, 'r') as f:
                    models_metrics = json.load(f)
                
                print(f"  Found metrics file with {len(models_metrics)} models")
                
                # 创建每个模型的结果字典
                dataset_results = {}
                for model_name, model_metrics in models_metrics.items():
                    dataset_results[model_name] = {
                        'avg_rmse': model_metrics.get('avg_rmse', 0),
                        'avg_mae': model_metrics.get('avg_mae', 0),
                        'avg_r2': model_metrics.get('avg_r2', 0),
                        'details': model_metrics.get('details', [])
                    }
                
                all_results[dataset_name] = dataset_results
                
                # 绘制该数据集下所有模型的比较图
                plt.figure(figsize=(15, 5))
                
                # RMSE比较
                plt.subplot(1, 3, 1)
                model_names = list(models_metrics.keys())
                rmse_values = [models_metrics[name].get('avg_rmse', 0) for name in model_names]
                
                # 修剪长值以保持图表可读性
                max_display = 1000  # 设置一个合理的最大显示值
                trimmed_rmse = [min(v, max_display) for v in rmse_values]
                bars = plt.bar(model_names, trimmed_rmse)
                
                # 为被修剪的值添加标签
                for i, (v, tv) in enumerate(zip(rmse_values, trimmed_rmse)):
                    if v > max_display:
                        plt.text(i, tv, f"{v:.1f}", ha='center', va='bottom', rotation=90, fontsize=8)
                
                plt.title(f'{dataset_name} - RMSE by Model')
                plt.xticks(rotation=45)
                plt.ylabel('RMSE')
                
                # MAE比较
                plt.subplot(1, 3, 2)
                mae_values = [models_metrics[name].get('avg_mae', 0) for name in model_names]
                
                # 修剪长值
                trimmed_mae = [min(v, max_display) for v in mae_values]
                bars = plt.bar(model_names, trimmed_mae)
                
                # 为被修剪的值添加标签
                for i, (v, tv) in enumerate(zip(mae_values, trimmed_mae)):
                    if v > max_display:
                        plt.text(i, tv, f"{v:.1f}", ha='center', va='bottom', rotation=90, fontsize=8)
                
                plt.title(f'{dataset_name} - MAE by Model')
                plt.xticks(rotation=45)
                plt.ylabel('MAE')
                
                # R²比较
                plt.subplot(1, 3, 3)
                r2_values = [models_metrics[name].get('avg_r2', 0) for name in model_names]
                plt.bar(model_names, r2_values)
                plt.title(f'{dataset_name} - R² by Model')
                plt.xticks(rotation=45)
                plt.ylabel('R²')
                
                plt.tight_layout()
                
                # 保存到数据集特定目录
                plots_dir = os.path.join(base_dir, dataset_dir, 'plots')
                os.makedirs(plots_dir, exist_ok=True)
                plt.savefig(os.path.join(plots_dir, 'models_metrics_comparison.png'))
                plt.close()
                
                # 为每个模型创建详细的特征性能报告
                for model_name, model_metrics in models_metrics.items():
                    if 'details' in model_metrics:
                        details = model_metrics['details']
                        
                        # 创建每个特征的性能条形图
                        if details and len(details) > 0:
                            plt.figure(figsize=(15, 10))
                            
                            # 提取特征名称和指标
                            feature_names = [d.get('feature', f'Feature {i}') for i, d in enumerate(details)]
                            feature_rmse = [d.get('rmse', 0) for d in details]
                            feature_mae = [d.get('mae', 0) for d in details]
                            feature_r2 = [d.get('r2', 0) for d in details]
                            
                            # RMSE
                            plt.subplot(3, 1, 1)
                            plt.bar(feature_names, feature_rmse)
                            plt.title(f'{model_name} - RMSE by Feature')
                            plt.xticks(rotation=45)
                            plt.ylabel('RMSE')
                            
                            # MAE
                            plt.subplot(3, 1, 2)
                            plt.bar(feature_names, feature_mae)
                            plt.title(f'{model_name} - MAE by Feature')
                            plt.xticks(rotation=45)
                            plt.ylabel('MAE')
                            
                            # R²
                            plt.subplot(3, 1, 3)
                            plt.bar(feature_names, feature_r2)
                            plt.title(f'{model_name} - R² by Feature')
                            plt.xticks(rotation=45)
                            plt.ylabel('R²')
                            
                            plt.tight_layout()
                            plt.savefig(os.path.join(plots_dir, f'{model_name}_feature_metrics.png'))
                            plt.close()
                    
            except Exception as e:
                print(f"  Error processing {dataset_name}: {e}")
                import traceback
                traceback.print_exc()
    
    # 生成数据集间的比较图表
    if all_results:
        # 创建一个数据集到最佳模型的映射
        dataset_best_models = {}
        dataset_avg_metrics = {}
        
        for dataset_name, models in all_results.items():
            # 找到RMSE最低的模型
            best_model = min(models.items(), key=lambda x: x[1]['avg_rmse'])
            dataset_best_models[dataset_name] = {
                'model': best_model[0],
                'metrics': best_model[1]
            }
            
            # 计算该数据集所有模型的平均指标
            avg_rmse = sum(m['avg_rmse'] for m in models.values()) / len(models)
            avg_mae = sum(m['avg_mae'] for m in models.values()) / len(models)
            avg_r2 = sum(m['avg_r2'] for m in models.values()) / len(models)
            
            dataset_avg_metrics[dataset_name] = {
                'avg_rmse': avg_rmse,
                'avg_mae': avg_mae,
                'avg_r2': avg_r2
            }
        
        # 保存最佳模型信息
        best_models_path = os.path.join(summary_dir, 'best_models.json')
        with open(best_models_path, 'w') as f:
            json.dump(dataset_best_models, f, indent=4)
        
        # 保存平均指标信息
        avg_metrics_path = os.path.join(summary_dir, 'dataset_avg_metrics.json')
        with open(avg_metrics_path, 'w') as f:
            json.dump(dataset_avg_metrics, f, indent=4)
        
        # 绘制数据集间最佳模型比较图
        plt.figure(figsize=(15, 10))
        
        # RMSE比较
        plt.subplot(1, 3, 1)
        dataset_names = list(dataset_best_models.keys())
        rmse_values = [dataset_best_models[name]['metrics']['avg_rmse'] for name in dataset_names]
        
        # 修剪长值
        max_display = 1000
        trimmed_rmse = [min(v, max_display) for v in rmse_values]
        bars = plt.bar(dataset_names, trimmed_rmse)
        
        # 为被修剪的值添加标签
        for i, (v, tv) in enumerate(zip(rmse_values, trimmed_rmse)):
            if v > max_display:
                plt.text(i, tv, f"{v:.1f}", ha='center', va='bottom', rotation=90, fontsize=8)
            plt.text(i, tv/2, dataset_best_models[dataset_names[i]]['model'], 
                    ha='center', va='center', rotation=90, fontsize=8)
        
        plt.title('Best Model RMSE by Dataset')
        plt.xticks(rotation=45)
        plt.ylabel('RMSE')
        
        # MAE比较
        plt.subplot(1, 3, 2)
        mae_values = [dataset_best_models[name]['metrics']['avg_mae'] for name in dataset_names]
        
        # 修剪长值
        trimmed_mae = [min(v, max_display) for v in mae_values]
        bars = plt.bar(dataset_names, trimmed_mae)
        
        # 为被修剪的值添加标签
        for i, (v, tv) in enumerate(zip(mae_values, trimmed_mae)):
            if v > max_display:
                plt.text(i, tv, f"{v:.1f}", ha='center', va='bottom', rotation=90, fontsize=8)
        
        plt.title('Best Model MAE by Dataset')
        plt.xticks(rotation=45)
        plt.ylabel('MAE')
        
        # R²比较
        plt.subplot(1, 3, 3)
        r2_values = [dataset_best_models[name]['metrics']['avg_r2'] for name in dataset_names]
        bars = plt.bar(dataset_names, r2_values)
        
        plt.title('Best Model R² by Dataset')
        plt.xticks(rotation=45)
        plt.ylabel('R²')
        
        plt.tight_layout()
        plt.savefig(os.path.join(summary_dir, 'best_models_comparison.png'))
        plt.close()
        
        # 生成摘要报告
        report_path = os.path.join(summary_dir, 'summary_report.txt')
        with open(report_path, 'w') as f:
            f.write("BiLSTM Multivariate Time Series Prediction Experiment Summary Report\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("Best model for each dataset:\n")
            f.write("-" * 70 + "\n")
            f.write(f"{'Dataset':<25} {'Best Model':<15} {'RMSE':<15} {'MAE':<15} {'R²':<15}\n")
            f.write("-" * 70 + "\n")
            
            for dataset_name in dataset_names:
                best = dataset_best_models[dataset_name]
                f.write(f"{dataset_name:<25} {best['model']:<15} "
                        f"{best['metrics']['avg_rmse']:<15.4f} "
                        f"{best['metrics']['avg_mae']:<15.4f} "
                        f"{best['metrics']['avg_r2']:<15.4f}\n")
            
            f.write("\n\nAverage metrics across all models for each dataset:\n")
            f.write("-" * 70 + "\n")
            f.write(f"{'Dataset':<25} {'Average RMSE':<15} {'Average MAE':<15} {'Average R²':<15}\n")
            f.write("-" * 70 + "\n")
            
            for dataset_name, metrics in dataset_avg_metrics.items():
                f.write(f"{dataset_name:<25} {metrics['avg_rmse']:<15.4f} "
                        f"{metrics['avg_mae']:<15.4f} {metrics['avg_r2']:<15.4f}\n")
        
        print("\nBest model for each dataset:")
        print("-" * 70)
        print(f"{'Dataset':<25} {'Best Model':<15} {'RMSE':<15} {'MAE':<15} {'R²':<15}")
        print("-" * 70)
        
        for dataset_name in dataset_names:
            best = dataset_best_models[dataset_name]
            print(f"{dataset_name:<25} {best['model']:<15} "
                  f"{best['metrics']['avg_rmse']:<15.4f} "
                  f"{best['metrics']['avg_mae']:<15.4f} "
                  f"{best['metrics']['avg_r2']:<15.4f}")
    
    print(f"\nFinal report generated successfully! Results saved to {summary_dir}")

if __name__ == "__main__":
    # 指定结果目录的路径
    base_dir = "./results"  # 更改为你的结果目录路径
    generate_final_report(base_dir) 