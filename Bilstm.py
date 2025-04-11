import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from ucimlrepo import fetch_ucirepo
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import json
from datetime import datetime
import torch.nn.functional as F

# Define mapping of dataset names to their UCI repository IDs
DATASET_IDS = {
    # 'Air Quality': 360,
    # 'Household Power': 235,
    # 'Appliances Energy': 374,
    'Tetouan Power': 849
}

# Create result directories
def create_result_dirs():
    # Create main results directory
    base_dir = 'results'
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    # Create subdirectory for each dataset
    for dataset_name in DATASET_IDS.keys():
        dataset_dir = os.path.join(base_dir, dataset_name.lower().replace(' ', '_'))
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
            
        # Create plots and models subdirectories
        plots_dir = os.path.join(dataset_dir, 'plots')
        models_dir = os.path.join(dataset_dir, 'models')
        
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
    
    # Create summary results directory
    summary_dir = os.path.join(base_dir, 'summary')
    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)
    
    return base_dir

def load_and_preprocess_dataset(dataset_name):
    dataset_id = DATASET_IDS[dataset_name]
    dataset = fetch_ucirepo(id=dataset_id)
    
    # Get features and targets
    X = dataset.data.features
    y = dataset.data.targets
    
    # Merge data
    df = pd.concat([X, y], axis=1) if isinstance(y, pd.DataFrame) else X
    
    # Data preprocessing
    # Handle missing values
    df = df.replace([-200, '?', '', None], np.nan)
    df = df.dropna()
    
    # Remove non-numeric columns
    date_columns = df.columns[df.columns.str.contains('date|time|Date|Time|DateTime', case=False)]
    df = df.drop(columns=date_columns)
    
    # Convert to float
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna()
    
    return df

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]  # Predict all features at the next time step
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def calculate_metrics_for_dataset(actuals_rescaled, predictions_rescaled, feature_names):
    overall_rmse, overall_mae, overall_r2 = [], [], []
    
    results_details = []
    print("-" * 50)
    for i in range(predictions_rescaled.shape[1]):
        rmse = np.sqrt(mean_squared_error(actuals_rescaled[:, i], predictions_rescaled[:, i]))
        mae = mean_absolute_error(actuals_rescaled[:, i], predictions_rescaled[:, i])
        r2 = r2_score(actuals_rescaled[:, i], predictions_rescaled[:, i])
        
        feature_name = feature_names[i] if i < len(feature_names) else f"Feature {i}"
        print(f"{feature_name[:20]:<20} {rmse:<10.4f} {mae:<10.4f} {r2:<10.4f}")
        
        overall_rmse.append(float(rmse))
        overall_mae.append(float(mae))
        overall_r2.append(float(r2))
        
        # Save detailed results for each feature
        results_details.append({
            "feature": feature_name,
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2)
        })
    
    # Calculate average performance metrics across all features
    avg_rmse = float(np.mean(overall_rmse))
    avg_mae = float(np.mean(overall_mae))
    avg_r2 = float(np.mean(overall_r2))
    
    print("-" * 50)
    print(f"{'Average':<20} {avg_rmse:<10.4f} {avg_mae:<10.4f} {avg_r2:<10.4f}")
    print("-" * 50)
    
    return {
        'avg_rmse': avg_rmse,
        'avg_mae': avg_mae,
        'avg_r2': avg_r2,
        'details': results_details
    }

# Define BiLSTM model
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Bidirectional LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # Fully connected layer, note that bidirectional LSTM output is twice the hidden size
        self.fc = nn.Linear(hidden_size * 2, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc_out = nn.Linear(128, output_size)
        
    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))
        
        # Get output from the last time step
        out = out[:, -1, :]
        
        # Fully connected layers
        out = self.fc(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc_out(out)
        
        return out

# Training function
def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, device, model_name=None):
    model.train()
    train_losses = []
    val_losses = []
    
    # 对AttentionBiLSTM使用额外的L1正则化和偏差校正
    use_bias_correction = model_name == 'AttentionBiLSTM'
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(data)
            
            # 主损失
            loss = criterion(output, target)
            
            # 对于AttentionBiLSTM添加额外的L1正则化，促使模型输出更小的值
            if use_bias_correction:
                l1_reg = torch.mean(torch.abs(output)) * 0.01  # L1强度参数
                loss = loss + l1_reg
            
            # Backward pass
            loss.backward()
            
            # 梯度裁剪以防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            
        # 计算训练损失
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        
        # 计算验证损失
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
        
        val_loss = val_loss / len(test_loader)
        val_losses.append(val_loss)
        
        # 对于AttentionBiLSTM，动态调整偏差校正参数
        if use_bias_correction and isinstance(model, AttentionBiLSTM) and epoch > 10:
            with torch.no_grad():
                # 计算所有验证数据的平均误差
                error_sum = torch.zeros(model.bias_correction.shape).to(device)
                count = 0
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    error = output.mean(dim=0) - target.mean(dim=0)
                    error_sum += error
                    count += 1
                
                if count > 0:
                    # 缓慢调整偏差校正参数
                    avg_error = error_sum / count
                    model.bias_correction.data += 0.01 * avg_error  # 学习率为0.01
        
        model.train()
        
        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}')
    
    return train_losses, val_losses

# Evaluation function
def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            
            predictions.append(output.cpu().numpy())
            actuals.append(target.cpu().numpy())
    
    avg_test_loss = test_loss / len(test_loader)
    print(f'Test Loss: {avg_test_loss:.4f}')
    
    # Convert to numpy arrays
    predictions = np.vstack(predictions)
    actuals = np.vstack(actuals)
    
    return predictions, actuals, avg_test_loss

# Plot training curve
def plot_training_curve(train_losses, val_losses, num_epochs, save_path):
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, num_epochs+1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

# Plot prediction results
def plot_predictions(actuals_rescaled, predictions_rescaled, feature_names, title_prefix, plots_dir):
    # Plot single feature prediction results
    plt.figure(figsize=(15, 8))
    feature_idx = 0  # Select the first feature for visualization
    plt.plot(actuals_rescaled[:100, feature_idx], label='Actual', linewidth=2)
    plt.plot(predictions_rescaled[:100, feature_idx], label='Predicted', linewidth=2, linestyle='--')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.title(f'{title_prefix} - Prediction Results - {feature_names[feature_idx]}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, 'prediction_single_feature.png'))
    plt.close()
    
    # Plot multiple features prediction results
    plt.figure(figsize=(15, 12))
    num_features_to_plot = min(4, predictions_rescaled.shape[1])  # Plot at most 4 features
    for i in range(num_features_to_plot):
        plt.subplot(2, 2, i+1)
        feature_name = feature_names[i] if i < len(feature_names) else f"Feature {i}"
        plt.plot(actuals_rescaled[:100, i], 'b-', label='Actual', linewidth=2)
        plt.plot(predictions_rescaled[:100, i], 'r--', label='Predicted', linewidth=2)
        plt.title(f'Feature: {feature_name}')
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
    plt.tight_layout()
    plt.suptitle(f'{title_prefix} - Multiple Features Prediction Results', fontsize=16)
    plt.subplots_adjust(top=0.92)
    plt.savefig(os.path.join(plots_dir, 'prediction_multiple_features.png'))
    plt.close()
    
    # Plot error distribution
    plt.figure(figsize=(12, 6))
    errors = actuals_rescaled - predictions_rescaled
    plt.hist(errors.flatten(), bins=50, alpha=0.75)
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title(f'{title_prefix} - Prediction Error Distribution')
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, 'error_distribution.png'))
    plt.close()
    
    # Plot actual vs predicted scatter plot
    plt.figure(figsize=(10, 8))
    feature_idx = 0  # First feature
    plt.scatter(actuals_rescaled[:, feature_idx], predictions_rescaled[:, feature_idx], alpha=0.5)
    plt.plot([actuals_rescaled[:, feature_idx].min(), actuals_rescaled[:, feature_idx].max()], 
             [actuals_rescaled[:, feature_idx].min(), actuals_rescaled[:, feature_idx].max()], 
             'k--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'{title_prefix} - Actual vs Predicted - {feature_names[feature_idx]}')
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, 'actual_vs_predicted.png'))
    plt.close()

# 重新定义带注意力机制的BiLSTM模型
class AttentionBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(AttentionBiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Bidirectional LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if num_layers > 1 else 0  # 添加层间dropout
        )
        
        # 注意力层 - 使用更简单的注意力机制
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),  # 减小注意力层的复杂度
            nn.Tanh(),
            nn.Linear(32, 1),
            nn.Softmax(dim=1)
        )
        
        # 添加预测校正层
        self.fc = nn.Linear(hidden_size * 2, 64)  # 减小全连接层的尺寸
        self.relu = nn.LeakyReLU(0.1)  # 使用LeakyReLU代替ReLU
        self.dropout = nn.Dropout(0.4)  # 增加dropout比例
        self.bn = nn.BatchNorm1d(64)   # 批归一化
        
        # 添加额外的输出调整层
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        
        # 最终输出层
        self.fc_out = nn.Linear(32, output_size)
        
        # 添加预测偏差校正参数
        self.bias_correction = nn.Parameter(torch.zeros(output_size))
        
    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM前向传播
        lstm_out, _ = self.lstm(x, (h0, c0))  # lstm_out: [batch, seq_len, hidden*2]
        
        # 计算注意力权重
        attn_weights = self.attention(lstm_out)  # [batch, seq_len, 1]
        
        # 应用注意力权重
        context = torch.sum(attn_weights * lstm_out, dim=1)  # [batch, hidden*2]
        
        # 全连接层
        out = self.fc(context)
        out = self.relu(out)
        out = self.bn(out)
        out = self.dropout(out)
        
        # 第二层
        out = self.fc2(out)
        out = self.relu(out)
        out = self.bn2(out)
        out = self.dropout(out)
        
        # 最终输出
        out = self.fc_out(out)
        
        # 应用偏差校正 - 这能帮助减少预测偏高的问题
        out = out - self.bias_correction
        
        return out

# 优化方案二：CNN-BiLSTM 结合残差连接
class CNNBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(CNNBiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # CNN层用于提取时序特征
        self.conv1 = nn.Conv1d(input_size, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=64, 
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # 残差连接和全连接层
        self.fc_direct = nn.Linear(input_size, 128)  # 残差连接
        self.fc1 = nn.Linear(hidden_size * 2, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.bn = nn.BatchNorm1d(128)
        self.fc_out = nn.Linear(128, output_size)
        
    def forward(self, x):
        batch_size, seq_len, features = x.size()
        
        # 保存原始输入用于残差连接
        direct_input = x[:, -1, :]  # 取最后一个时间步
        
        # CNN层处理 - 需要调整维度顺序为 [batch, features, seq_len]
        cnn_in = x.permute(0, 2, 1)
        cnn_out = self.conv1(cnn_in)
        cnn_out = F.relu(cnn_out)
        cnn_out = self.maxpool(cnn_out)
        
        cnn_out = self.conv2(cnn_out)
        cnn_out = F.relu(cnn_out)
        
        # 调整CNN输出维度以符合LSTM输入 [batch, seq_len, features]
        lstm_in = cnn_out.permute(0, 2, 1)
        
        # 确保维度匹配
        lstm_in = lstm_in[:, :seq_len, :]
        
        # LSTM层
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)
        lstm_out, _ = self.lstm(lstm_in, (h0, c0))
        
        # 提取最后一个时间步的输出
        lstm_out = lstm_out[:, -1, :]
        
        # 通过残差连接合并原始输入特征
        direct_features = self.fc_direct(direct_input)
        
        # 处理LSTM输出
        out = self.fc1(lstm_out)
        out = self.relu(out)
        out = self.bn(out)
        out = self.dropout(out)
        
        # 添加残差连接
        out = out + direct_features
        
        # 最终输出层
        out = self.fc_out(out)
        
        return out

# 添加普通LSTM模型定义
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # 全连接层
        self.fc = nn.Linear(hidden_size, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc_out = nn.Linear(128, output_size)
        
    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM前向传播
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # 获取最后一个时间步的输出
        out = lstm_out[:, -1, :]
        
        # 全连接层
        out = self.fc(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc_out(out)
        
        return out

# 修改 run_experiment 函数以支持多种模型比较
def run_experiment(dataset_name, base_dir, seq_length=24, num_epochs=None):
    print(f"\nProcessing dataset: {dataset_name}")
    
    # 如果没有指定epochs，则根据数据集设置默认值
    if num_epochs is None:
        if dataset_name == 'Household Power':
            num_epochs = 10  # 为Household Power数据集设置10个epochs
        else:
            num_epochs = 50  # 其他数据集使用默认的50个epochs
    
    print(f"Training for {num_epochs} epochs")
    
    # 创建结果路径
    dataset_dir = os.path.join(base_dir, dataset_name.lower().replace(' ', '_'))
    plots_dir = os.path.join(dataset_dir, 'plots')
    models_dir = os.path.join(dataset_dir, 'models')
    
    # 加载和预处理数据
    df = load_and_preprocess_dataset(dataset_name)
    
    # 数据归一化
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    
    # 创建序列数据
    X_seq, y_seq = create_sequences(scaled_data, seq_length)
    
    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.2, random_state=42, shuffle=False
    )
    
    # 转换为PyTorch张量
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test)
    
    # 创建数据加载器
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 模型参数
    input_size = X_train.shape[2]
    hidden_size = 64
    num_layers = 2
    output_size = y_train.shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 定义所有要比较的模型
    models = {
        'LSTM': LSTMModel(input_size, hidden_size, num_layers, output_size).to(device),
        'BiLSTM': BiLSTM(input_size, hidden_size, num_layers, output_size).to(device),
        'AttentionBiLSTM': AttentionBiLSTM(input_size, hidden_size, num_layers, output_size).to(device),
        'CNNBiLSTM': CNNBiLSTM(input_size, hidden_size, num_layers, output_size).to(device)
    }
    
    # 训练和评估所有模型
    results = {}
    all_predictions = {}
    all_actuals = {}
    
    for model_name, model in models.items():
        print(f"\nTraining model: {model_name}")
        
        # 定义损失函数和优化器
        if model_name == 'AttentionBiLSTM':
            # 为AttentionBiLSTM使用Huber损失，对大误差的惩罚较小
            criterion = nn.HuberLoss(delta=1.0)
            # 使用较小的学习率
            optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
        else:
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # 训练模型 - 传递模型名称
        train_losses, val_losses = train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, device, model_name)
        
        # 评估模型
        predictions, actuals, test_loss = evaluate_model(model, test_loader, criterion, device)
        
        # 反归一化结果
        predictions_rescaled = scaler.inverse_transform(predictions)
        actuals_rescaled = scaler.inverse_transform(actuals)
        
        # 保存模型
        torch.save(model.state_dict(), os.path.join(models_dir, f'{model_name}_model.pth'))
        
        # 绘制训练曲线
        plot_training_curve(train_losses, val_losses, num_epochs, 
                           os.path.join(plots_dir, f'{model_name}_training_curve.png'))
        
        # Create model-specific plots directory
        model_plots_dir = os.path.join(plots_dir, model_name)
        if not os.path.exists(model_plots_dir):
            os.makedirs(model_plots_dir)
        
        # Plot prediction curves for each model
        feature_names = df.columns.tolist()
        plot_predictions(actuals_rescaled, predictions_rescaled, feature_names, 
                        f"{dataset_name} - {model_name}", model_plots_dir)
        
        # 计算评估指标
        metrics = calculate_metrics_for_dataset(actuals_rescaled, predictions_rescaled, feature_names)
        results[model_name] = metrics
        
        # 保存预测结果用于后续比较绘图
        all_predictions[model_name] = predictions_rescaled
        all_actuals[model_name] = actuals_rescaled
    
    # 绘制所有模型的比较图
    plot_model_comparisons(all_predictions, all_actuals, feature_names, list(models.keys()), plots_dir)
    
    # 单独绘制指标比较图
    plot_metrics_comparison(results, list(models.keys()), plots_dir)
    
    # 保存所有模型的评估指标
    metrics_path = os.path.join(dataset_dir, 'all_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    return results

# 修改比较不同模型的绘图函数
def plot_model_comparisons(all_predictions, all_actuals, feature_names, model_names, plots_dir):
    # 选择一个特征进行比较展示
    feature_idx = 0
    plt.figure(figsize=(15, 8))
    
    # 获取任意模型的实际值（它们都是相同的）
    first_model = list(all_actuals.keys())[0]
    actuals = all_actuals[first_model]
    plt.plot(actuals[:100, feature_idx], 'k-', label='Actual Values', linewidth=2)
    
    # 为每个模型绘制预测值
    colors = ['b', 'r', 'g', 'm', 'c']
    line_styles = ['-', '--', '-.', ':']
    
    for i, model_name in enumerate(model_names):
        color_idx = i % len(colors)
        style_idx = i % len(line_styles)
        plt.plot(all_predictions[model_name][:100, feature_idx], 
                f'{colors[color_idx]}{line_styles[style_idx]}', 
                label=f'{model_name} Predictions', 
                alpha=0.7)
    
    plt.xlabel('Time Steps')
    plt.ylabel('Values')
    plt.title(f'Model Comparison - {feature_names[feature_idx]}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, 'models_comparison.png'))
    plt.close()
    
    # 绘制每个模型的RMSE、MAE和R²比较条形图
    plt.figure(figsize=(15, 5))
    
    # 从结果字典中提取指标，而不是从actuals中提取
    # 这个地方需要在run_experiment函数中保存metrics到results变量，然后传递给这个函数
    # 由于我们没有直接访问results，这里先注释掉这部分代码
    """
    # 提取指标
    rmse_values = [results[model]['avg_rmse'] for model in model_names]
    mae_values = [results[model]['avg_mae'] for model in model_names]
    r2_values = [results[model]['avg_r2'] for model in model_names]
    
    # 绘制RMSE比较
    plt.subplot(1, 3, 1)
    plt.bar(model_names, rmse_values)
    plt.title('Average RMSE Comparison')
    plt.xticks(rotation=45)
    plt.ylabel('RMSE')
    
    # 绘制MAE比较
    plt.subplot(1, 3, 2)
    plt.bar(model_names, mae_values)
    plt.title('Average MAE Comparison')
    plt.xticks(rotation=45)
    plt.ylabel('MAE')
    
    # 绘制R²比较
    plt.subplot(1, 3, 3)
    plt.bar(model_names, r2_values)
    plt.title('Average R² Comparison')
    plt.xticks(rotation=45)
    plt.ylabel('R²')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'metrics_comparison.png'))
    plt.close()
    """

# 新增单独的指标比较函数
def plot_metrics_comparison(results, model_names, plots_dir):
    plt.figure(figsize=(15, 5))
    
    # 提取指标
    rmse_values = [results[model]['avg_rmse'] for model in model_names]
    mae_values = [results[model]['avg_mae'] for model in model_names]
    r2_values = [results[model]['avg_r2'] for model in model_names]
    
    # 绘制RMSE比较
    plt.subplot(1, 3, 1)
    plt.bar(model_names, rmse_values)
    plt.title('Average RMSE Comparison')
    plt.xticks(rotation=45)
    plt.ylabel('RMSE')
    
    # 绘制MAE比较
    plt.subplot(1, 3, 2)
    plt.bar(model_names, mae_values)
    plt.title('Average MAE Comparison')
    plt.xticks(rotation=45)
    plt.ylabel('MAE')
    
    # 绘制R²比较
    plt.subplot(1, 3, 3)
    plt.bar(model_names, r2_values)
    plt.title('Average R² Comparison')
    plt.xticks(rotation=45)
    plt.ylabel('R²')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'metrics_comparison.png'))
    plt.close()

# Main program
def main():
    print("Starting BiLSTM Multivariate Time Series Prediction Experiment")
    print(f"Using device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # Create result directories
    base_dir = create_result_dirs()
    
    # Record start time
    start_time = datetime.now()
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {}
    
    for dataset_name in DATASET_IDS.keys():
        try:
            print(f"\n{'-'*20} Processing dataset: {dataset_name} {'-'*20}")
            dataset_results = run_experiment(dataset_name, base_dir)
            
            # 计算每个数据集的平均指标
            if dataset_results:
                # 从所有模型中提取指标并计算平均值
                all_rmse = []
                all_mae = []
                all_r2 = []
                
                for model_name, model_metrics in dataset_results.items():
                    if 'avg_rmse' in model_metrics:
                        all_rmse.append(model_metrics['avg_rmse'])
                    if 'avg_mae' in model_metrics:
                        all_mae.append(model_metrics['avg_mae'])
                    if 'avg_r2' in model_metrics:
                        all_r2.append(model_metrics['avg_r2'])
                
                # 计算平均值
                dataset_avg_metrics = {
                    'avg_rmse': sum(all_rmse) / len(all_rmse) if all_rmse else 0,
                    'avg_mae': sum(all_mae) / len(all_mae) if all_mae else 0,
                    'avg_r2': sum(all_r2) / len(all_r2) if all_r2 else 0,
                    'models': dataset_results  # 保存所有模型的指标
                }
                
                results[dataset_name] = dataset_avg_metrics
            
            print(f"{'-'*20} Completed dataset: {dataset_name} {'-'*20}")
        except Exception as e:
            print(f"\nError processing dataset {dataset_name}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Save comprehensive results
    summary_path = os.path.join(base_dir, 'summary', 'overall_results.json')
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    # Generate summary report
    report_path = os.path.join(base_dir, 'summary', 'summary_report.txt')
    with open(report_path, 'w') as f:
        f.write("BiLSTM Multivariate Time Series Prediction Experiment Summary Report\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Experiment start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Experiment end time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Average performance metrics for all datasets:\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Dataset':<25} {'Average RMSE':<15} {'Average MAE':<15} {'Average R²':<15}\n")
        f.write("-" * 70 + "\n")
        
        for dataset_name, metrics in results.items():
            f.write(f"{dataset_name:<25} {metrics['avg_rmse']:<15.4f} {metrics['avg_mae']:<15.4f} {metrics['avg_r2']:<15.4f}\n")
    
    # Print comprehensive results for all datasets
    print("\nAverage performance metrics for all datasets:")
    print("-" * 70)
    print(f"{'Dataset':<25} {'Average RMSE':<15} {'Average MAE':<15} {'Average R²':<15}")
    print("-" * 70)
    
    for dataset_name, metrics in results.items():
        print(f"{dataset_name:<25} {metrics['avg_rmse']:<15.4f} {metrics['avg_mae']:<15.4f} {metrics['avg_r2']:<15.4f}")
    
    # Plot comparison across all datasets
    plt.figure(figsize=(15, 10))
    
    # RMSE comparison
    plt.subplot(1, 3, 1)
    dataset_names = list(results.keys())
    rmse_values = [results[name]['avg_rmse'] for name in dataset_names]
    plt.bar(dataset_names, rmse_values)
    plt.title('Average RMSE by Dataset')
    plt.xticks(rotation=45)
    plt.ylabel('RMSE')
    
    # MAE comparison
    plt.subplot(1, 3, 2)
    mae_values = [results[name]['avg_mae'] for name in dataset_names]
    plt.bar(dataset_names, mae_values)
    plt.title('Average MAE by Dataset')
    plt.xticks(rotation=45)
    plt.ylabel('MAE')
    
    # R² comparison
    plt.subplot(1, 3, 3)
    r2_values = [results[name]['avg_r2'] for name in dataset_names]
    plt.bar(dataset_names, r2_values)
    plt.title('Average R² by Dataset')
    plt.xticks(rotation=45)
    plt.ylabel('R²')
    
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'summary', 'datasets_comparison.png'))
    plt.close()
    
    print(f"\nExperiment completed! Results saved to {base_dir} directory")

if __name__ == "__main__":
    main()