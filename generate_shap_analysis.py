"""
SHAP Analysis Script for Anomaly Transformer
生成模型的SHAP可解释性分析图
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import sys
import json
import matplotlib.pyplot as plt
from datetime import datetime
import random
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from model.AnomalyTransformer import AnomalyTransformer
from data_factory.data_loader import get_loader_segment

# 尝试导入SHAP
try:
    import shap
except ImportError:
    print("SHAP库未安装，正在尝试安装...")
    os.system("pip install shap")
    import shap


class ModelWrapper(nn.Module):
    """
    包装Anomaly Transformer模型以便SHAP分析
    将模型输出转换为异常分数
    """
    def __init__(self, model, win_size, k=3, temperature=50):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.win_size = win_size
        self.k = k
        self.temperature = temperature
        self.criterion = nn.MSELoss(reduction='none')
        
    def my_kl_loss(self, p, q):
        """计算KL散度"""
        res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
        return torch.mean(torch.sum(res, dim=-1), dim=1)
    
    def forward(self, x):
        """
        前向传播，返回每个时间步的异常分数（per-timestep）
        输入: x [B, L, D] - batch_size, window_size, features
        输出: anomaly_score [B, L] - 每个时间步的异常分数
        
        注意：为了SHAP能够计算梯度，这里不使用.detach()
        输入数据应该在外部已经设置了requires_grad=True
        """
        # 获取模型输出
        model_output = self.model(x)
        
        # 处理模型输出（可能是元组或单个张量）
        if isinstance(model_output, tuple):
            output, series, prior, _ = model_output
        else:
            # 如果模型只返回输出，需要重新调用获取attention信息
            output, series, prior, _ = self.model(x)
        
        # 计算重构损失（每个时间步）
        loss = torch.mean(self.criterion(x, output), dim=-1)  # [B, L]
        
        # 计算association discrepancy
        # 保留.detach()以符合原始测试逻辑
        series_loss = 0.0
        prior_loss = 0.0
        for u in range(len(prior)):
            prior_sum = torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.win_size)
            prior_normalized = prior[u] / (prior_sum + 1e-8)

            if u == 0:
                series_loss = self.my_kl_loss(series[u], prior_normalized.detach()) * self.temperature
                prior_loss  = self.my_kl_loss(prior_normalized, series[u].detach()) * self.temperature
            else:
                series_loss += self.my_kl_loss(series[u], prior_normalized.detach()) * self.temperature
                prior_loss  += self.my_kl_loss(prior_normalized, series[u].detach()) * self.temperature
        
        # 计算每个时间步的异常分数
        metric = torch.softmax((-series_loss - prior_loss), dim=-1)  # [B, L]
        anomaly_score = metric * loss  # [B, L]
        
        # 返回每个时间步的异常分数，不做平均
        # SHAP会对每个输出位置单独计算贡献
        return anomaly_score  # [B, L]


def load_model_and_config(model_path, config_path):
    """加载模型和配置"""
    # 读取配置
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # 创建模型
    model = AnomalyTransformer(
        win_size=config['win_size'],
        enc_in=config['input_c'],
        c_out=config['output_c'],
        e_layers=3
    )
    
    # 加载权重
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print(f"模型加载成功: {model_path}")
    print(f"配置: win_size={config['win_size']}, input_c={config['input_c']}, output_c={config['output_c']}")
    
    return model, config, device


def prepare_background_data(data_loader, n_samples=100, seed=42):
    """
    准备背景数据集用于SHAP分析
    使用固定随机种子确保可重复性
    """
    # 设置随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    background_samples = []
    count = 0
    
    for input_data, _ in data_loader:
        background_samples.append(input_data)
        count += input_data.shape[0]
        if count >= n_samples:
            break
    
    background_data = torch.cat(background_samples, dim=0)[:n_samples]
    # 确保数据需要梯度（SHAP需要）
    background_data = background_data.requires_grad_(True)
    print(f"背景数据集准备完成: {background_data.shape}")
    
    return background_data


def generate_shap_values(wrapped_model, background_data, test_data, device, batch_size=100):
    """
    生成SHAP值
    使用批处理以避免内存问题
    """
    print("\n开始计算SHAP值...")
    print(f"背景数据: {background_data.shape}")
    print(f"测试数据: {test_data.shape}")
    
    # 将数据移到设备上，并确保需要梯度（SHAP需要）
    background_data = background_data.to(device).float()
    test_data = test_data.to(device).float()
    
    # 创建SHAP解释器 (使用DeepExplainer，适用于深度学习模型)
    print("初始化SHAP解释器...")
    explainer = shap.DeepExplainer(wrapped_model, background_data)
    
    # 批处理计算SHAP值以避免内存问题
    # 注意：不能使用torch.no_grad()，因为SHAP需要梯度计算
    n_samples = test_data.shape[0]
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    print(f"分 {n_batches} 批计算SHAP值 (每批 {batch_size} 个样本)...")
    shap_values_list = []
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_samples)
        batch_data = test_data[start_idx:end_idx]
        
        print(f"  处理批次 {i+1}/{n_batches} (样本 {start_idx}-{end_idx-1})...")
        
        # SHAP内部会处理梯度计算，我们不需要手动设置requires_grad
        batch_shap = explainer.shap_values(batch_data,check_additivity=False)
        
        # 处理SHAP返回值（可能是列表或数组）
        if isinstance(batch_shap, list):
            # 如果是列表，取第一个元素（通常是输出）
            batch_shap = batch_shap[0]
        
        # 确保是numpy数组
        if isinstance(batch_shap, torch.Tensor):
            batch_shap = batch_shap.detach().cpu().numpy()
        
        shap_values_list.append(batch_shap)
    
    # 合并所有批次的SHAP值
    shap_values = np.concatenate(shap_values_list, axis=0)
    
    print(f"SHAP值计算完成: {shap_values.shape}")
    
    return shap_values, explainer


def plot_shap_summary(shap_values, test_data, feature_names, save_dir, test_labels=None):
    """
    绘制SHAP汇总图（保持时间×特征结构，不flatten）
    
    shap_values: [n_samples, n_outputs, n_features, win_size]
                 对于每个输出时间步，有对应的SHAP值
                 实际形状: (50, 30, 27, 30) = [n_samples, n_outputs, n_features, win_size]
    test_labels: [n_samples] 测试样本的标签（0=正常，1=异常）
    """
    # SHAP返回的形状: [n_samples, n_outputs, n_features, win_size]
    # 其中 n_outputs = win_size （每个时间步一个输出）
    n_samples, n_outputs, n_features, win_size = shap_values.shape
    
    print(f"\nSHAP值形状: {shap_values.shape}")
    print(f"  n_samples={n_samples}, n_outputs={n_outputs}, n_features={n_features}, win_size={win_size}")
    
    # ========== 方法1: 对所有输出和时间聚合，得到每个特征的总贡献 ==========
    # 对每个特征，在所有样本、所有输出、所有时间步上求和/平均
    # shap_values: [n_samples, n_outputs, n_features, win_size]
    # 我们要得到: [n_samples, n_features]
    
    # 策略：对每个样本，把所有输出时间步的SHAP值聚合
    # 由于每个输出时间步对应一个预测，我们可以：
    # 1. 对角线聚合：output_t 对 input_t 的影响
    # 2. 全局聚合：所有output对所有input的平均影响
    
    # 这里采用全局聚合：对output和time维度取平均
    # 注意：对于beeswarm图，我们需要保留符号信息，所以先不取绝对值
    shap_values_by_feature_signed = np.mean(shap_values, axis=(1, 3))  # [n_samples, n_features]
    shap_values_by_feature = np.abs(shap_values_by_feature_signed)  # [n_samples, n_features] 用于重要性排序

    print(f"特征维度SHAP值: {shap_values_by_feature.shape}")
    
    # 准备测试数据用于beeswarm图
    # test_data形状: [n_samples, win_size, n_features]
    # 对时间维度取平均，得到每个特征的平均值
    if len(test_data.shape) == 3:
        test_data_by_feature = np.mean(test_data, axis=1)  # [n_samples, n_features]
    else:
        # 如果已经是2D，直接使用
        test_data_by_feature = test_data
    
    # 1. SHAP Summary Plot (beeswarm) - 显示每个特征的影响分布
    plt.figure(figsize=(12, 10))
    shap.summary_plot(
        shap_values_by_feature_signed,  # 使用带符号的SHAP值
        test_data_by_feature,
        feature_names=feature_names,
        max_display=20,  # 只显示前20个最重要的特征
        show=False
    )
    plt.tight_layout()
    save_path = os.path.join(save_dir, "shap_summary_beeswarm.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"保存SHAP汇总图(蜂群图): {save_path}")
    plt.close()

    
    # 2. 保存特征重要性数值
    mean_abs_shap = np.mean(shap_values_by_feature, axis=0)
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': mean_abs_shap
    }).sort_values('mean_abs_shap', ascending=False)
    
    csv_path = os.path.join(save_dir, "feature_importance.csv")
    feature_importance_df.to_csv(csv_path, index=False)
    print(f"保存特征重要性数据: {csv_path}")
    
    print("\n特征重要性排名 (Top 10):")
    print(feature_importance_df.head(10).to_string(index=False))
    

    
    # ========== 方法4: 热图 - 特征×时间 ==========
    # 平均跨样本和输出，得到 [n_features, win_size] 的热图
    # shap_values: [n_samples, n_outputs, n_features, win_size]
    # 转置为 [n_features, win_size] 以便绘制
    heatmap_data = np.mean(np.abs(shap_values), axis=(0, 1))  # [n_features, win_size]
    
    plt.figure(figsize=(14, 10))
    # heatmap_data已经是 [n_features, win_size]，直接使用
    im = plt.imshow(heatmap_data, aspect='auto', cmap='YlOrRd')
    plt.colorbar(im, label='Mean |SHAP value|')
    plt.xlabel('Time step in window')
    plt.ylabel('Feature')
    plt.yticks(range(n_features), feature_names, fontsize=8)
    plt.title('Feature×Time SHAP Heatmap')
    plt.tight_layout()
    save_path = os.path.join(save_dir, "shap_feature_time_heatmap.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"保存特征×时间热图: {save_path}")
    plt.close()


def main():
    """主函数"""
    # ==================== 配置参数 ====================
    # MODEL_PATH = "experiments/checkpoints/checkpoints_k_analysis/pcb_k5_20251119_093327/pcb_cleaned_1minut_20250928_161509_checkpoint_2025-11-19_09-33-30/model.pth"
    # CONFIG_PATH = "experiments/checkpoints/checkpoints_k_analysis/pcb_k5_20251119_093327/pcb_cleaned_1minut_20250928_161509_checkpoint_2025-11-19_09-33-30/config.json"
    # DATA_PATH = "dataset/dataset/downsampleData_scratch_1minut/pcb/pcb_cleaned_1minut_20250928_161509.parquet"
      
    MODEL_PATH = "experiments/checkpoints/checkpoints_batch_size_analysis/contact_batch_size64_20251101_144126/contact_cleaned_1minut_20250928_172122_checkpoint_2025-11-01_14-41-29/model.pth"
    CONFIG_PATH = "experiments/checkpoints/checkpoints_batch_size_analysis/contact_batch_size64_20251101_144126/contact_cleaned_1minut_20250928_172122_checkpoint_2025-11-01_14-41-29/config.json"
    DATA_PATH = "dataset/dataset/downsampleData_scratch_1minut/contact/contact_cleaned_1minut_20250928_172122.parquet"
    
    # MODEL_PATH = "experiments/checkpoints/checkpoints_anormly_ratio_analysis/ring_anormly_ratio3.0_20251101_140809/Ring_cleaned_1minut_20250928_170147_checkpoint_2025-11-01_14-08-13/model.pth"
    # CONFIG_PATH = "experiments/checkpoints/checkpoints_anormly_ratio_analysis/ring_anormly_ratio3.0_20251101_140809/Ring_cleaned_1minut_20250928_170147_checkpoint_2025-11-01_14-08-13/config.json"
    # DATA_PATH = "dataset/dataset/downsampleData_scratch_1minut/ring/Ring_cleaned_1minut_20250928_170147.parquet"
    
    # SHAP参数
    N_BACKGROUND_SAMPLES = 200  # 背景数据集样本数
    N_TEST_SAMPLES = 50  # 用于计算SHAP值的测试样本数
    
    # 输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #output_dir = f"/home/wanting/Anomaly-Transformer/experiments/shap_analysis_{timestamp}_pcb"
    output_dir = f"/home/wanting/Anomaly-Transformer/experiments/shap_analysis_{timestamp}_contact"
    #output_dir = f"/home/wanting/Anomaly-Transformer/experiments/shap_analysis_{timestamp}_ring"

    os.makedirs(output_dir, exist_ok=True)
    print(f"\n输出目录: {output_dir}\n")
    
    # ==================== 1. 加载模型 ====================
    model, config, device = load_model_and_config(MODEL_PATH, CONFIG_PATH)
    
    # 创建包装模型
    wrapped_model = ModelWrapper(
        model=model,
        win_size=config['win_size'],
        k=config.get('k', 3)
    )
    wrapped_model.to(device)
    wrapped_model.eval()
    
    # ==================== 2. 加载数据 ====================
    print("\n加载数据...")
    
    # 加载训练数据作为背景数据
    train_loader = get_loader_segment(
        DATA_PATH,
        batch_size=32,
        win_size=config['win_size'],
        mode='train',
        dataset='Custom'
    )
    
    # 加载测试数据
    test_loader = get_loader_segment(
        DATA_PATH,
        batch_size=32,
        win_size=config['win_size'],
        mode='test',
        dataset='Custom'
    )
    
    # 设置随机种子确保可重复性
    RANDOM_SEED = 42
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    
    # 准备背景数据
    background_data = prepare_background_data(train_loader, n_samples=N_BACKGROUND_SAMPLES, seed=RANDOM_SEED)
    
    # 准备测试数据 - 使用分层采样确保包含异常样本
    print("\n准备测试数据（分层采样）...")
    all_test_samples = []
    all_test_labels = []
    
    # 首先收集所有测试数据
    for input_data, labels in test_loader:
        all_test_samples.append(input_data)
        all_test_labels.append(labels)
    
    all_test_data = torch.cat(all_test_samples, dim=0)
    all_test_labels = torch.cat(all_test_labels, dim=0)
    
    # 分离正常和异常样本
    # 检查每个窗口是否包含异常（至少有一个时间点是异常）
    anomaly_mask = (all_test_labels.sum(dim=1) > 20)  # [N] bool tensor
    normal_indices = torch.where(~anomaly_mask)[0]
    anomaly_indices = torch.where(anomaly_mask)[0]
    
    print(f"测试集总样本数: {len(all_test_data)}")
    print(f"  正常样本数: {len(normal_indices)} ({100*len(normal_indices)/len(all_test_data):.1f}%)")
    print(f"  异常样本数: {len(anomaly_indices)} ({100*len(anomaly_indices)/len(all_test_data):.1f}%)")
    
    # 分层采样：确保包含一定比例的异常样本
    n_anomaly_samples = min(int(N_TEST_SAMPLES * 0.5), len(anomaly_indices))  # 至少30%异常样本
    n_normal_samples = N_TEST_SAMPLES - n_anomaly_samples
    
    # 如果异常样本不够，调整比例
    if n_anomaly_samples > len(anomaly_indices):
        n_anomaly_samples = len(anomaly_indices)
        n_normal_samples = N_TEST_SAMPLES - n_anomaly_samples
    
    # 随机选择样本（使用固定种子）
    torch.manual_seed(RANDOM_SEED)
    selected_anomaly_idx = anomaly_indices[torch.randperm(len(anomaly_indices))[:n_anomaly_samples]]
    selected_normal_idx = normal_indices[torch.randperm(len(normal_indices))[:n_normal_samples]]
    
    # 合并并打乱
    selected_indices = torch.cat([selected_anomaly_idx, selected_normal_idx])
    selected_indices = selected_indices[torch.randperm(len(selected_indices))]
    
    # 提取选中的样本
    test_data = all_test_data[selected_indices]
    test_labels_array = all_test_labels[selected_indices]
    
    # 确保测试数据需要梯度（SHAP需要）
    test_data = test_data.requires_grad_(True)
    
    print(f"\n最终测试数据: {test_data.shape}")
    print(f"  包含异常样本: {n_anomaly_samples} ({100*n_anomaly_samples/N_TEST_SAMPLES:.1f}%)")
    print(f"  包含正常样本: {n_normal_samples} ({100*n_normal_samples/N_TEST_SAMPLES:.1f}%)")
    
    # 获取特征名称
    df = pd.read_parquet(DATA_PATH)
    feature_cols = [col for col in df.columns if col not in ['TimeStamp', 'anomaly_label']]
    print(f"\n特征数量: {len(feature_cols)}")
    print(f"特征列表: {feature_cols[:10]}..." if len(feature_cols) > 10 else f"特征列表: {feature_cols}")
    
    # ==================== 3. 计算SHAP值 ====================
    # 使用较小的批处理大小以避免内存问题
    shap_batch_size = min(50, N_TEST_SAMPLES)  # 每批最多50个样本
    shap_values, explainer = generate_shap_values(
        wrapped_model,
        background_data,
        test_data,
        device,
        batch_size=shap_batch_size
    )
    
    # 确保SHAP值是numpy数组，形状为 [n_samples, win_size, n_features]
    if isinstance(shap_values, torch.Tensor):
        shap_values = shap_values.cpu().numpy()
    elif isinstance(shap_values, list):
        # 如果是列表，取第一个元素
        shap_values = shap_values[0]
        if isinstance(shap_values, torch.Tensor):
            shap_values = shap_values.cpu().numpy()
    
    # 处理SHAP值形状
    # ModelWrapper返回 [B, L]，SHAP会对每个输出单独计算SHAP值
    # 实际返回形状: [n_samples, n_outputs, n_features, win_size]
    # 其中 n_outputs = win_size（每个时间步一个输出）
    print(f"SHAP值原始形状: {shap_values.shape}")
    
    # SHAP对于多输出模型，返回: [n_samples, n_outputs, ...input_shape]
    # 我们的情况：输入 [B, L, D]，输出 [B, L]
    # SHAP返回: [n_samples, n_outputs, n_features, win_size]
    # 表示：第i个输出对第j个特征的每个时间步的SHAP值
    
    if len(shap_values.shape) != 4:
        raise ValueError(f"期望SHAP值形状为4D，但得到: {shap_values.shape}")
    
    n_samples, n_outputs, dim1, dim2 = shap_values.shape
    # 根据实际形状判断：应该是 [n_samples, n_outputs, n_features, win_size]
    # 其中 n_features=27, win_size=30
    if dim1 == len(feature_cols):
        # dim1是特征数
        print(f"SHAP值形状: [n_samples={n_samples}, n_outputs={n_outputs}, n_features={dim1}, win_size={dim2}]")
    else:
        # 可能是 [n_samples, n_outputs, win_size, n_features]
        print(f"SHAP值形状: [n_samples={n_samples}, n_outputs={n_outputs}, dim1={dim1}, dim2={dim2}]")
        print(f"  注意：需要根据实际维度调整代码")
    
    # 转换测试数据为numpy（需要先detach，因为数据设置了requires_grad）
    if isinstance(test_data, torch.Tensor):
        test_data_np = test_data.detach().cpu().numpy()
    else:
        test_data_np = test_data
    
    # ==================== 4. 生成可视化 ====================
    print("\n生成SHAP可视化图...")
    
    # 转换标签为numpy数组
    if isinstance(test_labels_array, torch.Tensor):
        test_labels_np = test_labels_array.cpu().numpy()
    else:
        test_labels_np = test_labels_array
    
    # 新的可视化函数已经包含多种分析方法
    plot_shap_summary(shap_values, test_data_np, feature_cols, output_dir, test_labels_np)
    
    # ==================== 5. 保存SHAP值 ====================
    shap_save_path = os.path.join(output_dir, "shap_values.npy")
    np.save(shap_save_path, shap_values)
    print(f"\n保存SHAP值: {shap_save_path}")
    
    # 保存配置信息
    info = {
        "model_path": MODEL_PATH,
        "data_path": DATA_PATH,
        "n_background_samples": N_BACKGROUND_SAMPLES,
        "n_test_samples": N_TEST_SAMPLES,
        "win_size": config['win_size'],
        "input_c": config['input_c'],
        "output_c": config['output_c'],
        "feature_names": feature_cols,
        "timestamp": timestamp
    }
    
    info_path = os.path.join(output_dir, "analysis_info.json")
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(f"保存分析信息: {info_path}")
    
    print(f"\n✅ SHAP分析完成！")
    print(f"所有结果保存在: {output_dir}")
    print("\n生成的文件:")
    for f in os.listdir(output_dir):
        print(f"  - {f}")


if __name__ == "__main__":
    main()

