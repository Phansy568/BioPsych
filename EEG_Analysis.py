import mne
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel
from statsmodels.stats.anova import AnovaRM
import datetime

# 1.1 数据读取
def read_csv_data(file_path, marker_df, file_conditions_order=None):
    """读取CSV格式的EEG数据，并根据标记时间表提取所有实验段，返回[(raw, condition_label), ...]"""
    # 在函数开头初始化empty_segments
    empty_segments = {}

    data = pd.read_csv(file_path)
    columns = data.columns

    # 获取被试ID
    filename = os.path.basename(file_path)
    subject_id = filename.split('-')[1] if '-' in filename else ""
    
    print(f"处理被试 {subject_id} 的数据...")
    
    # 从标记时间表中获取该被试的标记
    subject_markers = marker_df[marker_df['被试ID'] == subject_id]
    
    if len(subject_markers) == 0:
        print(f"警告：在标记时间表中未找到被试 {subject_id} 的标记")
        return []
    
    # 按时间排序标记
    subject_markers = subject_markers.sort_values('时间')
    
    # 检查是否有足够的标记（至少需要8个标记，4段的开始和结束）
    if len(subject_markers) < 8:
        print(f"警告：被试 {subject_id} 的标记数量不足，只有 {len(subject_markers)} 个")
        return []
    
    # 文件名顺序标签，如果提供了file_conditions_order，则使用它
    file_conditions = file_conditions_order if file_conditions_order else ['长', '短', '漫']
    
    print(f"使用条件顺序: {file_conditions}")
    
    # 将标记分为4段（每段2个标记）
    segments = []
    condition_labels = []
    
    # 将时间字符串转换为秒数
    def time_to_seconds(time_str):
        time_parts = time_str.split(':')
        hours = int(time_parts[0])
        minutes = int(time_parts[1])
        seconds = float(time_parts[2])
        return hours * 3600 + minutes * 60 + seconds
    
    # 获取所有标记的时间（秒）
    marker_times = []
    for _, row in subject_markers.iterrows():
        marker_times.append(time_to_seconds(row['时间']))
    
    # 确保数据中有Elapsed Time列
    if 'Elapsed Time' not in data.columns:
        print(f"警告：文件 {filename} 中没有Elapsed Time列，无法按时间分段")
        return []
    
    # 将数据的Elapsed Time转换为秒数
    data_times = []
    for t in data['Elapsed Time']:
        data_times.append(time_to_seconds(t) if isinstance(t, str) else t)
    
    data['Elapsed Time (seconds)'] = data_times
    
    # 分割4段数据
    for i in range(0, len(marker_times), 2):
        if i+1 >= len(marker_times):
            break
        
        start_time = marker_times[i]
        end_time = marker_times[i+1]
        
        # 找到对应时间范围的数据
        segment_data = data[(data['Elapsed Time (seconds)'] >= start_time) & 
                            (data['Elapsed Time (seconds)'] <= end_time)]
        
        if len(segment_data) == 0:
            print(f"警告：未找到时间范围 {start_time}-{end_time} 的数据")
            continue
        
        segments.append(segment_data)
        
        # 第一段为静息段，其余三段按文件名顺序标记
        if i == 0:
            condition_labels.append('静息')
        else:
            # 计算当前是第几个非静息段（从0开始）
            exp_idx = (i // 2) - 1
            if exp_idx < len(file_conditions):
                condition_labels.append(file_conditions[exp_idx])
            else:
                condition_labels.append(f"未知_{exp_idx}")
    
    # 提取EEG数据并生成raw对象
    results = []
    for exp_data, cond_label in zip(segments, condition_labels):
        eeg_cols = [col for col in exp_data.columns if col.lower() in ['fp1', 'fp2']]
        if len(eeg_cols) < 2:
            non_time_cols = [col for col in exp_data.columns if col not in ['Elapsed Time', 'Elapsed Time (seconds)']]
            if len(non_time_cols) < 2:
                print(f"警告：找不到足够的EEG通道，只有 {len(non_time_cols)} 个")
                continue
            eeg_data = exp_data[non_time_cols[:2]].copy()
        else:
            eeg_data = exp_data[eeg_cols].copy()
        
        # 在创建RawArray前对eeg_data进行NaN插值
        if eeg_data.isnull().values.any():
            print(f"INFO: 被试 {subject_id}, 条件 {cond_label} - eeg_data 包含NaN，尝试插值...")
            for col in eeg_data.columns:
                if eeg_data[col].isnull().any():
                    # 确保列是数值类型，非数值转为NaN
                    eeg_data[col] = pd.to_numeric(eeg_data[col], errors='coerce')
                    
                    if eeg_data[col].isnull().all():
                        print(f"警告：被试 {subject_id}, 条件 {cond_label}, 列 {col} 全是NaN。将用0填充。")
                        eeg_data[col] = eeg_data[col].fillna(0)
                        continue
                    
                    # 使用pandas内置插值
                    try:
                        eeg_data[col] = eeg_data[col].interpolate(method='linear', limit_direction='both', axis=0)
                        # 如果线性插值后仍有NaN (通常是序列开头或末尾的NaN)，尝试用前后值填充
                        if eeg_data[col].isnull().any():
                            eeg_data[col] = eeg_data[col].fillna(method='bfill').fillna(method='ffill')
                        
                        remaining_nans = eeg_data[col].isnull().sum()
                        if remaining_nans > 0:
                            print(f"警告：被试 {subject_id}, 条件 {cond_label}, 列 {col} 插值后仍有 {remaining_nans} 个NaN。将用0填充剩余NaN。")
                            eeg_data[col] = eeg_data[col].fillna(0)
                        else:
                            print(f"INFO: 被试 {subject_id}, 条件 {cond_label}, 列 {col} 插值完成。")
                            
                    except Exception as e_interp:
                        print(f"错误：被试 {subject_id}, 条件 {cond_label}, 列 {col} 插值失败: {e_interp}。将用0填充NaN。")
                        eeg_data[col] = eeg_data[col].fillna(0)

            if eeg_data.isnull().values.any():
                 print(f"严重警告：被试 {subject_id}, 条件 {cond_label} - eeg_data 在插值尝试后仍包含NaN。这可能导致后续处理问题。")

        ch_names = list(eeg_data.columns)
        ch_types = ['eeg'] * len(ch_names)
        info = mne.create_info(ch_names=ch_names, sfreq=1000, ch_types=ch_types)
        print(f"条件 {cond_label} 的EEG数据形状: {eeg_data.shape}")
        raw = mne.io.RawArray(eeg_data.T.values, info)
        results.append((raw, cond_label))
    
    return results

# 1.2 解析文件名获取实验条件
def parse_filename(filename):
    """从文件名解析被试信息和实验条件顺序"""
    parts = os.path.basename(filename).split('-')
    subject_id = parts[1] if len(parts) > 1 else parts[0]  # 使用第二部分作为被试ID
    gender = parts[2] if len(parts) > 2 else ""  # F或M表示性别
    
    # 获取可能的条件部分
    potential_conditions = parts[3:] if len(parts) > 3 else []
    
    # 将条件映射为a(短视频)、b(长视频)、c(漫画)
    condition_map = {'短': 'a', '长': 'b', '漫': 'c'}
    
    # 过滤掉可能的非条件字符串（如.csv后缀）
    valid_conditions = []
    for cond in potential_conditions:
        # 移除可能的文件扩展名
        clean_cond = cond.split('.')[0] if '.' in cond else cond
        if clean_cond in condition_map:
            valid_conditions.append(clean_cond)
    
    # 确保至少有一个有效条件
    if not valid_conditions:
        print(f"警告：在文件 {filename} 中未找到有效的实验条件")
        # 默认返回所有可能的条件，以便程序继续运行
        valid_conditions = ['长', '短', '漫']
    
    return subject_id, gender, valid_conditions

# 1.3 批量处理所有被试数据
def process_all_subjects(data_folder, marker_file):
    """批量处理所有被试的数据"""
    all_subjects_data = []
    
    # 读取标记时间表
    marker_df = pd.read_csv(marker_file)
    
    # 获取所有CSV文件，不进行筛选
    csv_files_to_process = [f for f in os.listdir(data_folder)
                     if f.endswith('.csv') and not f.startswith('bcrx')]

    print(f"将要处理的被试文件: {csv_files_to_process}")

    # 存储每个被试的静息段数据，用于后续标准化
    rest_data = {}

    for csv_file in csv_files_to_process:
        file_path = os.path.join(data_folder, csv_file)
        subject_id, gender, conditions = parse_filename(csv_file)

        print(f"处理被试 {subject_id} 的数据...")
        
        # 将条件转换为中文标签
        file_conditions_order = conditions
        
        # 读取数据，返回(raw, label)对，传递条件顺序和标记时间表
        raw_label_list = read_csv_data(file_path, marker_df, file_conditions_order)
        
        # 存储该被试的所有条件数据，用于后续计算标准化指标
        subject_data = {}
        
        for (raw, cond_label) in raw_label_list:
            # 2.1 预处理
            raw.filter(l_freq=1, h_freq=30, method='fir', phase='zero-double')  # 带通滤波1-30Hz
            raw.notch_filter(freqs=50)  # 陷波滤波去除工频干扰
            
            # 2.2 功率谱密度计算
            try:
                # 检查数据是否全为零或接近零
                if np.allclose(raw.get_data(), 0, atol=1e-6):
                    print(f"警告：被试 {subject_id} 条件 {cond_label} 的数据全为零，跳过功率谱计算")
                    continue
                    
                # 添加额外的数据检查
                data_var = np.var(raw.get_data())
                if data_var < 1e-10:
                    print(f"警告：被试 {subject_id} 条件 {cond_label} 的数据方差过小 ({data_var})，跳过功率谱计算")
                    continue
                    
                freqs = np.arange(8, 30, 1)
                # 计算功率谱
                psds, freqs = mne.time_frequency.psd_array_welch(
                    raw.get_data(),
                    sfreq=raw.info['sfreq'],
                    fmin=8,
                    fmax=30,
                    n_jobs=4,
                    n_fft=1000,
                    n_per_seg=1000,
                    n_overlap=500,
                    average='mean'
                )
            except Exception as e:
                print(f"警告：被试 {subject_id} 条件 {cond_label} 的功率谱计算失败: {str(e)}")
                continue
            
            # 2.3 提取左右额区电极
            left_frontal_chs = [i for i, ch in enumerate(raw.ch_names) if ch.lower() in ['fp1']]
            right_frontal_chs = [i for i, ch in enumerate(raw.ch_names) if ch.lower() in ['fp2']]
            
            if not left_frontal_chs or not right_frontal_chs:
                print(f"警告：被试 {subject_id} 条件 {cond_label} 未找到左右额区电极")
                continue
            
            # 2.4 计算α波和β波能量
            alpha_mask = (freqs >= 8) & (freqs <= 12)
            beta_mask = (freqs > 13) & (freqs <= 30)
            
            # 左侧额区
            left_alpha_power = np.mean(psds[left_frontal_chs][:, alpha_mask], axis=1)[0]
            left_beta_power = np.mean(psds[left_frontal_chs][:, beta_mask], axis=1)[0]
            
            # 右侧额区
            right_alpha_power = np.mean(psds[right_frontal_chs][:, alpha_mask], axis=1)[0]
            right_beta_power = np.mean(psds[right_frontal_chs][:, beta_mask], axis=1)[0]
            
            # 计算α偏侧化指数（FP2右-FP1左的平均功率）
            alpha_lateralization = right_alpha_power - left_alpha_power
            
            # 计算注意力投入程度β/α（alpha的平均功率除以beta）
            attention_engagement = left_beta_power / left_alpha_power if left_alpha_power > 0 else np.nan
            
            # 存储该条件的数据前添加异常点检测
            subject_data[cond_label] = {
                'left_alpha': left_alpha_power,
                'right_alpha': right_alpha_power,
                'left_beta': left_beta_power,
                'right_beta': right_beta_power,
                'alpha_lateralization': alpha_lateralization,
                'attention_engagement': attention_engagement
            }
            
            # 如果是静息段，存储到rest_data中
            if cond_label == '静息':
                rest_data[subject_id] = subject_data[cond_label]
        
        # 计算标准化指标并添加到结果中
        for cond_label, data in subject_data.items():
            if cond_label != '静息' and subject_id in rest_data:
                # 标准化α偏侧化指数（α偏侧化指数除以静息段的平均功率）
                norm_alpha_lateralization = data['alpha_lateralization'] / rest_data[subject_id]['alpha_lateralization'] if rest_data[subject_id]['alpha_lateralization'] != 0 else np.nan
                
                # 标准化注意力投入程度β/α（注意力投入程度β/α除以静息段的平均功率）
                norm_attention_engagement = data['attention_engagement'] / rest_data[subject_id]['attention_engagement'] if rest_data[subject_id]['attention_engagement'] != 0 else np.nan
                
                all_subjects_data.append({
                    'Subject': subject_id,
                    'Gender': gender,
                    'Condition': cond_label,
                    'AlphaPower': data['left_alpha'],
                    'BetaPower': data['left_beta'],
                    'BetaAlphaRatio': data['left_beta'] / data['left_alpha'] if data['left_alpha'] > 0 else np.nan,
                    'AlphaLateralization': data['alpha_lateralization'],
                    'AttentionEngagement': data['attention_engagement'],
                    'NormAlphaLateralization': norm_alpha_lateralization,
                    'NormAttentionEngagement': norm_attention_engagement
                })
    
    # 转换为DataFrame
    return pd.DataFrame(all_subjects_data)

# 3. 主函数
def main():
    # 设置中文字体
    plt.rcParams["font.sans-serif"] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams["axes.unicode_minus"] = False
    
    # 设置数据文件夹路径
    csv_folder = r"D:\BaiduSyncdisk\大学\课程学习\大二下\生理心理学\大作业\csv格式实验数据"
    marker_file = r"D:\BaiduSyncdisk\大学\课程学习\大二下\生理心理学\大作业\标记时间表.csv"
    
    # 处理所有被试数据
    results_df = process_all_subjects(csv_folder, marker_file)
    
    # 3.1 统计分析
    # 检查每个被试在每个条件下的观察值数量
    print("\n每个被试在每个条件下的观察值数量:")
    obs_counts = results_df.groupby(['Subject', 'Condition']).size().reset_index(name='count')
    print(obs_counts[obs_counts['count'] > 1])  # 打印有多个观察值的组合

    # 聚合数据，确保每个被试在每个条件下只有一个观察值
    aggregated_df = results_df.groupby(['Subject', 'Condition']).agg({
        'Gender': 'first',
        'AlphaPower': 'mean',
        'BetaPower': 'mean',
        'BetaAlphaRatio': 'mean',
        'AlphaLateralization': 'mean',
        'AttentionEngagement': 'mean',
        'NormAlphaLateralization': 'mean',
        'NormAttentionEngagement': 'mean'
    }).reset_index()

    print(f"原始数据行数: {len(results_df)}, 聚合后数据行数: {len(aggregated_df)}")

    # 使用聚合后的数据进行统计分析
    # 检查数据平衡性
    print("\n数据平衡性检查:")
    for subject in results_df['Subject'].unique():
        conditions = results_df[results_df['Subject'] == subject]['Condition'].unique()
        print(f"被试 {subject}: {len(conditions)} 个条件 - {', '.join(conditions)}")
    
    # 尝试使用能处理不平衡数据的方法进行统计分析
    try:
        # 重复测量方差分析 - 标准化α偏侧化指数
        anova_alpha_lat = AnovaRM(results_df, 'NormAlphaLateralization', 'Subject', within=['Condition']).fit()
        print("\n标准化α偏侧化指数重复测量方差分析结果:")
        print(anova_alpha_lat.summary())
        
        # 重复测量方差分析 - 标准化注意力投入程度
        anova_attention = AnovaRM(results_df, 'NormAttentionEngagement', 'Subject', within=['Condition']).fit()
        print("\n标准化注意力投入程度重复测量方差分析结果:")
        print(anova_attention.summary())
    except ValueError as e:
        print(f"\n方差分析失败: {str(e)}")
        print("尝试使用配对t检验进行条件间比较...")
    
    # 3.2 事后配对t检验（Bonferroni校正）
    print("\n事后配对t检验结果:")
    conditions = ['长', '短', '漫']
    adjusted_alpha = 0.05 / 3  # 校正三组比较
    
    for i, cond1 in enumerate(conditions):
        for cond2 in conditions[i+1:]:
            # 标准化α偏侧化指数比较
            data1 = results_df[results_df.Condition==cond1].NormAlphaLateralization
            data2 = results_df[results_df.Condition==cond2].NormAlphaLateralization
            t_stat, p_val = ttest_rel(data1, data2)
            # 计算效应量Cohen's d
            diff = data1 - data2
            cohens_d = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) != 0 else float('nan')
            sig = "*" if p_val < adjusted_alpha else "ns"
            print(f"{cond1} vs {cond2} (标准化α偏侧化指数): t={t_stat:.3f}, p={p_val:.4f}, d={cohens_d:.3f}, {sig}")
            
            # 标准化注意力投入程度比较
            data1 = results_df[results_df.Condition==cond1].NormAttentionEngagement
            data2 = results_df[results_df.Condition==cond2].NormAttentionEngagement
            t_stat, p_val = ttest_rel(data1, data2)
            diff = data1 - data2
            cohens_d = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) != 0 else float('nan')
            sig = "*" if p_val < adjusted_alpha else "ns"
            print(f"{cond1} vs {cond2} (标准化注意力投入程度): t={t_stat:.3f}, p={p_val:.4f}, d={cohens_d:.3f}, {sig}")
    
    # 3.3 可视化结果
    plt.figure(figsize=(5, 10))
    
    # 标准化α偏侧化指数箱线图
    plt.subplot(2, 2, 1)
    sns.boxplot(x='Condition', y='NormAlphaLateralization', data=results_df, showfliers=False)
    plt.title('不同阅读材料的标准化α偏侧化指数')
    plt.xlabel('阅读材料')
    plt.ylabel('标准化α偏侧化指数')
    plt.gca().set_autoscale_on(True)

    # 标准化注意力投入程度箱线图
    plt.subplot(2, 2, 2)
    sns.boxplot(x='Condition', y='NormAttentionEngagement', data=results_df, showfliers=False)
    plt.title('不同阅读材料的标准化注意力投入程度')
    plt.xlabel('阅读材料')
    plt.ylabel('标准化注意力投入程度')

    # α偏侧化指数箱线图
    plt.subplot(2, 2, 3)
    sns.boxplot(x='Condition', y='AlphaLateralization', data=results_df, showfliers=False)
    plt.title('不同阅读材料的α偏侧化指数')
    plt.xlabel('阅读材料')
    plt.ylabel('α偏侧化指数')
    plt.gca().set_autoscale_on(True)

    # 注意力投入程度箱线图
    plt.subplot(2, 2, 4)
    sns.boxplot(x='Condition', y='AttentionEngagement', data=results_df, showfliers=False)
    plt.title('不同阅读材料的注意力投入程度')
    plt.xlabel('阅读材料')
    plt.ylabel('注意力投入程度')
    
    plt.tight_layout()
    plt.savefig('EEG_results.png', dpi=300)
    plt.show()
    
    # 保存结果到CSV
    results_df.to_csv('EEG_analysis_results.csv', index=False)
    print("\n分析结果已保存到 EEG_analysis_results.csv")

if __name__ == "__main__":
    main()