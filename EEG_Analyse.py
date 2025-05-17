import mne
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel
from statsmodels.stats.anova import AnovaRM

# 1.1 数据读取
def read_csv_data(file_path):
    """读取CSV格式的EEG数据，并根据标记提取实验段"""
    # 读取CSV文件
    data = pd.read_csv(file_path)
    
    # 确定标记列名
    columns = data.columns
    start_markers = ['start', 's', 'jingxi', 'strat', 'JINGXI']
    end_markers = ['end', 'e', 'END']
    
    # 查找start标记的索引
    start_idx = None
    for marker in start_markers:
        if marker in columns:
            try:
                start_idx = data[data[marker] == 1].index[0]
                break
            except (IndexError, KeyError):
                continue
    
    # 查找end标记的索引
    end_idx = None
    for marker in end_markers:
        if marker in columns:
            try:
                end_idx = data[data[marker] == 1].index[0]
                break
            except (IndexError, KeyError):
                continue
    
    # 如果没有找到标记，使用整个数据集
    if start_idx is None or end_idx is None:
        print(f"警告：在文件 {os.path.basename(file_path)} 中未找到有效的start或end标记，使用整个数据集")
        exp_data = data
    else:
        # 确保end_idx大于start_idx
        if end_idx < start_idx:
            end_idx, start_idx = start_idx, end_idx
        # 提取实验段数据
        exp_data = data.iloc[start_idx:end_idx+1, :]
    
    # 提取时间和EEG数据列
    time_col = exp_data['Elapsed Time']
    
    # 查找EEG数据列（Fp1/fp1和Fp2/fp2）
    eeg_cols = []
    for col in columns:
        if col.lower() in ['fp1', 'fp2']:
            eeg_cols.append(col)
    
    if len(eeg_cols) < 2:
        print(f"警告：在文件 {os.path.basename(file_path)} 中未找到足够的EEG数据列")
        # 使用前两列作为EEG数据（除时间列外）
        eeg_data = exp_data.iloc[:, 1:3]
    else:
        eeg_data = exp_data[eeg_cols]
    
    # 创建MNE Raw对象
    ch_names = list(eeg_data.columns)
    ch_types = ['eeg'] * len(ch_names)
    info = mne.create_info(ch_names=ch_names, sfreq=1000, ch_types=ch_types)
    raw = mne.io.RawArray(eeg_data.T.values, info)
    
    return raw

# 1.2 解析文件名获取实验条件
def parse_filename(filename):
    """从文件名解析被试信息和实验条件顺序"""
    parts = os.path.basename(filename).split('-')
    subject_id = parts[0]
    gender = parts[1]  # F或M表示性别
    
    # 获取可能的条件部分
    potential_conditions = parts[2:]
    
    # 将条件映射为a(短视频)、b(长视频)、c(漫画)
    condition_map = {'短': 'a', '长': 'b', '漫': 'c'}
    
    # 过滤掉可能的非条件字符串（如.csv后缀）
    valid_conditions = []
    for cond in potential_conditions:
        # 移除可能的文件扩展名
        clean_cond = cond.split('.')[0] if '.' in cond else cond
        if clean_cond in condition_map:
            valid_conditions.append(clean_cond)
    
    mapped_conditions = [condition_map[cond] for cond in valid_conditions]
    
    # 确保至少有一个有效条件
    if not mapped_conditions:
        print(f"警告：在文件 {filename} 中未找到有效的实验条件")
        # 默认返回所有可能的条件，以便程序继续运行
        mapped_conditions = ['a', 'b', 'c']
    
    return subject_id, gender, mapped_conditions

# 1.3 批量处理所有被试数据
def process_all_subjects(data_folder):
    """批量处理所有被试的数据"""
    all_subjects_data = []
    
    # 获取所有CSV文件
    csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv') and not f.startswith('bcrx')]
    
    for csv_file in csv_files:
        file_path = os.path.join(data_folder, csv_file)
        subject_id, gender, conditions = parse_filename(csv_file)
        
        print(f"处理被试 {subject_id} 的数据...")
        
        # 读取数据
        raw = read_csv_data(file_path)
        
        # 2.1 预处理
        # 滤波处理 (参考文献: Klimesch, W. (1999). EEG alpha and theta oscillations reflect cognitive and memory performance)
        raw.filter(l_freq=0.5, h_freq=40, method='fir', phase='zero-double')  # 带通滤波0.5-40Hz
        raw.notch_filter(freqs=50)  # 陷波滤波去除工频干扰
        
        # 2.2 功率谱密度计算
        freqs = np.arange(1, 40, 1)  # 1-40Hz频段
        psds, freqs = mne.time_frequency.psd_welch(raw, fmin=1, fmax=40, n_jobs=4)
        
        # 2.3 提取左侧额区电极
        left_frontal_chs = [i for i, ch in enumerate(raw.ch_names) if ch in ['Fp1']]
        
        # 2.4 计算α波和β波能量
        alpha_power = np.mean(psds[left_frontal_chs, (freqs >= 8) & (freqs <= 13)], axis=1)
        beta_power = np.mean(psds[left_frontal_chs, (freqs > 13) & (freqs <= 30)], axis=1)
        beta_alpha_ratio = beta_power / alpha_power
        
        # 2.5 保存结果
        for i, condition in enumerate(conditions):
            all_subjects_data.append({
                'Subject': subject_id,
                'Gender': gender,
                'Condition': condition,
                'AlphaPower': alpha_power[0],
                'BetaPower': beta_power[0],
                'BetaAlphaRatio': beta_alpha_ratio[0]
            })
    
    # 转换为DataFrame
    return pd.DataFrame(all_subjects_data)

# 3. 主函数
def main():
    # 设置数据文件夹路径
    csv_folder = r"D:\BaiduSyncdisk\大学\课程学习\大二下\生理心理学\大作业\csv格式实验数据"
    
    # 处理所有被试数据
    results_df = process_all_subjects(csv_folder)
    
    # 3.1 统计分析
    # 重复测量方差分析 - α波能量
    anova_alpha = AnovaRM(results_df, 'AlphaPower', 'Subject', within=['Condition']).fit()
    print("\nα波能量重复测量方差分析结果:")
    print(anova_alpha.summary())
    
    # 重复测量方差分析 - β/α比值
    anova_ratio = AnovaRM(results_df, 'BetaAlphaRatio', 'Subject', within=['Condition']).fit()
    print("\nβ/α比值重复测量方差分析结果:")
    print(anova_ratio.summary())
    
    # 3.2 事后配对t检验（Bonferroni校正）
    print("\n事后配对t检验结果:")
    conditions = ['a', 'b', 'c']
    condition_names = {'a': '短视频', 'b': '长视频', 'c': '漫画'}
    adjusted_alpha = 0.05 / 3  # 校正三组比较
    
    for i, cond1 in enumerate(conditions):
        for cond2 in conditions[i+1:]:
            # β/α比值比较
            t_stat, p_val = ttest_rel(
                results_df[results_df.Condition==cond1].BetaAlphaRatio,
                results_df[results_df.Condition==cond2].BetaAlphaRatio)
            sig = "*" if p_val < adjusted_alpha else "ns"
            print(f"{condition_names[cond1]} vs {condition_names[cond2]} (β/α比值): t={t_stat:.3f}, p={p_val:.4f}, {sig}")
            
            # α波能量比较
            t_stat, p_val = ttest_rel(
                results_df[results_df.Condition==cond1].AlphaPower,
                results_df[results_df.Condition==cond2].AlphaPower)
            sig = "*" if p_val < adjusted_alpha else "ns"
            print(f"{condition_names[cond1]} vs {condition_names[cond2]} (α波能量): t={t_stat:.3f}, p={p_val:.4f}, {sig}")
    
    # 3.3 可视化结果
    plt.figure(figsize=(12, 5))
    
    # β/α比值箱线图
    plt.subplot(1, 2, 1)
    sns.boxplot(x='Condition', y='BetaAlphaRatio', data=results_df)
    plt.title('不同阅读材料的β/α比值')
    plt.xlabel('阅读材料')
    plt.ylabel('β/α比值')
    plt.xticks([0, 1, 2], ['短视频', '长视频', '漫画'])
    
    # α波能量箱线图
    plt.subplot(1, 2, 2)
    sns.boxplot(x='Condition', y='AlphaPower', data=results_df)
    plt.title('不同阅读材料的α波能量')
    plt.xlabel('阅读材料')
    plt.ylabel('α波能量')
    plt.xticks([0, 1, 2], ['短视频', '长视频', '漫画'])
    
    plt.tight_layout()
    plt.savefig('EEG_results.png', dpi=300)
    plt.show()
    
    # 保存结果到CSV
    results_df.to_csv('EEG_analysis_results.csv', index=False)
    print("\n分析结果已保存到 EEG_analysis_results.csv")

# 执行主函数
if __name__ == "__main__":
    main()