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
    """读取CSV格式的EEG数据，并根据标记提取所有实验段，返回[(raw, condition_label), ...]"""
    # 在函数开头初始化empty_segments
    empty_segments = {}
    
    # 定义标记到标签的映射关系
    key_to_label = {
        'long': '长',
        'short': '短', 
        'manga': '漫',
        'manhua': '漫',
    }
    
    data = pd.read_csv(file_path)
    columns = data.columns

    # 获取被试姓名
    filename = os.path.basename(file_path)
    subject_name = filename.split('-')[1] if '-' in filename else ""

    # 标记定义
    start_keys = [k for k in columns if k.lower() in ['start', 's', 'strat']]
    end_keys = [k for k in columns if k.lower() in ['end', 'e', 'end']]
    seg_keys = [k for k in columns if k.lower() in ['long','short','manga','manhua','LONG', 'SHORT']]
    jingxi_keys = [k for k in columns if k.lower() == 'jingxi']

    segments = []
    condition_labels = []

    # 文件名顺序标签
    file_conditions = ['长', '短', '漫']

    if subject_name == "JF" and jingxi_keys:
        # JF：最后一个JINGXI段为manhua
        idxs = data.index[data[jingxi_keys[0]] == 1].tolist()
        for i, s in enumerate(idxs):
            if i < len(idxs) - 1:
                e = idxs[i+1] - 1
            else:
                e = len(data) - 1
            exp_data = data.iloc[s:e+1, :]
            if i == len(idxs) - 1:
                label = "漫"
            else:
                label = file_conditions[i] if i < 2 else "漫"
            segments.append(exp_data)
            condition_labels.append(label)
    elif subject_name == "RHY" and jingxi_keys:
        # RHY：中间的jingxi段为manhua
        idxs = data.index[data[jingxi_keys[0]] == 1].tolist()
        for i, s in enumerate(idxs):
            if i < len(idxs) - 1:
                e = idxs[i+1] - 1
            else:
                e = len(data) - 1
            exp_data = data.iloc[s:e+1, :]
            if i == 2:
                label = "漫"
            else:
                label = file_conditions[i] if i < 2 else "漫"
            segments.append(exp_data)
            condition_labels.append(label)
    elif subject_name == "MPX" and end_keys:
        # MPX：end为结束，end前一个marker为开始，取后三个时间段
        end_col = end_keys[0]
        end_idxs = data.index[data[end_col] == 1].tolist()
        # 取后三个end
        end_idxs = end_idxs[-3:]
        start_idxs = []
        for e in end_idxs:
            # 找end前面最近的一个非end的marker为1的行
            found = False
            for i in range(e-1, -1, -1):
                row = data.iloc[i]
                for k in columns:
                    if k != end_col and row.get(k, 0) == 1:
                        start_idxs.append(i)
                        found = True
                        break
                if found:
                    break
        for s, e, label in zip(start_idxs, end_idxs, file_conditions[-3:]):
            exp_data = data.iloc[s:e+1, :]
            segments.append(exp_data)
            condition_labels.append(label)
    elif subject_name == "LXY" and end_keys:
        # LXY：end为结束，end前一个marker为开始，取后三个时间段
        end_col = end_keys[0]
        end_idxs = data.index[data[end_col] == 1].tolist()
        # 取后三个end
        end_idxs = end_idxs[-3:]
        start_idxs = []
        for e in end_idxs:
            # 找end前面最近的一个非end的marker为1的行
            found = False
            for i in range(e-1, -1, -1):
                row = data.iloc[i]
                for k in columns:
                    if k != end_col and row.get(k, 0) == 1:
                        start_idxs.append(i)
                        found = True
                        break
                if found:
                    break
        for s, e, label in zip(start_idxs, end_idxs, file_conditions[-3:]):
            exp_data = data.iloc[s:e+1, :]
            segments.append(exp_data)
            condition_labels.append(label)
    elif subject_name in ["YYH", "YYX", "ZJM", "ZHR", "WZM", "WQX"] and end_keys:
        # 这些被试：end为结束，end前一个marker为开始，取后三个时间段
        end_col = end_keys[0]
        end_idxs = data.index[data[end_col] == 1].tolist()
        # 取后三个end
        end_idxs = end_idxs[-3:]
        start_idxs = []
        for e in end_idxs:
            # 找end前面最近的一个非end的marker为1的行
            found = False
            for i in range(e-1, -1, -1):
                row = data.iloc[i]
                for k in columns:
                    if k != end_col and row.get(k, 0) == 1:
                        start_idxs.append(i)
                        found = True
                        break
                if found:
                    break
        # 直接使用file_conditions中的标签，不再映射jingxi
        for s, e, label in zip(start_idxs, end_idxs, file_conditions[-3:]):
            exp_data = data.iloc[s:e+1, :]
            segments.append(exp_data)
            condition_labels.append(label)
    elif start_keys and end_keys:
        # 通用start/end分段
        start_col = start_keys[0]
        end_col = end_keys[0]
        start_idxs = data.index[data[start_col] == 1].tolist()
        end_idxs = data.index[data[end_col] == 1].tolist()
        seg_count = min(len(start_idxs), len(end_idxs), 3)
        for i in range(seg_count):
            s, e = start_idxs[i], end_idxs[i]
            if e < s:
                s, e = e, s
            exp_data = data.iloc[s:e+1, :]
            segments.append(exp_data)
            condition_labels.append(file_conditions[i])
    elif seg_keys:
        # 直接用long/short/manga等分段
        seg_keys = [k for k in columns if k.lower() in key_to_label.keys()]
        seg_count = min(len(seg_keys), 3)
        print(f"被试 {subject_name} 的分段标记: {seg_keys}")
        
        for i, key in enumerate(seg_keys[:3]):
            idxs = data.index[data[key] == 1].tolist()
            print(f"标记 {key} 的索引位置: {idxs}")
            if not idxs:
                print(f"警告：标记 {key} 没有找到有效数据点")
                empty_segments[key] = "未找到标记点"
                continue
                
            s = idxs[0]
            if i+1 < len(seg_keys):
                next_idxs = data.index[data[seg_keys[i+1]] == 1].tolist()
                e = next_idxs[0]-1 if next_idxs else len(data)-1
            else:
                e = len(data)-1
                
            # 添加更严格的分段范围验证
            if s >= len(data) or e >= len(data) or s > e:
                print(f"警告：无效的分段范围 {s}-{e}，跳过此分段")
                empty_segments[key] = f"无效范围 {s}-{e}"
                continue
                
            # 检查分段长度是否合理（至少1000个数据点，约1秒数据）
            if (e - s) < 1000:
                print(f"警告：分段 {key} 数据点不足({e-s})，可能导致功率谱计算失败")
                empty_segments[key] = f"数据点不足 {e-s}"
                continue
                
            exp_data = data.iloc[s:e+1, :]
            
            # 检查数据是否全为零或接近零
            if np.allclose(exp_data.select_dtypes(include=[np.number]).values, 0, atol=1e-6):
                print(f"警告：分段 {key} 数据全为零或接近零，跳过")
                empty_segments[key] = "数据全为零"
                continue
                
            label = key_to_label.get(key.lower(), f"未知_{key}")
            segments.append(exp_data)
            condition_labels.append(label)
    else:
        print(f"警告：文件 {os.path.basename(file_path)} 未检测到有效分段标记，使用全段")
        segments = [data]
        condition_labels = [file_conditions[0]]

    # 提取EEG数据并生成raw对象
    results = []
    for exp_data, cond_label in zip(segments, condition_labels):
        eeg_cols = [col for col in exp_data.columns if col.lower() in ['fp1', 'fp2']]
        if len(eeg_cols) < 2:
            non_time_cols = [col for col in exp_data.columns if col != 'Elapsed Time']
            eeg_data = exp_data[non_time_cols[:2]]
        else:
            eeg_data = exp_data[eeg_cols]
        ch_names = list(eeg_data.columns)
        ch_types = ['eeg'] * len(ch_names)
        info = mne.create_info(ch_names=ch_names, sfreq=1000, ch_types=ch_types)
        raw = mne.io.RawArray(eeg_data.T.values, info)
        results.append((raw, cond_label))
        
    # 在函数末尾输出空数据段记录
    if empty_segments:
        print("\n空数据段记录:")
        for key, info in empty_segments.items():
            print(f"标记 {key}: {info}")
            
    return results

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
    csv_files = [f for f in os.listdir(data_folder) 
                if f.endswith('.csv') and not f.startswith('bcrx')
                and any(f.startswith(f"{num}-") for num in ["35", "36"])]  # 只处理35和36编号
    
    for csv_file in csv_files:
        file_path = os.path.join(data_folder, csv_file)
        subject_id, gender, conditions = parse_filename(csv_file)
        
        print(f"处理被试 {subject_id} 的数据...")
        
        # 读取数据，返回(raw, label)对
        raw_label_list = read_csv_data(file_path)
        
        for (raw, cond_label) in raw_label_list:
            # 2.1 预处理
            raw.filter(l_freq=0.5, h_freq=40, method='fir', phase='zero-double')  # 带通滤波0.5-40Hz
            raw.notch_filter(freqs=50)  # 陷波滤波去除工频干扰
            
            # 2.2 功率谱密度计算 - 添加数据有效性检查
            if len(raw.get_data()) == 0:
                print(f"警告：被试 {subject_id} 条件 {cond_label} 的数据为空，跳过")
                continue
                
            try:
                # 检查数据是否全为零或接近零
                if np.allclose(raw.get_data(), 0, atol=1e-6):
                    print(f"警告：被试 {subject_id} 条件 {cond_label} 的数据全为零，跳过功率谱计算")
                    continue
                    
                freqs = np.arange(1, 40, 1)
                psds, freqs = mne.time_frequency.psd_array_welch(
                    raw.get_data(),
                    sfreq=raw.info['sfreq'],
                    fmin=1,
                    fmax=40,
                    n_jobs=4,
                    n_fft=2048,  # 增加FFT点数以提高频率分辨率
                    n_overlap=512  # 增加重叠以提高估计稳定性
                )
            except Exception as e:
                print(f"警告：被试 {subject_id} 条件 {cond_label} 的功率谱计算失败: {str(e)}")
                continue
                
            # 2.3 提取左侧额区电极
            left_frontal_chs = [i for i, ch in enumerate(raw.ch_names) if ch in ['Fp1']]
            
            # 2.4 计算α波和β波能量
            alpha_mask = (freqs >= 8) & (freqs <= 13)
            beta_mask = (freqs > 13) & (freqs <= 30)
            alpha_power = np.mean(psds[left_frontal_chs][:, alpha_mask], axis=1)
            beta_power = np.mean(psds[left_frontal_chs][:, beta_mask], axis=1)
            beta_alpha_ratio = beta_power / alpha_power
            
            all_subjects_data.append({
                'Subject': subject_id,
                'Gender': gender,
                'Condition': cond_label,
                'AlphaPower': alpha_power[0] if len(alpha_power) > 0 else np.nan,
                'BetaPower': beta_power[0] if len(beta_power) > 0 else np.nan,
                'BetaAlphaRatio': beta_alpha_ratio[0] if len(beta_alpha_ratio) > 0 else np.nan
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