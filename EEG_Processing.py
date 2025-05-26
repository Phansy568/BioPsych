import mne
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import datetime

mne.set_log_level('ERROR')  # 可选等级：'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'

def read_csv_data(file_path, marker_df, file_conditions_order=None):
    """读取CSV格式的EEG数据，并根据标记时间表提取所有实验段的多个时间段，返回[(raw, condition_label, segment_type), ...]"""
    # 在函数开头初始化empty_segments
    empty_segments = {}

    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误：文件 {file_path} 不存在")
        return []
        
    try:
        data = pd.read_csv(file_path)
    except Exception as e:
        print(f"错误：无法读取文件 {file_path}：{str(e)}")
        return []
        
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
    segment_types = []
    
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
    
    # 分割4段数据，每段截取多个时间段
    for i in range(0, len(marker_times), 2):
        if i+1 >= len(marker_times):
            break
        
        start_time = marker_times[i]
        end_time = marker_times[i+1]
        total_duration = end_time - start_time
        
        # 定义多个时间段的分割点
        # 前1/3段
        early_start = start_time
        early_end = start_time + total_duration / 3
        
        # 中间1/3段
        middle_start = start_time + total_duration / 3
        middle_end = start_time + 2 * total_duration / 3
        
        # 后1/3段
        late_start = start_time + 2 * total_duration / 3
        late_end = end_time
        
        # 找到对应时间范围的前1/3段数据
        early_segment = data[(data['Elapsed Time (seconds)'] >= early_start) & 
                            (data['Elapsed Time (seconds)'] < early_end)]
        
        # 找到对应时间范围的中间1/3段数据
        middle_segment = data[(data['Elapsed Time (seconds)'] >= middle_start) & 
                             (data['Elapsed Time (seconds)'] < middle_end)]
        
        # 找到对应时间范围的后1/3段数据
        late_segment = data[(data['Elapsed Time (seconds)'] >= late_start) & 
                           (data['Elapsed Time (seconds)'] <= late_end)]
        
        # 检查各段数据是否为空
        if len(early_segment) == 0:
            print(f"警告：未找到时间范围 {early_start}-{early_end} 的数据")
        else:
            segments.append(early_segment)
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
            segment_types.append('前段')
        
        if len(middle_segment) == 0:
            print(f"警告：未找到时间范围 {middle_start}-{middle_end} 的数据")
        else:
            segments.append(middle_segment)
            if i == 0:
                condition_labels.append('静息')
            else:
                exp_idx = (i // 2) - 1
                if exp_idx < len(file_conditions):
                    condition_labels.append(file_conditions[exp_idx])
                else:
                    condition_labels.append(f"未知_{exp_idx}")
            segment_types.append('中段')
        
        if len(late_segment) == 0:
            print(f"警告：未找到时间范围 {late_start}-{late_end} 的数据")
        else:
            segments.append(late_segment)
            if i == 0:
                condition_labels.append('静息')
            else:
                exp_idx = (i // 2) - 1
                if exp_idx < len(file_conditions):
                    condition_labels.append(file_conditions[exp_idx])
                else:
                    condition_labels.append(f"未知_{exp_idx}")
            segment_types.append('后段')
    
    # 提取EEG数据并生成raw对象
    results = []
    for exp_data, cond_label, segment_type in zip(segments, condition_labels, segment_types):
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
            print(f"INFO: 被试 {subject_id}, 条件 {cond_label}, 段落 {segment_type} - eeg_data 包含NaN，尝试插值...")
            for col in eeg_data.columns:
                if eeg_data[col].isnull().any():
                    # 确保列是数值类型，非数值转为NaN
                    eeg_data[col] = pd.to_numeric(eeg_data[col], errors='coerce')
                    
                    if eeg_data[col].isnull().all():
                        print(f"警告：被试 {subject_id}, 条件 {cond_label}, 段落 {segment_type}, 列 {col} 全是NaN。将用0填充。")
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
                            print(f"警告：被试 {subject_id}, 条件 {cond_label}, 段落 {segment_type}, 列 {col} 插值后仍有 {remaining_nans} 个NaN。将用0填充剩余NaN。")
                            eeg_data[col] = eeg_data[col].fillna(0)
                        else:
                            print(f"INFO: 被试 {subject_id}, 条件 {cond_label}, 段落 {segment_type}, 列 {col} 插值完成。")
                            
                    except Exception as e_interp:
                        print(f"错误：被试 {subject_id}, 条件 {cond_label}, 段落 {segment_type}, 列 {col} 插值失败: {e_interp}。将用0填充NaN。")
                        eeg_data[col] = eeg_data[col].fillna(0)

            if eeg_data.isnull().values.any():
                 print(f"严重警告：被试 {subject_id}, 条件 {cond_label}, 段落 {segment_type} - eeg_data 在插值尝试后仍包含NaN。这可能导致后续处理问题。")

        ch_names = list(eeg_data.columns)
        ch_types = ['eeg'] * len(ch_names)
        info = mne.create_info(ch_names=ch_names, sfreq=1000, ch_types=ch_types)
        print(f"条件 {cond_label}, 段落 {segment_type} 的EEG数据形状: {eeg_data.shape}")
        raw = mne.io.RawArray(eeg_data.T.values, info)
        results.append((raw, cond_label, segment_type))
    
    return results

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

def preprocess_eeg(raw, subject_id, cond_label, segment_type):
    """
    对EEG数据进行预处理，包括滤波、去除眼动伪迹等
    
    参数:
    raw - MNE Raw对象
    subject_id - 被试ID
    cond_label - 条件标签
    segment_type - 段落类型
    
    返回:
    处理后的MNE Raw对象，如果处理失败则返回None
    """
    try:
        print(f"[DEBUG] 进入3-sigma异常值剔除流程，被试: {subject_id}, 条件: {cond_label}, 段落: {segment_type}")
        # 1.3-sigma原则
        data = raw.get_data()
        std_dev = np.std(data)
        mean_val = np.mean(data)
        threshold = 3 * std_dev
        print(f"[DEBUG] 均值: {mean_val}, 标准差: {std_dev}, 阈值: {threshold}")
        mask = np.abs(data - mean_val) <= threshold
        # 对每个通道分别处理异常值
        for i in range(data.shape[0]):
            channel_mask = mask[i]
            if not np.all(channel_mask):
                # 用线性插值填补异常值
                channel_data = data[i]
                good_idx = np.where(channel_mask)[0]
                bad_idx = np.where(~channel_mask)[0]
                print(f"[DEBUG] 通道 {raw.ch_names[i]} 异常点索引: {bad_idx}")
                if len(good_idx) > 1:
                    channel_data[bad_idx] = np.interp(bad_idx, good_idx, channel_data[good_idx])
                else:
                    channel_data[bad_idx] = mean_val
                data[i] = channel_data
                print(f"被试 {subject_id} 条件 {cond_label}, 段落 {segment_type} - 通道 {raw.ch_names[i]} 剔除异常值")
        raw._data = data
        print(f"[DEBUG] 3-sigma异常值剔除结束，被试: {subject_id}, 条件: {cond_label}, 段落: {segment_type}")

        # 2. 检查电压跳变
        threshold = 100  # 微伏，可根据实际数据调整
        if np.any(np.abs(raw.get_data()) > threshold):
            print(f"警告：被试 {subject_id} 条件 {cond_label}, 段落 {segment_type} 检测到电压跳变")
        print(f"[DEBUG] 电压跳变检测结束")

        # 3. 带通滤波1-30Hz
        print(f"[DEBUG] 开始带通滤波1-30Hz")
        raw.filter(l_freq=1, h_freq=30, method='fir', phase='zero-double')
        print(f"[DEBUG] 带通滤波结束")

        # 4. 陷波滤波去除工频干扰
        print(f"[DEBUG] 开始陷波滤波去除工频干扰")
        raw.notch_filter(freqs=50)
        print(f"[DEBUG] 陷波滤波结束")
        
        # 5. 检查处理后的数据质量
        data_var = np.var(raw.get_data())
        if data_var < 1e-10:
            print(f"警告：被试 {subject_id} 条件 {cond_label}, 段落 {segment_type} 预处理后数据方差过小 ({data_var})")
        
        return raw
        
    except Exception as e:
        print(f"错误：被试 {subject_id} 条件 {cond_label}, 段落 {segment_type} 的预处理失败: {str(e)}")
        return None

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
        
        # 读取数据，返回(raw, label, segment_type)对，传递条件顺序和标记时间表
        raw_label_list = read_csv_data(file_path, marker_df, file_conditions_order)
        
        # 存储该被试的所有条件数据，用于后续计算标准化指标
        subject_data = {}
        
        for (raw, cond_label, segment_type) in raw_label_list:
            # 2.1 预处理
            raw = preprocess_eeg(raw, subject_id, cond_label, segment_type)
            if raw is None:
                print(f"警告：被试 {subject_id} 条件 {cond_label}, 段落 {segment_type} 的预处理失败，跳过后续分析")
                continue
            
            # 2.2 功率谱密度计算
            try:
                # 检查数据是否全为零或接近零
                if np.allclose(raw.get_data(), 0, atol=1e-6):
                    print(f"警告：被试 {subject_id} 条件 {cond_label}, 段落 {segment_type} 的数据全为零，跳过功率谱计算")
                    continue
                    
                # 添加额外的数据检查
                data_var = np.var(raw.get_data())
                if data_var < 1e-10:
                    print(f"警告：被试 {subject_id} 条件 {cond_label}, 段落 {segment_type} 的数据方差过小 ({data_var})，跳过功率谱计算")
                    continue
                
                # 检查数据长度是否足够进行功率谱分析
                data_length = raw.get_data().shape[1]
                if data_length < 1000:  # 假设至少需要1秒的数据
                    print(f"警告：被试 {subject_id} 条件 {cond_label}, 段落 {segment_type} 的数据长度不足 ({data_length} 点)，跳过功率谱计算")
                    continue
                    
                freqs = np.arange(8, 30, 1)
                # 计算功率谱
                psds, freqs = mne.time_frequency.psd_array_welch(
                    raw.get_data(),
                    sfreq=raw.info['sfreq'],
                    fmin=8,
                    fmax=30,
                    n_jobs=4,
                    n_fft=min(1000, data_length),  # 确保n_fft不超过数据长度
                    n_per_seg=min(1000, data_length),  # 确保n_per_seg不超过数据长度
                    n_overlap=min(500, data_length // 2),  # 确保overlap合理
                    average='mean'
                )
            except Exception as e:
                print(f"警告：被试 {subject_id} 条件 {cond_label}, 段落 {segment_type} 的功率谱计算失败: {str(e)}")
                continue
            
            # 2.3 提取左右额区电极
            left_frontal_chs = [i for i, ch in enumerate(raw.ch_names) if ch.lower() in ['fp1']]
            right_frontal_chs = [i for i, ch in enumerate(raw.ch_names) if ch.lower() in ['fp2']]
            
            if not left_frontal_chs or not right_frontal_chs:
                print(f"警告：被试 {subject_id} 条件 {cond_label}, 段落 {segment_type} 未找到左右额区电极")
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
            segment_key = f"{cond_label}_{segment_type}"
            subject_data[segment_key] = {
                'left_alpha': left_alpha_power,
                'right_alpha': right_alpha_power,
                'left_beta': left_beta_power,
                'right_beta': right_beta_power,
                'alpha_lateralization': alpha_lateralization,
                'attention_engagement': attention_engagement
            }
            
            # 如果是静息段，存储到rest_data中
            if cond_label == '静息':
                rest_key = f"{subject_id}_{segment_type}"
                rest_data[rest_key] = subject_data[segment_key]
        
        # 计算标准化指标并添加到结果中
        for segment_key, data in subject_data.items():
            if not segment_key.startswith('静息'):
                cond_label, segment_type = segment_key.split('_')
                rest_key = f"{subject_id}_{segment_type}"
                
                if rest_key in rest_data:
                    # 标准化α偏侧化指数（α偏侧化指数除以静息段的平均功率）
                    norm_alpha_lateralization = data['alpha_lateralization'] / rest_data[rest_key]['alpha_lateralization'] if rest_data[rest_key]['alpha_lateralization'] != 0 else np.nan
                    
                    # 标准化注意力投入程度β/α（注意力投入程度β/α除以静息段的平均功率）
                    norm_attention_engagement = data['attention_engagement'] / rest_data[rest_key]['attention_engagement'] if rest_data[rest_key]['attention_engagement'] != 0 else np.nan
                    
                    all_subjects_data.append({
                        'Subject': subject_id,
                        'Gender': gender,
                        'Condition': cond_label,
                        'SegmentType': segment_type,
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

# 主函数
def main():
    # 设置中文字体 - 修复字体缺失问题
    try:
        plt.rcParams["font.sans-serif"] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'SimSun', 'NSimSun', 'FangSong', 'KaiTi']
        plt.rcParams["axes.unicode_minus"] = False
        
        # 检查字体是否可用
        from matplotlib.font_manager import FontManager
        fm = FontManager()
        font_names = [f.name for f in fm.ttflist]
        
        # 尝试找到可用的中文字体
        available_chinese_fonts = [f for f in plt.rcParams["font.sans-serif"] if f in font_names]
        
        if available_chinese_fonts:
            print(f"使用可用的中文字体: {available_chinese_fonts[0]}")
            plt.rcParams["font.sans-serif"] = [available_chinese_fonts[0]] + plt.rcParams["font.sans-serif"]
        else:
            print("警告: 未找到可用的中文字体，图表中的中文可能无法正确显示")
            # 尝试使用系统默认字体
            import matplotlib.font_manager as fm
            plt.rcParams["font.sans-serif"] = ['sans-serif']
            plt.rcParams["font.family"] = 'sans-serif'
    except Exception as e:
        print(f"设置字体时出错: {str(e)}")
        # 使用备选方案
        plt.rcParams["font.sans-serif"] = ['sans-serif']
        plt.rcParams["font.family"] = 'sans-serif'
    
    # 设置数据文件夹路径
    csv_folder = r"D:\BaiduSyncdisk\大学\课程学习\大二下\生理心理学\大作业\csv格式实验数据"
    marker_file = r"D:\BaiduSyncdisk\大学\课程学习\大二下\生理心理学\大作业\标记时间表.csv"
    
    # 处理所有被试数据
    results_df = process_all_subjects(csv_folder, marker_file)
    
    # 保存处理后的数据
    results_df.to_csv("processed_eeg_results.csv", index=False, encoding='utf-8-sig')
    print("数据处理完成，已保存到processed_eeg_results.csv")

if __name__ == "__main__":
    main()