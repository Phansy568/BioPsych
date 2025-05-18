import mne
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel
from statsmodels.stats.anova import AnovaRM

# 1.1 数据读取
def read_csv_data(file_path, file_conditions_order=None):
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
    subject_name = filename.split('-')[1] if '-' in filename else "" # 您已有的代码

    # 标记定义
    start_keys = [k for k in columns if k.lower() in ['start', 's', 'strat']]
    end_keys = [k for k in columns if k.lower() in ['end', 'e', 'end']]
    seg_keys = [k for k in columns if k.lower() in ['long','short','manga','manhua','LONG', 'SHORT',"jingxi"]]

    segments = []
    condition_labels = []

    # 文件名顺序标签，如果提供了file_conditions_order，则使用它
    file_conditions = file_conditions_order if file_conditions_order else ['长', '短', '漫']
    
    print(f"使用条件顺序: {file_conditions}")

    if subject_name in ["MPX","LXY","YYH", "YYX", "ZJM", "ZHR", "WZM", "WQX"] and end_keys:
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
        print(f"--- 处理文件: {os.path.basename(file_path)} ---") # 添加打印
        print(f"被试 {subject_name} 的分段标记: {seg_keys}") # 添加打印

        # 收集所有有效的数据段
        valid_segments = []
        
        for key in seg_keys:
            print(f"\n--- 处理标记: {key} ---") # 添加打印
            idxs = data.index[data[key] == 1].tolist()
            print(f"标记 {key} 的索引位置: {idxs}") # 添加打印
            
            # 检查是否有至少两个标记点
            if len(idxs) < 2:
                print(f"警告：标记 {key} 没有找到两个标记点（开始和结束），只有 {len(idxs)} 个点")
                empty_segments[key] = f"标记点不足，只有 {len(idxs)} 个点"
                continue

            # 使用同一标记的第一个位置作为开始，第二个位置作为结束
            s = idxs[0]
            e = idxs[1]
            
            print(f"计算的分段范围: {s}-{e} (同一标记的第一次和第二次出现)") # 添加打印

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
            print(f"分段 {key} 成功提取，标签为: {label}") # 添加打印
            
            # 将有效的数据段添加到列表中，包含所有必要信息
            valid_segments.append({
                'key': key,
                'start': s,
                'end': e,
                'data': exp_data,
                'label': label
            })
        
        # 如果有超过3个有效数据段，只取后三个
        if len(valid_segments) > 3:
            print(f"检测到{len(valid_segments)}个有效数据段，只使用后三个")
            valid_segments = valid_segments[-3:]
        
        # 将有效数据段添加到结果中
        for segment in valid_segments:
            segments.append(segment['data'])
            condition_labels.append(segment['label'])
            print(f"使用数据段: {segment['key']}，标签: {segment['label']}，范围: {segment['start']}-{segment['end']}")
    else:
        print(f"警告：文件 {os.path.basename(file_path)} 未检测到有效分段标记，使用全段")
        segments = [data]
        condition_labels = [file_conditions[0]]

    # 提取EEG数据并生成raw对象
    results = []
    for exp_data, cond_label in zip(segments, condition_labels):
        # ---- 您已有的调试打印 ----
        filename = os.path.basename(file_path) # 获取文件名
        subject_name_parts = filename.split('-')
        subject_name = ""
        if len(subject_name_parts) > 1:
            subject_name = subject_name_parts[1] # 假设被试ID是文件名的第二部分

        if subject_name == "ZYH": # 确保这里的 "ZYH" 与被试35文件名中的ID一致
            print(f"\nDEBUG: Subject {subject_name}, Condition {cond_label}")
            print("exp_data.head(10):") # 打印原始分段数据的前10行
            print(exp_data.head(10))
            print("exp_data.isnull().sum():") # 检查原始分段数据中各列的NaN数量
            print(exp_data.isnull().sum())
            print("exp_data.dtypes:") # 检查原始分段数据中各列的数据类型
            print(exp_data.dtypes)
        # ---- 调试打印结束 ----

        eeg_cols = [col for col in exp_data.columns if col.lower() in ['fp1', 'fp2']]
        if len(eeg_cols) < 2:
            non_time_cols = [col for col in exp_data.columns if col != 'Elapsed Time']
            eeg_data = exp_data[non_time_cols[:2]].copy() # 使用 .copy() 避免 SettingWithCopyWarning
        else:
            eeg_data = exp_data[eeg_cols].copy() # 使用 .copy() 避免 SettingWithCopyWarning
        
        # ---- 新增：在创建RawArray前对eeg_data进行NaN插值 ----
        if eeg_data.isnull().values.any():
            print(f"INFO: 被试 {subject_name}, 条件 {cond_label} - eeg_data 包含NaN，尝试插值...")
            for col in eeg_data.columns:
                if eeg_data[col].isnull().any():
                    # 确保列是数值类型，非数值转为NaN
                    eeg_data[col] = pd.to_numeric(eeg_data[col], errors='coerce')
                    
                    if eeg_data[col].isnull().all():
                        print(f"警告：被试 {subject_name}, 条件 {cond_label}, 列 {col} 全是NaN。将用0填充。")
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
                            print(f"警告：被试 {subject_name}, 条件 {cond_label}, 列 {col} 插值后仍有 {remaining_nans} 个NaN。将用0填充剩余NaN。")
                            eeg_data[col] = eeg_data[col].fillna(0) # 对插值后仍存在的NaN用0填充
                        else:
                            print(f"INFO: 被试 {subject_name}, 条件 {cond_label}, 列 {col} 插值完成。")
                            
                    except Exception as e_interp:
                        print(f"错误：被试 {subject_name}, 条件 {cond_label}, 列 {col} 插值失败: {e_interp}。将用0填充NaN。")
                        eeg_data[col] = eeg_data[col].fillna(0) # 插值失败则用0填充

            if eeg_data.isnull().values.any():
                 print(f"严重警告：被试 {subject_name}, 条件 {cond_label} - eeg_data 在插值尝试后仍包含NaN。这可能导致后续处理问题。")
        # ---- 新增插值结束 ----

        ch_names = list(eeg_data.columns)
        ch_types = ['eeg'] * len(ch_names)
        info = mne.create_info(ch_names=ch_names, sfreq=1000, ch_types=ch_types)
        print(eeg_data.head() ,eeg_data.dtypes)
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

    # 定义结果文件路径
    results_file_path = os.path.join(data_folder, 'EEG_analysis_results.csv') # 或者直接指定绝对路径
    processed_subject_ids = set()

    # 尝试读取已处理的被试列表
    try:
        if os.path.exists(results_file_path):
            results_df = pd.read_csv(results_file_path)
            # 确保 'Subject' 列存在且数据类型正确，以便进行比较
            if 'Subject' in results_df.columns:
                # 将结果文件中的 Subject ID 转换为字符串，以匹配从文件名解析的 ID 类型
                processed_subject_ids = set(results_df['Subject'].astype(str).unique())
                print(f"已在结果文件中找到的被试ID: {processed_subject_ids}")
            else:
                print(f"警告：结果文件 {results_file_path} 中缺少 'Subject' 列。")
        else:
            print(f"信息：结果文件 {results_file_path} 不存在，将处理所有找到的被试。")
    except Exception as e:
        print(f"读取结果文件 {results_file_path} 时出错: {e}。将尝试处理所有找到的被试。")

    # 获取所有CSV文件 (您可以保留或修改这里的筛选逻辑)
    # 例如，如果您只想处理特定编号的被试，可以保留 any(f.startswith(f"{num}-") for num in ["35", "36"])
    # 如果要处理所有符合命名规范的被试，可以移除该部分
    csv_files_all = [f for f in os.listdir(data_folder)
                     if f.endswith('.csv') and not f.startswith('bcrx')] # 初始筛选

    csv_files_to_process = []
    for csv_file in csv_files_all:
        # 从文件名解析被试ID以进行检查
        # 注意：这里的 parse_filename 返回的 subject_id 应该是字符串类型
        temp_subject_id, _, _ = parse_filename(csv_file) # 假设 parse_filename 返回 (subject_id, gender, conditions)
        
        # 检查此被试是否已在结果文件中
        if temp_subject_id in processed_subject_ids:
            print(f"信息：被试 {temp_subject_id} 的数据已存在于 {results_file_path}，跳过处理。")
            continue
        
        # 如果您仍想只处理特定编号的、且未在结果文件中的被试，可以在此添加额外筛选
        # 例如，如果您只想处理 "35" 或 "36" 中未被处理的：
        # if not any(csv_file.startswith(f"{num}-") for num in ["35", "36"]):
        #     continue
            
        csv_files_to_process.append(csv_file)
    
    if not csv_files_to_process:
        print("没有需要处理的新被试数据。")
        return pd.DataFrame(all_subjects_data) # 返回空或已有的数据

    print(f"将要处理的被试文件: {csv_files_to_process}")

    for csv_file in csv_files_to_process: # 现在只遍历筛选后的文件列表
        file_path = os.path.join(data_folder, csv_file)
        subject_id, gender, conditions = parse_filename(csv_file) # subject_id 在这里被重新赋值

        print(f"处理被试 {subject_id} 的数据...")
        
        # 将条件代码转换为中文标签
        condition_map_reverse = {'a': '短', 'b': '长', 'c': '漫'}
        file_conditions_order = [condition_map_reverse[c] for c in conditions if c in condition_map_reverse]
        
        # 如果没有有效的条件顺序，使用默认顺序
        if not file_conditions_order:
            file_conditions_order = ['长', '短', '漫']
            
        print(f"从文件名解析的条件顺序: {file_conditions_order}")

        # 读取数据，返回(raw, label)对，传递条件顺序
        raw_label_list = read_csv_data(file_path, file_conditions_order)
        
        for (raw, cond_label) in raw_label_list:
            # 打印滤波前的数据统计
            try:
                data_before_filter = raw.copy().get_data() # 使用copy以防影响后续操作
                print(f"被试 {subject_id} 条件 {cond_label} 的数据统计 (滤波前):")
                print(f"  形状: {data_before_filter.shape}")
                print(f"  是否包含NaN: {np.isnan(data_before_filter).any()}")
                if data_before_filter.size > 0 and not np.isnan(data_before_filter).all():
                    print(f"  均值: {np.mean(data_before_filter)}")
                    print(f"  方差: {np.var(data_before_filter)}")
                    print(f"  最小值: {np.min(data_before_filter)}")
                    print(f"  最大值: {np.max(data_before_filter)}")
                    print(f"  零值比例: {np.sum(data_before_filter == 0) / data_before_filter.size}")
                elif np.isnan(data_before_filter).all():
                    print(f"  数据在滤波前已全是NaN")
                else:
                    print(f"  数据为空或无法计算统计值")
            except Exception as e:
                print(f"警告：被试 {subject_id} 条件 {cond_label} 滤波前统计失败: {str(e)}")


            # 2.1 预处理
            raw.filter(l_freq=1, h_freq=30, method='fir', phase='zero-double')  # 带通滤波0.5-40Hz
            raw.notch_filter(freqs=50)  # 陷波滤波去除工频干扰
            
            # 2.2 功率谱密度计算 - 添加数据有效性检查
            # 在功率谱计算前添加 (这是您已有的打印，现在明确其为滤波后)
            data_array = raw.get_data()
            print(f"被试 {subject_id} 条件 {cond_label} 的数据统计 (滤波后):") # 修改了标签以示区别
            print(f"  形状: {data_array.shape}")
            print(f"  均值: {np.mean(data_array)}")
            print(f"  方差: {np.var(data_array)}")
            print(f"  最小值: {np.min(data_array)}")
            print(f"  最大值: {np.max(data_array)}")
            print(f"  零值比例: {np.sum(data_array == 0) / data_array.size if data_array.size > 0 else 'N/A'}")

            # 检查并处理NaN值
            data_array = raw.get_data() # 确保这是最新的数据
            if np.isnan(data_array).any():
                print(f"警告：被试 {subject_id} 条件 {cond_label} 的数据包含NaN值，尝试插值替换")
                # 对每个通道分别处理NaN值
                for ch_idx in range(data_array.shape[0]):
                    # 获取非NaN的索引和值
                    non_nan_idx = np.where(~np.isnan(data_array[ch_idx]))[0]
                    if len(non_nan_idx) == 0:
                        print(f"错误：通道 {ch_idx} 全是NaN值，无法修复")
                        continue
                    non_nan_values = data_array[ch_idx, non_nan_idx]
                    # 创建插值函数
                    if len(non_nan_idx) > 1:  # 至少需要两个点才能插值
                        from scipy import interpolate
                        f = interpolate.interp1d(non_nan_idx, non_nan_values, 
                                                kind='linear', bounds_error=False, 
                                                fill_value=(non_nan_values[0], non_nan_values[-1]))
                        # 生成所有索引
                        all_idx = np.arange(len(data_array[ch_idx]))
                        # 替换原始数据
                        data_array[ch_idx] = f(all_idx)
                    else:  # 如果只有一个非NaN值，用该值填充所有NaN
                        data_array[ch_idx] = non_nan_values[0]
                
                # 更新raw对象的数据
                raw._data = data_array
                
                # 再次检查
                if np.isnan(raw.get_data()).any():
                    print(f"错误：被试 {subject_id} 条件 {cond_label} 的数据仍然包含NaN值，跳过处理")
                    continue
            
            if len(raw.get_data()) == 0:
                print(f"警告：被试 {subject_id} 条件 {cond_label} 的数据为空，跳过")
                continue
                
            # 在process_all_subjects函数中，修改功率谱计算部分
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
                    
                freqs = np.arange(1, 40, 1)
                # 修改功率谱计算参数，尝试解决"Weights sum to zero"问题
                psds, freqs = mne.time_frequency.psd_array_welch(
                    raw.get_data(),
                    sfreq=raw.info['sfreq'],
                    fmin=1,
                    fmax=40,
                    n_jobs=4,
                    n_fft=1024,  # 减小FFT点数
                    n_overlap=256,  # 减小重叠
                    average='mean'  # 明确指定平均方法
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
    # 检查数据平衡性
    print("\n数据平衡性检查:")
    for subject in results_df['Subject'].unique():
        conditions = results_df[results_df['Subject'] == subject]['Condition'].unique()
        print(f"被试 {subject}: {len(conditions)} 个条件 - {', '.join(conditions)}")
    
    # 尝试使用能处理不平衡数据的方法进行统计分析
    try:
        # 重复测量方差分析 - α波能量
        anova_alpha = AnovaRM(results_df, 'AlphaPower', 'Subject', within=['Condition']).fit()
        print("\nα波能量重复测量方差分析结果:")
        print(anova_alpha.summary())
        
        # 重复测量方差分析 - β/α比值
        anova_ratio = AnovaRM(results_df, 'BetaAlphaRatio', 'Subject', within=['Condition']).fit()
        print("\nβ/α比值重复测量方差分析结果:")
        print(anova_ratio.summary())
    except ValueError as e:
        print(f"\n方差分析失败: {str(e)}")
        print("尝试使用配对t检验进行条件间比较...")
        
        # 如果方差分析失败，直接进行配对t检验
        # 这里不需要平衡数据，只比较有共同数据的被试
    
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