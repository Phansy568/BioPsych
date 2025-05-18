import os
import pandas as pd
import csv
from prettytable import PrettyTable

def list_markers_in_csv(csv_folder, output_file=None):
    """列出CSV文件中出现的所有标记（markers）及其对应的时间，并输出到表格文件"""
    # 非标记列名
    non_marker_columns = ['Elapsed Time', 'Fp1', 'Fp2', 'BioRadio Event']
    
    # 存储所有文件中发现的标记
    all_markers = {}
    file_markers = {}
    marker_times = {}  # 存储每个标记对应的时间
    
    # 准备输出到CSV的数据
    output_data = []  # 将存储[被试ID, 标记, 时间]格式的数据
    
    # 获取所有CSV文件
    csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv') and not f.startswith('bcrx')]
    
    for csv_file in csv_files:
        file_path = os.path.join(csv_folder, csv_file)
        print(f"分析文件: {csv_file}")
        
        # 从文件名中提取被试ID
        parts = os.path.basename(csv_file).split('-')
        subject_id = parts[1] if len(parts) > 1 else parts[0]  # 假设被试ID是文件名的第二部分，如果没有则使用第一部分
        
        try:
            # 读取CSV文件
            data = pd.read_csv(file_path)
            columns = data.columns.tolist()
            
            # 记录该文件中的标记
            file_markers[csv_file] = []
            
            # 将除了非标记列之外的所有列视为标记
            marker_columns = [col for col in columns if col not in non_marker_columns]
            
            for marker in marker_columns:
                if marker not in all_markers:
                    all_markers[marker] = 0
                    marker_times[marker] = {}
                all_markers[marker] += 1
                file_markers[csv_file].append(marker)
                
                # 查找标记为1的行，记录对应的时间
                marker_indices = data.index[data[marker] == 1].tolist()
                if marker_indices:
                    if csv_file not in marker_times[marker]:
                        marker_times[marker][csv_file] = []
                    
                    for idx in marker_indices:
                        if 'Elapsed Time' in data.columns:
                            time = data.loc[idx, 'Elapsed Time']
                            marker_times[marker][csv_file].append(time)
                            
                            # 添加到输出数据
                            output_data.append([subject_id, marker, time])
        
        except Exception as e:
            print(f"处理文件 {csv_file} 时出错: {e}")
    
    # 打印结果
    print("\n所有CSV文件中发现的标记:")
    for marker, count in sorted(all_markers.items(), key=lambda x: (-x[1], x[0])):
        print(f"{marker}: 出现在 {count} 个文件中")
    
    print("\n每个文件中的标记:")
    for file, markers in file_markers.items():
        if markers:
            print(f"{file}: {', '.join(markers)}")
        else:
            print(f"{file}: 未发现标记")
    
    # 以表格形式打印每个标记及对应的时间
    print("\n标记时间表:")
    for marker in sorted(marker_times.keys()):
        table = PrettyTable()
        table.field_names = ["文件名", "标记时间"]
        
        for file in sorted(marker_times[marker].keys()):
            times = marker_times[marker][file]
            if times:
                # 限制显示的时间数量，避免表格过长
                if len(times) > 5:
                    time_str = ", ".join(map(str, times[:5])) + f"... (共{len(times)}个)"
                else:
                    time_str = ", ".join(map(str, times))
                table.add_row([file, time_str])
        
        print(f"\n标记: {marker}")
        print(table)
    
    # 将数据输出到CSV文件
    if output_file:
        # 创建DataFrame并保存为CSV
        output_df = pd.DataFrame(output_data, columns=["被试ID", "标记", "时间"])
        # 按被试ID和时间排序
        output_df = output_df.sort_values(by=["被试ID", "时间"])
        output_df.to_csv(output_file, index=False, encoding='utf-8-sig')  # 使用utf-8-sig编码以支持中文
        print(f"\n标记时间数据已保存到: {output_file}")
        
        return output_df

if __name__ == "__main__":
    # 设置数据文件夹路径
    csv_folder = r"D:\BaiduSyncdisk\大学\课程学习\大二下\生理心理学\大作业\csv格式实验数据"
    
    # 设置输出文件路径
    output_file = os.path.join(os.path.dirname(csv_folder), "标记时间表.csv")
    
    # 检查是否安装了prettytable
    try:
        import prettytable
    except ImportError:
        print("需要安装prettytable库来显示表格。请运行: pip install prettytable")
        exit(1)
        
    list_markers_in_csv(csv_folder, output_file)