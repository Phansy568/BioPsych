import os
import pandas as pd
import csv

def list_markers_in_csv(csv_folder):
    """列出CSV文件中出现的所有标记（markers）"""
    # 非标记列名
    non_marker_columns = ['Elapsed Time', 'Fp1', 'Fp2']
    
    # 存储所有文件中发现的标记
    all_markers = {}
    file_markers = {}
    
    # 获取所有CSV文件
    csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv') and not f.startswith('bcrx')]
    
    for csv_file in csv_files:
        file_path = os.path.join(csv_folder, csv_file)
        print(f"分析文件: {csv_file}")
        
        try:
            # 尝试读取CSV文件的列名
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                columns = next(reader)  # 获取第一行（列名）
            
            # 记录该文件中的标记
            file_markers[csv_file] = []
            
            # 将除了非标记列之外的所有列视为标记
            for col in columns:
                if col not in non_marker_columns:
                    if col not in all_markers:
                        all_markers[col] = 0
                    all_markers[col] += 1
                    file_markers[csv_file].append(col)
        
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

if __name__ == "__main__":
    # 设置数据文件夹路径
    csv_folder = r"D:\BaiduSyncdisk\大学\课程学习\大二下\生理心理学\大作业\csv格式实验数据"
    list_markers_in_csv(csv_folder)