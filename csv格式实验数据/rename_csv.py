import os
import re

# 文件夹路径
bcrx_folder = r"D:\BaiduSyncdisk\大学\课程学习\大二下\生理心理学\大作业\bcrx格式实验数据"

# 获取所有bcrx文件和csv文件
files = os.listdir(bcrx_folder)
bcrx_files = [f for f in files if f.lower().endswith('.bcrx')]
csv_files = [f for f in files if f.lower().endswith('.csv')]

# 按序号排序bcrx文件
bcrx_files.sort()

# 创建序号到bcrx文件名的映射
bcrx_map = {}
for bcrx_file in bcrx_files:
    # 提取序号
    match = re.match(r'^(\d+)-', bcrx_file)
    if match:
        file_id = int(match.group(1))
        bcrx_map[file_id] = bcrx_file

# 重命名csv文件
renamed_count = 0
failed_count = 0
for csv_file in csv_files:
    # 提取序号
    match = re.match(r'^(\d+)-', csv_file)
    if match:
        file_id = int(match.group(1))
        # 查找对应的bcrx文件
        if file_id in bcrx_map:
            # 获取对应的bcrx文件名（不含后缀）
            bcrx_base = os.path.splitext(bcrx_map[file_id])[0]
            # 创建新的csv文件名
            new_csv_name = f"{bcrx_base}.csv"
            
            # 如果新文件名与当前文件名不同，则重命名
            if new_csv_name != csv_file:
                old_path = os.path.join(bcrx_folder, csv_file)
                new_path = os.path.join(bcrx_folder, new_csv_name)
                
                # 检查目标文件是否已存在
                if os.path.exists(new_path):
                    print(f"跳过: {csv_file} -> {new_csv_name} (目标文件已存在)")
                    failed_count += 1
                    continue
                    
                try:
                    print(f"重命名: {csv_file} -> {new_csv_name}")
                    os.rename(old_path, new_path)
                    renamed_count += 1
                except Exception as e:
                    print(f"错误: 无法重命名 {csv_file} -> {new_csv_name}")
                    print(f"      {str(e)}")
                    failed_count += 1

# 处理特殊情况：没有序号或序号格式不标准的文件
special_cases = []
for csv_file in csv_files:
    if not re.match(r'^\d+-', csv_file):
        special_cases.append(csv_file)

if special_cases:
    print("\n以下文件没有标准序号格式，需要手动处理:")
    for case in special_cases:
        print(f"- {case}")

print(f"\n完成! 共重命名 {renamed_count} 个CSV文件。")