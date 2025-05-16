import os
import re

folder = r"D:\BaiduSyncdisk\大学\课程学习\大二下\生理心理学\大作业\原始实验数据"
existing_ids = set()
file_info = []

# 1. 预扫描所有文件，分离 ID 和主体信息
for fname in os.listdir(folder):
    if not fname.endswith(".bcrx"):
        continue
    base = fname[:-5]  # remove .bcrx
    id_match = re.match(r"^(\d+)[-_]", base)
    if id_match:
        file_id = int(id_match.group(1))
        existing_ids.add(file_id)
        rest = base[len(id_match.group(0)):]
    else:
        file_id = None
        rest = base
    file_info.append((fname, file_id, rest))

# 2. 分配新 ID 的生成器
new_id = 1
def get_next_id():
    global new_id
    while new_id in existing_ids:
        new_id += 1
    existing_ids.add(new_id)
    return new_id 

# 3. 中文字符逐字分词
def segment_chinese(text):
    return list(text)

# 4. 批量重命名逻辑
for fname, file_id, rest in file_info:
    parts = rest.split("-")
    idx = 0

    # 分配 ID
    assigned_id = file_id if file_id else get_next_id()

    # 获取 Name
    name = parts[idx] if idx < len(parts) else "UNKNOWN"
    idx += 1

    # 获取性别并标准化
    sex = parts[idx] if idx < len(parts) else "U"
    idx += 1
    if sex in ["女", "F"]:
        sex = "F"
    elif sex in ["男", "M"]:
        sex = "M"
    else:
        sex = "U"

    # 获取类型字段
    type_raw = parts[idx:] if idx < len(parts) else []
    type_clean = []
    for t in type_raw:
        if re.search(r"[\u4e00-\u9fff]", t):
            type_clean.extend(segment_chinese(t))
        else:
            type_clean.append(t)

    # 重组文件名
    assigned_id_str = f"{assigned_id:02d}"
    new_fname = f"{assigned_id_str}-{name}-{sex}"
    if type_clean:
        new_fname += "-" + "-".join(type_clean)
    new_fname += ".bcrx"

    # 重命名
    src = os.path.join(folder, fname)
    dst = os.path.join(folder, new_fname)
    print(f"{fname} -> {new_fname}")
    os.rename(src, dst)
