import pyautogui
import time

import os

# 创建文件名数组
folder_path = r"D:\BaiduSyncdisk\大学\课程学习\大二下\生理心理学\大作业\bcrx格式实验数据"  # 替换为你的文件夹路径

# 只替换后缀，保留中文文件名
csv_filenames = [
    fname[:-5] + ".csv" if fname.lower().endswith(".bcrx") else fname
    for fname in os.listdir(folder_path)
    if fname.lower().endswith(".bcrx")
]

csv_filenames.sort()
print("CSV 文件名列表(按名称排序):")
for i, name in enumerate(csv_filenames):
    print(f"{i+1}: {name}")

for i in range(1, 37):

    pyautogui.press('enter')
    # 使用 Win 打开搜索，输入“biocapture”，回车启动软件
    pyautogui.press('win')
    time.sleep(1)
    pyautogui.write('biocapture')
    time.sleep(2)
    pyautogui.press('enter')
    time.sleep(5)  # 等待软件启动
    # 按 space, tab, enter
    pyautogui.press('space')
    pyautogui.press('tab')
    pyautogui.press('enter')
    time.sleep(5)

    # 按 alt, F, O, R
    pyautogui.press('alt')
    pyautogui.press('f')
    pyautogui.press('o')
    pyautogui.press('r')
    time.sleep(5)

    # 输入“循环次数-”，个位数前补0
    pyautogui.write(f"{i:02d}-")
    time.sleep(1)
    pyautogui.press('down')
    pyautogui.press('enter')
    time.sleep(8)  # 等待文件导入

    # 按 alt, a, e
    pyautogui.press('alt')
    pyautogui.press('a')
    pyautogui.press('e')
    pyautogui.press('enter')
    time.sleep(5)

    # 命名文件
    pyautogui.write(csv_filenames[i-1])
    pyautogui.press('enter')
    pyautogui.press('enter', presses=2)

    # 等待文件导出，以减轻运行压力
    time.sleep(40)