import numpy as np
import pyedflib
import os
import xml.etree.ElementTree as ET

# 解析header.xml获取实际参数
header_path = os.path.join(os.path.dirname(__file__), 'header.xml')
tree = ET.parse(header_path)
root = tree.getroot()

# 获取采样率和通道信息
sample_rate = int(root.find('.//SampleRate').text)
channels = []
for channel in root.findall('.//BioPotentialChannelConfiguration'):
    if channel.find('Enabled').text == 'true':
        channels.append(channel.find('Name').text)

n_channels = len(channels)
signal_labels = channels
physical_min, physical_max = -187.5, 187.5  # µV

# 从 .rec 文件中读取原始数据（以24位采样为例）
try:
    rec_path = os.path.join(os.path.dirname(__file__), '0_0.rec')
    raw = np.fromfile(rec_path, dtype=np.int32)
    # Calculate expected samples per channel
    samples_per_channel = len(raw) // n_channels
    # Reshape ensuring proper channel count
    raw = raw[:n_channels * samples_per_channel].reshape((n_channels, samples_per_channel))
except FileNotFoundError:
    print(f"[错误] 无法找到文件: {rec_path}")
    sys.exit(1)

# 转换为 µV
# 原始数据是24位有符号整数，范围是-2^23到2^23-1
# 物理范围是±187.5µV，所以转换公式应为：
raw = raw * (physical_max / (2**23 - 1))  # 按增益和单位换算

# 验证数据范围是否合理
if np.any(np.abs(raw) > physical_max * 1.1):
    print(f"[警告] 数据超出预期范围: 最大值{np.max(np.abs(raw)):.2f}µV")
    print("建议检查原始数据或转换公式")

# 写入 EDF
edf = pyedflib.EdfWriter("output.edf", n_channels, file_type=pyedflib.FILETYPE_EDFPLUS)
channel_info = [{
    'label': signal_labels[i],
    'dimension': 'uV',
    'sample_frequency': sample_rate,
    'physical_min': physical_min,
    'physical_max': physical_max,
    'digital_min': -32768,
    'digital_max': 32768 - 1,
    'transducer': '',
    'prefilter': ''
} for i in range(n_channels)]

# 确保数据维度正确
if raw.ndim == 1:
    raw = raw.reshape((1, -1))

edf.setSignalHeaders(channel_info)
edf.writeSamples(raw)
edf.close()
print("EDF文件已成功生成: output.edf")
