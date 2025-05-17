import os
import xml.etree.ElementTree as ET
import struct
import csv
from datetime import datetime

def parse_header_xml(xml_path):
    """解析header.xml文件，获取通道信息和采样率"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # 获取采样率
    sample_rate = int(root.find('.//SampleRate').text)
    
    # 获取所有启用的EEG通道
    channels = []
    for channel in root.findall('.//BioPotentialChannelConfiguration'):
        if channel.find('Enabled').text == 'true':
            channels.append({
                'index': int(channel.find('ChannelIndex').text),
                'name': channel.find('Name').text
            })
    
    return sample_rate, channels

def parse_rec_file(rec_path, sample_rate, channels):
    """解析.rec文件，获取EEG数据"""
    data = []
    with open(rec_path, 'rb') as f:
        # 读取文件头
        header = f.read(16)
        packet_size = 4 * len(channels)
        frame_idx = 0
        while True:
            packet = f.read(packet_size)
            if not packet:
                break
            if len(packet) != packet_size:
                print(f"[警告] {rec_path} 第{frame_idx}帧数据长度不符: 期望{packet_size}字节, 实际{len(packet)}字节")
                break
            try:
                values = struct.unpack('<' + 'i' * len(channels), packet)
            except struct.error as e:
                print(f"[错误] unpack失败: {e}，文件: {rec_path}，帧: {frame_idx}")
                break
            # 转换为微伏
            microvolts = [v / 1000.0 for v in values]
            data.append(microvolts)
            frame_idx += 1
    # 生成时间戳
    timestamps = [i/sample_rate for i in range(len(data))]
    return timestamps, data

def convert_to_csv(bcrx_folder, output_csv):
    """将bcrx文件夹中的数据转换为CSV文件"""
    # 解析header.xml
    header_path = os.path.join(bcrx_folder, 'header.xml')
    sample_rate, channels = parse_header_xml(header_path)
    
    # 获取所有.rec文件
    rec_files = [f for f in os.listdir(bcrx_folder) if f.endswith('.rec')]
    rec_files.sort()  # 按文件名排序
    
    # 准备CSV写入
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # 写入表头
        headers = ['timestamp'] + [ch['name'] for ch in channels]
        writer.writerow(headers)
        
        # 处理每个.rec文件
        for rec_file in rec_files:
            rec_path = os.path.join(bcrx_folder, rec_file)
            timestamps, data = parse_rec_file(rec_path, sample_rate, channels)
            
            # 写入数据
            for ts, values in zip(timestamps, data):
                row = [ts] + values
                writer.writerow(row)

if __name__ == '__main__':
    # 示例用法
    input_folder = 'bcrx格式实验数据/02-XLL-F-长-漫-短'
    output_file = 'csv格式实验数据/02-XLL-F-长-漫-短.csv'
    convert_to_csv(input_folder, output_file)