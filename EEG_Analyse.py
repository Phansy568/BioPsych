import mne
import numpy as np
import os

# 1.1 数据读取
def read_biocapture(file_path):
    """自定义Biocapture文件读取函数（需根据实际文件结构编写）"""
    raw = mne.io.read_raw_edf(file_path, preload=True)  # 读取XML格式的EEG数据
    return raw

# 示例调用
xml_folder = r"D:\BaiduSyncdisk\大学\课程学习\大二下\生理心理学\大作业\xml格式实验数据"
xml_file = os.path.join(xml_folder, '1-JKH-F-长-漫-短.xml')
raw = read_biocapture(xml_file)

# 1.2 设置基本参数
raw.info['sfreq'] = 1000  # 明确采样率（需根据实际设备设置，如NeuroScan常用1000Hz）

# 1.3 滤波处理
raw.filter(l_freq=0.5, h_freq=40,  # 带通滤波0.5-40Hz（抑制低频漂移和高频肌电）
           method='fir', phase='zero-double')  
raw.notch_filter(freqs=50)  # 陷波滤波去除工频干扰（国内50Hz）

# 可视化检查原始信号
raw.plot(n_channels=32, duration=5, scalings=dict(eeg=1e-4))

# 2.1 事件标记与分段
events = mne.find_events(raw, stim_channel='STI 014')  # 根据刺激通道标记事件
epochs = mne.Epochs(raw, events, tmin=-0.2, tmax=10, 
                    baseline=(-0.2, 0), preload=True)  # 分段为10s试次

# 2.2 独立成分分析（ICA）去伪迹
ica = mne.preprocessing.ICA(n_components=20, random_state=97)
ica.fit(epochs)
ica.plot_components()  # 手动标记眼动/眨眼成分
ica.apply(epochs)  # 去除选定成分

# 2.3 重参考与坏道插值
epochs.set_eeg_reference(ref_channels='average')  # 平均参考（参考文献：Delorme & Makeig, 2004）
epochs.interpolate_bads(reset_bads=True)  # 插值修复坏导

# 3.1 功率谱密度计算
freqs = np.arange(8, 30, 1)  # 关注α(8-13Hz)和β(13-30Hz)频段
psds, freqs = mne.time_frequency.psd_multitaper(
    epochs, fmin=8, fmax=30, n_jobs=4)

# 3.2 提取左侧额区电极（如F3, F1）
left_frontal_chs = ['F3', 'F1']
ch_idx = [epochs.ch_names.index(ch) for ch in left_frontal_chs]

# 3.3 计算β/α比值
alpha_power = psds[:, ch_idx, (freqs >= 8) & (freqs <= 13)].mean(axis=2)
beta_power = psds[:, ch_idx, (freqs > 13) & (freqs <= 30)].mean(axis=2)
beta_alpha_ratio = beta_power / alpha_power

from statsmodels.stats.anova import AnovaRM

# 4.1 数据整理为DataFrame
import pandas as pd
df = pd.DataFrame({
    'Subject': np.repeat(np.arange(36), 3),
    'Condition': np.tile(['a', 'b', 'c'], 36),
    'AlphaPower': alpha_power.flatten(),
    'BetaAlphaRatio': beta_alpha_ratio.flatten()
})

# 4.2 重复测量方差分析
anova_alpha = AnovaRM(df, 'AlphaPower', 'Subject', within=['Condition']).fit()
print(anova_alpha.summary())

# 4.3 事后配对t检验（Bonferroni校正）
from scipy.stats import ttest_rel
t_stat, p_val = ttest_rel(
    df[df.Condition=='a'].BetaAlphaRatio,
    df[df.Condition=='b'].BetaAlphaRatio)
adjusted_alpha = 0.05 / 3  # 校正三组比较

# 5.1 地形图对比
mne.viz.plot_compare_evokeds(
    dict(ShortVideo=epochs['a'].average(),
         LongVideo=epochs['b'].average()),
    picks='eeg', 
    title='ERP对比：短视频 vs 长视频')

# 5.2 β/α比值统计图
import seaborn as sns
sns.boxplot(x='Condition', y='BetaAlphaRatio', data=df)
plt.annotate('*', xy=(0, 1.2), ha='center', color='red')  # 标记显著性