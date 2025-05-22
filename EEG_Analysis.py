import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import ttest_rel
from statsmodels.stats.anova import AnovaRM

# 设置中文字体函数
def setup_chinese_fonts():
    """设置matplotlib中文字体支持"""
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

# 统计分析函数
def perform_statistical_analysis(results_df):
    """对处理后的EEG数据进行统计分析"""
    # 3.1 统计分析
    # 检查每个被试在每个条件下的观察值数量
    print("\n每个被试在每个条件和段落下的观察值数量:")
    obs_counts = results_df.groupby(['Subject', 'Condition', 'SegmentType']).size().reset_index(name='count')
    print(obs_counts[obs_counts['count'] > 1])  # 打印有多个观察值的组合

    # 聚合数据，确保每个被试在每个条件和段落下只有一个观察值
    aggregated_df = results_df.groupby(['Subject', 'Condition', 'SegmentType']).agg({
        'Gender': 'first',
        'AlphaPower': 'mean',
        'BetaPower': 'mean',
        'BetaAlphaRatio': 'mean',
        'AlphaLateralization': 'mean',
        'AttentionEngagement': 'mean',
        'NormAlphaLateralization': 'mean',
        'NormAttentionEngagement': 'mean'
    }).reset_index()

    print(f"原始数据行数: {len(results_df)}, 聚合后数据行数: {len(aggregated_df)}")

    # 使用聚合后的数据进行统计分析
    # 检查数据平衡性
    print("\n数据平衡性检查:")
    for subject in aggregated_df['Subject'].unique():
        for segment_type in aggregated_df['SegmentType'].unique():
            segment_data = aggregated_df[(aggregated_df['Subject'] == subject) & (aggregated_df['SegmentType'] == segment_type)]
            conditions = segment_data['Condition'].unique()
            print(f"被试 {subject}, 段落 {segment_type}: {len(conditions)} 个条件 - {', '.join(conditions)}")
    
    # 增加条件组合平衡性验证
    print("\n条件组合平衡性验证:")
    # 计算每个条件和段落的样本数
    condition_segment_counts = aggregated_df.groupby(['Condition', 'SegmentType']).size().reset_index(name='count')
    print("各条件和段落样本数:")
    print(condition_segment_counts)
    
    # 检查是否所有被试都完成了所有条件和段落
    expected_conditions = ['长', '短', '漫']
    expected_segments = ['前段', '中段', '后段']
    all_balanced = True
    missing_combinations = {}
    
    for subject in aggregated_df['Subject'].unique():
        for segment in expected_segments:
            subject_segment_conditions = set(aggregated_df[(aggregated_df['Subject'] == subject) & 
                                                      (aggregated_df['SegmentType'] == segment)]['Condition'].unique())
            missing = [cond for cond in expected_conditions if cond not in subject_segment_conditions]
            if missing:
                all_balanced = False
                missing_combinations[f"{subject}_{segment}"] = missing
    
    if all_balanced:
        print("所有被试都完成了所有条件和段落组合")
    else:
        print("以下被试和段落缺少某些条件:")
        for combo, missing in missing_combinations.items():
            print(f"{combo}: 缺少 {', '.join(missing)}")
    
    # 执行重复测量方差分析
    try:
        # 对注意力投入度进行重复测量方差分析
        print("\n注意力投入度的重复测量方差分析:")
        attention_data = aggregated_df.dropna(subset=['AttentionEngagement'])
        
        # 检查是否有足够的数据进行分析
        if len(attention_data) > 10:
            aovrm = AnovaRM(attention_data, 'AttentionEngagement', 'Subject', within=['Condition', 'SegmentType'])
            res = aovrm.fit()
            print(res)
            
            # 添加事后检验 - 如果条件主效应显著，进行配对t检验
            if res.anova_table.loc['Condition', 'Pr > F'] < 0.05:
                print("\n条件主效应显著，进行事后配对t检验:")
                
                # 创建一个字典存储每个条件的平均值，用于结果解释
                condition_means = {}
                
                # 对每对条件进行配对t检验
                for i, cond1 in enumerate(expected_conditions):
                    condition_means[cond1] = attention_data[attention_data['Condition'] == cond1]['AttentionEngagement'].mean()
                    for cond2 in expected_conditions[i+1:]:
                        # 提取两个条件的数据
                        cond1_subjects = {}
                        cond2_subjects = {}
                        
                        # 对每个被试，计算其在不同段落下的平均值
                        for subject in attention_data['Subject'].unique():
                            cond1_data = attention_data[(attention_data['Subject'] == subject) & 
                                                      (attention_data['Condition'] == cond1)]['AttentionEngagement']
                            cond2_data = attention_data[(attention_data['Subject'] == subject) & 
                                                      (attention_data['Condition'] == cond2)]['AttentionEngagement']
                            
                            if len(cond1_data) > 0 and len(cond2_data) > 0:
                                cond1_subjects[subject] = cond1_data.mean()
                                cond2_subjects[subject] = cond2_data.mean()
                        
                        # 确保两组数据包含相同的被试
                        common_subjects = set(cond1_subjects.keys()) & set(cond2_subjects.keys())
                        
                        if len(common_subjects) > 1:
                            # 创建配对数据
                            cond1_values = [cond1_subjects[s] for s in common_subjects]
                            cond2_values = [cond2_subjects[s] for s in common_subjects]
                            
                            # 执行配对t检验
                            t_stat, p_val = ttest_rel(cond1_values, cond2_values)
                            
                            # 应用Bonferroni校正
                            alpha_corrected = 0.05 / 3  # 3对比较
                            
                            print(f"{cond1} vs {cond2}: t = {t_stat:.3f}, p = {p_val:.4f}, " + 
                                  f"{'显著' if p_val < alpha_corrected else '不显著'} (校正后α = {alpha_corrected:.4f}, n = {len(common_subjects)})")
                            
                            # 添加均值比较，帮助解释结果
                            print(f"  {cond1}平均值: {condition_means[cond1]:.4f}, {cond2}平均值: {condition_means[cond2]:.4f}, " +
                                  f"差值: {condition_means[cond1] - condition_means[cond2]:.4f}")
                
                # 添加结果解释
                print("\n事后检验结果解释:")
                sorted_conditions = sorted(condition_means.items(), key=lambda x: x[1], reverse=True)
                print(f"注意力投入度从高到低排序: {' > '.join([f'{cond}({val:.4f})' for cond, val in sorted_conditions])}")
        else:
            print("数据不足，无法进行重复测量方差分析")
            
        # 对α偏侧化指数进行重复测量方差分析
        print("\nα偏侧化指数的重复测量方差分析:")
        alpha_data = aggregated_df.dropna(subset=['AlphaLateralization'])
        
        if len(alpha_data) > 10:
            aovrm = AnovaRM(alpha_data, 'AlphaLateralization', 'Subject', within=['Condition', 'SegmentType'])
            res = aovrm.fit()
            print(res)
            
            # 添加事后检验 - 如果条件主效应显著，进行配对t检验
            if res.anova_table.loc['Condition', 'Pr > F'] < 0.05:
                print("\n条件主效应显著，进行事后配对t检验:")
                
                # 创建一个字典存储每个条件的平均值，用于结果解释
                condition_means = {}
                
                # 对每对条件进行配对t检验
                for i, cond1 in enumerate(expected_conditions):
                    condition_means[cond1] = alpha_data[alpha_data['Condition'] == cond1]['AlphaLateralization'].mean()
                    for cond2 in expected_conditions[i+1:]:
                        # 提取两个条件的数据
                        cond1_subjects = {}
                        cond2_subjects = {}
                        
                        # 对每个被试，计算其在不同段落下的平均值
                        for subject in alpha_data['Subject'].unique():
                            cond1_data = alpha_data[(alpha_data['Subject'] == subject) & 
                                                  (alpha_data['Condition'] == cond1)]['AlphaLateralization']
                            cond2_data = alpha_data[(alpha_data['Subject'] == subject) & 
                                                  (alpha_data['Condition'] == cond2)]['AlphaLateralization']
                            
                            if len(cond1_data) > 0 and len(cond2_data) > 0:
                                cond1_subjects[subject] = cond1_data.mean()
                                cond2_subjects[subject] = cond2_data.mean()
                        
                        # 确保两组数据包含相同的被试
                        common_subjects = set(cond1_subjects.keys()) & set(cond2_subjects.keys())
                        
                        if len(common_subjects) > 1:
                            # 创建配对数据
                            cond1_values = [cond1_subjects[s] for s in common_subjects]
                            cond2_values = [cond2_subjects[s] for s in common_subjects]
                            
                            # 执行配对t检验
                            t_stat, p_val = ttest_rel(cond1_values, cond2_values)
                            
                            # 应用Bonferroni校正
                            alpha_corrected = 0.05 / 3  # 3对比较
                            
                            print(f"{cond1} vs {cond2}: t = {t_stat:.3f}, p = {p_val:.4f}, " + 
                                  f"{'显著' if p_val < alpha_corrected else '不显著'} (校正后α = {alpha_corrected:.4f}, n = {len(common_subjects)})")
                            
                            # 添加均值比较，帮助解释结果
                            print(f"  {cond1}平均值: {condition_means[cond1]:.4f}, {cond2}平均值: {condition_means[cond2]:.4f}, " +
                                  f"差值: {condition_means[cond1] - condition_means[cond2]:.4f}")
                
                # 添加结果解释
                print("\n事后检验结果解释:")
                sorted_conditions = sorted(condition_means.items(), key=lambda x: x[1], reverse=True)
                print(f"α偏侧化指数从高到低排序: {' > '.join([f'{cond}({val:.4f})' for cond, val in sorted_conditions])}")
        else:
            print("数据不足，无法进行重复测量方差分析")
    except Exception as e:
        print(f"进行重复测量方差分析时出错: {str(e)}")
        print("尝试使用配对t检验进行条件间比较...")
        
        # 使用配对t检验比较不同条件
        for segment in expected_segments:
            segment_data = aggregated_df[aggregated_df['SegmentType'] == segment]
            
            # 比较长视频和短视频
            long_data = segment_data[segment_data['Condition'] == '长']['AttentionEngagement'].dropna()
            short_data = segment_data[segment_data['Condition'] == '短']['AttentionEngagement'].dropna()
            
            if len(long_data) > 5 and len(short_data) > 5:
                t_stat, p_val = ttest_rel(long_data, short_data)
                print(f"{segment} - 长视频 vs 短视频的注意力投入度: t={t_stat:.3f}, p={p_val:.3f}")
            
            # 比较长视频和漫画
            comic_data = segment_data[segment_data['Condition'] == '漫']['AttentionEngagement'].dropna()
            
            if len(long_data) > 5 and len(comic_data) > 5:
                t_stat, p_val = ttest_rel(long_data, comic_data)
                print(f"{segment} - 长视频 vs 漫画的注意力投入度: t={t_stat:.3f}, p={p_val:.3f}")
            
            # 比较短视频和漫画
            if len(short_data) > 5 and len(comic_data) > 5:
                t_stat, p_val = ttest_rel(short_data, comic_data)
                print(f"{segment} - 短视频 vs 漫画的注意力投入度: t={t_stat:.3f}, p={p_val:.3f}")
    
    return aggregated_df

# 可视化函数
def create_visualizations(results_df):
    """创建EEG数据的可视化图表"""
    # 3.2 可视化
    # 创建结果文件夹
    results_folder = "EEG_Results"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    
    # 设置绘图风格
    sns.set(style="whitegrid")
    
    # 确保中文标签正确显示
    try:
        # 检查当前字体设置
        current_font = plt.rcParams["font.sans-serif"][0]
        print(f"当前使用的字体: {current_font}")
        
        # 测试字体是否能正确显示中文
        fig, ax = plt.figure(), plt.subplot(111)
        ax.set_title('测试中文显示')
        plt.close(fig)
    except Exception as e:
        print(f"字体测试时出错: {str(e)}")
        # 尝试使用其他字体
        try:
            import matplotlib.font_manager as fm
            # 查找系统中的中文字体
            chinese_fonts = [f.name for f in fm.fontManager.ttflist if '黑体' in f.name or '宋体' in f.name or 'SimSun' in f.name or 'SimHei' in f.name]
            if chinese_fonts:
                plt.rcParams["font.sans-serif"] = chinese_fonts + ['sans-serif']
                print(f"切换到字体: {chinese_fonts[0]}")
        except:
            print("无法找到合适的中文字体，将使用默认字体")
    
    # 绘制不同条件下的注意力投入度
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='Condition', y='AttentionEngagement', hue='SegmentType', data=results_df)
    plt.title('不同条件下的注意力投入度', fontsize=16)
    plt.xlabel('实验条件', fontsize=14)
    plt.ylabel('注意力投入度 (β/α)', fontsize=14)
    
    # 添加条件标签映射，确保中文显示正确
    condition_labels = {'长': '长视频', '短': '短视频', '漫': '漫画'}
    ax = plt.gca()
    if ax.get_xticklabels():
        labels = [condition_labels.get(label.get_text(), label.get_text()) for label in ax.get_xticklabels()]
        ax.set_xticklabels(labels)
    
    plt.savefig(os.path.join(results_folder, 'attention_engagement_by_condition.png'), dpi=300, bbox_inches='tight')
    
    # 绘制不同条件下的α偏侧化指数
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='Condition', y='AlphaLateralization', hue='SegmentType', data=results_df)
    plt.title('不同条件下的α偏侧化指数', fontsize=16)
    plt.xlabel('实验条件', fontsize=14)
    plt.ylabel('α偏侧化指数 (右-左)', fontsize=14)
    
    # 添加条件标签映射
    ax = plt.gca()
    if ax.get_xticklabels():
        labels = [condition_labels.get(label.get_text(), label.get_text()) for label in ax.get_xticklabels()]
        ax.set_xticklabels(labels)
    
    plt.savefig(os.path.join(results_folder, 'alpha_lateralization_by_condition.png'), dpi=300, bbox_inches='tight')
    
    # 绘制不同段落的时间变化趋势
    plt.figure(figsize=(14, 8))
    
    # 创建段落顺序映射
    segment_order = {'前段': 0, '中段': 1, '后段': 2}
    results_df['SegmentOrder'] = results_df['SegmentType'].map(segment_order)
    
    # 按条件和段落分组计算平均值
    trend_data = results_df.groupby(['Condition', 'SegmentType']).agg({
        'AttentionEngagement': 'mean',
        'SegmentOrder': 'first'
    }).reset_index()
    
    # 按段落顺序排序
    trend_data = trend_data.sort_values('SegmentOrder')
    
    # 为每个条件绘制线图
    for condition in trend_data['Condition'].unique():
        condition_data = trend_data[trend_data['Condition'] == condition]
        plt.plot(condition_data['SegmentType'], condition_data['AttentionEngagement'], 
                 marker='o', linewidth=2, label=condition_labels.get(condition, condition))
    
    plt.title('注意力投入度随时间的变化趋势', fontsize=16)
    plt.xlabel('实验段落', fontsize=14)
    plt.ylabel('平均注意力投入度 (β/α)', fontsize=14)
    plt.legend(title='实验条件')
    plt.savefig(os.path.join(results_folder, 'attention_trend_over_time.png'), dpi=300, bbox_inches='tight')
    
    print(f"可视化结果已保存到 {results_folder} 文件夹")

# 主函数
def main():
    # 设置中文字体
    setup_chinese_fonts()
    
    # 读取处理后的数据
    try:
        results_df = pd.read_csv("processed_eeg_results.csv", encoding='utf-8-sig')
        print(f"成功读取处理后的数据，共 {len(results_df)} 行")
    except Exception as e:
        print(f"读取数据时出错: {str(e)}")
        print("请先运行 EEG_Processing.py 生成处理后的数据文件")
        return
    
    # 进行统计分析
    aggregated_df = perform_statistical_analysis(results_df)
    
    # 创建可视化
    create_visualizations(aggregated_df)

if __name__ == "__main__":
    main()