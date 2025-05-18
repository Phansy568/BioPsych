import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# 修改导入部分，在现有导入语句后添加
from scipy.stats import ttest_rel, shapiro
from statsmodels.stats.anova import AnovaRM
import os
import pingouin as pg  # 用于球形性检验，需要安装: pip install pingouin

def generate_eeg_report(results_csv='EEG_analysis_results.csv', output_md='EEG_results_report.md'):
    """
    生成EEG分析结果的详细说明文档
    
    参数:
        results_csv: 分析结果CSV文件路径
        output_md: 输出的Markdown文档路径
    """
    # 检查结果文件是否存在
    if not os.path.exists(results_csv):
        print(f"错误: 找不到结果文件 {results_csv}")
        return
    
    # 读取分析结果
    print(f"正在读取分析结果: {results_csv}")
    results_df = pd.read_csv(results_csv)
    
    # 直接使用原始数据，不进行聚合
    aggregated_df = results_df.copy()
    
    # 检查缺失的被试-条件组合
    all_combinations = pd.MultiIndex.from_product(
        [results_df['Subject'].unique(), results_df['Condition'].unique()],
        names=['Subject', 'Condition']
    )
    missing_combinations = all_combinations.difference(aggregated_df.set_index(['Subject', 'Condition']).index)
    print("缺失的被试-条件组合:\n", missing_combinations)
    
    # 创建Markdown文档
    with open(output_md, 'w', encoding='utf-8') as f:
        # 1. 标题和介绍
        f.write('# EEG分析结果报告\n\n')
        f.write('## 研究概述\n')
        f.write('本研究探究了不同阅读材料（长视频、短视频、漫画）对脑电活动的影响。通过分析前额区域的脑电数据，')
        f.write('我们计算了α偏侧化指数和注意力投入程度等指标，以评估不同阅读材料对大脑活动的影响。\n\n')
        
        # 2. 数据概览
        f.write('## 数据概览\n')
        f.write(f'- 总被试数: {len(aggregated_df["Subject"].unique())}\n')
        f.write(f'- 实验条件: {", ".join(aggregated_df["Condition"].unique())}\n')
        
        # 性别分布
        gender_counts = aggregated_df.groupby('Subject')['Gender'].first().value_counts()
        f.write('- 性别分布:\n')
        for gender, count in gender_counts.items():
            f.write(f'  - {gender}: {count}人\n')
        f.write('\n')
        
        # 3. 指标解释
        f.write('## 指标解释\n')
        f.write('### 主要指标\n')
        f.write('- **α偏侧化指数**: FP2(右)-FP1(左)的平均功率，反映了大脑左右半球在α波段的活动差异。\n')
        f.write('  - 正值表示右半球α波活动更强，负值表示左半球α波活动更强。\n')
        f.write('  - α波与放松状态相关，因此这一指标可能反映了阅读材料引起的情绪或认知加工偏好。\n\n')
        
        f.write('- **注意力投入程度**: α波与β波的比值(α/β)，反映了注意力的投入程度。\n')
        f.write('  - 较低的值表示β波相对更强，可能对应更高的注意力投入。\n')
        f.write('  - 较高的值表示α波相对更强，可能对应更放松的状态。\n\n')
        
        f.write('### 标准化指标\n')
        f.write('为控制个体差异，我们计算了相对于静息状态的标准化指标：\n')
        f.write('- **标准化α偏侧化指数**: α偏侧化指数除以静息段的相应值。\n')
        f.write('- **标准化注意力投入程度**: 注意力投入程度除以静息段的相应值。\n\n')
        
        # 在"尝试进行方差分析"之前添加前提检验代码
        # 4. 统计分析结果
        f.write('## 统计分析结果\n')
        
        # 前提检验
        f.write('### 前提检验\n')
        
        # 1. 正态性检验 (Shapiro-Wilk)
        f.write('#### 正态性检验 (Shapiro-Wilk)\n')
        f.write('正态性检验用于评估数据是否符合正态分布，p > 0.05表示数据符合正态分布。\n\n')
        
        f.write('| 条件 | 变量 | W统计量 | p值 | 结论 |\n')
        f.write('|------|------|---------|-----|------|\n')
        
        conditions = sorted(aggregated_df["Condition"].unique())
    
        for cond in conditions:
            # 标准化α偏侧化指数正态性检验
            alpha_data = aggregated_df[aggregated_df.Condition == cond]['NormAlphaLateralization'].dropna()
            if len(alpha_data) >= 3:  # Shapiro-Wilk要求至少3个样本
                w_alpha, p_alpha = shapiro(alpha_data)
                conclusion_alpha = "符合正态分布" if p_alpha > 0.05 else "不符合正态分布"
                f.write(f'| {cond} | 标准化α偏侧化指数 | {w_alpha:.3f} | {p_alpha:.4f} | {conclusion_alpha} |\n')
            else:
                f.write(f'| {cond} | 标准化α偏侧化指数 | - | - | 样本量不足 |\n')
            
            # 标准化注意力投入程度正态性检验
            attention_data = aggregated_df[aggregated_df.Condition == cond]['NormAttentionEngagement'].dropna()
            if len(attention_data) >= 3:
                w_attention, p_attention = shapiro(attention_data)
                conclusion_attention = "符合正态分布" if p_attention > 0.05 else "不符合正态分布"
                f.write(f'| {cond} | 标准化注意力投入程度 | {w_attention:.3f} | {p_attention:.4f} | {conclusion_attention} |\n')
            else:
                f.write(f'| {cond} | 标准化注意力投入程度 | - | - | 样本量不足 |\n')
        
        f.write('\n')
        
        # 2. 球形性检验 (Mauchly's Test)
        f.write('#### 球形性检验 (Mauchly\'s Test)\n')
        f.write('球形性检验用于评估不同条件间的方差是否相等，p > 0.05表示满足球形性假设。\n\n')
        
        try:
            # 准备数据用于球形性检验
            # 将数据转换为宽格式
            wide_alpha = aggregated_df.pivot(index='Subject', columns='Condition', values='NormAlphaLateralization')
            wide_attention = aggregated_df.pivot(index='Subject', columns='Condition', values='NormAttentionEngagement')
            
            # 进行球形性检验
            sphericity_alpha = pg.sphericity(wide_alpha.dropna(), subject=None, method='mauchly')
            sphericity_attention = pg.sphericity(wide_attention.dropna(), subject=None, method='mauchly')
            
            f.write('##### 标准化α偏侧化指数\n')
            f.write('```\n')
            f.write(str(sphericity_alpha))
            f.write('\n```\n\n')
            
            f.write('##### 标准化注意力投入程度\n')
            f.write('```\n')
            f.write(str(sphericity_attention))
            f.write('\n```\n\n')
            
            # 解释球形性检验结果
            f.write('#### 球形性检验结果解释\n')
            
            # α偏侧化指数球形性检验结果解释
            if isinstance(sphericity_alpha, tuple):
                # 处理元组类型结果 (statistic, pval, ...)
                p_sphericity_alpha = sphericity_alpha[1] if len(sphericity_alpha) > 1 else np.nan
            else:
                # 处理 SpherResults 对象
                p_sphericity_alpha = sphericity_alpha.pval if hasattr(sphericity_alpha, 'pval') else np.nan

            
            if not np.isnan(p_sphericity_alpha):
                if p_sphericity_alpha > 0.05:
                    f.write('- **标准化α偏侧化指数**: 满足球形性假设 (W={:.3f}, p={:.4f})，可以直接使用重复测量方差分析。\n'.format(
                        sphericity_alpha['W'].iloc[0], p_sphericity_alpha))
                else:
                    f.write('- **标准化α偏侧化指数**: 违反球形性假设 (W={:.3f}, p={:.4f})，需要使用Greenhouse-Geisser校正。\n'.format(
                        sphericity_alpha['W'].iloc[0], p_sphericity_alpha))
                    f.write('  Greenhouse-Geisser校正系数 (ε) = {:.3f}\n'.format(sphericity_alpha['GG'].iloc[0]))
            else:
                f.write('- **标准化α偏侧化指数**: 无法进行球形性检验，可能是由于数据不完整。\n')
            
            # 注意力投入程度球形性检验结果解释
            if isinstance(sphericity_attention, tuple):
                # 处理元组类型结果 (statistic, pval, ...)
                p_sphericity_attention = sphericity_attention[1] if len(sphericity_attention) > 1 else np.nan
            else:
                # 处理 SpherResults 对象
                p_sphericity_attention = sphericity_attention.pval if hasattr(sphericity_attention, 'pval') else np.nan

            if not np.isnan(p_sphericity_attention):
                if p_sphericity_attention > 0.05:
                    f.write('- **标准化注意力投入程度**: 满足球形性假设 (W={:.3f}, p={:.4f})，可以直接使用重复测量方差分析。\n\n'.format(
                        sphericity_attention['W'].iloc[0], p_sphericity_attention))
                else:
                    f.write('- **标准化注意力投入程度**: 违反球形性假设 (W={:.3f}, p={:.4f})，需要使用Greenhouse-Geisser校正。\n'.format(
                        sphericity_attention['W'].iloc[0], p_sphericity_attention))
                    f.write('  Greenhouse-Geisser校正系数 (ε) = {:.3f}\n\n'.format(sphericity_attention['GG'].iloc[0]))
            else:
                f.write('- **标准化注意力投入程度**: 无法进行球形性检验，可能是由于数据不完整。\n\n')
                
        except Exception as e:
            f.write(f'球形性检验未能成功执行: {str(e)}\n\n')
        
        # 尝试进行方差分析
        try:
            # 在方差分析部分修改代码，添加Greenhouse-Geisser校正
            # 重复测量方差分析 - 标准化α偏侧化指数
            anova_alpha_lat = AnovaRM(aggregated_df, 'NormAlphaLateralization', 'Subject', within=['Condition']).fit()
            f.write('### 重复测量方差分析\n')
            f.write('#### 标准化α偏侧化指数\n')
            f.write('```\n')
            f.write(str(anova_alpha_lat.summary()))
            f.write('\n```\n\n')
            
            # 如果违反球形性假设，添加Greenhouse-Geisser校正结果
            if 'p_sphericity_alpha' in locals() and p_sphericity_alpha < 0.05 and 'sphericity_alpha' in locals():
                gg_correction = sphericity_alpha['GG'].iloc[0]
                f.write('##### Greenhouse-Geisser校正后的结果\n')
                f.write('由于违反球形性假设，应用Greenhouse-Geisser校正：\n')
                
                # 校正自由度和p值
                df_num = anova_alpha_lat.anova_table['num DF']['Condition']
                df_den = anova_alpha_lat.anova_table['den DF']['Condition']
                f_val = anova_alpha_lat.anova_table['F']['Condition']
                
                corrected_df_num = df_num * gg_correction
                corrected_df_den = df_den * gg_correction
                
                # 使用F分布计算校正后的p值
                from scipy.stats import f
                corrected_p = 1 - f.cdf(f_val, corrected_df_num, corrected_df_den)
                
                f.write(f'校正后的自由度: {corrected_df_num:.2f}, {corrected_df_den:.2f}\n')
                f.write(f'F值: {f_val:.3f}\n')
                f.write(f'校正后的p值: {corrected_p:.4f}\n\n')
            
            # 重复测量方差分析 - 标准化注意力投入程度
            anova_attention = AnovaRM(aggregated_df, 'NormAttentionEngagement', 'Subject', within=['Condition']).fit()
            f.write('#### 标准化注意力投入程度\n')
            f.write('```\n')
            f.write(str(anova_attention.summary()))
            f.write('\n```\n\n')
            
            # 如果违反球形性假设，添加Greenhouse-Geisser校正结果
            if 'p_sphericity_attention' in locals() and p_sphericity_attention < 0.05 and 'sphericity_attention' in locals():
                gg_correction = sphericity_attention['GG'].iloc[0]
                f.write('##### Greenhouse-Geisser校正后的结果\n')
                f.write('由于违反球形性假设，应用Greenhouse-Geisser校正：\n')
                
                # 校正自由度和p值
                df_num = anova_attention.anova_table['num DF']['Condition']
                df_den = anova_attention.anova_table['den DF']['Condition']
                f_val = anova_attention.anova_table['F']['Condition']
                
                corrected_df_num = df_num * gg_correction
                corrected_df_den = df_den * gg_correction
                
                # 使用F分布计算校正后的p值
                corrected_p = 1 - f.cdf(f_val, corrected_df_num, corrected_df_den)
                
                f.write(f'校正后的自由度: {corrected_df_num:.2f}, {corrected_df_den:.2f}\n')
                f.write(f'F值: {f_val:.3f}\n')
                f.write(f'校正后的p值: {corrected_p:.4f}\n\n')
            
            # 解释方差分析结果
            f.write('#### 方差分析结果解释\n')
            
            # α偏侧化指数结果解释
            p_alpha = anova_alpha_lat.anova_table['Pr > F']['Condition']
            f.write('- **标准化α偏侧化指数**: ')
            if p_alpha < 0.05:
                f.write(f"存在显著差异 (F={anova_alpha_lat.anova_table['F']['Condition']:.3f}, p={p_alpha:.4f})。\n")
                f.write('  这表明不同阅读材料对大脑左右半球α波活动的影响存在显著差异。\n')
            else:
                f.write(f"不存在显著差异 (F={anova_alpha_lat.anova_table['F']['Condition']:.3f}, p={p_alpha:.4f})。\n")
                f.write('  这表明不同阅读材料对大脑左右半球α波活动的影响没有显著差异。\n')
            
            # 注意力投入程度结果解释
            p_attention = anova_attention.anova_table['Pr > F']['Condition']
            f.write('- **标准化注意力投入程度**: ')
            if p_attention < 0.05:
                f.write(f"存在显著差异 (F={anova_attention.anova_table['F']['Condition']:.3f}, p={p_attention:.4f})。\n")
                f.write('  这表明不同阅读材料需要的注意力投入程度存在显著差异。\n\n')
            else:
                f.write(f"不存在显著差异 (F={anova_attention.anova_table['F']['Condition']:.3f}, p={p_attention:.4f})。\n")
                f.write('  这表明不同阅读材料需要的注意力投入程度没有显著差异。\n\n')
                
        except Exception as e:
            f.write(f'方差分析未能成功执行: {str(e)}\n\n')
        
        # 配对t检验结果
        f.write('### 配对t检验结果\n')
        f.write('使用Bonferroni校正进行多重比较，校正后的显著性水平为0.05/3 = 0.0167。\n\n')
        
        conditions = ['长', '短', '漫']
        adjusted_alpha = 0.05 / 3  # Bonferroni校正
        
        f.write('#### 标准化α偏侧化指数\n')
        for i, cond1 in enumerate(conditions):
            for cond2 in conditions[i+1:]:
                try:
                    # 确保每个条件下有足够的数据
                    data1 = aggregated_df[aggregated_df.Condition==cond1].NormAlphaLateralization
                    data2 = aggregated_df[aggregated_df.Condition==cond2].NormAlphaLateralization
                    
                    # 移除NaN值并确保数据配对
                    valid_indices = ~(np.isnan(data1) | np.isnan(data2))
                    if sum(valid_indices) < 2:
                        f.write(f"- {cond1} vs {cond2}: 有效数据不足，无法进行检验\n")
                        continue
                        
                    t_stat, p_val = ttest_rel(data1[valid_indices], data2[valid_indices])
                    sig = "显著" if p_val < adjusted_alpha else "不显著"
                    f.write(f"- {cond1} vs {cond2}: t={t_stat:.3f}, p={p_val:.4f}, {sig}\n")
                except Exception as e:
                    f.write(f"- {cond1} vs {cond2}: 检验失败 ({str(e)})\n")
        
        f.write('\n#### 标准化注意力投入程度\n')
        for i, cond1 in enumerate(conditions):
            for cond2 in conditions[i+1:]:
                try:
                    # 确保每个条件下有足够的数据
                    data1 = aggregated_df[aggregated_df.Condition==cond1].NormAttentionEngagement
                    data2 = aggregated_df[aggregated_df.Condition==cond2].NormAttentionEngagement
                    
                    # 移除NaN值并确保数据配对
                    valid_indices = ~(np.isnan(data1) | np.isnan(data2))
                    if sum(valid_indices) < 2:
                        f.write(f"- {cond1} vs {cond2}: 有效数据不足，无法进行检验\n")
                        continue
                        
                    t_stat, p_val = ttest_rel(data1[valid_indices], data2[valid_indices])
                    sig = "显著" if p_val < adjusted_alpha else "不显著"
                    f.write(f"- {cond1} vs {cond2}: t={t_stat:.3f}, p={p_val:.4f}, {sig}\n")
                except Exception as e:
                    f.write(f"- {cond1} vs {cond2}: 检验失败 ({str(e)})\n")
        
        f.write('\n### 检验结果解释\n')
        f.write('- **标准化α偏侧化指数**: 配对t检验结果表明，不同阅读材料之间的α偏侧化指数差异...\n')
        f.write('  (根据实际结果补充解释)\n\n')
        f.write('- **标准化注意力投入程度**: 配对t检验结果表明，不同阅读材料之间的注意力投入程度差异...\n')
        f.write('  (根据实际结果补充解释)\n\n')
        
        # 5. 描述性统计
        f.write('## 描述性统计\n')
        
        # 计算各条件下的均值和标准差
        desc_stats = aggregated_df.groupby('Condition').agg({
            'NormAlphaLateralization': ['mean', 'std', 'count'],
            'NormAttentionEngagement': ['mean', 'std', 'count']
        })
        
        f.write('### 标准化α偏侧化指数\n')
        f.write('| 条件 | 均值 | 标准差 | 样本量 |\n')
        f.write('|------|------|--------|--------|\n')
        for cond in conditions:
            if cond in desc_stats.index:
                mean = desc_stats.loc[cond, ('NormAlphaLateralization', 'mean')]
                std = desc_stats.loc[cond, ('NormAlphaLateralization', 'std')]
                count = desc_stats.loc[cond, ('NormAlphaLateralization', 'count')]
                f.write(f'| {cond} | {mean:.4f} | {std:.4f} | {int(count)} |\n')
        
        f.write('\n### 标准化注意力投入程度\n')
        f.write('| 条件 | 均值 | 标准差 | 样本量 |\n')
        f.write('|------|------|--------|--------|\n')
        for cond in conditions:
            if cond in desc_stats.index:
                mean = desc_stats.loc[cond, ('NormAttentionEngagement', 'mean')]
                std = desc_stats.loc[cond, ('NormAttentionEngagement', 'std')]
                count = desc_stats.loc[cond, ('NormAttentionEngagement', 'count')]
                f.write(f'| {cond} | {mean:.4f} | {std:.4f} | {int(count)} |\n')
        
        # 6. 图表说明
        f.write('\n## 图表说明\n')
        f.write('### 箱线图解释\n')
        f.write('分析结果包含四个箱线图，分别展示了不同阅读材料条件下的以下指标：\n\n')
        
        f.write('1. **标准化α偏侧化指数**\n')
        f.write('   - 箱体中线表示中位数\n')
        f.write('   - 箱体上下边界分别表示第一和第三四分位数\n')
        f.write('   - 离群点表示异常值\n')
        f.write('   - 这一指标反映了不同阅读材料对大脑左右半球α波活动的影响\n\n')
        
        f.write('2. **标准化注意力投入程度**\n')
        f.write('   - 较低的值表示更高的注意力投入\n')
        f.write('   - 较高的值表示更放松的状态\n')
        f.write('   - 这一指标反映了不同阅读材料需要的注意力投入程度\n\n')
        
        f.write('3. **α偏侧化指数**（未标准化）\n')
        f.write('   - 展示了原始的α偏侧化指数，未经过静息状态标准化\n')
        f.write('   - 可用于比较标准化前后的差异\n\n')
        
        f.write('4. **注意力投入程度**（未标准化）\n')
        f.write('   - 展示了原始的注意力投入程度，未经过静息状态标准化\n')
        f.write('   - 可用于比较标准化前后的差异\n\n')
        
        # 7. 研究结论
        f.write('## 研究结论\n')
        f.write('根据以上分析结果，我们可以得出以下初步结论：\n\n')
        
        f.write('1. 不同阅读材料对被试的脑电活动有不同程度的影响。\n')
        f.write('2. 标准化注意力投入程度在不同阅读材料间可能存在显著差异，这表明不同类型的阅读材料可能需要不同程度的注意力投入。\n')
        f.write('3. α偏侧化指数的差异反映了不同阅读材料可能激活了大脑的不同区域。\n\n')
        
        # 8. 研究局限性
        f.write('## 研究局限性\n')
        f.write('1. 样本量有限，结果可能受到个体差异的影响。\n')
        f.write('2. 只使用了前额区域的两个电极(FP1和FP2)，无法全面反映整个大脑的活动情况。\n')
        f.write('3. 环境因素和实验过程中的干扰可能影响了数据的质量。\n')
        f.write('4. 部分数据存在缺失或异常值，可能影响了统计分析的准确性。\n\n')
        
        # 9. 未来研究方向
        f.write('## 未来研究方向\n')
        f.write('1. 增加样本量，提高统计检验的效力。\n')
        f.write('2. 使用更多电极位置，获取更全面的脑电活动信息。\n')
        f.write('3. 结合其他生理指标（如心率、皮电等），进行多模态分析。\n')
        f.write('4. 探究不同阅读材料对不同人群（如不同年龄、教育背景）的影响差异。\n')
    
    print(f"报告已生成: {output_md}")
    # 可选：生成简单的HTML版本
    try:
        import markdown
        with open(output_md, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        html_content = markdown.markdown(md_content, extensions=['tables'])
        
        with open(output_md.replace('.md', '.html'), 'w', encoding='utf-8') as f:
            f.write(f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>EEG分析结果报告</title>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; max-width: 900px; margin: 0 auto; padding: 20px; }}
        h1, h2, h3 {{ color: #2c3e50; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        code {{ background-color: #f8f8f8; padding: 2px 5px; border-radius: 3px; }}
        pre {{ background-color: #f8f8f8; padding: 10px; border-radius: 5px; overflow-x: auto; }}
    </style>
</head>
<body>
    {html_content}
</body>
</html>""")
        
        print(f"HTML报告已生成: {output_md.replace('.md', '.html')}")
    except ImportError:
        print("提示: 安装'markdown'包可以生成HTML版本的报告")

if __name__ == "__main__":
    generate_eeg_report()
