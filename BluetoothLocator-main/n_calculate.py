import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def calculate_n(rssi, distance, rssi0=-56.9, d0=1):
    """
    rssi (float): 接收信号强度
    distance (float): 距离
    rssi0 (float): 参考距离d0处的RSSI值，默认为-56.9
    d0 (float): 参考距离，默认为1m
    """
    # 避免除零和无效对数输入
    if distance <= 0 or np.isclose(distance, d0):
        return None
    
    try:
        # 确保对数输入为正
        log_term = np.log10(distance/d0)
        if log_term <= 0:
            return None
            
        n = (rssi0 - rssi) / (10 * log_term)
        # 过滤不合理的n值(通常n在1-7之间)
        if n < 0.5 or n > 10:
            return None
        return n
    except (ValueError, ZeroDivisionError):
        return None

def main():
    # 读取Excel文件
    try:
        df = pd.read_excel('./rssi_filtered.xlsx')
    except FileNotFoundError:
        print("错误：找不到文件rssi_filtered.xlsx，请确保文件在正确的路径下。")
        return
    
    # 检查必要的列是否存在
    required_columns = ['rssi', 'distance']
    for col in required_columns:
        if col not in df.columns:
            print(f"错误：文件中缺少必要的列 '{col}'")
            return
    
    # 过滤无效数据：距离必须大于0且不等于参考距离d0(1m)
    filtered_df = df[(df['distance'] > 0) & 
                     (~np.isclose(df['distance'], 1.0)) & 
                     (df['rssi'] < 0)].copy()
    
    if filtered_df.empty:
        print("错误：过滤后没有有效数据用于计算")
        return
    
    # 计算每个数据点的n值
    filtered_df['n'] = filtered_df.apply(lambda row: calculate_n(row['rssi'], row['distance']), axis=1)
    
    # 过滤掉计算失败的n值和NaN
    valid_n_df = filtered_df.dropna(subset=['n'])
    
    if valid_n_df.empty:
        print("错误：没有成功计算出有效的n值")
        return
    
    # 计算n的统计量，使用nan-safe函数
    n_mean = np.nanmean(valid_n_df['n'])
    n_median = np.nanmedian(valid_n_df['n'])
    n_std = np.nanstd(valid_n_df['n'])
    
    # 输出结果
    print(f"计算完成，共处理{len(valid_n_df)}个有效数据点")
    print(f"路径损耗指数n的统计结果:")
    print(f"  平均值: {n_mean:.4f}")
    print(f"  中位数: {n_median:.4f}")
    print(f"  标准差: {n_std:.4f}")
    
    # 可视化n的分布
    plt.figure(figsize=(10, 6))
    
    # 过滤掉无穷大值
    finite_n = valid_n_df['n'][np.isfinite(valid_n_df['n'])]
    
    if len(finite_n) > 0:
        plt.hist(finite_n, bins=20, alpha=0.7, color='skyblue')
        plt.axvline(n_mean, color='red', linestyle='dashed', linewidth=1, label=f'平均值: {n_mean:.4f}')
        plt.axvline(n_median, color='green', linestyle='dashed', linewidth=1, label=f'中位数: {n_median:.4f}')
        plt.title('路径损耗指数n的分布')
        plt.xlabel('路径损耗指数n')
        plt.ylabel('频率')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
    else:
        print("警告：没有有效的有限n值用于绘图")
    
    # 保存结果到新的Excel文件
    output_file = 'rssi_n_calculation_results.xlsx'
    valid_n_df.to_excel(output_file, index=False)
    print(f"详细计算结果已保存到 {output_file}")
    
    # 显示图形
    plt.show()

if __name__ == "__main__":
    main()    
