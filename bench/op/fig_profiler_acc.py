#!/usr/bin/env python3
"""
fig_profiler_acc.py

绘制柱状图，对比 LoadAttention 的 SM-level 和 CTA-level cost 估计
相对于 NCU profile 实测 sm__cycles_active 的误差（支持多种度量）。

度量选项 (METRIC):
  - "mape"  : Mean Absolute Percentage Error (MAPE)，对极端值敏感
  - "smape" : Symmetric MAPE，范围固定 [0%, 200%]，更稳定
  - "mdape" : Median APE，用中位数抗极端值
  - "log"   : Log-ratio Error，对 ratio 度量最自然，天然对称
  - "all"   : 同时打印所有度量，但不画图（仅数值比较）

    "mape":          (compute_mape,          "MAPE (%)",          "lower"),
    "smape":         (compute_smape,         "SMAPE (%)",         "lower"),
    "mdape":         (compute_mdape,         "MdAPE (%)",         "lower"),
    "log":           (compute_log_ratio_error, "Log-ratio Error",  "lower"),
    "pearson":       (compute_pearson_r,     "Pearson r",         "higher"),
    "r2_calib":      (compute_r2_calibrated, "R² (calibrated)",   "higher"),
    "hitrate":       (compute_hit_rate,      "Hit Rate (%)",      "higher"),
    "smape_calib":   (compute_smape_calibrated, "SMAPE-calib (%)","lower"),

横坐标：模型名称
每组两根柱子：SM-level (蓝色) 和 CTA-level (橙色)

数据来源：
  - op_est_cost.csv (sm_lb_metric, cta_lb_metric) → 估计值
  - perf/op_util/op_util_model_*_method_load_attn.csv → sm__cycles_active 实测值
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===================== 路径常量 =====================
SRC  = os.path.dirname(os.path.abspath(__file__))
ROOT = SRC
CSV_PATH    = os.path.join(ROOT, 'op_est_cost.csv')
PROFILE_DIR = os.path.join(ROOT, 'op_util')
OUTPUT_DIR  = os.path.join(ROOT, 'fig')

# ===================== 模型名称映射 =====================
MODEL_DISPLAY_NAMES = {
    # MHA
    'progen2-tp1':   'Progen2-6B',
    'llama-7b-tp1':   'Llama2-7B',
    'llama-7b-tp2':   'Llama2-7B',
    'llama-7b-tp4':   'Llama2-7B',
    'llama-13b-tp2':  'Llama2-13B',
    # GQA
    'llama-3-8b-tp1': 'Llama3-8B',
    'llama-3-8b-tp2': 'Llama3-8B',
    'llama-3-8b-tp8': 'Llama3-8B',
    'yi-6b-tp1':      'Yi-6B',
    'yi-6b-tp2':      'Yi-6B',
    'yi-34b-tp1':     'Yi-34B',
    'yi-34b-tp2':     'Yi-34B',
    # GQA Sparse
    'mistral-7b-tp1': 'Mistral-7B',
    'mistral-7b-tp2': 'Mistral-7B',
    'mistral-7b-tp4': 'Mistral-7B',
}


# # ===================== 可配置参数 =====================
# # 选择画图使用的误差度量: "mape" / "smape" / "mdape" / "log"  / "pearson" / "r2_calib" / "hitrate" / "smape_calib" / "all"
# METRIC = "pearson"

# # 候选模型：当前两个数据源共有 load_attn 数据的模型
# CANDIDATE_MODELS = [
#         # MHA
#         # 'llama-7b-tp1',
#         # 'llama-7b-tp2',
#         'llama-7b-tp4',
#         # 'llama-13b-tp2',
#         # GQA
#         'llama-3-8b-tp1',
#         # 'llama-3-8b-tp2',
#         # 'llama-3-8b-tp8',
#         # 'yi-6b-tp1',
#         'yi-6b-tp2',
#         # 'yi-34b-tp1',
#         'yi-34b-tp2',
#         # GQA Sparse
#         # 'mistral-7b-tp1',
#         # 'mistral-7b-tp2',
#         # 'mistral-7b-tp4',
#     ]

# ===================== 可配置参数 =====================
# 选择画图使用的误差度量: "mape" / "smape" / "mdape" / "log"  / "pearson" / "r2_calib" / "hitrate" / "smape_calib" / "all"
METRIC = "smape_calib"

# 候选模型：当前两个数据源共有 load_attn 数据的模型
CANDIDATE_MODELS = [
        # MHA
        'progen2-tp1',
        # 'llama-7b-tp1',
        # 'llama-7b-tp2',
        # 'llama-7b-tp4',
        'llama-3-8b-tp8',
        'llama-13b-tp2',
        # GQA
        # 'llama-3-8b-tp1',
        # 'llama-3-8b-tp2',
        # 'yi-6b-tp1',
        # 'yi-6b-tp2',
        'yi-34b-tp1',
        # 'yi-34b-tp2',
        # GQA Sparse
        'mistral-7b-tp1',
        # 'mistral-7b-tp2',
        # 'mistral-7b-tp4',
        'progen2'
    ]


# ===================== 数据读取 =====================

def read_op_est_cost(model_name):
    """
    从 op_est_cost.csv 中读取指定模型的 sm_lb_metric 和 cta_lb_metric。

    参数:
        model_name: str, 如 'llama-7b-tp4'

    返回:
        (sm_lb_est, cta_lb_est): 两个 np.ndarray，长度等于该模型的配置数
    """
    df = pd.read_csv(CSV_PATH)
    model_df = df[df['model'] == model_name]
    if len(model_df) == 0:
        raise ValueError(f"Model '{model_name}' not found in op_est_cost.csv")

    sm_cost_max  = model_df['sm_cost_max'].astype(float).values
    cta_cost_max = model_df['cta_cost_max'].astype(float).values

    sm_cost_avg  = model_df['sm_cost_avg'].astype(float).values
    cta_cost_avg = model_df['cta_cost_avg'].astype(float).values

    sm_cost_min  = model_df['sm_cost_min'].astype(float).values
    cta_cost_min = model_df['cta_cost_min'].astype(float).values

    return sm_cost_max / sm_cost_min, cta_cost_max / cta_cost_min


    # sm_lb  = model_df['sm_lb_metric'].astype(float).values
    # cta_lb = model_df['cta_lb_metric'].astype(float).values

    # return 1 / sm_lb, 1 / cta_lb

    # return sm_lb, cta_lb


def read_profile_lb(model_name):
    """
    从 NCU profile CSV 读取 sm__cycles_active，计算 lb_metric = avg / (max - min)。

    参数:
        model_name: str, 如 'llama-7b-tp4'

    返回:
        sm_lb_profile: np.ndarray
    """
    filepath = os.path.join(
        PROFILE_DIR, f'op_util_model_{model_name}_method_load_attn.csv'
    )
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Profile file not found: {filepath}")

    df = pd.read_csv(filepath, skiprows=3, thousands=',')
    df = df.dropna(subset=['ID'])

    def to_numeric(series):
        return pd.to_numeric(
            series.astype(str).str.replace(',', ''), errors='coerce'
        )

    sm_max = to_numeric(df['sm__cycles_active.max'])
    sm_avg = to_numeric(df['sm__cycles_active.avg'])
    sm_min = to_numeric(df['sm__cycles_active.min'])

    # sm_diff = sm_max - sm_min
    sm_diff = sm_max / sm_min
    return sm_diff.dropna().values
    # sm_lb = sm_avg / sm_diff
    # sm_lb = sm_diff / sm_avg
    # return sm_lb.dropna().values


# ===================== 误差度量 =====================

def compute_mape(est, actual):
    """MAPE = mean(|est - actual| / actual) × 100%"""
    valid = actual > 0
    if valid.sum() == 0:
        return np.nan
    ape = np.abs(est[valid] - actual[valid]) / actual[valid]
    return float(np.mean(ape) * 100)


def compute_smape(est, actual):
    """SMAPE = mean(|est - actual| / ((|est| + |actual|) / 2)) × 100%，范围 [0, 200]"""
    denom = (np.abs(est) + np.abs(actual)) / 2
    valid = denom > 0
    if valid.sum() == 0:
        return np.nan
    return float(np.mean(np.abs(est[valid] - actual[valid]) / denom[valid]) * 100)


def compute_mdape(est, actual):
    """MdAPE = median(|est - actual| / actual) × 100%，抗极端值"""
    valid = actual > 0
    if valid.sum() == 0:
        return np.nan
    ape = np.abs(est[valid] - actual[valid]) / actual[valid]
    return float(np.median(ape) * 100)


def compute_log_ratio_error(est, actual):
    """Log-ratio Error = mean(|ln(est) - ln(actual)|)，可转为 exp(LRE)-1 百分比"""
    valid = (est > 0) & (actual > 0)
    if valid.sum() == 0:
        return np.nan
    lre = np.mean(np.abs(np.log(est[valid]) - np.log(actual[valid])))
    return float(lre)


# ---- 线性校准辅助函数 ----

def _linear_calibrate(est, actual):
    """
    用最小二乘法拟合 actual = a + b * est。
    返回 (intercept, slope, predicted, r2)
    """
    X = np.column_stack([np.ones(len(est)), est])
    coeff, _, _, _ = np.linalg.lstsq(X, actual, rcond=None)
    intercept, slope = coeff[0], coeff[1]
    predicted = X @ coeff
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return intercept, slope, predicted, r2


# ---- 积极度量（越高越好） ----

def compute_pearson_r(est, actual):
    """Pearson 相关系数：衡量线性关系强度，范围 [-1, 1]，越高越好"""
    # 使用 np.corrcoef
    valid = np.isfinite(est) & np.isfinite(actual)
    if valid.sum() < 3:
        return np.nan
    corr = np.corrcoef(est[valid], actual[valid])[0, 1]
    return float(corr)


def compute_r2_calibrated(est, actual):
    """R² after linear calibration：校准后估计值解释实际方差的比例，范围 [0, 1]，越高越好"""
    _, _, _, r2 = _linear_calibrate(est, actual)
    return float(r2)


def compute_hit_rate(est, actual):
    """
    成对命中率：任意两配置比较时，估计正确判断哪个负载更不均衡的比例。
    范围 [0, 100]%，越高越好，>50% 优于随机猜测。
    """
    n = len(est)
    hits, total = 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            if actual[i] != actual[j]:
                total += 1
                if (est[i] - est[j]) * (actual[i] - actual[j]) > 0:
                    hits += 1
    if total == 0:
        return np.nan
    return float(hits / total * 100)


# ---- 校准后误差度量（越低越好） ----

def compute_smape_calibrated(est, actual):
    """SMAPE after linear calibration：消除系统性偏差后的误差，越低越好"""
    _, _, predicted, _ = _linear_calibrate(est, actual)
    return compute_smape(predicted, actual)


# 度量注册表: (函数, 标签, 方向)
# direction: "lower" = 越低越好, "higher" = 越高越好
METRIC_FUNCTIONS = {
    "mape":          (compute_mape,          "MAPE (%)",          "lower"),
    "smape":         (compute_smape,         "SMAPE (%)",         "lower"),
    "mdape":         (compute_mdape,         "MdAPE (%)",         "lower"),
    "log":           (compute_log_ratio_error, "Log-ratio Error",  "lower"),
    "pearson":       (compute_pearson_r,     "Pearson r",         "higher"),
    "r2_calib":      (compute_r2_calibrated, "R² (calibrated)",   "higher"),
    "hitrate":       (compute_hit_rate,      "Hit Rate (%)",      "higher"),
    "smape_calib":   (compute_smape_calibrated, "SMAPE-calib (%)","lower"),
}

METRIC_LABELS = {k: v[1] for k, v in METRIC_FUNCTIONS.items()}
METRIC_DIRECTIONS = {k: v[2] for k, v in METRIC_FUNCTIONS.items()}


def compute_error(est, actual, metric):
    """根据 metric 名称计算对应的误差值。"""
    func, _, _ = METRIC_FUNCTIONS[metric]
    return func(est, actual)


# ===================== 绘图 =====================

def plot_bar_chart(models, sm_errors, cta_errors, metric):
    """
    绘制分组柱状图，横轴为模型，纵轴为误差值。

    参数:
        models:     list[str], 模型名称 (内部 key)
        sm_errors:  list[float], SM-level error
        cta_errors: list[float], CTA-level error
        metric:     str, 度量名称 ("mape"/"smape"/"mdape"/"log"/"pearson"/"r2_calib"/"hitrate"/"smape_calib")
    """
    # 样式
    plt.rcParams.update({'font.size': 18})
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False

    colors = {'sm': '#D674FF', 'cta': '#5CAFFF'}
    labels = {'sm': 'SM-level', 'cta': 'CTA-level'}

    # 对非百分比的度量做 ×100 缩放（Pearson/r2_calib/log 原始值为 0~1 或小数）
    if metric in ('pearson', 'r2_calib', 'log'):
        sm_vals = [v * 100 for v in sm_errors]
        cta_vals = [v * 100 for v in cta_errors]
        fmt = '{:.1f}%'
    else:
        sm_vals = list(sm_errors)
        cta_vals = list(cta_errors)
        fmt = '{:.1f}%'

    n = len(models)
    x = np.arange(n)
    width = 0.30
    opacity = 0.65

    fig, ax = plt.subplots(figsize=(10, 5))

    bar_sm = ax.bar(
        x - width / 2, sm_vals, width,
        color=colors['sm'], alpha=opacity,
        edgecolor='black', linewidth=0.8, label=labels['sm']
    )
    bar_cta = ax.bar(
        x + width / 2, cta_vals, width,
        color=colors['cta'], alpha=opacity,
        edgecolor='black', linewidth=0.8, label=labels['cta']
    )

    # 柱顶标注数值
    for bar in [bar_sm, bar_cta]:
        for rect in bar:
            h = rect.get_height()
            if np.isfinite(h):
                ax.text(
                    rect.get_x() + rect.get_width() / 2., h + 0.5,
                    fmt.format(h), ha='center', va='bottom', fontsize=14
                )

    # 横轴标签（加粗，不含 TP）
    display_names = [MODEL_DISPLAY_NAMES.get(m, m) for m in models]
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, fontsize=22, fontweight='bold')

    ax.set_ylabel(METRIC_LABELS.get(metric, 'Error'), fontsize=24, fontweight='bold')
    ax.legend(loc='upper right', fontsize=16, frameon=False, prop={'weight': 'bold'})
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    # 纵轴固定 0-105%，确保 100% 在图中可见，柱子紧贴底部
    ax.set_ylim(0, 105)
    # 去掉 x 轴两侧多余留白，让柱子紧贴坐标轴
    ax.margins(x=0.05)

    fig.tight_layout(pad=0.5)

    # 保存
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pdf_path = os.path.join(OUTPUT_DIR, 'profiler_acc.pdf')
    svg_path = os.path.join(OUTPUT_DIR, 'profiler_acc.svg')
    plt.savefig(pdf_path, bbox_inches='tight', dpi=300)
    plt.savefig(svg_path, format='svg', bbox_inches='tight', dpi=300, transparent=True)
    print(f"Saved: {pdf_path}")
    print(f"Saved: {svg_path}")
    plt.show()


# ===================== 主流程 =====================

def main():

    models_ok = []
    sm_errors  = []   # 用于画图的 SM 误差
    cta_errors = []   # 用于画图的 CTA 误差

    # 同时收集所有度量用于打印
    all_metrics = {m: {'sm': {}, 'cta': {}} for m in CANDIDATE_MODELS}

    for model in CANDIDATE_MODELS:
        try:
            sm_est, cta_est = read_op_est_cost(model)
            sm_profile = read_profile_lb(model)

            if len(sm_est) != len(sm_profile):
                print(f"[SKIP] {model}: row count mismatch "
                      f"(est={len(sm_est)}, profile={len(sm_profile)})")
                continue

            # 计算所有度量
            for m_name in METRIC_FUNCTIONS:
                all_metrics[model]['sm'][m_name]  = compute_error(sm_est,  sm_profile, m_name)
                all_metrics[model]['cta'][m_name] = compute_error(cta_est, sm_profile, m_name)

            # 提取画图用的度量
            sm_val  = all_metrics[model]['sm'][METRIC]
            cta_val = all_metrics[model]['cta'][METRIC]

            if np.isnan(sm_val) or np.isnan(cta_val):
                print(f"[SKIP] {model}: {METRIC} is NaN")
                continue

            models_ok.append(model)
            sm_errors.append(sm_val)
            cta_errors.append(cta_val)

        except FileNotFoundError as e:
            print(f"[SKIP] {model}: {e}")
        except ValueError as e:
            print(f"[SKIP] {model}: {e}")
        except Exception as e:
            print(f"[ERROR] {model}: {e}")

    # ---- 打印所有度量对比表 ----
    if models_ok:
        print(f"\n{'='*90}")
        print(f"  Error metrics comparison (plotted: {METRIC})")
        dir_symbol = '↑ higher=better' if METRIC_DIRECTIONS.get(METRIC) == 'higher' else '↓ lower=better'
        print(f"  Direction: {dir_symbol}")
        print(f"{'='*90}")
        header = f"{'Model':<20} {'Lvl':<4}"
        for m in METRIC_FUNCTIONS:
            header += f" {m:>12}"
        print(header)
        print("-" * 90)
        for model in models_ok:
            for level in ['sm', 'cta']:
                row = f"{model:<20} {level.upper():<4}"
                for m_name in METRIC_FUNCTIONS:
                    val = all_metrics[model][level][m_name]
                    if m_name in ('log',):
                        row += f" {val:>12.4f}"
                    elif m_name in ('pearson', 'r2_calib'):
                        row += f" {val:>12.3f}"
                    else:
                        row += f" {val:>11.1f}%"
                print(row)

            # 打印校准公式
            sm_est, cta_est = read_op_est_cost(model)
            sm_actual = read_profile_lb(model)
            _, _, _, _ = _linear_calibrate(sm_est, sm_actual)
            # re-read for cta
            a_sm, b_sm, _, _ = _linear_calibrate(sm_est, sm_actual)
            a_cta, b_cta, _, _ = _linear_calibrate(cta_est, sm_actual)
            print(f"  {'':24}calibration: actual = {a_sm:.3f} + {b_sm:.3f}×SM_est,  actual = {a_cta:.3f} + {b_cta:.3f}×CTA_est")
            print()
        print(f"{'='*90}\n")

    if len(models_ok) == 0:
        print("No valid models to plot. Exiting.")
        return

    if METRIC == "all":
        print("METRIC='all': skipping plot, see comparison table above.")
        return

    plot_bar_chart(models_ok, sm_errors, cta_errors, METRIC)


if __name__ == '__main__':
    main()
