#!/usr/bin/env python3
# -*- coding: utf-8 -*-\
from collections import defaultdict
import os
import matplotlib
import numpy as np
import typing as t
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.font_manager import FontProperties
from service import get_db
from scipy.interpolate import interp1d
from scipy.interpolate import make_interp_spline
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

from utils.common import get_all_enum_class_values
from utils.path import get_project_root_path
from constant.enum import DatasetChoice, AttackChoice, DefenderChoice

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# 注册字体
font_path = os.path.join(get_project_root_path(), "data", "ARIAL.TTF")
if os.path.exists(font_path):
    font_prop = FontProperties(fname=font_path)
    plt.rcParams["font.family"] = font_prop.get_name()

db = get_db()

# global config
st.set_page_config(
    page_title="对比图表",
    layout='wide',
    page_icon="🌟",
    initial_sidebar_state="auto",
)

st.markdown("# 🌟 对比图表")
st.sidebar.markdown("攻击防御算法对比图表")


def interp1d_data(x,  y, n_pointes: int = 1000):
    """ 
    插值，使得数据更加平滑
    @param n_pointes: 细粒度之后的点数
    """
    if len(x) <= 4 or n_pointes == 0:
        # 只有1个数据点无法平滑
        return x, y
    if isinstance(x, t.List):
        x = np.array(x)
    if isinstance(y, t.List):
        y = np.array(y)
    # f_linear = interp1d(x, y, kind='slinear')
    f_linear = interp1d(x, y, kind='quadratic')
    x_new = np.linspace(x.min(), x.max(), n_pointes)
    y_new = f_linear(x_new)
    return x_new, y_new


def draw_sub_fig(ax: Axes, attack: AttackChoice, dataset: DatasetChoice):
    # 画图参数
    line_width, marker_size = 6, 15
    # method_styles = {
    #     DefenderChoice.TAMARAW.value: {"marker": "o", "linestyle": "-", "linecolor": "#1d39c4"},
    #     DefenderChoice.WTF_PAD.value: {"marker": "s", "linestyle": "--", "linecolor": "#e65100"},
    #     DefenderChoice.FRONT.value: {"marker": "^", "linestyle": "-.", "linecolor": "#2b8a3e"},
    #     DefenderChoice.MOCKING_BIRD.value: {"marker": "d", "linestyle": ":", "linecolor": "#faad14"},
    #     DefenderChoice.ALERT.value: {"marker": "p", "linestyle": "-", "linecolor": "#7F2F8F"},
    #     DefenderChoice.AWA.value: {"marker": "p", "linestyle": "-", "linecolor": "#7F2F8F"},
    #     DefenderChoice.AWA_SAME.value: {"marker": "*", "linestyle": "--", "linecolor": "#b50000"},
    #     DefenderChoice.STINGER.value: {"marker": "P", "linestyle": "--", "linecolor": "#00b8d4"},
    # }

    method_styles = {
        DefenderChoice.TAMARAW.value: {"marker": "o", "linestyle": "-", "linecolor": "#1d39c4"},
        DefenderChoice.WTF_PAD.value: {"marker": "o", "linestyle": "--", "linecolor": "#bd4300"},
        DefenderChoice.FRONT.value: {"marker": "o", "linestyle": "-.", "linecolor": "#267a37"},
        DefenderChoice.MOCKING_BIRD.value: {"marker": "o", "linestyle": ":", "linecolor": "#d19111"},
        DefenderChoice.ALERT.value: {"marker": "o", "linestyle": "-", "linecolor": "#6d287a"},
        DefenderChoice.AWA.value: {"marker": "o", "linestyle": "-", "linecolor": "#6d287a"},
        DefenderChoice.AWA_SAME.value: {"marker": "o", "linestyle": "--", "linecolor": "#a30000"},
        DefenderChoice.STINGER.value: {"marker": "o", "linestyle": "--", "linecolor": "#00a6bf"},
    }

    ax.grid(linestyle=":")
    for spine in ax.spines.values():
        # 设置子图边框宽度
        spine.set_linewidth(4)

    # 遍历每一个防御算法
    for defense in get_all_enum_class_values(DefenderChoice, append_all=False):
        if defense == DefenderChoice.NO_DEF.value:
            continue
        if defense.find("Stinger-") > -1:
            continue
        # 获取数据
        records = db.query_models(
            dataset=dataset.value,
            attack_method=attack.value,
            defense_method=defense,
            use_for_plot=True,
        )
        # 筛选没有 sdr 的记录
        records = list(filter(lambda x: x[6] != "", records))
        if len(records) == 0:
            continue
        # 按照 overhead 进行分组
        group_by_overhead = defaultdict(list)
        for r in records:
            overhead, sdr = r[4], float(r[6])
            # 过滤掉 overhead 大于 100 的
            if overhead >= 102:
                continue
            group_by_overhead[overhead].append(sdr)
        # 转换成画图需要的数据
        x_list = sorted(list(group_by_overhead.keys()))
        y_list, y_min_list, y_max_list = [], [], []
        for x in x_list:
            y_points = group_by_overhead[x]
            y_min, y_max, y_avg = min(y_points), max(y_points), sum(y_points) / len(y_points)
            if y_min != y_avg:
                y_min -= line_width / 2
            if y_max != y_avg:
                y_max += line_width / 2
            y_min_list.append(y_min)
            y_max_list.append(y_max)
            y_list.append(y_avg)


        # 平滑处理
        n_pointes = 0
        _x_list, y_list = interp1d_data(x_list, y_list, n_pointes=n_pointes)
        _, y_max_list = interp1d_data(x_list, y_max_list, n_pointes=n_pointes)
        _, y_min_list = interp1d_data(x_list, y_min_list, n_pointes=n_pointes)

        # 先画主线
        ax.plot(
            _x_list, y_list,
            label=defense,
            linewidth=line_width,
            color=method_styles[defense]["linecolor"],
            marker=method_styles[defense]["marker"],  # 标记图形
            markeredgecolor="black",  # 外圈颜色
            markeredgewidth=4.5,
            markersize=marker_size,
        )
        # 填充上下区域
        # ax.fill_between(
        #     _x_list, y_max_list, y_min_list,
        #     facecolor=method_styles[defense]["linecolor"],  # 填充颜色
        #     # edgecolor='black, # 边界颜色
        #     linewidth=line_width,
        #     alpha=0.3, # 透明度
        # )

    plt.tick_params(axis='both', which='major', labelsize=16)
    ax.set_ylim(0, 105)  # 假定 Y 轴范围为 0-100
    # ax.set_xlim(5, 105)  # 假定 Y 轴范围为 0-100


# 创建图，2个数据集 * 6 个攻击算法，图的规模是 4 * 3
n_cols, n_rows = 4, 3
fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(40, 6.0* n_rows))


# 定义攻击方法的顺序和位置
# 6 个攻击方法，12 个子图从中间分开，两边代表不同的数据集
attack_fig_index = [] # type: t.List[t.List[t.Tuple[AttackChoice, DatasetChoice]]]
for i in range(n_rows):
    if i == 0:
        # 第一行是 CUMUL、DF 攻击
        attack_methods = [AttackChoice.CUMUL, AttackChoice.DF]
    elif i == 1:
        # 第二行是 AWF-SDAE、AWF-CNN 攻击
        attack_methods = [AttackChoice.AWF_SDAE, AttackChoice.AWF_CNN]
    else:
        # 第三行是 Var-CNN、GANDaLF
        attack_methods = [AttackChoice.VAR_CNN, AttackChoice.GANDaLF]

    sub_row = []
    # 每一行是 4 个图
    for dataset in [DatasetChoice.DF, DatasetChoice.AWF]:
        for attack_method in attack_methods:
            sub_row.append((attack_method, dataset))
    attack_fig_index.append(sub_row)


# 遍历进行画图
for row_idx in range(len(attack_fig_index)):
    for col_idx in range(len(attack_fig_index[0])):
        attack_method, dataset = attack_fig_index[row_idx][col_idx]
        ax = axes[row_idx][col_idx] # type: Axes
        draw_sub_fig(ax, attack_method, dataset)
        # 只有第一列展示 y 刻度
        if col_idx == 0:
            ax.set_yticklabels(["0", "20", "40", "60", "80", "100"], fontsize=36)
            ax.set_ylabel('SDR (%)', fontsize=35)
        else:
            ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

        if row_idx != n_rows - 1:
            # 只有最后一行展示 x 刻度
            ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            # 子图下方的小标题，显示攻击方法, 通过调整 labelpad 调节偏移位置
            ax.set_xlabel(attack_method.value, fontsize=35, labelpad=9)
        else:
            ax.tick_params(axis='x', which='both', labelsize=35)
            # 显示横坐标轴标题
            ax.text(
                0.5, -0.25,  # 坐标值，x = 0.5 表示居中，y = -0.15 表示在轴的下方
                "bandwidth overhead (%)",
                fontsize=38,
                ha='center',  # 水平居中对齐
                transform=ax.transAxes  # 使用轴的坐标系
            )
            ax.set_xlabel(attack_method.value, fontsize=35, labelpad=42,)
            # 最后一行的第 1 列和第 3 列显示数据集名称
            if col_idx == 0:
                ax.text(
                    1.0, -0.58,  # 坐标值，x = 0.5 表示居中，y = -0.15 表示在轴的下方
                    "(a)DF dataset",
                    fontsize=40,
                    ha='center',  # 水平居中对齐
                    transform=ax.transAxes  # 使用轴的坐标系
                )
            if col_idx == 2:
                ax.text(
                    1.0, -0.58,  # 坐标值，x = 0.5 表示居中，y = -0.15 表示在轴的下方
                    "(b)AWF dataset",
                    fontsize=40,
                    ha='center',  # 水平居中对齐
                    transform=ax.transAxes  # 使用轴的坐标系
                )

# 添加全局图例
handles, labels = axes[0][0].get_legend_handles_labels()  # 获取图例信息
axes[0][0].legend(
    handles, labels,
    loc=[0.45, 1.04],  # 图例位于图表顶部中心
    handletextpad=0.3,
    ncol=6,  # 图例按列排列
    columnspacing=0.5,
    fontsize=36,  # 图例字体大小
    frameon=False,  # 去掉图例边框
    handlelength=3,  # 增大图例线条长度
    handleheight=1.5,
    markerscale=1.2,
    labelspacing=4
)
# 调整子图布局，避免图例与子图重叠
plt.subplots_adjust(top=0.87,wspace=0.06, hspace=0.15,right=0.95,bottom=0.13)  # 调整顶部、子图间宽度和高度
plt.savefig("./attack.pdf")
# 在 Streamlit 中显示子图
st.pyplot(fig)