#!/usr/bin/env python3
# -*- coding: utf-8 -*-\
import matplotlib
import logging
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib import rcParams
from service import get_db
from matplotlib.axes import Axes
from constant.enum import DatasetChoice, AttackChoice, DefenderChoice


matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams["font.family"] = "Times New Roman"

# global config
st.set_page_config(
    page_title="换头实验",
    page_icon="🤯",
    initial_sidebar_state="auto",
)

st.markdown("# 🤯 换头实验")
st.sidebar.markdown("stinger 切换不同的对抗算法")

db = get_db()

defense_methods = [
    DefenderChoice.STINGER_GandLAF,
    DefenderChoice.STINGER_VAR_CNN,
    DefenderChoice.STINGER_DF,
    DefenderChoice.STINGER_AWF_SDAE,
    DefenderChoice.STINGER_AWF_CNN,
]

attack_methods = [
    AttackChoice.AWF_CNN,
    AttackChoice.AWF_SDAE,
    AttackChoice.DF,
    AttackChoice.VAR_CNN,
    AttackChoice.GANDaLF
]

def get_result(dataset: str):
    # 查询换头实验的记录
    result = []
    for d in defense_methods:
        row = []
        for a in attack_methods:
            records = db.query_models(
                dataset=dataset,
                attack_method=a.value,
                defense_method=d.value,
            )
            records = list(filter(lambda x: x[6] != "", records))
            if len(records) == 0:
                row.append(101)
                continue
            row.append(float(records[0][6]))
        result.append(row)
    return np.array(result)


def draw_label(ax: Axes, result: np.ndarray):
    # 将 SDR 值标记在图上
    row, col = result.shape
    for i in range(row):
        for j in range(col):
            color = 'black'
            # value = result[row-i-1][col-j-1]
            value = round(result[i][j], 2)
            # if value < 80:
            #     color = 'white'
            ax.text(j+0.3, i+0.4, value, color=color)



def draw_color_table(dataset: DatasetChoice):
    fig = plt.figure(figsize=(6, 3))
    result = get_result(dataset.value)
    ax = fig.add_subplot(111)
    plt.pcolor(
        result, edgecolors='k',linewidths=3, cmap='RdYlGn'
    )
    plt.colorbar()
    draw_label(ax, result)

    ax.set_ylabel("Defense Method")
    ax.set_yticklabels(
        [d.value.strip("Stinger-") for d in defense_methods],
    )
    ax.set_yticks([i + 0.5 for i in range(len(defense_methods))])

    # 将底部的坐标轴（也就是默认的X轴）隐藏
    ax.spines['bottom'].set_visible(False)
    # 将顶部的坐标轴设置为X轴
    ax.xaxis.set_ticks_position('top')
    ax.set_xticks([j + 0.5 for j in range(len(attack_methods))])
    ax.set_xticklabels(
        [a.value for a in attack_methods],
    )
    # plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9) # 调整子图的边距
    # if dataset == DatasetChoice.AWF:
    #     ax.set_xlabel("Attack Method")
    st.header(dataset.value + " Dataset")
    st.pyplot(fig)
    plt.savefig(f"./换头实验-{dataset.value}-Dataset.pdf", bbox_inches = 'tight')


# DF 数据集
draw_color_table(DatasetChoice.DF)
draw_color_table(DatasetChoice.AWF)
