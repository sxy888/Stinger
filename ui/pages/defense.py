#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import streamlit as st

from service import get_db
from utils import get_all_enum_class_values
from constant.enum import DatasetChoice, AttackChoice, DefenderChoice

pd.set_option('display.float_format', '{:.2f}'.format)

# global config
st.set_page_config(
    page_title="防御记录",
    layout='wide',
    page_icon="🛡️",
    initial_sidebar_state="auto",
)

# ui
st.markdown("# 🛡️防御记录")
st.sidebar.markdown("防御算法运行实验结果")

# the search form
form = st.form(key="search")
col1, col2 = form.columns([1, 1])
dataset_option = col1.selectbox(
    '数据集', get_all_enum_class_values(DatasetChoice)
)
defense_option = col2.selectbox(
    '防御方法', get_all_enum_class_values(DefenderChoice)
)

form.form_submit_button(label="搜索")


# convert the records to data frame
columns = [
    "运行时间", '数据集',
    "防御方法", "防御 Overhead(%)",
    '防御耗时', "防御参数",
]

# the sqlite database store attack and defense records
db = get_db()
db_filter = {}
if dataset_option != "全部":
    db_filter['dataset'] = dataset_option
if defense_option != "全部":
    db_filter['defense_method'] = defense_option
records = db.query_models(**db_filter)
# filter has the sdr record
records = list(filter(lambda x: x[6] == "", records))
records = sorted(records, key=lambda x: (str(x[3]), str(x[4])))
df_index = [r[0] for r in records]
# filter some fields
df_data = [[r[1], r[2], r[3], r[4], r[7], r[9]] �qS )r   �   r   r   �   r   �   r   )r   r    r   r   r   r!   D   s    )�columns�indexu
   导出 CSVF)r&   zutf-8z
attack.csvztext/csvzdownload-attack-csv)*�pandas�pd�	streamlit�st�servicer   �utilsr   �