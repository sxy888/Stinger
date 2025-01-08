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
    page_title="攻击记录",
    layout='wide',
    page_icon="🔫",
    initial_sidebar_state="auto",
)

# ui
st.markdown("# 🔫攻击记录")
st.sidebar.markdown("攻击算法运行实验结果")

# the search form
form = st.form(key="search")
col1, col2, col3 = form.columns([1, 1, 1])
dataset_option = col1.selectbox(
    '数据集', get_all_enum_class_values(DatasetChoice)
)
defense_option = col2.selectbox(
    '防御方法', get_all_enum_class_values(DefenderChoice) + ["stinger-like"]
)
attack_option = col3.selectbox(
    '攻击方法', get_all_enum_class_values(AttackChoice)
)

form.form_submit_button(label="搜索")


# convert the records to data frame
columns = [
    "运行时间", '数据集',
    "防御方法", "防御 Overhead(%)",
    '攻击方法', "SDR(%)", "攻击耗时",
]

# the sqlite database store attack and defense records
db = get_db()
db_filter = {}
if dataset_option != "全部":
    db_filter['dataset'] = dataset_option
if defense_option not in ["全部", "stinger-like"]:
    db_filter['defense_method'] = defense_option
if attack_option not in  ["全部"]:
    db_filter['attack_method'] = attack_option
records = db.query_models(**db_filter)
# filter has the sdr record
records = list(filter(lambda x: x[6] != "", records))
records = sorted(records, key=lambda x: (x[3], x[4]))


if defense_option == "stinger-like":
    records = list(filter(lambda x: x[3].find("Stinger-") > -1, records))


df_index = [r[0] for r in records]
# filter some fields
df_data = [[r[1], r[2], r[3], r[4], r[5], r[6], r[8]] for r in records]
df = pd.DataFrame(df_data, columns=columns, index=df_index)

st.table(df)

st.download_button(
   "导出 CSV",
   df.to_csv(index=False).encode('utf-8'),
   "attack.csv",
   "text/csv",
   key='download-attack-csv'
)
