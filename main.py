# -*- coding: utf-8 -*-
'''
title           : main.py
description     : for classification ai model monitoring
author          : Ling
date_created    : 20210127
date_modified   : -
version         : 2.0
usage           : main python file
python_version  : 3.8
web reference   : http://awesome-streamlit.org/ (Kickstarter Dashboard)
'''

# ## import

import numpy as np
import pandas as pd
import streamlit as st
import os
import datetime
from monitor_package import *
import configure as cg

# Global variable
GV = cg.InfoConfigure()
MIN_MAX_DICT = GV.min_max_dict 
CAT_COL_DICT = GV.cat_dict
NA_DICT = GV.na_dict
CAT_WARN_DICT = GV.cat_warn_dict
COL_DATE = GV.main_date
COL_LEVEL = GV.col_level
COL_LABEL = GV.col_label
COL_SCORE = GV.col_score
COL_Y = GV.col_y
USE_COLS = GV.use_cols
COLORS = GV.color_list
# -

# ## set screen

## Web Style 
STYLE = """
<style>
    img {
        max-width: 100%;
    }
    .block-container {{
        max-width: "1400px";
        padding: 5rem 1rem 10rem;
        color: "#FFF";
        background-color: "#000";
    }}         
</style>
"""

st.set_page_config(layout="wide")
st.markdown(STYLE, unsafe_allow_html=True)
st.set_option('deprecation.showfileUploaderEncoding', False)

# 程式功能說明
st.markdown(
    """
    <h1 style= 'font-family: 微軟正黑體';>
    MODEL DASHBOARD
    </a>
    </h1>
    """,
    unsafe_allow_html=True)
st.sidebar.markdown(
    """   
    <h3 style= 'font-family: 微軟正黑體';> 
    請上傳以下檔案(以csv檔匯入)：\n
    1.變數規格檔\n
    2.比較資料集(可上傳複數檔案)\n
    3.主要資料集\n
    </h3>
    """,
    unsafe_allow_html=True)

# ## Read Data

# +
format_csv = st.sidebar.file_uploader('UPLOAD FORMAT CSV FILE', type='csv')

if format_csv:
    df_format = load_data(format_csv, index_col=0)
    dict_asign, str_type, num_type = get_dict(df_format)
    compare_csv = st.sidebar.file_uploader('UPLOAD COMPARE DATA', type='csv', accept_multiple_files=True)
    
    if compare_csv:
        df_all = pd.DataFrame()
        for file in compare_csv:
            df_compare = pd.read_csv(file, usecols = USE_COLS, dtype=dict_asign)
            if COL_DATE!='Null':
                df_compare[COL_DATE] = pd.to_datetime(df_compare[COL_DATE])
            df_compare["FILE_TAG"] = file.name.split(".")[0]
            df_all = pd.concat([df_all, df_compare], axis = 0)

        main_csv = st.sidebar.file_uploader('UPLOAD MAIN DATA', type='csv')
        
        if main_csv:
            df_main = pd.read_csv(main_csv, usecols = USE_COLS, dtype=dict_asign)
            if COL_DATE!='Null':
                df_main[COL_DATE] = pd.to_datetime(df_main[COL_DATE])
            df_main["FILE_TAG"] = main_csv.name.split(".")[0]
            df_all = pd.concat([df_all, df_main], axis = 0)
            list_datasets = df_all.FILE_TAG.unique()
        else:
            st.error('MAIN DATA NOT FOUND')
    else:
            st.error('COMPARE DATA NOT FOUND')
else:
    st.error('FORMAT FILE NOT FOUND')
# -

# ### testing data description

main_data_description = get_data_description(df_main[USE_COLS], df_format.iloc[1:,])

# ### num: max min check; str: catogery check

# +
num_status_report, str_status_report, status = get_status_report(
    df_description=main_data_description,
    str_type=str_type[1:],
    num_type=num_type,
    min_max_dict=MIN_MAX_DICT,
    cat_dict=CAT_COL_DICT,
    na_dict = NA_DICT)

status = status.join(df_format[['TYPE']], how="left")
status = status[['TYPE', 'DESCRIPTION', "NA_WARNING"]]
# -

st.markdown(
    """   
    <h3 style= 'font-family: 微軟正黑體';> 
    GENERAL INFO OF DATA SET
    </h3>
    """,
    unsafe_allow_html=True)

df_basic_info = get_basic_info(df_all, list_datasets, str_type[1:], num_type, COL_DATE)
st.dataframe(df_basic_info)

# +
# thres = GV.get_threshold(df_all[df_all["FILE_TAG"]=="2020Q4"], "CASE_RISK_SCORE", "TRUE_Y")
# st.write(thres)
# -

st.markdown(
    """   
    <h3 style= 'font-family: 微軟正黑體';> 
    REPORT OF CASE RISK LEVEL
    </h3>
    """,
    unsafe_allow_html=True)

testing_level_ratio = get_ratio_df(df_main, COL_LEVEL, COL_Y, False, True)

# check if low level y ratio >= 0.015, if ture bgcolor turns red
st.write(
    testing_level_ratio.style.apply(GV.accuracy_warning,
                                       axis=1).format({
                                           'Y_RATIO':
                                           "{:.2%}",
                                           'SUM_RATIO':
                                           '{:.2%}'
                                       }))

st.markdown(
    """   
    <h3 style= 'font-family: 微軟正黑體';> 
    CASE RISK SCORE CHECK
    </h3>
    """,
    unsafe_allow_html=True)

df_model_score = check_model_score(df_all, list_datasets, COL_SCORE, COL_LABEL ,COL_Y)
st.dataframe(df_model_score)
fig = plot_box(df_all, COL_SCORE)
st.plotly_chart(fig)

st.markdown(
    """   
    <h3 style= 'font-family: 微軟正黑體';> 
    TESTING DATA WARNING LIST
    </h3>
    """,
    unsafe_allow_html=True)

st.dataframe(status)

st.markdown(
    """   
    <h3 style= 'font-family: 微軟正黑體';> 
    TESTING DATA DESCRIPTION
    </h3>
    """,
    unsafe_allow_html=True)

st.dataframe(main_data_description.style.format({'NA_DISTRIBUTION': "{:.2%}", 
                                            '0_DISTRIBUTION': '{:.2%}',
                                           "MAX":"{:.2f}",
                                           "MIN":"{:.2f}",
                                           "0_COUNT": "{:.0f}"}))

# ### find error value

st.markdown(
    """   
    <h3 style= 'font-family: 微軟正黑體';> 
    WARNING DATA
    </h3>
    """,
    unsafe_allow_html=True)
with st.expander("HIDE WARNING DATA OR NOT"):
    find_num_index(df_num_report = num_status_report, df_testing = df_main[USE_COLS], min_max_dict = MIN_MAX_DICT)
    find_str_index(df_str_report = str_status_report, df_testing = df_main[USE_COLS], cat_dict = CAT_COL_DICT)

# ### categorical data check

st.markdown(
    """   
    <h3 style= 'font-family: 微軟正黑體';> 
    CATEGORICAL DATA CHECKING
    </h3>
    """,
    unsafe_allow_html=True)

df_cat = show_cat_warning(df_all, list_datasets, COL_Y, CAT_COL_DICT, CAT_WARN_DICT)
st.dataframe(
    df_cat.style.format("{:.2%}",
                        subset=[
                            i for i in df_cat.columns if "RATIO" in i
                        ]).set_properties(**{'background-color': "#A5CCFA"},
                                          subset=[list_datasets[-1]+'_SUM_RATIO', list_datasets[-1]+"_Y_RATIO"]))

fig_bar = plot_bar(df_cat, list_datasets, COLORS)
st.plotly_chart(fig_bar)

st.markdown(
    """   
    <h3 style= 'font-family: 微軟正黑體';> 
    NUMERIC DATA CHECKING
    </h3>
    """,
    unsafe_allow_html=True)

fig_hist = show_num_plot(df_all, list_datasets, MIN_MAX_DICT)
st.plotly_chart(fig_hist)
