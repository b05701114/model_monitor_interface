# -*- coding: utf-8 -*-
'''
title           : monitor_package.py
description     : functions for classification model monitor
author          : Ling
date_created    : 20210127
date_modified   : -
version         : 2.0
usage           : functions for main.py
python_version  : 3.8
'''

import numpy as np
import pandas as pd
import streamlit as st
import plotly.figure_factory as ff
import plotly_express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_auc_score


@st.cache
def load_data(df, usecols=None, index_col=None, dtype=None):
    df = pd.read_csv(df, usecols=usecols, index_col=index_col, dtype=dtype)
    return df


@st.cache
def get_dict(df_format: pd.DataFrame) -> (dict, list, list):
    """
    Get all columns' data type that need to be transfer
    Arg:
        df_format: pd.Dataframe
           dataframe that is loaded in the beginning 
    Return:
        dict: dictionary that record all columns data type, used as read.csv
        list: list of all categorical columns and date columns
        list: list of all numeric columns
    """
    keys = df_format.index
    values = np.select([
        df_format['TYPE'] == 'str', df_format['TYPE'] == 'object',
        df_format['TYPE'] == 'date', df_format['TYPE'] == 'int',
        df_format['TYPE'] == 'float'
    ], ['str', 'str', 'str', pd.Int64Dtype(), 'float'])
    dict_asign = dict(zip(keys, values))

    str_type = list(
        df_format[(df_format['TYPE'] == 'str') | (df_format['TYPE'] == 'date')
                  | (df_format['TYPE'] == 'object')].index)
    num_type = list(df_format[(df_format['TYPE'] == 'int') |
                              (df_format['TYPE'] == 'float')].index)

    return dict_asign, str_type, num_type


@st.cache
def get_cols_cat(df: pd.DataFrame, str_type: list) -> pd.Series:
    """
    Get categorical columns' category
    Arg:
        df : pd.Dataframe
        str_type: list
            list of categorical columns
    Return:
        pd.Series: all categorical columns' category
    """
    series_out = pd.Series(dtype='object')
    for col in str_type:
        series_out = series_out.append(pd.Series([sorted(list(df[col].value_counts().index))], index=[col]))
    return series_out


@st.cache
def count_zero_num(df: pd.DataFrame, num_type: list) -> (pd.Series, pd.Series):
    """
    Count zero number and percentage for each numeric columns
    Arg:
        df: pd.Dataframe
        num_type: list
            list of numeric columns
    Return:
        pd.Series: number of zeros in each numeric columns
        pd.Series: percentage of zeros in each numeric columns
    """
    count_out = pd.Series(dtype='int')
    distribution_out = pd.Series(dtype='object')
    for col in num_type:
        count_out = count_out.append(
            pd.Series([sum(df[col].dropna() == 0)], index=[col]))
        distribution_out = distribution_out.append(
            pd.Series(
                [sum(df[col].dropna() == 0) / df.shape[0]],
                index=[col]))
    return count_out, distribution_out


@st.cache
def count_na_num(df: pd.DataFrame) -> pd.Series:
    """
    Count na number and percentage for each columns
    Arg:
        df: pd.Dataframe
    Return:
        pd.Series: number of zeros in each columns
        pd.Series: percentage of zeros in each columns
    """
    count_out = pd.Series(dtype='int')
    distribution_out = pd.Series(dtype='object')
    for col in df.describe(include='all').T.index:
        count_out = count_out.append(pd.Series([df[col].isnull().sum()], index=[col]))
        distribution_out = distribution_out.append(pd.Series([(df[col].isnull().sum() /df.shape[0])], index=[col]))
    return count_out, distribution_out


def get_year_month(df, date_col):
    """
    Get the year-month of a period of time
    Arg:
        df: pd.Dataframe
        date_col: str
            column name of date
    Return:
        pd.Series: number of zeros in each columns
        pd.Series: percentage of zeros in each columns
    """
    min_year_month = str(df[date_col].min().year) + "/" + str(df[date_col].min().month)
    max_year_month = str(df[date_col].max().year) + "/" + str(df[date_col].max().month)
    if min_year_month == max_year_month:
        year_month_str = min_year_month
    else:
        year_month_str = min_year_month + "-" + max_year_month
    return year_month_str


def get_basic_info(df, list_df, str_type, num_type, date_col) -> pd.DataFrame:
    """
    Get basic information of the three datasets(such as rows, column numbers...)
    Arg:
        df: pd.DataFrame
        list_df: list
            list of file tags, which means how many files to upload
        str_type: list
            list of categorical columns, which will be generate by function get_dict()
        num_type: list
            list of numeric columns, which will be generate by function get_dict()
        date_col: str
            column name of date(transaction date)
    Return:
        pd.Dataframe: dataframe with basic information of all upload datasets
    """
    date_list = []
    row_list = []
    for i in list_df:
        if date_col != 'Null':
            date_list.append(get_year_month(df[df["FILE_TAG"]==i], date_col))
        else:
            date_list.append('Null')
        row_list.append(df[df["FILE_TAG"]==i].shape[0])
        
    featrue_list = [str(len(str_type) + len(num_type))] * len(list_df)
    cat_col_list = [str(len(str_type))] * len(list_df)
    num_col_list = [+len(num_type)] * len(list_df)
    df_basic_info = pd.DataFrame(np.array([date_list, row_list, featrue_list, cat_col_list, num_col_list]),
        columns = list_df)
    df_basic_info["INFO"] = ["DATE", "ROWS", "FEATRUES", "CAT_COLS", "NUM_COLS"]
    df_basic_info = df_basic_info.set_index("INFO")
    return df_basic_info


def get_df_quatile(df):
    """
    Get min, max, quantile of a dataframe(which only with a column)
    Arg:
        df: pd.Dataframe
    Return:
        list:the min, PR25, PR50, mean, PR75, PR95, max number of the dataframe(which only with a column)
    """
    quantile_list = [
        df.min(),
        df.quantile(0.25),
        df.quantile(0.5),
        df.mean(),
        df.quantile(0.75),
        df.quantile(0.95),
        df.max()
    ]
    return quantile_list


def check_model_score(df, list_df, col_score, col_label,col_y):
    """
    Show the three dataset's model score distribution
    Arg:
        df: pd.DataFrame
        list_df: list
            list of file tags, which means how many files to upload
        col_score: str
            column name of model score
        col_y: str
            column name of y variable
    Return:
        pd.Dataframe: max, min, mean, quantile, auc of the three datasets' model score
    """
    df_description = get_num_info(df, list_df, col_score)
    auc_list = []
    for i in list_df:
        auc_list.append(roc_auc_score(df[df["FILE_TAG"]==i][col_y],df[df["FILE_TAG"]==i][col_label]))
        
    df_description["AUC"] = auc_list
    
    df_description = df_description.style.format("{:.5f}").set_properties(**{'background-color': "#A5CCFA"},
                          subset=['AUC'])
    return df_description


def plot_box(df, col_score):
    """
    df: pd.DataFrame
    col_score: str
        column name of model score
    """
    fig = px.box(df, x="FILE_TAG", y=col_score)
    fig.update_layout(title="MODEL SCORE", width=700, height=600)
    return fig


def get_data_description(df: pd.DataFrame,
                     df_format: pd.DataFrame) -> pd.DataFrame:
    """
    Get df.describe() info and 0, na counts
    Arg:
        df: pd.Dataframe
        df_format: pd.Dataframe
            format dataframe that was loaded at the beginning
    Return:
        pd.Dataframe: the description of the dataset
    """
    describe = df.describe(include='all').T
    df_describe = df_format.copy()
    df_describe['MAX'] = describe['max']
    df_describe['MIN'] = describe['min']
    df_describe['UNIQUE'] = describe['unique']
    df_describe['CATEGORY'] = get_cols_cat(df, get_dict(df_format)[1])
    df_describe['0_COUNT'], df_describe['0_DISTRIBUTION'] = count_zero_num(df,get_dict(df_format)[2])
    df_describe['NA_COUNT'], df_describe['NA_DISTRIBUTION'] = count_na_num(df)
    return df_describe


def get_status_report(df_description: pd.DataFrame,
                      str_type: list,
                      num_type: list,
                      min_max_dict: dict,
                      cat_dict: dict, 
                      na_dict: dict) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    Check if there is anything strange for each columns, including category check, min max check and na check
    Arg:
        df_description: pd.Dataframe
            dataframe that returned by function get_data_description()
        str_type: list
            list of categorical columns, which will be generate by function get_dict()
        num_type: list
            list of numeric columns, which will be generate by function get_dict()
        min_max_dict: dict
            dict that records training data's each numeric columns' min and max
        cat_dict: dict
            dict that records training data's each categorical columns' categories
    Return:
        pd.Dataframe: status report of all numeric columns(return for function find_num_index)
        pd.Dataframe: status report of all categotical columns(return for function find_str_index)
        pd.Dataframe: status report of all columns
    """
    testing_num_status = get_num_report(df_description, num_type, min_max_dict)
    testing_str_status = get_cat_report(df_description, str_type, cat_dict)
    all_status = testing_str_status.append(testing_num_status)
    # check NA percentage
    df_na_warnings = check_na_warnings(df_description, na_dict)
    all_status = all_status.append(df_na_warnings)
    return testing_num_status, testing_str_status, all_status


def check_na_warnings(df_description, na_dict):
    """
    Check if columns na distribution are out of range
    Arg:
        df_description: pd.DataFrame
        na_dict: na dictionary import from configure.py
    Return:
        pd.DataFrame: only with index that needs to be warned
    """
    df_na_warning = pd.DataFrame()
    default_list = list(set(df_description.index) - set(na_dict.keys()))
    
    for i in list(na_dict.keys()):
        if i != "DEFAULT":
            df_temp = df_description.loc[[i], ["NA_DISTRIBUTION"]]
            df_na_warning = df_na_warning.append(
                df_temp[df_temp["NA_DISTRIBUTION"] > na_dict[i]])
        else:
            df_temp = df_description.loc[default_list, ["NA_DISTRIBUTION"]]
            df_na_warning = df_na_warning.append(
                df_temp[df_temp["NA_DISTRIBUTION"] > na_dict["DEFAULT"]])
            
    if df_na_warning.shape[0] != 0:
        df_na_warning["NA_WARNING"] = "NA DISTRIBUTION IS OUT OF RANGE"
    
    return df_na_warning


def get_num_report(df_description: pd.DataFrame, num_type: list,
                   min_max_dict: dict):
    """
    Get numeric columns report of min max check
    Arg: 
        df_description:pd.DataFrame
        num_type: list
            list of numeric columns, which will be generate by function get_dict()
        min_max_dict: dict
            dict that records training data's each numeric columns' min and max
    Return:
        pd.DataFrame: only with index that needs to be warned
    """

    testing_num_status = pd.DataFrame(columns=["COL_NAME", 'DESCRIPTION'])
    df_min_max = pd.DataFrame.from_dict(min_max_dict).T.rename(columns={
        0: 'MIN',
        1: "MAX"
    })

    training_max = df_min_max.filter(items=min_max_dict.keys(), axis=0)['MAX']
    training_min = df_min_max.filter(items=min_max_dict.keys(), axis=0)['MIN']
    testing_max = df_description.filter(items=min_max_dict.keys(),
                                        axis=0)['MAX']
    testing_min = df_description.filter(items=min_max_dict.keys(),
                                        axis=0)['MIN']

    compare_max = testing_max.sort_index() > training_max.sort_index()
    compare_min = testing_min.sort_index() < training_min.sort_index()

    for idx in min_max_dict.keys():
        if compare_max[idx] and compare_min[idx]:
            testing_num_status = testing_num_status.append(
                {
                    'COL_NAME': idx,
                    "DESCRIPTION": "MAX AND MIN VALUE ERROR"
                },
                ignore_index=True)
        elif (not compare_max[idx]) and compare_min[idx]:
            testing_num_status = testing_num_status.append(
                {
                    'COL_NAME': idx,
                    "DESCRIPTION": "MIN VALUE ERROR"
                },
                ignore_index=True)
        elif compare_max[idx] and (not compare_min[idx]):
            testing_num_status = testing_num_status.append(
                {
                    'COL_NAME': idx,
                    "DESCRIPTION": "MAX VALUE ERROR"
                },
                ignore_index=True)

    testing_num_status = testing_num_status.set_index("COL_NAME")
    return testing_num_status


def get_cat_report(df_description: pd.DataFrame, str_type: list,
                   cat_dict: dict):
    """
    Get categorical columns report of category check
    Arg: 
        df_description:pd.DataFrame
        str_type: list
            list of categorical columns, which will be generate by function get_dict()
        cat_dict: dict
            dict that records training data's each categorical columns' categories
    Return:
        pd.DataFrame: only with index that needs to be warned
    """
    testing_str_status = pd.DataFrame(columns=['COL_NAME', 'DESCRIPTION'])

    df_cat = pd.DataFrame.from_dict(cat_dict).T.rename(columns={0: "CATEGORY"})
    training_category = df_cat.filter(items=str_type, axis=0)['CATEGORY']
    testing_category = df_description.filter(items=str_type,
                                             axis=0)['CATEGORY']

    for idx in str_type:
        # check category set
        if len(list(set(testing_category[idx]) -
                    set(training_category[idx]))) > 0:
            testing_str_status = testing_str_status.append(
                {
                    'COL_NAME':
                    idx,
                    "DESCRIPTION":
                    'UNKNOWN SET: ' + str(
                        set(testing_category[idx]) -
                        set(training_category[idx]))
                },
                ignore_index=True)

    testing_str_status = testing_str_status.set_index("COL_NAME")
    return testing_str_status


def find_num_index(df_num_report: pd.DataFrame,
                   df_testing: pd.DataFrame,
                   min_max_dict: dict) -> list:
    """
    Find strange data of numeric columns that was warned by function get_status_report
    Arg:
        df_num_report: pd.Dataframe
            dataframe that generate from function get_status_report
        df_testing: pd.Dataframe
            dataframe of this month's data
        cat_dict: dict
            dict that records training data's each categorical columns' categories
    Show:
        specific strange rows of the this month's dataset
    """
    df_testing = df_testing.reset_index()
    for idx in df_num_report.index:
        txt = idx + ': ' + df_num_report['DESCRIPTION'][idx]
        st.warning(txt)
        st.text('INDEX OF LIST:')
        st.text(
            np.where(
                ((df_testing[idx] > min_max_dict[idx][1]) |
                 (df_testing[idx] < min_max_dict[idx][0])
                 ).fillna(False))[0])
        warn_list = list(np.where(
                 ((df_testing[idx] > min_max_dict[idx][1]) |
                  (df_testing[idx] < min_max_dict[idx][0])
                   ).fillna(False))[0])
        st.text('WARNING DATA:')
        st.dataframe(
            df_testing.filter(items=warn_list, axis=0))


def find_str_index(df_str_report: pd.DataFrame,
                   df_testing: pd.DataFrame,
                   cat_dict: dict) -> list:
    """
    Find strange data of categorical columns that was warned by function get_status_report
    Arg:
        df_str_report: pd.Dataframe
            dataframe that generate from function get_status_report
        df_testing: pd.Dataframe
            dataframe of this month's data
        cat_dict: dict
            dict that records training data's each categorical columns' categories
    Show:
        specific strange rows of the this month's dataset
    """
    df_testing = df_testing.reset_index()
    for idx in df_str_report.index:
        txt = idx + ': ' + df_str_report['DESCRIPTION'][idx]
        st.warning(txt)
        st.text('INDEX OF LIST:')
        st.text(
            np.where(
                 (df_testing[idx].isin(cat_dict[idx][0])==False).fillna(False))[0])
        st.text('WARNING DATA:')
        st.dataframe(
            df_testing.filter(items=np.where(
                (df_testing[idx].isin(cat_dict[idx][0])==False).fillna(False))[0],
                                axis=0))


def get_ratio_df(df, col_cat, col_y, to_set_index, do_total_sum):
    """
    Get a dataframe that contains a categorial column and true y's confusion matrix
    Arg:
        df : pd.Dataframe
        col_cat: str
            the categorical column name to show
        col_y: str
            the y variable
        to_set_index: boolen
            do this dataframe need to set index
        do_total_sum: 
            do this dataframe need to get total info
    Return:
        pd.Dataframe
    """
    cat_levl = pd.crosstab(df[col_cat], df[col_y]).reset_index()
    if len(cat_levl.columns) != 3:
        # check if there is no y=0 or y=1 in a specific category
        if cat_levl.columns[1] != "0":
            cat_levl["0"] = 0
        else:
            cat_levl["1"] = 0
    cat_levl = cat_levl.rename(columns={"0": "NO", "1": "YES"})
    cat_levl["SUM"] = cat_levl["NO"] + cat_levl["YES"]
    cat_levl["SUM_RATIO"] = cat_levl["SUM"] / cat_levl["SUM"].sum()

    if do_total_sum:
        # add a row to record total information
        df_sum = pd.DataFrame([[
            "Total", cat_levl["NO"].sum(), cat_levl["YES"].sum(),
            cat_levl["SUM"].sum(), cat_levl["SUM_RATIO"].sum()
        ]], columns = [col_cat, "NO", "YES", "SUM", "SUM_RATIO"])
        cat_levl = pd.concat([cat_levl, df_sum], axis = 0)
        cat_levl = cat_levl.reset_index(drop=True)
        
    if to_set_index:
        # if the dataframe need to set index for columns' category
        cat_levl = cat_levl.set_index(col_cat)
        
    cat_levl["Y_RATIO"] = cat_levl["YES"] / cat_levl["SUM"]
    return cat_levl


def get_cat_df(df, list_df, col, col_y, warn_num,spe_cols=[]):
    """
    Concat the three dataset's get_ratio_df
    Arg:
        df: pd.DataFrame
        list_df: list
            list of file tags, which means how many files to upload
        col: str
            the column name
        col_y: str
            the column name of y variable
        spe_cols: list
            categorical columns that are numeric-like
    Return:
        pd.Dataframe: the dataframe that concat by the three datasets
    """
    main_cat_levl = get_ratio_df(df[df["FILE_TAG"] == list_df[-1]], col, col_y,
                                 True, False)
    main_cat_levl = main_cat_levl.rename(
        columns={
            "SUM_RATIO": list_df[-1] + "_SUM_RATIO",
            "Y_RATIO": list_df[-1] + "_Y_RATIO"
        })
    for i in list_df[:-1]:
        cat_levl_temp = get_ratio_df(df[df["FILE_TAG"] == i], col, col_y, True,
                                     False)
        main_cat_levl[i + "_SUM_RATIO"] = cat_levl_temp["SUM_RATIO"]
        main_cat_levl[i + "_Y_RATIO"] = cat_levl_temp["Y_RATIO"]
        main_cat_levl[i + "_SUM_R_WARNING"] = np.where(
        abs(main_cat_levl[i + "_SUM_RATIO"] - main_cat_levl[list_df[-1] + "_SUM_RATIO"]) > warn_num,
        "WARNING", "-")
        main_cat_levl[i + "_Y_R_WARNING"] = np.where(
        abs(main_cat_levl[i + "_Y_RATIO"] - main_cat_levl[list_df[-1] + "_Y_RATIO"]) > warn_num,
        "WARNING", "-")

    list_sort = ["NO", "YES", "SUM"]
    list_sum_ratio = [i for i in main_cat_levl.columns if "SUM_R" in i]
    list_y_ratio = [i for i in main_cat_levl.columns if "Y_R" in i]
    list_sort.extend(list_sum_ratio)
    list_sort.extend(list_y_ratio)
    main_cat_levl = main_cat_levl[list_sort]

    if col in spe_cols:
        #for cat cols that are numeric-like
        main_cat_levl.index = main_cat_levl.index.map(int)
        main_cat_levl = main_cat_levl.sort_index()

    return main_cat_levl


def check_col_warning(df, list_df, col_y, cat_dict, cat_warn_dict):
    show_list = []
    for col in cat_dict.keys():
        if col in list(cat_warn_dict.keys()):
            df_cat = get_cat_df(df, list_df, col, col_y, cat_warn_dict[col], spe_cols = ["SUSP_CODE"])
        else:
            df_cat = get_cat_df(df, list_df, col, col_y, cat_warn_dict["DEFAULT"], spe_cols = ["SUSP_CODE"])
            
        if "WARNING" in df_cat.values:
            show_list.append(col)
    return show_list


def show_cat_warning(df,list_df, col_y, cat_dict, cat_warn_dict):
    """
    Show sum_ratio, y_ratio and plot of a categorical column
    Arg:
        df: pd.DataFrame
        list_df: list
            list of file tags, which means how many files to upload
        col_y: str
            column name of y variable
        cat_dict: dict
            dictionary that has all the categorical columns
    Return:
        pd.DataFrame: sum_ratio, y_ratio information of specific categorical column
    """
    
    show_list = check_col_warning(df, list_df, col_y, cat_dict, cat_warn_dict)
    col = st.selectbox('SELECT CATEGORICAL VARIABLE：', show_list)
    if col in list(cat_warn_dict.keys()):
        df_cat = get_cat_df(df, list_df, col, col_y, cat_warn_dict[col], spe_cols = ["SUSP_CODE"])
    else:
        df_cat = get_cat_df(df, list_df, col, col_y, cat_warn_dict["DEFAULT"], spe_cols = ["SUSP_CODE"])
    return df_cat


def plot_bar(df_cat, list_df, colors):
    """
    Show sum_ratio, y_ratio plot of a categorical column
    Arg:
        df: pd.DataFrame
        list_df: list
            list of file tags, which means how many files to upload
        colors: list
            list of line plot's colors
    Return:
        plot: sum_ratio, y_ratio plot of specific categorical column
    """
    fig_bar = go.Figure(data = [go.Bar(name= i + '_SUM_RATIO',
               x=df_cat.index,
               y=list(df_cat[i + '_SUM_RATIO'])) for i in list_df])
    
    for i in range(len(list_df)):
        fig_bar.add_trace(go.Scatter(name=list_df[i] + '_Y_RATIO',
        x=df_cat.index,y=list(df_cat[list_df[i] + "_Y_RATIO"]), line=dict(color=colors[i])))
        
    fig_bar.update_layout(xaxis=dict(tickmode='array', tickvals=df_cat.index),
                       title="SUM_RATIO.BAR | Y_RATIO.LINE",  width=1500, height=550)
    return fig_bar


def get_num_info(df, list_df, col):
    """
    Get dataframe of a numeric column's min, max, quantile info
    Arg:
        df: pd.DataFrame
        list_df: list
            list of file tags, which means how many files to upload
        col: str
            numeric column name to observe
    Return:
        pd.Dataframe: min, max and quantile data of the numeric column
    """
    quantile_list = []
    for i in list_df:
        quantile_list_temp = get_df_quatile(df[df["FILE_TAG"]==i][col])
        quantile_list.append(quantile_list_temp)

    df_num_info = pd.DataFrame(
        np.array(quantile_list),
        columns = ["MIN", "Q1", "MEDIAN","MEAN", "Q3","95%", "MAX"])
    df_num_info["DATA"] = list_df
    df_num_info = df_num_info.set_index("DATA")
    
    return df_num_info


def show_num_plot(df, list_df,
                  num_dict):
    """
    Show min, max, quantile info and plot of a numeric column
    Arg:
        df: pd.DataFrame
        list_df: list
            list of file tags, which means how many files to upload
        num_dict: dict
            dictionary that has all the numeric columns
    Return:
        plot: distplot of a numeric colnumn
    """
    col = st.selectbox('SELECT NUMERIC VARIABLE：', list(num_dict.keys()))
    log = st.selectbox('SELECT GET LOG OR NOT：', [False, True])
    col_info = get_num_info(df, list_df, col)
    st.dataframe(col_info)
    colors = px.colors.qualitative.Plotly
    if log : 
        labels = [i + "_log" for i in list_df]
        hist_data = [np.log10((df[df["FILE_TAG"]==i][col].dropna()) + 1).to_list() for i in list_df]
        
    else:
        labels = list_df
        hist_data = [df[df["FILE_TAG"]==i][col].dropna().to_list() for i in list_df]
        
    fig_hist = ff.create_distplot(hist_data, labels, show_hist=False, colors = colors[:len(hist_data)])
    fig_hist.update_layout(title_text='NUMERICAL DATA', width=1500, height=550)
    return fig_hist
