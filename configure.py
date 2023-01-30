# -*- coding: utf-8 -*-
import numpy as np
class InfoConfigure():
    def __init__(self):
        self.min_max_dict = {'Age': [0.42, 74.0],
         'SibSp': [0, 8],
         'Parch': [0, 5],
         'Fare': [0.0, 512.3292],
         'Score': [0, 1]}
        
        self.cat_dict = {'Survived': [[0, 1]],
                         'Pclass': [['1', '2', '3']],
                         'Sex': [['female', 'male']],
                         'Embarked': [['C', 'Q', 'S']],
                         'Ticket_for': [['A5', 'CA', 'PC', 'STO', 'num', 'oth']],
                         'Cabin_first': [['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T']],
                         'title': [['Master', 'Miss', 'Mr', 'Mrs', 'others']],
                         'Label': [["0", "1"]],
                         'Level':[["high", "low"]]}
        
        self.use_cols = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
       'Embarked', 'Ticket_for', 'Cabin_first', 'title', 'Label', 'Score', 'Level']
        # dictionary for each columns' limit percentage of na distribution
        self.na_dict = {"DEFAULT": 0.05}
        
        # dictionary for each categorical columns' limit percentage of warnings
        self.cat_warn_dict = {"DEFAULT": 0.05}
        
        self.main_date = "Null"
        self.col_level = "Level"
        self.col_label = "Label"
        self.col_score = "Score"
        self.col_y = "Survived"
        self.color_list = ["#3F51B5", "#DC3912", "#1B9E77", "#8E44AD", "#CA6F1E", "#2E86C1", "#D81B60", "#8BC34A"]
    
    def accuracy_warning(self, df):
        """
        Customize your model's each case risk level's accuracy
        """
        if df["Level"]=="High" and df["Y_RATIO"] < 0.85 :
            return ['background-color: #FFB6C1'] * (df.shape[0])
        else:
            return ['background-color: white']  * (df.shape[0])

    def get_threshold(self, df, col_score, col_y):
        """
        Customize how your model's thershold is calculated
        while running the function, 
            it will get the best threshold by calculating low risk accuracy when it meeets 99%

        Arg:
        df: pd.Dataframe
            data used to run theshold, which needs to contain six month transactions, with columns: col_score, col_y
        col_score: str
            the column name of model score
        col_y: str
            the column name of y
        Note:
          print final threshold score and its precentile
        """
        thres_list = df[col_score].quantile(np.arange(.01, 1, 0.001)).to_list()
        thres_index = df[col_score].quantile(np.arange(
        .01, 1, 0.001)).to_frame().reset_index()
        min_level = 0
        for score in thres_list:
            low_risk_accuracy = df[(df[col_score] < score)
                               & (df[col_y] == "0")].shape[0] / df[
                                   df[col_score] < score].shape[0]
            if low_risk_accuracy > 0.99:
                max_level = score

        thres_index = thres_index.rename(columns={"index": "PERCENTILE"})
        percentile = thres_index[thres_index[col_score] ==
                             max_level].index[0]
        text = "THIS MONTH'S THRESHOLD : " + str(
            round(max_level, 5)) + " | PERCENTILE： " + str(
                round(thres_index.at[percentile, "PERCENTILE"],2))
        return text
