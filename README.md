# model_monitor_interface
Usage: This interface is for classification model monitoring.  
Features:  
1. Compare multiple datasets with main dataset(eg. training and testing dataset, this month and last 5 months datasets) using plots and tables.
2. Detect columns with strange percentage of na or 0 values.
3. Check if there are extreme values in compared datasets.(base on configure.py)
## Get Started
1. Create virtual env
```
conda create env --name monitor_interface python=3.8
conda activate monitor_interface
```
2. Install required packages
```
pip install -r requirments.txt
```
3. Run main.py
```
streamlit run main.py
```
4. Upload data/format.csv, prediction_test.csv, prediction_train.csv sequentially
