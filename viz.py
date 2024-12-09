import os
import tempfile
import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.express as px
import matplotlib.pyplot as plt
import networkx as nx
from pyvis.network import Network
st.set_page_config(page_title="Scopus dataset VIZ" ,layout="wide")

st.title("Visualization of Scopus Dataset")

@st.cache_data
def load_data():
    data = pd.read_parquet('viz_data.parquet.gzip')
    asjc_df = pd.read_csv('ASJC_cat.csv')  
    return data , asjc_df

scopus_data , asjc_df = load_data()


#drop null row & cast to int
month_stats_df = scopus_data.copy()
month_stats_df = scopus_data.dropna(subset=["ref_count" , "citedby_count"],axis=0 , how="any")
month_stats_df["ref_count"] = month_stats_df["ref_count"].astype(int)
month_stats_df["citedby_count"] = month_stats_df["citedby_count"].astype(int)

month_stats_df["year"] = month_stats_df["delivered_date"].dt.year
month_stats_df["month"] = month_stats_df["delivered_date"].dt.month

count_metric = st.selectbox("metric" , options=["Citation" , "Reference"])
metric_map = {"Citation" : "citedby_count" , "Reference" : "ref_count"}
st.subheader(f"Total {count_metric}: {len(month_stats_df[metric_map[count_metric]])}")
fig = px.histogram(month_stats_df,
                   x="month",
                   y = metric_map[count_metric] , 
                   animation_frame="year" ,
                   title = f"Number of {count_metric} used each month" , 
                   category_orders={"year" : [2019,2020,2021,2022,2023]},
                   range_x=[1,12],
                   range_y=[0,120000],
                   labels={"total"}
                   )
fig.update_layout(transition = {"duration" : 100} )
fig.update_xaxes(dtick= "M1" , tickformat="%B\n%y")
st.plotly_chart(fig)



st.header("Which journal get funded?")
# # convert to dict, (index='Code')
asjc_df.set_index('Code',inplace=True)
asjc_dict = asjc_df["ASJC category"].to_dict()
asjc_cat_dict = asjc_df["ASJC category"].to_dict()
asjc_subj_dict = asjc_df["Subject area"].to_dict()

scopus_data = scopus_data.explode(column="ASJC_code") #list of subject code to multiple rows
scopus_data["is_funding"] = scopus_data["is_funding"].fillna("0")
scopus_data["is_funding"] = scopus_data["is_funding"].astype(int)

funded_sbj_df:pd.DataFrame = scopus_data.groupby("ASJC_code",as_index=False)["is_funding"].sum()
funded_sbj_df["Category"] = funded_sbj_df["ASJC_code"].astype(int).map(asjc_cat_dict)
funded_sbj_df["Subject_area"] = funded_sbj_df["ASJC_code"].astype(int).map(asjc_subj_dict)

# get unique color map
unique_sbj_area = funded_sbj_df["Subject_area"].unique()
total_sbj_area = len(unique_sbj_area)
colormap = plt.get_cmap('hsv')
cluster_colors = {cluster: [int(x*255) for x in colormap(i/(total_sbj_area))[:3]]
 for i, cluster in enumerate(unique_sbj_area)}

funded_sbj_df["color"] = funded_sbj_df["Subject_area"].map(cluster_colors)

horizontal_bar_g = px.bar(funded_sbj_df ,title="Graph showing total number of journal which is funded", x="is_funding",y="Category" , color="Subject_area")
st.plotly_chart(horizontal_bar_g)