import streamlit as st
import pandas as pd
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

@st.cache
def load_data():

    df = pd.read_csv("https://raw.githubusercontent.com/ThuwarakeshM/PracticalML-KMeans-Election/master/voters_demo_sample.csv")
    return df

df = load_data()

# MAIN
st.title("Interactive K-Means Clustering")
st.write("Here is the dataset used in this analysis:")
#st.write(df)

# SIDEBAR
sidebar = st.sidebar
df_display = sidebar.checkbox("Display Raw Data", value = True)

if df_display:
    st.write(df)

n_clusters = sidebar.slider(
    "Select Number of Clusters",
    min_value = 2,
    max_value = 10,
)

#IMPORTS
# -----------------------------------------------------------------------------------------------------------------------------

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

#helper functions

def run_kmeans(df, n_clusters = 2):
    kmeans = KMeans(n_clusters, random_state =0).fit(df[["Age","Income"]])
    fig, ax = plt.subplots(figsize = (16,9))

    #Create Scatterplot

    ax = sns.scatterplot(
        ax = ax,
        x = df.Age,
        y = df.Income,
        hue = kmeans.labels_,
        palette = sns.color_palette("colorblind", n_colors = n_clusters),
        legend = None
    )

    return fig


with st.container():
    st.write("Result:")
    st.write(run_kmeans(df, n_clusters = n_clusters))

