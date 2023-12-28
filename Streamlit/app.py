# Pada file app.py
import streamlit as st
from Apps.overview import app as overview_app
from Apps.data_processing import app as data_processing_app
from Apps.classification import app as classification_app

def main():
    st.set_page_config(page_title="Sistem Klasifikasi 3D MRI", layout="wide")

    st.sidebar.title("Navigation")
    
    # Ganti st.sidebar.radio dengan st.sidebar.selectbox
    app_selection = st.sidebar.selectbox("Go to", ("Overview", "Data Processing", "Classification"))

    if app_selection == "Overview":
        overview_app()
    elif app_selection == "Data Processing":
        data_processing_app()
    elif app_selection == "Classification":
        classification_app()

if __name__ == "__main__":
    main()
