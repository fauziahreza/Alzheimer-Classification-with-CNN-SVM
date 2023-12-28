import streamlit as st

class MultiApp:
    def __init__(self):
        self.apps = []

    def add_app(self, title, func):
        self.apps.append({"title": title, "function": func})

    def run(self):
        app_names = [app["title"] for app in self.apps]
        app_selection = st.sidebar.selectbox("Pilih Aplikasi", app_names)

        for app in self.apps:
            if app["title"] == app_selection:
                app["function"]()

if __name__ == "__main__":
    st.set_page_config(page_title="Sistem Klasifikasi 3D MRI", layout="wide")

    from Apps.overview import app as overview_app
    from Apps.data_processing import app as data_processing_app
    from Apps.classification import app as classification_app

    multi_app = MultiApp()
    multi_app.add_app("Overview", overview_app)
    multi_app.add_app("Data Processing", data_processing_app)
    multi_app.add_app("Classification", classification_app)
    multi_app.run()