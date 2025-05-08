import streamlit as st

from ui import (
    ui_settings,
    ui_main,
)


st.set_page_config(layout="wide")

settings_layout = st.sidebar
data_layout = st.container()

settings = ui_settings(settings_layout)

ui_main(data_layout, settings)
