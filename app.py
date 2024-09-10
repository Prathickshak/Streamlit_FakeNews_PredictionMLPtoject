import streamlit as st
from predict_page import predict_page_show

page = st.sidebar.selectbox("Fake News Prediction", ("Get Started", "Nothanks"))

if page == "Get Started":
    predict_page_show()
else:
    qoutes = "Thanks for visiting our page..."
    st.success(qoutes)