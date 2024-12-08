import streamlit as st

# Set the page configuration
st.set_page_config(
    page_title="Data Driven Approach to Payment Fraud Detection",
    page_icon="💳",
    layout="wide",
)

# Sidebar for navigation
st.sidebar.title("Navigation")
database = st.sidebar.radio("Select Database:", ["Database 1", "Database 2"])

# Dynamic navigation options based on selected database
if database == "Database 1":
    page = st.sidebar.radio(
        "Go to",
        [
            "📊 Data Overview",
            "🌍 Geographical patterns",
            "🌏➡️📍 Geo-Distance",
            "🛍️ Product category"
        ]
    )

    # Navigation logic for Database 1
    if page == "📊 Data Overview":
        from pages.data_overview_db1 import run
        run()
    elif page == "🌍 Geographical patterns":
        from pages.visualizations_db1 import run
        run()
    elif page == "🌏➡️📍 Geo-Distance":
        from pages.visualizations_db1_2 import run
        run()
    elif page == "🛍️ Product category":
        from pages.visualizations_db1_3 import run
        run()

elif database == "Database 2":
    page = st.sidebar.radio(
        "Go to",
        [
            "📊 Data Overview",
            "👶👵 Customer age",
            "🕒 Time of day",
            "📅 Account age"
        ]
    )

    # Navigation logic for Database 2
    if page == "📊 Data Overview":
        from pages.data_overview_db2 import run
        run()
    elif page == "👶👵 Customer age":
        from pages.visualizations_db2 import run
        run()
    elif page == "🕒 Time of day":
        from pages.visualizations_db2_2 import run
        run()
    elif page == "📅 Account age":
        from pages.visualizations_db2_3 import run
        run()
