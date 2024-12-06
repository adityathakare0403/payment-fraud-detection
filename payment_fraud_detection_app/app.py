import streamlit as st

# Set the page configuration
st.set_page_config(
    page_title="Fraudulent Transactions Analysis",
    page_icon="💳",
    layout="wide",
)

st.sidebar.title("Navigation")
database = st.sidebar.radio("Select Database:", ["Database 1", "Database 2"])

if database == "Database 1":
    page = st.sidebar.radio("Go to", ["📊 Data Overview", "📈 Visualization 1", "📈 Visualization 2", "📈 Visualization 3"])

    if page == "📊 Data Overview":
        from pages.data_overview_db1 import run
        run()
    elif page == "📈 Visualization 1":
        from pages.visualizations_db1 import run
        run()
    elif page == "📈 Visualization 2":
        from pages.visualizations_db1_2 import run
        run()
    elif page == "📈 Visualization 3":
        from pages.visualizations_db1_3 import run
        run()

elif database == "Database 2":
    page = st.sidebar.radio("Go to", ["📊 Data Overview", "📈 Visualization 1", "📈 Visualization 2", "📈 Visualization 3"])

    if page == "📊 Data Overview":
        from pages.data_overview_db2 import run
        run()
    elif page == "📈 Visualization 1":
        from pages.visualizations_db2 import run
        run()
    elif page == "📈 Visualization 2":
        from pages.visualizations_db2_2 import run
        run()
    elif page == "📈 Visualization 3":
        from pages.visualizations_db2_3 import run
        run()
