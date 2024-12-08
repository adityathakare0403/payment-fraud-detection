import streamlit as st
from utils.data_loader import fetch_data_from_hbase, add_record_to_hbase, update_record_in_hbase, delete_record_from_hbase
import pandas as pd

def run():
    st.title("📊 Data Overview for Database 2")

    # Initialize session state for the DataFrame
    if "df_db2" not in st.session_state:
        try:
            st.session_state.df_db2 = fetch_data_from_hbase(limit=100, database="Database 2")
        except Exception as e:
            st.session_state.df_db2 = None  # Initialize to None if loading fails
            st.error(f"Error loading data from HBase: {e}")

    if st.session_state.df_db2 is not None and not st.session_state.df_db2.empty:
        df = st.session_state.df_db2.copy()

        # Add filters for the displayed table
        st.subheader("Filter Data")
        columns_to_filter = df.columns.tolist()
        filters = {}
        for col in columns_to_filter:
            unique_values = df[col].dropna().unique()
            if len(unique_values) < 50:  # Only allow filters for columns with fewer than 50 unique values
                selected = st.multiselect(f"Filter by {col}", options=unique_values)
                if selected:
                    filters[col] = selected

        # Apply filters
        filtered_df = df.copy()
        for col, selected_values in filters.items():
            filtered_df = filtered_df[filtered_df[col].isin(selected_values)]

        st.dataframe(filtered_df)  # Show filtered data

        # CRUD Operations
        st.subheader("Edit Data")
        
        # Add Record
        with st.expander("➕ Add New Record"):
            new_data = {}
            for col in df.columns:
                new_data[col] = st.text_input(f"Enter value for {col}:", key=f"add_db2_{col}")
            if st.button("Add Record"):
                try:
                    add_record_to_hbase(new_data, database="Database 2")
                    st.success("Record added to HBase successfully!")
                except Exception as e:
                    st.error(f"Error adding record: {e}")
        
        # Update Record
        with st.expander("✏️ Update Existing Record"):
            selected_row = st.selectbox("Select Transaction ID to Update", df["Transaction ID"].unique())
            if selected_row:
                row_data = df[df["Transaction ID"] == selected_row].iloc[0]
                updated_data = {}
                for col in df.columns:
                    updated_data[col] = st.text_input(f"Update {col}:", value=row_data[col], key=f"update_db2_{col}")
                if st.button("Update Record"):
                    try:
                        update_record_in_hbase(selected_row, updated_data, database="Database 2")
                        st.success("Record updated in HBase successfully!")
                    except Exception as e:
                        st.error(f"Error updating record: {e}")
        
        # Delete Record
        with st.expander("❌ Delete Record"):
            selected_row_to_delete = st.selectbox("Select Transaction ID to Delete", df["Transaction ID"].unique())
            if st.button("Delete Record"):
                try:
                    delete_record_from_hbase(selected_row_to_delete, database="Database 2")
                    st.success("Record deleted from HBase successfully!")
                except Exception as e:
                    st.error(f"Error deleting record: {e}")
    else:
        st.warning("No data found.")