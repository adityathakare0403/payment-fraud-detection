import streamlit as st
import happybase

# Function to connect to HBase and get data
def get_data_from_hbase():
    try:
        # Establish connection to HBase
        st.write("Connecting to HBase on localhost:9090...")
        connection = happybase.Connection('localhost', port=9090)  # Adjust if using remote machine
        connection.open()
        st.write("Connected to HBase.")

        # Connect to the table
        table_name = 'fraudulent_transactions'
        if table_name.encode() not in connection.tables():
            st.error(f"Table '{table_name}' does not exist in HBase. Available tables: {connection.tables()}")
            connection.close()
            return []

        table = connection.table(table_name)

        # Retrieve the first 10 rows
        st.write(f"Fetching the first 10 rows from the '{table_name}' table...")
        data = []
        for i, row in enumerate(table.scan()):
            if i >= 10:
                break
            row_key = row[0].decode()  # Decode row key
            row_data = {column.decode(): value.decode() for column, value in row[1].items()}  # Decode columns and values
            data.append({'row_key': row_key, **row_data})
        
        connection.close()
        st.write("Data fetched successfully.")
        return data

    except happybase.hbase.ttypes.IOError as e:
        st.error(f"IOError while connecting to HBase: {e}")
        return []

    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return []

# Streamlit UI components
def main():
    # Set up the title of the app
    st.title("HBase Fraudulent Transactions Viewer")

    # Retrieve data from HBase
    data = get_data_from_hbase()

    # Display data in a table
    if data:
        st.write("Showing first 10 rows from the 'fraudulent_transactions' table:")
        st.dataframe(data)  # Use Streamlit's dataframe display for better readability
    else:
        st.write("No data found in HBase.")

    # Sidebar options (optional for further development)
    st.sidebar.title("Options")
    st.sidebar.write("Select filters or actions here")

if __name__ == '__main__':
    main()
