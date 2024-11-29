import streamlit as st
import happybase

# Function to connect to HBase and get data
def get_data_from_hbase():
    # Establish connection to HBase
    connection = happybase.Connection('localhost', port=9090)  # Update if using remote machine
    connection.open()

    # Connect to the table
    table = connection.table('fraudulent_transactions')

    # Retrieve the first 10 rows (you can change this limit as needed)
    data = []
    for i, row in enumerate(table.scan()):
        if i >= 10:
            break
        row_key = row[0]
        row_data = {column.decode(): value.decode() for column, value in row[1].items()}
        data.append({'row_key': row_key, **row_data})
    
    connection.close()
    return data

# Streamlit UI components
def main():
    # Set up the title of the app
    st.title("HBase Fraudulent Transactions Viewer")

    # Retrieve data from HBase
    data = get_data_from_hbase()

    # Display data in a table
    if data:
        st.write("Showing first 10 rows from fraudulent_transactions table:")
        st.write(data)  # Displaying the data as a simple DataFrame-like table
    else:
        st.write("No data found in HBase.")

    # Add additional widgets or functionality as needed
    st.sidebar.title("Options")
    st.sidebar.write("Select filters or actions here")

if __name__ == '__main__':
    main()
