import happybase
import thriftpy2
from thriftpy2.transport import TTransportException
import streamlit as st

def get_data_from_hbase():
    """Retrieve data from HBase."""
    try:
        host = "localhost"
        port = 9090

        # Logging connection attempt
        st.write(f"Attempting to connect to HBase at {host}:{port}...")

        # Connect to HBase
        connection = happybase.Connection(host, port=port, autoconnect=False)
        connection.open()
        st.write("Connected to HBase.")

        # Fetch data from the HBase table
        table = connection.table("fraud_transactions")
        rows = table.scan()

        # Convert rows to a list of dictionaries
        data = []
        for key, value in rows:
            row_data = {"TransactionID": key.decode()}  # Include row key
            row_data.update({k.decode(): v.decode() for k, v in value.items()})
            data.append(row_data)

        st.write("Data fetched successfully.")
        return data

    except TTransportException as e:
        st.error(f"Thrift transport error: {e}")
        return []
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return []

def main():
    """Main function for Streamlit app."""
    st.title("HBase Fraudulent Transactions Viewer")

    # Retrieve data from HBase
    data = get_data_from_hbase()

    # Display data in a table
    if data:
        st.write("Displaying fetched data:")
        st.dataframe(data)
    else:
        st.warning("No data found in HBase.")

if __name__ == "__main__":
    main()
