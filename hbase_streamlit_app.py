import happybase
import streamlit as st

def get_data_from_hbase():
    try:
        st.write("Connecting to HBase on localhost:9090...")
        connection = happybase.Connection('localhost', port=9090, autoconnect=False)
        connection.open()
        st.write("Connected to HBase.")
        
        table = connection.table('fraud_transactions')
        st.write("Fetching data from HBase table...")
        
        data = []
        for key, value in table.scan():
            row = {'row_key': key.decode('utf-8')}
            row.update({k.decode('utf-8'): v.decode('utf-8') for k, v in value.items()})
            data.append(row)
        
        st.write("Data fetched successfully.")
        return data

    except happybase._thriftpy2.transport.TTransportException as e:
        st.error(f"Thrift transport error: {e}")
        return []
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return []

def main():
    st.title("HBase Fraudulent Transactions Viewer")
    
    # Retrieve data from HBase
    data = get_data_from_hbase()
    
    # Display data in a table
    if data:
        st.write("Fraudulent Transactions Data")
        st.dataframe(data)
    else:
        st.warning("No data found in HBase.")

if __name__ == '__main__':
    main()
