import pandas as pd
import happybase

# Define HBase connection parameters
HBASE_HOST = "192.168.1.4"
HBASE_PORT = 9090

def fetch_data_from_hbase(limit=None, database="Database 1"):
    """
    Fetch data from the specified HBase table.
    
    Parameters:
        limit (int): The number of records to fetch.
        database (str): The database to fetch data from. "Database 1" or "Database 2".
    
    Returns:
        pd.DataFrame: The data fetched from HBase.
    """
    try:
        connection = happybase.Connection(HBASE_HOST, port=HBASE_PORT)
        table_name = "fraudulent_transactions" if database == "Database 1" else "fraud_test_transactions"
        table = connection.table(table_name)

        rows = table.scan(limit=limit)
        data = []
        for key, row in rows:
            record = {"Transaction ID" if database == "Database 1" else "trans_num": key.decode()}
            for col, val in row.items():
                cf, col_name = col.decode().split(":")
                record[col_name] = val.decode()
            data.append(record)

        connection.close()
        return pd.DataFrame(data)
    except Exception as e:
        raise RuntimeError(f"Error fetching data from HBase: {e}")


def add_record_to_hbase(data, database="Database 1"):
    """
    Add a new record to the specified HBase table.

    Parameters:
        data (dict): The data to add.
        database (str): The database to add the record to. "Database 1" or "Database 2".
    """
    try:
        connection = happybase.Connection(HBASE_HOST, port=HBASE_PORT)
        table_name = "fraudulent_transactions" if database == "Database 1" else "fraud_test_transactions"
        table = connection.table(table_name)

        row_key = data.pop("Transaction ID" if database == "Database 1" else "trans_num")
        table.put(row_key, {f"{cf}:{col}": val for cf, col, val in format_data(data, database)})
        connection.close()
    except Exception as e:
        raise RuntimeError(f"Error adding record to HBase: {e}")


def update_record_in_hbase(row_key, updated_data, database="Database 1"):
    """
    Update an existing record in the specified HBase table.

    Parameters:
        row_key (str): The row key of the record to update.
        updated_data (dict): The updated data.
        database (str): The database to update the record in. "Database 1" or "Database 2".
    """
    try:
        connection = happybase.Connection(HBASE_HOST, port=HBASE_PORT)
        table_name = "fraudulent_transactions" if database == "Database 1" else "fraud_test_transactions"
        table = connection.table(table_name)

        table.put(row_key, {f"{cf}:{col}": val for cf, col, val in format_data(updated_data, database)})
        connection.close()
    except Exception as e:
        raise RuntimeError(f"Error updating record in HBase: {e}")


def delete_record_from_hbase(row_key, database="Database 1"):
    """
    Delete a record from the specified HBase table.

    Parameters:
        row_key (str): The row key of the record to delete.
        database (str): The database to delete the record from. "Database 1" or "Database 2".
    """
    try:
        connection = happybase.Connection(HBASE_HOST, port=HBASE_PORT)
        table_name = "fraudulent_transactions" if database == "Database 1" else "fraud_test_transactions"
        table = connection.table(table_name)

        table.delete(row_key)
        connection.close()
    except Exception as e:
        raise RuntimeError(f"Error deleting record from HBase: {e}")


def format_data(data, database):
    """
    Helper function to format data into column-family:column structure.

    Parameters:
        data (dict): The data to format.
        database (str): The database schema to use for formatting.
    
    Returns:
        list: A list of tuples with column-family, column, and value.
    """
    cf_mapping_db1 = {
        "address_info": ["shipping_address", "billing_address"],
        "customer_info": ["customer_age", "customer_id", "customer_location"],
        "product_info": ["product_category", "quantity", "payment_method", "device_used"],
        "transaction_info": ["transaction_amount", "transaction_date", "transaction_hour", "is_fraudulent", "account_age_days", "ip_address"],
    }
    
    cf_mapping_db2 = {
        "transaction_info": ["trans_date_trans_time", "cc_num", "merchant", "category", "amt", "unix_time", "is_fraud"],
        "customer_info": ["first_name", "last_name", "gender", "city_pop", "job", "dob"],
        "address_info": ["street", "city", "state", "zip", "lat", "long"],
        "merchant_info": ["merch_lat", "merch_long"],
    }

    cf_mapping = cf_mapping_db1 if database == "Database 1" else cf_mapping_db2

    formatted = []
    for key, value in data.items():
        for cf, cols in cf_mapping.items():
            if key in cols:
                formatted.append((cf, key, value))
                break
    return formatted
