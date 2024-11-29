import happybase

def test_connection():
    try:
        connection = happybase.Connection('localhost', port=9090)  # Update port if needed
        connection.open()
        print("Connected to HBase successfully.")
        connection.close()
    except Exception as e:
        print(f"Failed to connect to HBase: {e}")

test_connection()
