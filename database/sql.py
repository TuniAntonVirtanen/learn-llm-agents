import sqlite3
from typing import List


def run_query(query, db_path="database/jokes.db"):
    try:
        # Connect to the database
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

        # Execute the query
        cur.execute(query)

        # If the query is an INSERT, UPDATE, or DELETE, commit the transaction
        if query.strip().upper().startswith(("INSERT", "UPDATE", "DELETE")):
            conn.commit()
            result = "Query executed successfully."
        else:
            # For SELECT queries, fetch the results
            result = cur.fetchall()

        # Close the cursor and connection
        cur.close()
        conn.close()

        return result

    except sqlite3.Error as e:
        return f"SQLite error: {e}"
    except Exception as e:
        return f"General error: {e}"


# List all the tables in the database
def list_tables():
    print("Listing tables...")
    # Connect to the database
    conn = sqlite3.connect("database/jokes.db")
    # Assuming you have already connected to the database
    cur = conn.cursor()
    query = """
    SELECT name 
    FROM sqlite_master 
    WHERE type='table'
    ORDER BY name;
    """
    cur.execute(query)
    rows = cur.fetchall()
    return "\n".join(row[0] for row in rows)


# describe the tables in the database with their columns
def describe_table(table_names: List[str]):
    print("Describing tables...")
    conn = sqlite3.connect("database/jokes.db")
    c = conn.cursor()

    descriptions = {}
    for table_name in table_names:
        query = f"PRAGMA table_info('{table_name}');"
        c.execute(query)
        rows = c.fetchall()

        column_infos = []
        for row in rows:
            column_info = (
                f"{row[1]} {row[2]}"  # row[1] is column name, row[2] is data type
            )
            column_infos.append(column_info)

        if column_infos:
            descriptions[table_name] = column_infos
        else:
            descriptions[table_name] = ["Table does not exist or has no columns."]

    conn.close()

    return "\n".join(
        f"{table}: {', '.join(columns)}" for table, columns in descriptions.items()
    )
