import sqlite3
import pandas as pd

# Load the CSV file
csv_file = "ai_projects.csv"  # Update with your actual CSV file path
df = pd.read_csv(csv_file)

# Connect to the SQLite database
db_path = "instance/projects.db"  # Update the path if needed
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Insert data into the 'project' table
for _, row in df.iterrows():
    cursor.execute(
        "INSERT INTO project (title, summary, domain) VALUES (?, ?, ?)",
        (row["Title"], row["Summary"], row["Domain"])
    )

# Commit changes and close the connection
conn.commit()
conn.close()

print("Data inserted successfully!")
