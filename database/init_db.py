import sqlite3
import os


def initialize_database(db_path):
    print("Initializing the database...")
    # Check if the database file already exists
    db_exists = os.path.exists(db_path)

    # Connect to the database
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    if not db_exists:
        print("Creating the jokes table...")
        # Create a table
        c.execute(
            """
        CREATE TABLE jokes (
            topic TEXT NOT NULL,
            joke TEXT NOT NULL UNIQUE,
            rating INTEGER NOT NULL
        )
        """
        )

        # Insert some data
        jokes = [
            ("Hello World", "Hello World joke", 1),
            (
                "Programming",
                "Why do programmers prefer dark mode? Because the light attracts bugs!",
                4,
            ),
            (
                "SQL",
                "Why did the SQL query get so many dates? It knew how to join tables!",
                5,
            ),
            (
                "Database",
                "I would tell you a joke about a broken database, but there's no schema to follow.",
                3,
            ),
            (
                "Python",
                "Why did the Python programmer get rejected by the Java developer? Because they didn’t have enough class.",
                4,
            ),
            (
                "JavaScript",
                "Why was the JavaScript developer sad? Because they didn’t know how to null their feelings.",
                3,
            ),
            (
                "Algorithms",
                "Why do algorithms always know the best jokes? They always have the best punch(line).",
                4,
            ),
            (
                "Hello World",
                "The first program I wrote ran smoothly. It said Hello World and I said Goodbye Social Life.",
                2,
            ),
            (
                "Networking",
                "Why don’t network engineers get along with others? They can’t find common ground.",
                3,
            ),
            (
                "Binary",
                "There are 10 kinds of people in the world: those who understand binary and those who don’t.",
                5,
            ),
            (
                "Cloud Computing",
                "Why did the cloud break up with the server? It found someone more responsive.",
                4,
            ),
        ]

        # Insert data into the table
        c.executemany("INSERT INTO jokes (topic, joke, rating) VALUES (?, ?, ?)", jokes)

        # Commit the transaction
        conn.commit()

    # Fetch and print all rows from the jokes table
    c.execute("SELECT * FROM jokes")
    rows = c.fetchall()

    print("Jokes in the database:")
    for row in rows:
        print(f"Topic: {row[0]}, Joke: {row[1]}, Rating: {row[2]}")

    # Close the connection
    conn.close()


if __name__ == "__main__":
    # Define the path for the database
    db_path = "jokes.db"

    # Initialize the database
    initialize_database(db_path)

    print(f"Database '{db_path}' initialized successfully!")
