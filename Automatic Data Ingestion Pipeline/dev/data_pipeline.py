import sqlite3
import pandas as pd
import logging
from datetime import datetime
from typing import List
import unittest

# Formats the logger to only errors or above, the message will contain a timestamp and level
logging.basicConfig(
    filename='error_log.txt', 
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Keeps a record of any changes that will include a version and message
changelog = []

def log_change(version: str, message: str) -> None:
    """
    Log a change to the changelog.
    
    Args:
        version (str): Version number of the change.
        message (str): Description of the change.
    """
    changelog.append(f"Version {version}: {message}")
    
def write_changelog() -> None:
    """Write the changelog to a file."""
    with open('changelog.txt', 'w') as f:
        f.write('\n'.join(changelog))
        
def connect_to_db(db_name: str) -> sqlite3.Connection:
    """
    Connect to the SQLite database.

    Args:
        db_name (str): Name of the database file.

    Returns:
        sqlite3.Connection: Connection object to the database.
    """
    try:
        return sqlite3.connect(db_name)
    except sqlite3.Error as e:
        logging.error(f"Error connecting to database: {e}")
        raise
    
def get_table_names(cursor: sqlite3.Cursor) -> List[str]:
    """
    Get the names of all tables in the database.

    Args:
        cursor (sqlite3.Cursor): Cursor object for database operations.

    Returns:
        List[str]: List of table names.
    """
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    return [table[0] for table in cursor.fetchall()]

def load_and_clean_data(connection: sqlite3.Connection) -> pd.DataFrame:
    """
    Performs a comprehensive transformation pipeline to load in each SQLite
    table, clean each feature, and merge the individual tables into one
    dataframe where it can be used to create or update a new table/csv.

    Args:
        connection (sqlite3.Connection): Connection object to the database.

    Returns:
        pd.DataFrame: Cleaned and merged dataframe.
    """
    # Load courses
    courses = pd.read_sql_query("SELECT * FROM cademycode_courses", connection)
    courses['career_path_id'] = courses['career_path_id'].astype('Int16')
    courses['career_path_name'] = courses['career_path_name'].astype('string')

    # Load jobs
    jobs = pd.read_sql_query("SELECT * FROM cademycode_student_jobs", connection)
    jobs = jobs.drop_duplicates()
    jobs['job_id'] = jobs['job_id'].astype('Int8')

    # Load students
    students = pd.read_sql_query("SELECT * FROM cademycode_students", connection)
    students['name'] = students['name'].astype('string')
    students['dob'] = pd.to_datetime(students['dob'])
    students['sex'] = students['sex'].astype('category')
    students['job_id'] = pd.to_numeric(students['job_id'], errors='coerce').astype('Int8')
    students['num_course_taken'] = pd.to_numeric(students['num_course_taken'], errors='coerce').astype('Int64')
    students['current_career_path_id'] = pd.to_numeric(students['current_career_path_id'], errors='coerce').astype('Int16')
    students['time_spent_hrs'] = pd.to_numeric(students['time_spent_hrs'], errors='coerce').astype('Float64')

    # Merge dataframes
    df = pd.merge(students, courses, left_on='current_career_path_id', right_on='career_path_id', how='left')
    df = pd.merge(df, jobs, on='job_id', how='left')

    # Drop redundant columns and reorder
    df = df.drop(['current_career_path_id', 'career_path_id', 'job_id'], axis=1)
    df = df[[
        'uuid', 'name', 'dob', 'sex', 'contact_info', 'career_path_name',
        'num_course_taken', 'hours_to_complete', 'time_spent_hrs', 'job_category', 'avg_salary'
    ]]

    # Rename columns
    df.rename(columns={
        'career_path_name': 'course_name',
        'job_category': 'job',
        'num_course_taken': 'num_courses_taken'
    }, inplace=True)

    # Handle missing values
    df['course_name'].fillna('None', inplace=True)
    df[['num_courses_taken', 'hours_to_complete', 'time_spent_hrs']] = df[['num_courses_taken', 'hours_to_complete', 'time_spent_hrs']].fillna(0)
    df['job'].fillna('engineer', inplace=True)
    df['avg_salary'].fillna(101000, inplace=True)

    return df

def export_data(df: pd.DataFrame, connection: sqlite3.Connection) -> None:
    """
    Export the dataframe to CSV and SQLite. This will update the table and csv
    if they already exist.

    Args:
        df (pd.DataFrame): Dataframe to export.
        connection (sqlite3.Connection): Connection object to the database.
    """
    df.to_csv('cademy_stats.csv', index=False)
    df.to_sql('cademy_stats', connection, if_exists='replace', index=False)

class DatabaseTests(unittest.TestCase):
    """Unit tests for the database operations."""

    def setUp(self):
        self.connection = connect_to_db('cademycode.db')
        self.cursor = self.connection.cursor()

    def tearDown(self):
        self.connection.close()

    def test_schema(self):
        """Test if the updated database has the same schema as the original."""
        original_tables = set(get_table_names(self.cursor))
        self.assertIn('cademy_stats', original_tables)

    def test_join(self):
        """Test if the tables will join properly."""
        try:
            df = load_and_clean_data(self.connection)
            self.assertIsNotNone(df)
        except Exception as e:
            self.fail(f"Failed to join tables: {e}")

    def test_new_data(self):
        """Test if there is any new data."""
        before_count = pd.read_sql_query("SELECT COUNT(*) FROM cademy_stats", self.connection).iloc[0, 0]
        df = load_and_clean_data(self.connection)
        export_data(df, self.connection)
        after_count = pd.read_sql_query("SELECT COUNT(*) FROM cademy_stats", self.connection).iloc[0, 0]
        self.assertEqual(before_count, after_count)
        
def main() -> None:
    """Main function to run data pipeline."""
    try:
        connection = connect_to_db('cademycode.db') # connect to db
        df = load_and_clean_data(connection) # load in and transform data
        export_data(df, connection) # save and update data to db and csv
        
        # Run unit tests
        suite = unittest.TestLoader().loadTestsFromTestCase(DatabaseTests) # extract tests from class
        test_result = unittest.TextTestRunner(verbosity=2).run(suite) # run the tests with an output
        
        # Log the test results
        if test_result.wasSuccessful():
            log_change("1.0", f"Data updated successfully. Rows: {len(df)}")
        else:
            log_change("1.0", "Data update failed. Check error log for details.")
        
        write_changelog()
    
    except Exception as e:
        logging.error(f"An error occured: {e}")
    
    # Close the connection, even if an exception has occured
    finally:
        # If a connection was created earlier in the try block, then close it
        if 'connection' in locals():
            connection.close()
            
if __name__ == "__main__":
    main()