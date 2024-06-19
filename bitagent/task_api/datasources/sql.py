import sqlite3
from typing import List


class SQLDataset:
    def __init__(self,  table_name: str, columns: List[str], db_path: str = "", db_type: str = 'sqlite'):
        # Initialize dataset info
        self.db_path = db_path
        self.table_name = table_name
        self.columns = columns
        self.shuffled = False
        self.db_type = db_type
        self.current_index = 0
        
        if self.db_type == 'sqlite':
            # Connect to SQLite database
            self.connection = sqlite3.connect(db_path, check_same_thread=False)
            self.connection.row_factory = sqlite3.Row
        else:
            raise ValueError("Invalid database type or missing PostgreSQL configuration")
        cursor_args = {}
        self.cursor = self.connection.cursor(**cursor_args)
        self._create_table()
        
    def _create_table(self):
        columns = ', '.join([f"{column} TEXT" for column in self.columns])
        if self.db_type == 'sqlite':
            self.cursor.execute(f"CREATE TABLE IF NOT EXISTS {self.table_name} ({columns})")
        elif self.db_type == 'postgres':
            self.cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    {columns}
                )
            """)
        self.connection.commit()

    def add_item(self, item):
        # Ensure item keys match the dataset features
        if not all(key in self.columns for key in item):
            raise ValueError("Item keys must match dataset features")

        columns = ', '.join(item.keys())
        placeholders = ', '.join(['%s' if self.db_type == 'postgres' else '?' for _ in item])
        values = tuple(item.values())

        # Insert the new record into the database
        self.cursor.execute(f"INSERT INTO {self.table_name} ({columns}) VALUES ({placeholders})", values)
        self.connection.commit()

    def update_item(self, index, item):
        # Ensure item keys match the dataset features
        if not all(key in self.columns for key in item):
            raise ValueError("Item keys must match dataset features")

        columns = ', '.join([f"{key} = %s" if self.db_type == 'postgres' else f"{key} = ?" for key in item])
        values = tuple(item.values())

        # Update the record in the database
        self.cursor.execute(f"UPDATE {self.table_name} SET {columns} WHERE rowid = %s" if self.db_type == 'postgres' else f"UPDATE {self.table_name} SET {columns} WHERE rowid = ?", values + (index,))
        self.connection.commit()

    def find_item(self, item):
        query = ' AND '.join([f"{key} = %s" if self.db_type == 'postgres' else f"{key} = ?" for key in item.keys()])
        values = tuple(item.values())
        self.cursor.execute(f"SELECT rowid FROM {self.table_name} WHERE {query}", values)
        row = self.cursor.fetchone()
        return row['rowid'] if row else None

    def shuffle(self):
        # Enable shuffling for the next iteration
        self.shuffled = True

    def __iter__(self):
        if self.shuffled:
            query = f"SELECT * FROM {self.table_name} ORDER BY RANDOM()"
            self.shuffled = False
        else:
            query = f"SELECT * FROM {self.table_name}"
        self.cursor.execute(query)
        self.current_index = 0
        return self

    def __next__(self):
        if self.current_index >= len(self):
            raise StopIteration
        self.cursor.execute(f"SELECT * FROM {self.table_name} LIMIT 1 OFFSET {self.current_index}")
        row = self.cursor.fetchone()
        self.current_index += 1
        return dict(row)
    
    def __len__(self):
        self.cursor.execute(f"SELECT COUNT(*) FROM {self.table_name}")
        count = self.cursor.fetchone()
        return count[0]

    def __del__(self):
        self.cursor.close()
        self.connection.close()
