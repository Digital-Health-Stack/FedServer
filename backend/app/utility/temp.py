from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import text
import os
from db import get_db  # Ensure correct import

# Load environment variables
load_dotenv()

# Get the database URL from the environment
DATABASE_URL = os.getenv("DB_URL")
print(f"Database URL: {DATABASE_URL}")

db = next(get_db())

# Get a list of table names
tables = db.execute(text("SELECT name FROM sqlite_master WHERE type='table';")).fetchall()

# Print table names
print("Tables in the database:")
for table in tables:
    print(table[0])

db.close()

