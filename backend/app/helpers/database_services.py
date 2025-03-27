# import os
# from sqlalchemy import create_engine, Column, String, JSON
# from sqlalchemy.orm import sessionmaker
# from sqlalchemy.ext.declarative import declarative_base
# from sqlalchemy.exc import NoResultFound
# from dotenv import load_dotenv

# load_dotenv()

# # Get the database path from the environment
# DB_PATH = os.getenv("DB_PATH")

# # SQLite Database URL
# DATABASE_URL = f"sqlite:///{DB_PATH}"

# HDFS_RAW_DATASETS_DIR = os.getenv("HDFS_RAW_DATASETS_DIR")
# HDFS_PROCESSED_DATASETS_DIR = os.getenv("HDFS_PROCESSED_DATASETS_DIR")

# # SQLAlchemy Setup
# engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})  # SQLite-specific option
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
# Base = declarative_base()


# class RawDataset(Base):
#     __tablename__ = "raw_datasets"
#     file_name = Column(String, primary_key=True, index=True)
#     details = Column(JSON)

# class ProcessedDataset(Base):
#     __tablename__ = "processed_datasets"
#     file_name = Column(String,primary_key=True, index=True)
#     details = Column(JSON)

# # Create the database and tables (if they don't already exist)
# try:
#     Base.metadata.create_all(bind=engine)
# except Exception as e:
#     print(f"Error initializing database: {str(e)}")


# """
# NOTE: Database session doesn't hold much resource, so i'm creating a global session and return that when __init__ is called,
# otherwise i'll need to create a new session for each operation (even for deleting each file) so i'm using a global session

# """
# class DatabaseManager:
#     def __init__(self):
#         """Initialize the Database Manager."""
#         # Ensure database file exists
#         if not os.path.exists(DB_PATH):
#             print(f"Creating new database file at: {DB_PATH}")
#         self.Session = SessionLocal

#     async def add_dataset(self, directory: str, file_name: str, details: dict):
#         """Add a new dataset to the database."""
#         session = self.Session()
#         try:
#             # Check if the file already exists
#             tableclass = RawDataset if directory == HDFS_RAW_DATASETS_DIR else ProcessedDataset
#             existing = session.query(tableclass).filter(tableclass.file_name == file_name).first()
#             if existing:
#                 # update the details
#                 existing.details = details
#                 session.commit()
#                 print(f"Dataset stats of {file_name} updated successfully in DB!")
#                 return {"message": "Dataset details updated successfully"}

#             # Add new dataset
#             dataset = tableclass(file_name=file_name, details=details)
#             session.add(dataset)
#             session.commit()
#             session.refresh(dataset)
#             print(f"Dataset stats of {file_name} saved successfully in DB!")
#             return {"message": "Dataset details saved successfully"}
#         except Exception as e:
#             session.rollback()
#             return {"error": str(e)}
#         finally:
#             session.close()

#     async def get_dataset(self,directory: str, file_name: str):
#         """Retrieve a dataset by file name."""
#         session = self.Session()
#         try:
#             tableclass = RawDataset if directory == HDFS_RAW_DATASETS_DIR else ProcessedDataset
#             dataset = session.query(tableclass).filter(tableclass.file_name == file_name).one()
#             return dataset.details
#         except NoResultFound:
#             return {"error": "File not found"}
#         except Exception as e:
#             return {"error": str(e)}
#         finally:
#             session.close()

#     async def delete_dataset(self,directory:str, file_name: str):
#         """Delete a dataset by file name."""
#         session = self.Session()
#         try:
#             tableclass = RawDataset if directory == HDFS_RAW_DATASETS_DIR else ProcessedDataset
#             dataset = session.query(tableclass).filter(tableclass.file_name == file_name).first()
#             if not dataset:
#                 return {"error": "File not found"}
#             session.delete(dataset)
#             session.commit()
#             return {"message": f"Dataset {file_name} deleted successfully"}
#         except Exception as e:
#             session.rollback()
#             return {"error": str(e)}
#         finally:
#             session.close()

#     async def rename_dataset(self, directory: str, old_file_name: str, new_file_name: str):
#         """Rename a dataset by file name."""
#         session = self.Session()
#         try:
#             tableclass = RawDataset if directory == HDFS_RAW_DATASETS_DIR else ProcessedDataset
#             dataset = session.query(tableclass).filter(tableclass.file_name == old_file_name).first()
#             if not dataset:
#                 print(f"Dataset {directory/old_file_name} not found in DB!")
#                 return {"error": "File not found"}
#             dataset.file_name = new_file_name
#             session.commit()
#             return {"message": f"Dataset {old_file_name} renamed to {new_file_name} successfully"}
#         except Exception as e:
#             session.rollback()
#             return {"error": str(e)}
#         finally:
#             session.close()

#     async def list_all_datasets(self):
#         """Retrieve all datasets in the database."""
#         session = self.Session()
#         try:
#             raw_datasets = session.query(RawDataset).all()
#             uploads =  [dataset.file_name for dataset in raw_datasets]
#             processed_datasets = session.query(ProcessedDataset).all()
#             processed = [dataset.file_name for dataset in processed_datasets]
#             return {"contents": {HDFS_RAW_DATASETS_DIR: uploads, HDFS_PROCESSED_DATASETS_DIR: processed}}

#         except Exception as e:
#             return {"error": str(e)}
#         finally:
#             session.close() 