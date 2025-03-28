from sqlalchemy.orm import Session, load_only
from sqlalchemy.exc import SQLAlchemyError, IntegrityError, NoResultFound
from typing import Optional
from fastapi import HTTPException, status
from schemas.dataset import DatasetCreate
from models.Dataset import Dataset, RawDataset, Task

def create_raw_dataset(db: Session, dataset: DatasetCreate):
    try:
        db_dataset = RawDataset(**dataset.dict())
        db.add(db_dataset)
        db.commit()
        db.refresh(db_dataset)
        return db_dataset
    except IntegrityError:
        db.rollback()
        return {"error": "Raw dataset with this name already exists."}
    except SQLAlchemyError as e:
        db.rollback()
        return {"error": f"Database error: {e}"}

def delete_raw_dataset(db: Session, filename: str):
    try:
        dataset = db.query(RawDataset).filter(RawDataset.filename == filename).first()
        if not dataset:
            return {"error": "Raw dataset not found."}
        db.delete(dataset)
        db.commit()
        return {"message": "Raw dataset deleted successfully."}
    except SQLAlchemyError as e:
        db.rollback()
        return {"error": f"Database error: {e}"}

def rename_raw_dataset(db: Session, old_file_name: str, new_file_name: str):
    try:
        dataset = db.query(RawDataset).filter(RawDataset.filename == old_file_name).first()
        if not dataset:
            return {"error": "Raw dataset not found."}
        dataset.filename = new_file_name
        db.commit()
        return {"message": "Raw dataset renamed successfully."}
    except SQLAlchemyError as e:
        db.rollback()
        return {"error": f"Database error: {e}"}

def list_raw_datasets(db: Session, skip: int, limit: int):
    try:
        return db.query(RawDataset).options(load_only(RawDataset.filename)).offset(skip).limit(limit).all()
    except SQLAlchemyError as e:
        return {"error": f"Database error: {e}"}

def get_raw_dataset_stats(db: Session, filename: str):
    try:
        dataset = db.query(RawDataset).filter(RawDataset.filename == filename).first()
        return dataset.as_dict() if dataset else {"details": "File not found"}
    except NoResultFound:
            return {"error": "File not found"}
    except SQLAlchemyError as e:
        return {"error": f"Database error: {e}"}
    
def create_dataset(db: Session, dataset: DatasetCreate):
    try:
        db_dataset = Dataset(**dataset.dict())
        db.add(db_dataset)
        db.commit()
        db.refresh(db_dataset)
        return db_dataset
    except IntegrityError:
        db.rollback()
        return {"error": "Dataset with this name already exists."}
    except SQLAlchemyError as e:
        db.rollback()
        return {"error": f"Database error: {e}"}

def delete_dataset(db: Session, dataset_id: int):
    try:
        dataset = db.query(Dataset).filter(Dataset.dataset_id == dataset_id).first()
        if not dataset:
            return {"error": "Dataset not found."}
        db.delete(dataset)
        db.commit()
        return {"message": "Dataset deleted successfully."}
    except SQLAlchemyError as e:
        db.rollback()
        return {"error": f"Database error: {e}"}

def rename_dataset(db: Session, dataset_id: int, new_name: str):
    try:
        dataset = db.query(Dataset).filter(Dataset.dataset_id == dataset_id).first()
        if not dataset:
            return {"error": "Dataset not found."}
        dataset.filename = new_name
        db.commit()
        return {"message": "Dataset renamed successfully."}
    except SQLAlchemyError as e:
        db.rollback()
        return {"error": f"Database error: {e}"}

def list_datasets(db: Session, skip: int, limit: int):
    try:
        return db.query(Dataset).options(load_only(Dataset.filename)).offset(skip).limit(limit).all()
    except SQLAlchemyError as e:
        return {"error": f"Database error: {e}"}
    
def get_dataset_stats(db: Session, filename: str):
    try:
        dataset = db.query(Dataset).filter(Dataset.filename == filename).first()
        return dataset.as_dict() if dataset else {"details": "File not found"}
    except NoResultFound:
            return {"error": "File not found"}
    except SQLAlchemyError as e:
        return {"error": f"Database error: {e}"}

def get_dataset_by_filename(db: Session, filename: str):
    return db.query(Dataset).filter(Dataset.filename == filename).first()

# Dataset CRUD operations
def get_tasks_by_dataset_id(db: Session, dataset_id: int):
    return db.query(Task).filter(Task.dataset_id == dataset_id).all()

def get_tasks_by_dataset_filename(db: Session, filename: str):
    dataset = db.query(Dataset).filter(Dataset.filename == filename).first()
    if not dataset:
        return None
    return db.query(Task).filter(Task.dataset_id == dataset.dataset_id).all()
