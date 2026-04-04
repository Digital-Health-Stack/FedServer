from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError, IntegrityError, NoResultFound
from schemas.dataset import DatasetCreate, DatasetUpdate
from models.Dataset import RawDataset, Dataset
from dotenv import load_dotenv

load_dotenv()


def create_raw_dataset(db: Session, dataset: DatasetCreate):
    try:
        # print("dataset details", dataset)
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


def delete_raw_dataset(db: Session, dataset_id: str):
    try:
        dataset = (
            db.query(RawDataset).filter(RawDataset.dataset_id == dataset_id).first()
        )
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
        dataset = (
            db.query(RawDataset).filter(RawDataset.filename == old_file_name).first()
        )
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
        total = db.query(RawDataset).count()
        datasets = (
            db.query(RawDataset)
            .offset(skip)
            .limit(limit)
            .all()
        )
        return {"datasets": datasets, "total": total}
    except SQLAlchemyError as e:
        return {"error": f"Database error: {e}"}


def get_raw_data_filename_by_id(db: Session, dataset_id: str):
    try:
        dataset = (
            db.query(RawDataset).filter(RawDataset.dataset_id == dataset_id).first()
        )
        return dataset.filename if dataset else {"details": "File not found"}
    except NoResultFound:
        return {"error": "File not found"}
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


def edit_raw_dataset_details(db: Session, newdetails: DatasetUpdate):
    try:
        dataset = (
            db.query(RawDataset)
            .filter(RawDataset.dataset_id == newdetails.dataset_id)
            .first()
        )
        if not dataset:
            return {"error": "Raw dataset not found."}
        dataset.filename = newdetails.filename
        dataset.description = newdetails.description
        dataset.datastats["filename"] = newdetails.filename
        db.commit()
        return {"message": "Raw dataset details updated successfully."}
    except SQLAlchemyError as e:
        db.rollback()
        return {"error": f"Database error: {e}"}


#########################################################################
# CRUD operations for Dataset


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


def rename_dataset(db: Session, old_file_name: str, new_file_name: str):
    try:
        dataset = db.query(Dataset).filter(Dataset.filename == old_file_name).first()
        if not dataset:
            return {"error": "Dataset not found."}
        dataset.filename = new_file_name
        db.commit()
        return {"message": "Dataset renamed successfully."}
    except SQLAlchemyError as e:
        db.rollback()
        return {"error": f"Database error: {e}"}


def list_datasets(db: Session, skip: int, limit: int):
    try:
        total = db.query(Dataset).count()
        datasets = (
            db.query(Dataset)
            .offset(skip)
            .limit(limit)
            .all()
        )
        return {"datasets": datasets, "total": total}
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


def get_data_filename_by_id(db: Session, dataset_id: int):
    try:
        dataset = db.query(Dataset).filter(Dataset.dataset_id == dataset_id).first()
        return dataset.filename if dataset else {"details": "File not found"}
    except NoResultFound:
        return {"error": "File not found"}
    except SQLAlchemyError as e:
        return {"error": f"Database error: {e}"}


def edit_dataset_details(db: Session, newdetails: DatasetUpdate):
    try:
        dataset = (
            db.query(Dataset)
            .filter(Dataset.dataset_id == newdetails.dataset_id)
            .first()
        )
        if not dataset:
            return {"error": "Dataset not found."}
        dataset.filename = newdetails.filename
        dataset.description = newdetails.description
        # it's repeated but it's required for some reason
        # don't do like datastats.filename #datastats is a dict and filename is key not attribute
        dataset.datastats["filename"] = newdetails.filename
        db.commit()
        return {"message": "Dataset details updated successfully."}
    except SQLAlchemyError as e:
        db.rollback()
        return {"error": f"Database error: {e}"}


def update_dataset_stats(db: Session, filename: str, datastats: dict):
    try:
        dataset = db.query(Dataset).filter(Dataset.filename == filename).first()
        if not dataset:
            return {"error": "Dataset not found."}
        dataset.datastats = datastats
        db.commit()
        return {"message": "Dataset stats updated successfully."}
    except SQLAlchemyError as e:
        db.rollback()
        return {"error": f"Database error: {e}"}


def handle_file_renaming_during_processing(
    db: Session, old_file_name: str, new_file_name: str, dataset_type: str
):
    if dataset_type == "raw":
        result = rename_raw_dataset(db, old_file_name, new_file_name)
        if isinstance(result, dict) and "error" in result:
            return {"error": result["error"]}

    elif dataset_type == "processed":
        result = rename_dataset(db, old_file_name, new_file_name)
        if isinstance(result, dict) and "error" in result:
            return {"error": result["error"]}
    else:
        print(f"Invalid dataset_type: {dataset_type}")
        return {"error": "Invalid dataset type"}
