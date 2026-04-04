from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError, IntegrityError, NoResultFound
from schemas.dataset import DatasetCreate, DatasetUpdate
from models.Dataset import Dataset
from services.local_storage import LocalStorageManager


def _dataset_payload(dataset: DatasetCreate) -> dict:
    if hasattr(dataset, "model_dump"):
        return dataset.model_dump()
    return dataset.dict()


def create_dataset(db: Session, dataset: DatasetCreate):
    try:
        db_dataset = Dataset(**_dataset_payload(dataset))
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
        db.commit()
        return {"message": "Dataset details updated successfully."}
    except SQLAlchemyError as e:
        db.rollback()
        return {"error": f"Database error: {e}"}


def update_dataset_metadata_from_overview(filename: str, overview: dict):
    """Persist merged stats into metadata.json (not the SQL DB)."""
    storage = LocalStorageManager()
    meta = storage.read_metadata(filename) or {}
    meta["filename"] = filename
    if "datastats" in overview and isinstance(overview["datastats"], dict):
        meta["datastats"] = overview["datastats"]
    elif "numRows" in overview:
        meta["datastats"] = {
            "numRows": overview["numRows"],
            "numColumns": overview["numColumns"],
            "columnStats": overview.get("columnStats", []),
        }
    storage.write_metadata(filename, meta)
    return {"message": "Dataset metadata updated successfully."}
