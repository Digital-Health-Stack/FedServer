from sqlalchemy.orm import Session, load_only
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import desc
from models.TrainingDataTransfer import TrainingDataTransfer
from datetime import datetime, timezone

def create_transfer(db: Session, data: dict):
    try:
        transfer = TrainingDataTransfer(**data)
        db.add(transfer)
        db.commit()
        db.refresh(transfer)
        return transfer
    except SQLAlchemyError as e:
        db.rollback()
        return {"error": f"Database error: {e}"}

def get_all_transfers(db: Session):
    """ Returns all QPD records in desc sorted order by transferredAt """
    try:
        return db.query(TrainingDataTransfer).options(
            load_only(
                TrainingDataTransfer.id,
                TrainingDataTransfer.training_name,
                TrainingDataTransfer.num_datapoints,
                TrainingDataTransfer.data_path,
                TrainingDataTransfer.parent_filename,
                TrainingDataTransfer.transferredAt,
                TrainingDataTransfer.approvedAt,
                TrainingDataTransfer.federated_session_id
            ).order_by(desc(TrainingDataTransfer.transferredAt))
        ).execution_options(include_approved=True).all()
    except SQLAlchemyError as e:
        return {"error": f"Database error: {e}"}
    
def get_pending_transfers(db: Session):
    """ Returns all QPD records that are not approved """
    try:
        return db.query(TrainingDataTransfer).options(
            load_only(
                TrainingDataTransfer.id,
                TrainingDataTransfer.training_name,
                TrainingDataTransfer.num_datapoints,
                TrainingDataTransfer.data_path,
                TrainingDataTransfer.parent_filename,
                TrainingDataTransfer.transferredAt,
                TrainingDataTransfer.approvedAt,
                TrainingDataTransfer.federated_session_id
            ).order_by(desc(TrainingDataTransfer.transferredAt))
        ).filter(TrainingDataTransfer.approvedAt.is_(None)).all()
    except SQLAlchemyError as e:
        return {"error": f"Database error: {e}"}

def get_transfer_details(db: Session, transfer_id: int):
    try:
        return db.query(TrainingDataTransfer).execution_options(include_approved=True).filter(TrainingDataTransfer.id == transfer_id).first()
    except SQLAlchemyError as e:
        return {"error": f"Database error: {e}"}

def approve_transfer(db: Session, transfer_id: int):
    try:
        transfer = db.query(TrainingDataTransfer).filter(TrainingDataTransfer.id == transfer_id).first()
        if not transfer:
            return {"error": "Transferred data not found"}
        transfer.approvedAt = datetime.now(timezone.utc)
        db.commit()
        return transfer
    except SQLAlchemyError as e:
        db.rollback()
        return {"error": f"Database error: {e}"}

def delete_transfer(db: Session, transfer_id: int):
    try:
        transfer = db.query(TrainingDataTransfer).filter(TrainingDataTransfer.id == transfer_id).first()
        if not transfer:
            return {"error": "Transferred data not found"}
        db.delete(transfer)
        db.commit()
        return {"message": "Transferred data deleted successfully"}
    except SQLAlchemyError as e:
        db.rollback()
        return {"error": f"Database error: {e}"}