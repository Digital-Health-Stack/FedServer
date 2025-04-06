from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime
from typing import List
from crud.training_data_transfer_crud import (
    create_transfer,
    get_all_transfers,
    get_pending_transfers,
    get_transfer_details,
    approve_transfer,
    delete_transfer
)

from schemas.training_data_transfer import (
    TransferCreate,
    TransferListItem,
    TransferDetails
)

from utility.db import get_db

qpd_router = APIRouter(tags=["QPD"])

@qpd_router.post("/create-transferred-data/", response_model=TransferListItem)
def create_data_transfer(transfer_data: TransferCreate, db: Session = Depends(get_db)):
    try:
        result = create_transfer(db, transfer_data.dict())
        if isinstance(result, dict) and "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except Exception as e:
        print(f"Error creating transfer: {e}")
        return {"error": str(e)}

@qpd_router.get("/list-transferred-data/", response_model=List[TransferListItem])
def list_all_transfers(db: Session = Depends(get_db)):
    try:
        transfers = get_pending_transfers(db)
        if isinstance(transfers, dict):
            raise HTTPException(status_code=500, detail=transfers["error"])
        return transfers
    except Exception as e:
        print(f"Error listing pending transfers: {e}")
        return {"error": str(e)}

@qpd_router.get("/transferred-data-overview/{transfer_id}", response_model=TransferDetails)
def get_transfer_overview(transfer_id: int, db: Session = Depends(get_db)):
    try:
        details = get_transfer_details(db, transfer_id)
        if not details or isinstance(details, dict):
            raise HTTPException(status_code=404, detail="Transfer not found")
        return details
    except Exception as e:
        print(f"Error fetching transfer details: {e}")
        return {"error": str(e)}

@qpd_router.delete("/delete-transferred-data/{transfer_id}")
def delete_transfer_record(transfer_id: int, db: Session = Depends(get_db)):
    try:
        result = delete_transfer(db, transfer_id)
        if isinstance(result, dict) and "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except Exception as e:
        print(f"Error deleting transfer: {e}")
        return {"error": str(e)}
    
@qpd_router.post("/approve-transferred-data/{transfer_id}")
def approve_transfer_record(transfer_id: int, db: Session = Depends(get_db)):
    try:
        # merge all this in an function to pass to the executor
        # logic to read and merge data file with parent file in processed directory of hdfs
        result = approve_transfer(db, transfer_id)
        if isinstance(result, dict) and "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return {"message": "Approved successfully"}
    except Exception as e:
        print(f"Error approving transfer: {e}")
        return {"error": str(e)}
    
