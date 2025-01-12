from fastapi import APIRouter, Depends, HTTPException
from models.Dataset import Dataset
from models.Benchmark import Benchmark
from fastapi import APIRouter
from sqlalchemy.orm import Session
from db import get_db
from schemas.Dataset_Schema import DatasetCreate


dataset_router = APIRouter()


# Dataset Endpoints
@dataset_router.post("/datasets/add_dataset/")
def create_dataset(dataset: DatasetCreate, db: Session = Depends(get_db)):
    # Check if the dataset name already exists
    existing_dataset = db.query(Dataset).filter(Dataset.code == dataset.code).first()
    
    if existing_dataset:
        raise HTTPException(status_code=400, detail=f'Dataset with this code {dataset.code} already exists')
    
    # Create a new Dataset instance
    new_dataset = Dataset(
        name=dataset.name,
        code=dataset.code,
        description=dataset.description,
        source=dataset.source,
    )
    db.add(new_dataset)
    db.commit()
    db.refresh(new_dataset)
    return {
            'message':'Dataset has been added!'}

@dataset_router.get("/datasets/{code}")
def get_dataset(code: str, db: Session=Depends(get_db)):
    dataset = db.query(Dataset).filter(Dataset.code == code).first()
    if not dataset:
        raise HTTPException(status_code=404, detail=f"Dataset with code {code} does not exists")
    return dataset

@dataset_router.get("/datasets/")
def get_all_dataset(db: Session=Depends(get_db)):
    datasets = db.query(Dataset).all()
    return datasets






