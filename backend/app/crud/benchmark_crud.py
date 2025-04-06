from sqlalchemy.orm import Session
from models.Benchmark import Benchmark
from models.Dataset import Task
from schemas.dataset import BenchmarkCreate
from models.FederatedSession import FederatedSession
from models.Dataset import Dataset
from fastapi import HTTPException, status
from sqlalchemy.exc import SQLAlchemyError, IntegrityError

def delete_benchmark(db: Session, benchmark_id: int):
    try:
        benchmark = db.query(Benchmark).filter(Benchmark.benchmark_id == benchmark_id).first()
        if not benchmark:
            return {"error": "Benchmark not found."}
        db.delete(benchmark)
        db.commit()
        return {"message": "Benchmark deleted successfully."}
    except SQLAlchemyError as e:
        db.rollback()
        return {"error": f"Database error: {e}"}
    
def create_benchmark(db: Session, benchmark: BenchmarkCreate):
    try:
        db_benchmark = Benchmark(**benchmark.dict())
        db.add(db_benchmark)
        db.commit()
        db.refresh(db_benchmark)
        return db_benchmark
    except IntegrityError:
        db.rollback()
        return {"error": "Benchmark creation failed due to integrity constraints."}
    except SQLAlchemyError as e:
        db.rollback()
        return {"error": f"Database error: {e}"}

def get_benchmarks_by_task_id(db: Session, task_id: int):
    return db.query(Benchmark).filter(Benchmark.task_id == task_id).all()

def get_benchmarks_by_dataset_name_and_task_name(db: Session, dataset_name: str, task_name: str):
    dataset = db.query(Dataset).filter(Dataset.filename == dataset_name).first()
    if not dataset:
        return None
    
    task = db.query(Task).filter(
        Task.dataset_id == dataset.dataset_id,
        Task.task_name == task_name
    ).first()
    
    if not task:
        return None
    
    return db.query(Benchmark).filter(Benchmark.task_id == task.task_id).all()

def get_benchmarks_by_dataset_id_and_task_id(db: Session, dataset_id: int, task_id: int):
    # Verify the task belongs to the dataset
    task = db.query(Task).filter(
        Task.task_id == task_id,
        Task.dataset_id == dataset_id
    ).first()
    
    if not task:
        return None
    
    return db.query(Benchmark).filter(Benchmark.task_id == task_id).all()

def get_training_by_benchmark_id(db: Session, benchmark_id: int):
    benchmark = db.query(Benchmark).filter(Benchmark.benchmark_id == benchmark_id).first()
    if not benchmark:
        return None
    
    return db.query(FederatedSession).filter(FederatedSession.id == benchmark.training_id).first()