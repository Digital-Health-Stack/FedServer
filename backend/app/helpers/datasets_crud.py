from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
from schemas.Dataset_Schema import DatasetCreate, TaskCreate, BenchmarkCreate
from models.Dataset import Dataset, Task
from models.Benchmark import Benchmark

def create_dataset(db: Session, dataset: DatasetCreate):
    try:
        db_dataset = Dataset(**dataset.dict())
        db.add(db_dataset)
        db.commit()
        db.refresh(db_dataset)
        return db_dataset
    except IntegrityError:
        db.rollback()
        print("Error: Dataset with this name already exists.")
        return {"error": "Dataset with this name already exists."}
    except SQLAlchemyError as e:
        db.rollback()
        print(f"Database error: {e}")
        return {"error": "Database error occurred."}

def delete_dataset(db: Session, dataset_id: int):
    try:
        dataset = db.query(Dataset).filter(Dataset.dataset_id == dataset_id).first()
        if not dataset:
            print("Error: Dataset not found.")
            return {"error": "Dataset not found."}
        db.delete(dataset)
        db.commit()
        return {"message": "Dataset deleted successfully."}
    except SQLAlchemyError as e:
        db.rollback()
        print(f"Database error: {e}")
        return {"error": "Database error occurred."}

def create_task(db: Session, task: TaskCreate):
    try:
        db_task = Task(**task.dict())
        db.add(db_task)
        db.commit()
        db.refresh(db_task)
        return db_task
    except IntegrityError:
        db.rollback()
        print("Error: Task creation failed due to integrity constraints.")
        return {"error": "Task creation failed due to integrity constraints."}
    except SQLAlchemyError as e:
        db.rollback()
        print(f"Database error: {e}")
        return {"error": "Database error occurred."}

def delete_task(db: Session, task_id: int):
    try:
        task = db.query(Task).filter(Task.task_id == task_id).first()
        if not task:
            print("Error: Task not found.")
            return {"error": "Task not found."}
        db.delete(task)
        db.commit()
        return {"message": "Task deleted successfully."}
    except SQLAlchemyError as e:
        db.rollback()
        print(f"Database error: {e}")
        return {"error": "Database error occurred."}

def create_benchmark(db: Session, benchmark: BenchmarkCreate):
    try:
        db_benchmark = Benchmark(**benchmark.dict())
        db.add(db_benchmark)
        db.commit()
        db.refresh(db_benchmark)
        return db_benchmark
    except IntegrityError:
        db.rollback()
        print("Error: Benchmark creation failed due to integrity constraints.")
        return {"error": "Benchmark creation failed due to integrity constraints."}
    except SQLAlchemyError as e:
        db.rollback()
        print(f"Database error: {e}")
        return {"error": "Database error occurred."}

def delete_benchmark(db: Session, benchmark_id: int):
    try:
        benchmark = db.query(Benchmark).filter(Benchmark.benchmark_id == benchmark_id).first()
        if not benchmark:
            print("Error: Benchmark not found.")
            return {"error": "Benchmark not found."}
        db.delete(benchmark)
        db.commit()
        return {"message": "Benchmark deleted successfully."}
    except SQLAlchemyError as e:
        db.rollback()
        print(f"Database error: {e}")
        return {"error": "Database error occurred."}

def get_datasets(db: Session, skip: int, limit: int):
    try:
        return db.query(Dataset).offset(skip).limit(limit).all()
    except SQLAlchemyError as e:
        print(f"Database error: {e}")
        return {"error": "Database error occurred."}

def get_tasks_for_dataset(db: Session, dataset_id: int):
    try:
        return db.query(Task).filter(Task.dataset_id == dataset_id).all()
    except SQLAlchemyError as e:
        print(f"Database error: {e}")
        return {"error": "Database error occurred."}

def get_benchmark_for_task(db: Session, task_id: int, skip: int, limit: int):
    try:
        return db.query(Benchmark).filter(Benchmark.task_id == task_id).offset(skip).limit(limit).all()
    except SQLAlchemyError as e:
        print(f"Database error: {e}")
        return {"error": "Database error occurred."}