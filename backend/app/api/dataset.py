from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from utility.db import get_db
from schemas.Dataset_Schema import DatasetCreate, TaskCreate, BenchmarkCreate, DatasetResponse, TaskResponse, BenchmarkResponse
from helpers.datasets_crud import create_dataset, delete_dataset, create_task, delete_task, create_benchmark, delete_benchmark, get_datasets, get_tasks_for_dataset, get_benchmark_for_task

dataset_router = APIRouter()

# Add Dataset
@dataset_router.post("/datasets/", response_model=DatasetResponse)
def add_dataset(dataset: DatasetCreate, db: Session = Depends(get_db)):
    return create_dataset(db, dataset)

# Delete Dataset (Cascade Delete Tasks and Benchmarks)
@dataset_router.delete("/datasets/{dataset_id}")
def remove_dataset(dataset_id: int, db: Session = Depends(get_db)):
    delete_dataset(db, dataset_id)
    return {"message": "Dataset deleted successfully"}

# Add Task
@dataset_router.post("/tasks/", response_model=TaskResponse)
def add_task(task: TaskCreate, db: Session = Depends(get_db)):
    return create_task(db, task)

# Delete Task (Cascade Delete Benchmarks)
@dataset_router.delete("/tasks/{task_id}")
def remove_task(task_id: int, db: Session = Depends(get_db)):
    delete_task(db, task_id)
    return {"message": "Task deleted successfully"}

# Add Benchmark
@dataset_router.post("/benchmarks/", response_model=BenchmarkResponse)
def add_benchmark(benchmark: BenchmarkCreate, db: Session = Depends(get_db)):
    return create_benchmark(db, benchmark)

# Delete Benchmark
@dataset_router.delete("/benchmarks/{benchmark_id}")
def remove_benchmark(benchmark_id: int, db: Session = Depends(get_db)):
    delete_benchmark(db, benchmark_id)
    return {"message": "Benchmark deleted successfully"}

# Get Datasets (Paginated)
@dataset_router.get("/datasets/", response_model=list[DatasetResponse])
def list_datasets(skip: int = 0, limit: int = Query(20, le=100), db: Session = Depends(get_db)):
    return get_datasets(db, skip, limit)

# Get Tasks for a Dataset
@dataset_router.get("/datasets/{dataset_id}/tasks", response_model=list[TaskResponse])
def list_tasks(dataset_id: int, db: Session = Depends(get_db)):
    return get_tasks_for_dataset(db, dataset_id)

# Get Benchmarks for a Dataset and Task (Paginated)
@dataset_router.get("/tasks/{task_id}/benchmarks", response_model=list[BenchmarkResponse])
def list_benchmarks(task_id: int, skip: int = 0, limit: int = Query(20, le=100), db: Session = Depends(get_db)):
    return get_benchmark_for_task(db, task_id, skip, limit)
