from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from utility.db import get_db
from schemas.dataset import BenchmarkCreate

from schemas.dataset import (
    BenchmarkCreate,
    BenchmarkResponse,
)

from crud.benchmark_crud import (
    create_benchmark,
    delete_benchmark,
    get_benchmarks_by_task_id,
    get_benchmarks_by_dataset_name_and_task_name,
    get_benchmarks_by_dataset_id_and_task_id,
    get_training_by_benchmark_id
)

from dotenv import load_dotenv
load_dotenv()

benchmark_router = APIRouter(tags=["Benchmark"])

########### Benchmark Management Routes

@benchmark_router.post("/add-benchmarks", response_model=BenchmarkResponse)
def create_new_benchmark(benchmark: BenchmarkCreate, db: Session = Depends(get_db)):
    try:
        result = create_benchmark(db, benchmark)
        if isinstance(result, dict) and "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except Exception as e:
        print(f"Error creating benchmark: {e}")
        return {"error": str(e)}

@benchmark_router.delete("/delete-benchmarks/{benchmark_id}")
def delete_existing_benchmark(benchmark_id: int, db: Session = Depends(get_db)):
    try:
        result = delete_benchmark(db, benchmark_id)
        if isinstance(result, dict) and "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except Exception as e:
        print(f"Error deleting benchmark: {e}")
        return {"error": str(e)}

@benchmark_router.get("/get-benchmarks-with-taskid/{task_id}")
def read_benchmarks_by_task_id(task_id: int, db: Session = Depends(get_db)):
    try:
        benchmarks = get_benchmarks_by_task_id(db, task_id)
        if not benchmarks:
            raise HTTPException(status_code=404, detail="No benchmarks found for this task")
        return [benchmark.as_dict() for benchmark in benchmarks]
    except Exception as e:
        print(f"Error retrieving benchmarks: {e}")
        return {"error": str(e)}

@benchmark_router.get("/get-benchmarks-with-dataset-and-tasknames/{dataset_name}/{task_name}")
def read_benchmarks_by_dataset_and_task_names(
    dataset_name: str,
    task_name: str,
    db: Session = Depends(get_db)
):
    try:
        benchmarks = get_benchmarks_by_dataset_name_and_task_name(db, dataset_name, task_name)
        if benchmarks is None:
            raise HTTPException(status_code=404, detail="Dataset or task not found")
        if not benchmarks:
            raise HTTPException(status_code=404, detail="No benchmarks found")
        return [benchmark.as_dict() for benchmark in benchmarks]
    except Exception as e:
        print(f"Error retrieving benchmarks: {e}")
        return {"error": str(e)}

@benchmark_router.get("/get-benchmarks-with-dataset-and-taskid/{dataset_id}/{task_id}")
def read_benchmarks_by_dataset_and_task_ids(
    dataset_id: int,
    task_id: int,
    db: Session = Depends(get_db)
):
    try:
        benchmarks = get_benchmarks_by_dataset_id_and_task_id(db, dataset_id, task_id)
        if benchmarks is None:
            raise HTTPException(status_code=404, detail="Task not found in this dataset")
        if not benchmarks:
            raise HTTPException(status_code=404, detail="No benchmarks found")
        return [benchmark.as_dict() for benchmark in benchmarks]
    except Exception as e:
        print(f"Error retrieving benchmarks: {e}")
        return {"error": str(e)}

@benchmark_router.get("/get-training-with-benchmarkid/{benchmark_id}")
def read_training_by_benchmark_id(benchmark_id: int, db: Session = Depends(get_db)):
    try:
        training = get_training_by_benchmark_id(db, benchmark_id)
        if not training:
            raise HTTPException(status_code=404, detail="Benchmark or training not found")
        return training.as_dict()
    except Exception as e:
        print(f"Error retrieving training: {e}")
        return {"error": str(e)}
    