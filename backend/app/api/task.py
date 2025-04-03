from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from schemas.dataset import TaskCreate
from utility.db import get_db
from schemas.dataset import (TaskCreate,TaskResponse,)

from helpers.task_crud import (
    create_task,
    delete_task,
    get_tasks_by_dataset_id,
    get_tasks_by_dataset_name,
)

from dotenv import load_dotenv
load_dotenv()

task_router = APIRouter(tags=["Task"])
########## Task Management Routes

@task_router.post("/create-task", response_model=TaskResponse)
def create_new_task(task: TaskCreate, db: Session = Depends(get_db)):
    result = create_task(db, task)
    if isinstance(result, dict) and "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result

@task_router.delete("/delete-task/{task_id}")
def delete_existing_task(task_id: int, db: Session = Depends(get_db)):
    result = delete_task(db, task_id)
    if isinstance(result, dict) and "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result

@task_router.get("/list-tasks-with-datasetid/{dataset_id}")
def read_tasks_by_dataset_id(dataset_id: int, db: Session = Depends(get_db)):
    tasks = get_tasks_by_dataset_id(db, dataset_id)
    if not tasks:
        raise HTTPException(status_code=404, detail="No tasks found for this dataset")
    return [task.as_dict() for task in tasks]

@task_router.get("/list-tasks-with-datasetname/{filename}")
def read_tasks_by_dataset_filename(filename: str, db: Session = Depends(get_db)):
    tasks = get_tasks_by_dataset_name(db, filename)
    if tasks is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if not tasks:
        raise HTTPException(status_code=404, detail="No tasks found for this dataset")
    return [task.as_dict() for task in tasks]

