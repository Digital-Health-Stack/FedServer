from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from schemas.dataset import TaskCreate
from utility.db import get_db
from schemas.dataset import (TaskCreate,TaskResponse,LeaderboardResponse)

from crud.task_crud import (
    create_task,
    delete_task,
    get_tasks_by_dataset_id,
    get_tasks_by_dataset_name,
    get_task_by_id,
    update_task_by_id,
    get_leaderboard_by_task_id
)

from dotenv import load_dotenv
load_dotenv()

task_router = APIRouter(tags=["Task"])
########## Task Management Routes

@task_router.post("/create-task", response_model=TaskResponse)
def create_new_task(task: TaskCreate, db: Session = Depends(get_db)):
    try:
        result = create_task(db, task)
        if isinstance(result, dict) and "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except Exception as e:
        print(f"Error creating task: {e}")
        return {"error": str(e)}

@task_router.delete("/delete-task/{task_id}")
def delete_existing_task(task_id: int, db: Session = Depends(get_db)):
    try:
        result = delete_task(db, task_id)
        if isinstance(result, dict) and "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except Exception as e:
        print(f"Error deleting task: {e}")
        return {"error": str(e)}

@task_router.get("/list-tasks-with-datasetid/{dataset_id}")
def read_tasks_by_dataset_id(dataset_id: int, db: Session = Depends(get_db)):
    try:
        tasks = get_tasks_by_dataset_id(db, dataset_id)
        if not tasks:
            raise HTTPException(status_code=404, detail="No tasks found for this dataset")
        return [task.as_dict() for task in tasks]
    except Exception as e:
        print(f"Error retrieving tasks: {e}")
        return {"error": str(e)}

@task_router.get("/list-tasks-with-datasetname/{filename}")
def read_tasks_by_dataset_filename(filename: str, db: Session = Depends(get_db)):
    try:
        tasks = get_tasks_by_dataset_name(db, filename)
        if tasks is None:
            raise HTTPException(status_code=404, detail="Dataset not found")
        if not tasks:
            raise HTTPException(status_code=404, detail="No tasks found for this dataset")
        return [task.as_dict() for task in tasks]
    except Exception as e:
        print(f"Error retrieving tasks: {e}")
        return {"error": str(e)}

@task_router.get("/get-task/{task_id}", response_model=TaskResponse)
def read_task_by_id(task_id: int, db: Session = Depends(get_db)):
    try:
        task = get_task_by_id(db, task_id)
        return task
    except Exception as e:
        print(f"Error retrieving task: {e}")
        return {"error": str(e)}
    
@task_router.put("/update-task/{task_id}", response_model=TaskResponse)
def update_existing_task(task_id: int, task: TaskCreate, db: Session = Depends(get_db)):
    try:
        updated_task = update_task_by_id(db, task_id, task)
        return updated_task
    except Exception as e:
        print(f"Error updating task: {e}")
        return {"error": str(e)}
    
@task_router.get("/leaderboard/{task_id}", response_model=LeaderboardResponse)
def get_task_leaderboard(
    task_id: int,
    db: Session = Depends(get_db)
):
    """
    Get leaderboard for a specific task including benchmark comparison
    
    Returns:
    - task_name: Name of the task
    - metric: Primary metric for evaluation
    - benchmark: Benchmark value for the metric
    - sessions: List of sessions with their performance data
    """
    result = get_leaderboard_by_task_id(db, task_id)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result
