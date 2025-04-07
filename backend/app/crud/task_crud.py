from sqlalchemy.orm import Session
from models.Dataset import Task, Dataset
from schemas.dataset import TaskCreate
from typing import List
from crud.datasets_crud import get_dataset_by_filename
from fastapi import HTTPException, status
from sqlalchemy.exc import SQLAlchemyError, IntegrityError

def create_task(db: Session, task: TaskCreate) -> Task:
    """Create a new task in the database"""
    try:
        db_task = Task(**task.dict())
        db.add(db_task)
        db.commit()
        db.refresh(db_task)
        return db_task
    except IntegrityError as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Task creation failed due to integrity constraints: {str(e)}"
        )
    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}"
        )

def delete_task(db: Session, task_id: int) -> None:
    """Delete a task from the database"""
    try:
        task = db.query(Task).filter(Task.task_id == task_id).first()
        if not task:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Task not found"
            )
        
        db.delete(task)
        db.commit()
    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}"
        )

def get_tasks_by_dataset_name(db: Session, dataset_filename: str) -> List[Task]:
    """Get all tasks for a dataset by filename"""
    try:
        dataset = get_dataset_by_filename(db, dataset_filename)
        tasks = db.query(Task).filter(
            Task.dataset_id == dataset.dataset_id
        ).all()
        
        if not tasks:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No tasks found for dataset '{dataset_filename}'"
            )
            
        return tasks
    except HTTPException:
        raise  
    except SQLAlchemyError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}"
        )

def get_tasks_by_dataset_id(db: Session, dataset_id: int) -> List[Task]:
    """Get all tasks for a dataset by ID"""
    try:
        if not db.query(Dataset).filter(Dataset.dataset_id == dataset_id).first():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset with id {dataset_id} not found"
            )
            
        tasks = db.query(Task).filter(
            Task.dataset_id == dataset_id
        ).all()
        
        if not tasks:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No tasks found for dataset with id {dataset_id}"
            )
            
        return tasks
    except SQLAlchemyError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}"
        )

def get_task_by_id(db: Session, task_id: int) -> Task:
    """Get a task by its ID"""
    try:
        task = db.query(Task).filter(Task.task_id == task_id).first()
        if not task:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task with id {task_id} not found"
            )
        return task
    except SQLAlchemyError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}"
        )

def update_task_by_id(db: Session, task_id: int, task_update: TaskCreate) -> Task:
    """Update an existing task by its ID"""
    try:
        task = db.query(Task).filter(Task.task_id == task_id).first()
        if not task:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task with id {task_id} not found"
            )

        for key, value in task_update.dict(exclude_unset=True).items():
            setattr(task, key, value)

        db.commit()
        db.refresh(task)
        return task
    except IntegrityError as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Update failed due to integrity constraints: {str(e)}"
        )
    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}"
        )