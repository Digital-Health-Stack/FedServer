from sqlalchemy.orm import Session, joinedload
from models.FederatedSession import FederatedSession, FederatedTestResults
from models.User import User
from models.Dataset import Task, Dataset
from schemas.dataset import TaskCreate
from typing import List, Dict, Optional
from crud.datasets_crud import get_dataset_by_filename
from fastapi import HTTPException, status
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
from sqlalchemy import func


def create_task(db: Session, task: TaskCreate) -> Task:
    """Create a new task in the database"""

    try:

        db_task = Task(**task.model_dump())
        db.add(db_task)
        db.commit()
        db.refresh(db_task)
        return db_task
    except IntegrityError as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Task creation failed due to integrity constraints: {str(e)}",
        )
    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}",
        )


def delete_task(db: Session, task_id: int) -> None:
    """Delete a task from the database"""
    try:
        task = db.query(Task).filter(Task.task_id == task_id).first()
        if not task:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Task not found"
            )

        db.delete(task)
        db.commit()
    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}",
        )


def get_tasks_by_dataset_name(db: Session, dataset_filename: str) -> List[Task]:
    """Get all tasks for a dataset by filename"""
    try:
        dataset = get_dataset_by_filename(db, dataset_filename)
        if not dataset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset with filename {dataset_filename} not found",
            )
        tasks = db.query(Task).filter(Task.dataset_id == dataset.dataset_id).all()

        if not tasks:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No tasks found for dataset '{dataset_filename}'",
            )

        return tasks
    except HTTPException:
        raise
    except SQLAlchemyError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}",
        )


def get_tasks_by_dataset_id(db: Session, dataset_id: int) -> List[Task]:
    """Get all tasks for a dataset by ID"""
    try:
        if not db.query(Dataset).filter(Dataset.dataset_id == dataset_id).first():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset with id {dataset_id} not found",
            )

        tasks = db.query(Task).filter(Task.dataset_id == dataset_id).all()
        return tasks
    except SQLAlchemyError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}",
        )


def get_task_by_id(db: Session, task_id: int) -> Task:
    """Get a task by its ID"""
    try:
        task = db.query(Task).filter(Task.task_id == task_id).first()
        if not task:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task with id {task_id} not found",
            )
        return task
    except SQLAlchemyError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}",
        )


def update_task_by_id(db: Session, task_id: int, task_update: TaskCreate) -> Task:
    """Update an existing task by its ID"""
    try:
        task = db.query(Task).filter(Task.task_id == task_id).first()
        if not task:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task with id {task_id} not found",
            )

        for key, value in task_update.model_dump(exclude_unset=True).items():
            setattr(task, key, value)

        db.commit()
        db.refresh(task)
        return task
    except IntegrityError as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Update failed due to integrity constraints: {str(e)}",
        )
    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}",
        )


def get_leaderboard_by_task_id(db: Session, task_id: int) -> Dict:
    """
    Get leaderboard data for a specific task including benchmark comparison

    Args:
        db: Database session
        task_id: ID of the task to get leaderboard for

    Returns:
        Dictionary containing task info and leaderboard data
    """
    try:
        # Get task with benchmark metric
        task = db.query(Task).filter(Task.task_id == task_id).first()
        if not task:
            return {"error": "Task not found"}

        # Extract std_mean as benchmark value
        benchmark_value = None
        if task.benchmark and task.metric in task.benchmark:
            benchmark_value = task.benchmark[task.metric].get("std_mean")

        # Get sessions with their last round results
        sessions = (
            db.query(FederatedSession)
            .filter(FederatedSession.federated_info["dataset_info"]["task_id"].as_integer() == task_id)
            .options(joinedload(FederatedSession.results))
            .all()
        )
        print(f"Total sessions fetched: {len(sessions)}")
        leaderboard = []

        for session in sessions:
            # # Get last round result
            last_round_result = next(
                (r for r in session.results if r.round_number == session.max_round),
                None,
            )

            if not last_round_result or not last_round_result.metrics_report:
                continue
            if last_round_result:
                print(
                    f">>> Selected Last Round Result: Round {last_round_result.round_number} | Metrics: {last_round_result.metrics_report}"
                )
            else:
                print(">>> No result found for the max round.")

            metric_value = last_round_result.metrics_report.get(task.metric)
            if metric_value is None:
                continue

            # Determine if benchmark is met
            meets_benchmark = None
            if benchmark_value is not None:
                if task.metric in ["mae", "mse"]:
                    meets_benchmark = metric_value <= benchmark_value
                else:  # Accuracy, F1, etc.
                    meets_benchmark = metric_value >= benchmark_value

            # Get admin username
            admin_username = (
                db.query(User.username).filter(User.id == session.admin_id).scalar()
                or "Unknown"
            )

            leaderboard.append(
                {
                    "session_id": session.id,
                    "organisation_name": session.federated_info.get(
                        "organisation_name", "Unknown"
                    ),
                    "model_name": session.federated_info.get("model_name", "Unknown"),
                    "metric_value": metric_value,
                    "meets_benchmark": meets_benchmark,
                    "total_rounds": session.max_round,
                    "created_at": (
                        session.createdAt.isoformat() if session.createdAt else None
                    ),
                    "admin_username": admin_username,
                }
            )

        # Sort based on metric type
        reverse_sort = task.metric not in ["mae", "mse"]
        leaderboard.sort(key=lambda x: x["metric_value"], reverse=reverse_sort)

        return {
            "task_name": task.task_name,
            "metric": task.metric,
            "benchmark": benchmark_value,
            "created_at": task.created_at.isoformat() if task.created_at else None,
            "sessions": leaderboard,
        }

    except Exception as e:
        return {"error": f"Error fetching leaderboard: {str(e)}"}
