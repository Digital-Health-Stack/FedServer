from sqlalchemy.orm import Session
from models.FederatedSession import FederatedSession, FederatedTestResults
from models.User import User
from models.Dataset import Task, Dataset
from schemas.dataset import TaskCreate
from typing import List, Dict
from crud.datasets_crud import get_dataset_by_filename
from fastapi import HTTPException, status
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
from sqlalchemy import func, and_, desc, asc
import re


def sanitize_model_name(model_name: str) -> str:
    """
    Sanitize model names to make them more readable by converting camelCase/PascalCase
    to proper spacing and capitalization.

    Examples:
    - LinearRegression -> Linear Regression
    - LandMarkSVM -> Land Mark SVM
    - multiLayerPerceptron -> Multi Layer Perceptron
    """
    if not model_name or model_name == "Unknown":
        return model_name

    # Handle specific known cases first
    model_mappings = {
        "LinearRegression": "Linear Regression",
        "LandMarkSVM": "Land Mark SVM",
        "multiLayerPerceptron": "Multi Layer Perceptron",
    }

    if model_name in model_mappings:
        return model_mappings[model_name]

    # General camelCase/PascalCase to space-separated conversion
    # Insert space before uppercase letters that follow lowercase letters
    result = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", model_name)

    # Capitalize first letter and ensure proper spacing
    result = result.strip()
    if result:
        result = result[0].upper() + result[1:]

    return result


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
        print(f"Benchmark value: {benchmark_value}")

        # Create subquery to get the last round result for each session
        # This replaces the Python filtering with database-level filtering
        last_round_subquery = (
            db.query(
                FederatedTestResults.session_id,
                func.max(FederatedTestResults.round_number).label("max_round_number"),
            )
            .group_by(FederatedTestResults.session_id)
            .subquery()
        )

        # Determine sort order based on metric type
        reverse_sort = task.metric not in ["mae", "mse"]
        metric_order = desc if reverse_sort else asc

        # Main query with all optimizations:
        # 1. Join with last round subquery to get only final results
        # 2. Join with User table to avoid N+1 queries
        # 3. Filter sessions by task_id and ensure metrics exist
        # 4. Order by metric value in the database
        query_results = (
            db.query(
                FederatedSession.id.label("session_id"),
                FederatedSession.federated_info,
                FederatedSession.max_round.label("total_rounds"),
                FederatedSession.createdAt,
                FederatedTestResults.metrics_report,
                FederatedTestResults.round_number,
                User.name.label("admin_username"),
            )
            .join(
                FederatedTestResults,
                FederatedSession.id == FederatedTestResults.session_id,
            )
            .join(
                last_round_subquery,
                and_(
                    FederatedTestResults.session_id == last_round_subquery.c.session_id,
                    FederatedTestResults.round_number
                    == last_round_subquery.c.max_round_number,
                ),
            )
            .join(User, FederatedSession.admin_id == User.id)
            .filter(
                and_(
                    FederatedSession.federated_info["task_id"].as_integer() == task_id,
                    FederatedTestResults.metrics_report.isnot(None),
                    FederatedTestResults.metrics_report[task.metric].isnot(None),
                )
            )
            .order_by(
                metric_order(
                    FederatedTestResults.metrics_report[task.metric].as_float()
                )
            )
            .all()
        )

        print(f"Total sessions fetched: {len(query_results)}")
        leaderboard = []

        for result in query_results:
            metric_value = result.metrics_report.get(task.metric)
            if metric_value is None:
                continue

            print(
                f">>> Selected Last Round Result: Round {result.round_number} | Metrics: {result.metrics_report}"
            )

            # Determine if benchmark is met
            meets_benchmark = None
            if benchmark_value is not None:
                if task.metric in ["mae", "mse"]:
                    meets_benchmark = metric_value <= benchmark_value
                else:  # Accuracy, F1, etc.
                    meets_benchmark = metric_value >= benchmark_value

            leaderboard.append(
                {
                    "session_id": result.session_id,
                    "organisation_name": result.federated_info.get(
                        "organisation_name", "Unknown"
                    ),
                    "model_name": sanitize_model_name(
                        result.federated_info.get("model_name", "Unknown")
                    ),
                    "metric_value": metric_value,
                    "meets_benchmark": meets_benchmark,
                    "total_rounds": result.total_rounds,
                    "created_at": (
                        result.createdAt.isoformat() if result.createdAt else None
                    ),
                    "admin_username": result.admin_username or "Unknown",
                }
            )

        return {
            "task_name": task.task_name,
            "metric": task.metric,
            "benchmark": benchmark_value,
            "created_at": task.created_at.isoformat() if task.created_at else None,
            "sessions": leaderboard,
        }

    except Exception as e:
        return {"error": f"Error fetching leaderboard: {str(e)}"}
