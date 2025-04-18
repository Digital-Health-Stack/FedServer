from datetime import datetime, timedelta
from typing import List, Union
from sqlalchemy.orm import Session
from sqlalchemy.sql import and_
from models.Notification import Notification
from models.User import User

def add_notifications_for_recently_active_users(db: Session, message: dict, valid_until: datetime = None, excluded_users: List[User]  = []):
    # Calculate the time 24 hours ago
    last_updated_at = datetime.now() - timedelta(days=1)

    # Fetch users whose updatedAt is within the last 24 hours
    users_to_notify = db.query(User).filter(and_(
        User.updatedAt >= last_updated_at,
        User.role != "admin",
        ~User.id.in_([user.id for user in excluded_users])
    )).all()
    
    
    add_notifications_for(db, message, users_to_notify, valid_until)

    # Optionally, return the newly added notifications (for debugging or logging purposes)
    return users_to_notify

def add_notifications_for(db: Session, message: dict, users_to_notify: Union[List[int], List[User]], valid_until: datetime = None):

    # Create a new notification for each user
    for user in users_to_notify:
        user_id = user if isinstance(user, int) else user.id
        new_notification = Notification(
            user_id=user_id,
            message=message,
            created_at=datetime.now(),
            valid_until=valid_until
        )
        db.add(new_notification)

    # Commit the changes to the database
    db.commit()
    
def add_notifications_for_user(db: Session, user_id: int, message: dict, valid_until: datetime = None):
    """
    Add a notification for a specific user.

    Args:
        db (Session): SQLAlchemy database session.
        user_id (int): The ID of the user to notify.
        message (dict): The message to send as a notification.
        valid_until (datetime, optional): The expiry time of the notification. Defaults to None.
    """
    # Create a new notification for the specific user
    new_notification = Notification(
        user_id=user_id,
        message=message,
        created_at=datetime.now(),
        valid_until=valid_until
    )
    db.add(new_notification)

    # Commit the changes to the database
    db.commit()
