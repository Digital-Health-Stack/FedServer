from datetime import datetime
from models.User import User
from sqlalchemy import Null
from sqlalchemy.orm import Session

from models.Notification import Notification


def get_unnotified_notifications(user: User, db: Session):
    """
    Fetch all notifications for this user that have not been notified.
    """
    return (
        db.query(Notification)
        .filter(
            Notification.user_id == user.id,
            Notification.notified_at.is_(None),
            (
                Notification.valid_until.is_(None)
                | (Notification.valid_until > datetime.now())
            ),
        )
        .all()
    )
