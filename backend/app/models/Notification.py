from sqlalchemy import Column, Integer, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
from .Base import Base

class Notification(Base):
    __tablename__ = "notifications"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)  # Assuming a 'users' table exists
    message = Column(JSON, nullable=False)  # JSON field to store structured data
    created_at = Column(DateTime, default=datetime.now())
    notified_at = Column(DateTime, nullable=True)
    valid_until = Column(DateTime, nullable=True) 

    # Relationships (optional, depending on your app)
    user = relationship("User", back_populates="notifications")  # Assuming a User model exists

    def as_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "message": self.message,  # JSON is already serializable
            "created_at": self.created_at.isoformat(),
            "notified_at": self.notified_at.isoformat() if self.notified_at else None,
            "valid_until": self.valid_until.isoformat() if self.valid_until else None,
        }
