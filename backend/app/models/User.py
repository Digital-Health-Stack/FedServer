from datetime import datetime
from sqlalchemy import Column, DateTime, Integer, String
from sqlalchemy.orm import relationship
from .Base import Base
from .Notification import Notification


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    role = Column(String, nullable=False, default="client")
    data_url = Column(String)
    hashed_password = Column(String)
    refresh_token = Column(String, nullable=True)
    createdAt = Column(DateTime, default=lambda: datetime.now())
    updatedAt = Column(
        DateTime, default=lambda: datetime.now(), onupdate=lambda: datetime.now()
    )

    federated_sessions = relationship("FederatedSession", back_populates="admin")
    federated_session_clients = relationship(
        "FederatedSessionClient", back_populates="user"
    )
    notifications = relationship("Notification", back_populates="user")

    def as_dict(self):
        return {
            "id": self.id,
            "username": self.username,
            "role": self.role,
            "data_url": self.data_url,
            "createdAt": (
                self.createdAt.isoformat() if self.createdAt else None
            ),  # Convert DateTime to ISO format
            "updatedAt": (
                self.updatedAt.isoformat() if self.updatedAt else None
            ),  # Convert DateTime to ISO format
        }
