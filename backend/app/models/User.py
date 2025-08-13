from datetime import datetime
from sqlalchemy import DateTime, Integer, String
from sqlalchemy.orm import mapped_column, relationship
from .Base import Base
from .Notification import Notification


class User(Base):
    __tablename__ = "users"

    id = mapped_column(Integer, primary_key=True, index=True)
    username = mapped_column(String, unique=True, index=True)
    role = mapped_column(String, nullable=False, default="client")
    data_url = mapped_column(String)
    hashed_password = mapped_column(String)
    refresh_token = mapped_column(String, nullable=True)
    createdAt = mapped_column(DateTime, default=lambda: datetime.now())
    updatedAt = mapped_column(
        DateTime, default=lambda: datetime.now(), onupdate=lambda: datetime.now()
    )
    federated_sessions = relationship("FederatedSession", back_populates="admin")
    federated_session_clients = relationship(
        "FederatedSessionClient", back_populates="user"
    )
    notifications = relationship("Notification", back_populates="user")
    client_permissions = relationship("ClientPermission", back_populates="user")

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
