from dataclasses import dataclass
from datetime import datetime, timedelta
from dotenv import load_dotenv
from sqlalchemy import (
    JSON,
    Enum,
    DateTime,
    ForeignKey,
    Integer,
    Float,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, declared_attr, relationship, mapped_column
from .Base import Base
import os
from constant.enums import ClientStatus, TrainingStatus
import pytz

load_dotenv()


# Define IST timezone
IST = pytz.timezone("Asia/Kolkata")


class TimestampMixin:
    @declared_attr
    def createdAt(cls):
        return mapped_column(DateTime, default=lambda: datetime.now(IST))

    @declared_attr
    def updatedAt(cls):
        return mapped_column(
            DateTime,
            default=lambda: datetime.now(IST),
            onupdate=lambda: datetime.now(IST),
        )

    @declared_attr
    def deletedAt(cls):
        return mapped_column(DateTime, default=None, nullable=True)


class FederatedSessionLog(TimestampMixin, Base):
    __tablename__ = "federated_session_logs"

    id = mapped_column(Integer, primary_key=True, index=True)
    session_id = mapped_column(
        Integer, ForeignKey("federated_sessions.id", ondelete="CASCADE"), nullable=False
    )
    message = mapped_column(String, nullable=False)

    session = relationship("FederatedSession", back_populates="logs")

    def as_dict(self):
        return {
            "id": self.id,
            "session_id": self.session_id,
            "message": self.message,
            "created_at": self.createdAt.isoformat() if self.createdAt else None,
            "updated_at": self.updatedAt.isoformat() if self.updatedAt else None,
            "deleted_at": self.deletedAt.isoformat() if self.deletedAt else None,
        }


class FederatedSession(TimestampMixin, Base):
    __tablename__ = "federated_sessions"

    id = mapped_column(Integer, primary_key=True, index=True)
    federated_info = mapped_column(JSON, nullable=False)
    admin_id = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    curr_round = mapped_column(Integer, default=1, nullable=False)
    max_round = mapped_column(Integer, default=50, nullable=False)
    session_price = mapped_column(Float, default=0, nullable=True)
    training_status = mapped_column(
        Enum(TrainingStatus), default=TrainingStatus.CREATED, nullable=False
    )
    no_of_recieved_weights = mapped_column(Integer, default=0, nullable=False)
    no_of_left_clients = mapped_column(Integer, default=0, nullable=False)

    admin = relationship("User", back_populates="federated_sessions")
    clients = relationship("FederatedSessionClient", back_populates="session")
    logs = relationship(
        "FederatedSessionLog",
        order_by=FederatedSessionLog.createdAt,
        back_populates="session",
        cascade="all, delete-orphan",
    )
    results = relationship(
        "FederatedTestResults",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="FederatedTestResults.round_number",
    )

    def as_dict(self):
        return {
            "id": self.id,
            "federated_info": self.federated_info,
            "admin_id": self.admin_id,
            "curr_round": self.curr_round,
            "max_round": self.max_round,
            "training_status": self.training_status,
            "admin": (
                self.admin.as_dict() if self.admin else None
            ),  # Call as_dict on the related User
            "clients": [client.as_dict() for client in self.clients],
            "results": [result.as_dict() for result in self.results],
            "created_at": self.createdAt.isoformat() if self.createdAt else None,
        }


@dataclass
class FederatedSessionClient(TimestampMixin, Base):
    __tablename__ = "federated_session_clients"

    id = mapped_column(Integer, primary_key=True, index=True)
    user_id = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    session_id = mapped_column(
        Integer, ForeignKey("federated_sessions.id"), nullable=False
    )
    # Status values: 0 (means client accepted), 1 (means client initiated model)
    status = mapped_column(Integer, default=ClientStatus.JOINED.value, nullable=False)
    ip = mapped_column(String, nullable=False)

    user = relationship("User", back_populates="federated_session_clients")
    session = relationship("FederatedSession", back_populates="clients")

    def as_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "status": self.status,
            "ip": self.ip,
            "user": self.user.as_dict() if self.user else None,
        }


@dataclass
class FederatedRoundClientSubmission(TimestampMixin, Base):
    __tablename__ = "federated_round_client_submissions"

    id = mapped_column(Integer, primary_key=True, index=True)
    session_id = mapped_column(
        Integer, ForeignKey("federated_sessions.id"), nullable=False
    )
    user_id = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    round_number = mapped_column(Integer, nullable=False)
    metrics_report = mapped_column(JSON)

    __table_args__ = (
        UniqueConstraint(
            "session_id",
            "user_id",
            "round_number",
            name="unique_client_round_submission",
        ),
    )

    def as_dict(self):
        return {
            "id": self.id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "round_number": self.round_number,
            "metrics_report": self.metrics_report,
            "submitted_at": self.createdAt.isoformat() if self.createdAt else None,
        }


class FederatedTestResults(TimestampMixin, Base):
    __tablename__ = "federated_test_results"

    id = mapped_column(Integer, primary_key=True, index=True)
    session_id = mapped_column(
        Integer, ForeignKey("federated_sessions.id"), nullable=False
    )
    round_number = mapped_column(Integer, nullable=False)
    metrics_report = mapped_column(JSON, nullable=False)

    session = relationship("FederatedSession", back_populates="results")

    def as_dict(self):
        return {
            "id": self.id,
            "session_id": self.session_id,
            "round_number": self.round_number,
            "metrics_report": self.metrics_report,
            "created_at": self.createdAt.isoformat() if self.createdAt else None,
            "updated_at": self.updatedAt.isoformat() if self.updatedAt else None,
        }
