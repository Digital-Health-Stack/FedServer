from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from sqlalchemy import (
    JSON,
    Enum,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    Float,
    String,
    event,
    UniqueConstraint,
)
from sqlalchemy.orm import declared_attr, relationship, Session, with_loader_criteria
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
        return Column(DateTime, default=lambda: datetime.now(IST))

    @declared_attr
    def updatedAt(cls):
        return Column(
            DateTime,
            default=lambda: datetime.now(IST),
            onupdate=lambda: datetime.now(IST),
        )

    @declared_attr
    def deletedAt(cls):
        return Column(DateTime, default=None, nullable=True)


class FederatedSessionLog(TimestampMixin, Base):
    __tablename__ = "federated_session_logs"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(
        Integer, ForeignKey("federated_sessions.id", ondelete="CASCADE"), nullable=False
    )
    message = Column(String, nullable=False)

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


class FederatedSessionLogV2(TimestampMixin, Base):
    __tablename__ = "federated_session_logs_v2"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(
        Integer,
        ForeignKey("federated_sessions_v2.id", ondelete="CASCADE"),
        nullable=False,
    )
    message = Column(String, nullable=False)

    session = relationship("FederatedSessionsV2", back_populates="logs")

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

    id = Column(Integer, primary_key=True, index=True)
    federated_info = Column(JSON, nullable=False)
    admin_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    curr_round = Column(Integer, default=1, nullable=False)
    max_round = Column(Integer, default=50, nullable=False)
    session_price = Column(Float, default=0, nullable=True)
    training_status = Column(Integer, default=1, nullable=False)
    # Wait Time
    wait_till = Column(
        DateTime,
        default=lambda: datetime.now(IST)
        + timedelta(minutes=int(os.getenv("SESSION_WAIT_MINUTES"))),
    )

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
            "wait_till": (
                self.wait_till.isoformat() if self.wait_till else None
            ),  # Convert DateTime to ISO format
            "admin": (
                self.admin.as_dict() if self.admin else None
            ),  # Call as_dict on the related User
            "clients": [client.as_dict() for client in self.clients],
            "results": [result.as_dict() for result in self.results],
            "created_at": self.createdAt.isoformat() if self.createdAt else None,
        }


class FederatedSessionsV2(TimestampMixin, Base):
    __tablename__ = "federated_sessions_v2"

    id = Column(Integer, primary_key=True, index=True)
    federated_info = Column(JSON, nullable=False)
    admin_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    curr_round = Column(Integer, default=1, nullable=False)
    max_round = Column(Integer, default=50, nullable=False)
    session_price = Column(Float, default=0, nullable=True)
    training_status = Column(
        Enum(TrainingStatus), default=TrainingStatus.CREATED, nullable=False
    )
    # Wait Time
    wait_till = Column(
        DateTime,
        default=lambda: datetime.now(IST)
        + timedelta(minutes=int(self.federated_info.get("wait_time", 0))),
    )

    admin = relationship("User", back_populates="federated_sessions_v2")
    clients = relationship("FederatedSessionClientV2", back_populates="session")
    logs = relationship(
        "FederatedSessionLogV2",
        order_by=FederatedSessionLogV2.createdAt,
        back_populates="session",
        cascade="all, delete-orphan",
    )
    results = relationship(
        "FederatedTestResultsV2",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="FederatedTestResultsV2.round_number",
    )

    def as_dict(self):
        return {
            "id": self.id,
            "federated_info": self.federated_info,
            "admin_id": self.admin_id,
            "curr_round": self.curr_round,
            "max_round": self.max_round,
            "training_status": self.training_status,
            "wait_till": (
                self.wait_till.isoformat() if self.wait_till else None
            ),  # Convert DateTime to ISO format
            "admin": (
                self.admin.as_dict() if self.admin else None
            ),  # Call as_dict on the related User
            "clients": [client.as_dict() for client in self.clients],
            "results": [result.as_dict() for result in self.results],
            "created_at": self.createdAt.isoformat() if self.createdAt else None,
        }


class FederatedSessionClient(TimestampMixin, Base):
    __tablename__ = "federated_session_clients"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    session_id = Column(Integer, ForeignKey("federated_sessions.id"), nullable=False)
    # Status values: 0 (means client accepted), 1 (means client initiated model)
    status = Column(Integer, default=ClientStatus.JOINED.value, nullable=False)
    ip = Column(String, nullable=False)

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


class FederatedSessionClientV2(TimestampMixin, Base):
    __tablename__ = "federated_session_clients_v2"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    session_id = Column(Integer, ForeignKey("federated_sessions_v2.id"), nullable=False)
    # Status values: 0 (means client accepted), 1 (means client initiated model)
    status = Column(Integer, default=ClientStatus.JOINED.value, nullable=False)
    ip = Column(String, nullable=False)

    user = relationship("User", back_populates="federated_session_clients_v2")
    session = relationship("FederatedSessionsV2", back_populates="clients")

    def as_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "status": self.status,
            "ip": self.ip,
            "user": self.user.as_dict() if self.user else None,
        }


class FederatedRoundClientSubmission(TimestampMixin, Base):
    __tablename__ = "federated_round_client_submissions"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("federated_sessions.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    round_number = Column(Integer, nullable=False)
    metrics_report = Column(JSON)

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


class FederatedRoundClientSubmissionV2(TimestampMixin, Base):
    __tablename__ = "federated_round_client_submissions_v2"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("federated_sessions_v2.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    round_number = Column(Integer, nullable=False)
    metrics_report = Column(JSON)

    __table_args__ = (
        UniqueConstraint(
            "session_id",
            "user_id",
            "round_number",
            name="unique_client_round_submission_v2",
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

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("federated_sessions.id"), nullable=False)
    round_number = Column(Integer, nullable=False)
    metrics_report = Column(JSON, nullable=False)

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


class FederatedTestResultsV2(TimestampMixin, Base):
    __tablename__ = "federated_test_results_v2"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("federated_sessions_v2.id"), nullable=False)
    round_number = Column(Integer, nullable=False)
    metrics_report = Column(JSON, nullable=False)

    session = relationship("FederatedSessionsV2", back_populates="results")

    def as_dict(self):
        return {
            "id": self.id,
            "session_id": self.session_id,
            "round_number": self.round_number,
            "metrics_report": self.metrics_report,
            "created_at": self.createdAt.isoformat() if self.createdAt else None,
            "updated_at": self.updatedAt.isoformat() if self.updatedAt else None,
        }
