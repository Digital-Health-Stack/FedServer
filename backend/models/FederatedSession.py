from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from sqlalchemy import JSON, Column, DateTime, ForeignKey, Integer, Null, String, event
from sqlalchemy.orm import declared_attr, relationship, Session, with_loader_criteria
from .User import Base
import os

load_dotenv()


class TimestampMixin:
    @declared_attr
    def createdAt(cls):
        return Column(DateTime, default=lambda: datetime.now())
    @declared_attr
    def updatedAt(cls):
        return Column(DateTime, default=lambda: datetime.now(), onupdate=lambda: datetime.now())

    @declared_attr
    def deletedAt(cls):
        return Column(DateTime, nullable=True)

@event.listens_for(Session, "do_orm_execute")
def filter_soft_deleted(execute_state):
    if not execute_state.is_column_load and not execute_state.is_relationship_load:
        if not execute_state.execution_options.get("include_deleted", False):
            execute_state.statement = execute_state.statement.options(
                with_loader_criteria(
                    TimestampMixin,
                    lambda cls: cls.deletedAt.is_(None),
                    include_aliases=True
                )
            )

class FederatedSession(TimestampMixin, Base):
    __tablename__ = 'federated_sessions'
    
    id = Column(Integer, primary_key=True, index=True)
    federated_info = Column(JSON, nullable=False)
    admin_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    curr_round = Column(Integer, default=1, nullable=False)
    max_round = Column(Integer, default=10, nullable=False)
    global_parameters = Column(JSON, default='[]', nullable=False)
    training_status = Column(Integer, default=1, nullable=False) # 1 for server waiting for all clients and 2 for training starts, 3 for completed
    client_parameters = Column(JSON, default='{}', nullable=False)
    # Wait Time
    wait_till = Column(DateTime, default=lambda: datetime.now() + timedelta(minutes=int(os.getenv('SESSION_WAIT_MINUTES'))))
    
    admin = relationship("User", back_populates="federated_sessions")
    clients = relationship('FederatedSessionClient', back_populates='session')
    
class FederatedSessionClient(TimestampMixin, Base):
    __tablename__ = 'federated_session_clients'

    id = Column(Integer, primary_key=True, index=True)
    client_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    session_id = Column(Integer, ForeignKey('federated_sessions.id'), nullable=False)
    status = Column(Integer, default=1, nullable=False) # Status values: 1 (not responded), 2 (accepted), 3 (rejected)
    ip = Column(String, nullable=False)
    local_model_id = Column(String, nullable=True)
    
    client = relationship('User', back_populates="federated_session_clients")
    session = relationship('FederatedSession', back_populates='clients')
