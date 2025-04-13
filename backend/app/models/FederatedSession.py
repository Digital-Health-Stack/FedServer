from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from sqlalchemy import JSON, Column, DateTime, ForeignKey, Integer, Float, String, event, UniqueConstraint
from sqlalchemy.orm import declared_attr, relationship, Session, with_loader_criteria
from .Base import Base
import os
from constant.enums import ClientStatus

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

class FederatedSessionLog(Base):
    __tablename__ = 'federated_session_logs'
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey('federated_sessions.id', ondelete="CASCADE"), nullable=False)
    message = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    session = relationship('FederatedSession', back_populates='logs')



class FederatedSession(TimestampMixin, Base):
    __tablename__ = 'federated_sessions'
    
    id = Column(Integer, primary_key=True, index=True)
    federated_info = Column(JSON, nullable=False)
    admin_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    curr_round = Column(Integer, default=1, nullable=False)
    max_round = Column(Integer, default=3, nullable=False)
    # global_parameters = Column(JSON, default='[]', nullable=False)
    session_price = Column(Float, default= 0, nullable=True)
    # 1 for server waiting for admin to price, 2 for server waiting for all clients and 3 for training starts, 4 for completed
    # -1 if training fails
    training_status = Column(Integer, default=1, nullable=False)  
    # Wait Time
    wait_till = Column(DateTime, default=lambda: datetime.now() + timedelta(minutes=int(os.getenv('SESSION_WAIT_MINUTES'))))
    
    admin = relationship("User", back_populates="federated_sessions")
    clients = relationship('FederatedSessionClient', back_populates='session')
    logs = relationship("FederatedSessionLog", order_by=FederatedSessionLog.timestamp, back_populates="session", cascade="all, delete-orphan")
    global_model_weights = relationship("GlobalModelWeights", back_populates="session", cascade="all, delete-orphan",order_by="GlobalModelWeights.created_at")
    test_results = relationship("FederatedTestResults", back_populates="session", 
                                cascade="all, delete-orphan", order_by="FederatedTestResults.round_number")
    
    def as_dict(self):  
        return {
            "id": self.id,
            "federated_info": self.federated_info,
            "admin_id": self.admin_id,
            "curr_round": self.curr_round,
            "max_round": self.max_round,
            "training_status": self.training_status,
            "wait_till": self.wait_till.isoformat() if self.wait_till else None,  # Convert DateTime to ISO format
            "admin": self.admin.as_dict() if self.admin else None,               # Call as_dict on the related User
            "clients": [client.as_dict() for client in self.clients],            
            "test_results": [result.as_dict() for result in self.test_results]
        }

class FederatedSessionClient(TimestampMixin, Base):
    __tablename__ = 'federated_session_clients'

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    session_id = Column(Integer, ForeignKey('federated_sessions.id'), nullable=False)
    # Status values: 0 (means client accepted), 1 (means client initiated model)
    status = Column(Integer, default=ClientStatus.JOINED.value, nullable=False) 
    ip = Column(String, nullable=False)
  
    user = relationship('User', back_populates="federated_session_clients")
    session = relationship('FederatedSession', back_populates='clients')

    def as_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "status": self.status,
            "ip": self.ip,
            "user": self.user.as_dict() if self.user else None
        }

class FederatedRoundClientSubmission(Base):
    __tablename__ = 'federated_round_client_submissions'
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey('federated_sessions.id'), nullable=False)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    round_number = Column(Integer, nullable=False)
    submitted_at = Column(DateTime, default=lambda: datetime.now())
    
    model_weights = relationship("ClientModelWeights", back_populates="submission",cascade="all, delete-orphan", uselist=False)
    
    __table_args__ = (
        UniqueConstraint(
            'session_id',
            'user_id',
            'round_number',
            name='unique_client_round_submission'
        ),
    )
    def as_dict(self):
        return {
            "id": self.id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "round_number": self.round_number,
            "submitted_at": self.submitted_at.isoformat(),
            "model_weights": self.model_weights.weights if self.model_weights else None
        }

    
class ClientModelWeights(Base):
    __tablename__ = 'client_model_weights'
    
    id = Column(Integer, primary_key=True)
    submission_id = Column(Integer, ForeignKey('federated_round_client_submissions.id'), unique=True)
    weights = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now())
    submission = relationship("FederatedRoundClientSubmission", back_populates="model_weights")
    def as_dict(self):
        return {
            "id": self.id,
            "submission_id": self.submission_id,
            "weights": self.weights
        }

class GlobalModelWeights(Base):
    __tablename__ = 'global_model_weights'

    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey('federated_sessions.id'), nullable=False)
    weights = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now())
    updated_at = Column(DateTime, default=lambda: datetime.now())
    session = relationship("FederatedSession", back_populates="global_model_weights")
    

class FederatedTestResults(TimestampMixin, Base):
    __tablename__ = 'federated_test_results'
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey('federated_sessions.id'), nullable=False)
    round_number = Column(Integer, nullable=False)
    metrics = Column(JSON, nullable=False)

    session = relationship("FederatedSession", back_populates="test_results")
    
    def as_dict(self):
        return {
            "id": self.id,
            "session_id": self.session_id,
            "round_number": self.round_number,
            "metrics": self.metrics,
            "created_at": self.createdAt.isoformat() if self.createdAt else None,
            "updated_at": self.updatedAt.isoformat() if self.updatedAt else None
        }