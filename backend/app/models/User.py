from datetime import datetime
from sqlalchemy import Column, DateTime, Integer, String
from sqlalchemy.orm import relationship
from .Base import Base
from .Notification import Notification


class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    data_url = Column(String)
    # email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    refresh_token = Column(String, nullable=True)
    createdAt = Column(DateTime, default=lambda: datetime.now())
    updatedAt = Column(DateTime, default=lambda: datetime.now(), onupdate=lambda: datetime.now())
    
    federated_sessions = relationship('FederatedSession', back_populates='admin')
    federated_session_clients = relationship('FederatedSessionClient', back_populates='user')
    notifications = relationship('Notification', back_populates='user')
