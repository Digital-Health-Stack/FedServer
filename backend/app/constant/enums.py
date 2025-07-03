# app/enums.py or app/models/enums.py
from enum import IntEnum


class ClientStatus(IntEnum):
    # INVITED = -1         # Client invited but not joined
    JOINED = 0  # Joined session
    INITIATED = 1  # Config done
    # TRAINING_STARTED = 2
    # COMPLETED = 3


class TrainingStatus(IntEnum):
    # todo
    FAILED = -2
    CANCELLED = -1
    CREATED = 0
    ACCEPTING_CLIENTS = 1
    STARTED = 2
    COMPLETED = 3
