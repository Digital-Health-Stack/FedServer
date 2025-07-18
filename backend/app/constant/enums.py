# app/enums.py or app/models/enums.py
from enum import IntEnum, StrEnum


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


class FederatedSessionLogTag(StrEnum):
    INFO = "INFO"
    ERROR = "ERROR"
    SUCCESS = "SUCCESS"
    TRAINING = "TRAINING"
    CLIENT_JOINED = "CLIENT_JOINED"
    WEIGHTS_RECEIVED = "WEIGHTS_RECEIVED"
    AGGREGATED_WEIGHTS = "AGGREGATED_WEIGHTS"
    TEST_RESULTS = "TEST_RESULTS"
    CLIENT_LEFT = "CLIENT_LEFT"
    PRIZE_NEGOTIATION = "PRIZE_NEGOTIATION"
