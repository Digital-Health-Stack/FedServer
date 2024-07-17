from pydantic import BaseModel


class User(BaseModel):
    name: str
    data_path: str
    password: str


class Parameter(BaseModel):
    client_parameter: dict
    client_id: int


class FederatedLearningInfo(BaseModel):
    organisation_name: str
    dataset_info: dict
    model_name: str
    model_info: dict


class CreateFederatedLearning(BaseModel):
    fed_info: FederatedLearningInfo
    client_token: str


class ClientFederatedResponse(BaseModel):
    client_id: str
    session_id: str
    decision: int
