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
    model_name: str
    model_info: dict
    dataset_info: dict

    