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
    expected_results: dict
    # to resolve warning of protected namespace model_ for model_name and model_info
    class Config:
        protected_namespaces = ()

class CreateFederatedLearning(BaseModel):
    fed_info: FederatedLearningInfo
    # client_token: str


class ClientFederatedResponse(BaseModel):
    session_id: int
    decision: int
    
class ClientModelIdResponse(BaseModel):
    session_id: int

class ClientReceiveParameters(BaseModel):
    session_id: int
    client_parameter: dict
    metrics_report: dict

    