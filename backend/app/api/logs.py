from fastapi import APIRouter, Session, Depends
from models.FederatedSession import FederatedSessionLog

log_router = APIRouter()

@log_router.get("/{session_id}", response_model=list[FederatedSessionLogResponse])
def get_logs(session_id: int, db: Session = Depends(get_db)):
    """
    Retrieve all logs for a specific federated session.
    """
    logs = db.query(FederatedSessionLog).filter(FederatedSessionLog.session_id == session_id).order_by(FederatedSessionLog.timestamp).all()
    if not logs:
        raise HTTPException(status_code=404, detail="No logs found for this session")
    return logs

@log_router.delete("/session/{session_id}", response_model=dict)
def delete_all_logs_of_session(session_id: int, db: Session = Depends(get_db)):
    """
    Delete all log entries for a specific federated session.
    """
    logs = db.query(FederatedSessionLog).filter(FederatedSessionLog.session_id == session_id).all()
    
    if not logs:
        raise HTTPException(status_code=404, detail="No logs found for this session")
    
    # Delete all logs related to the session
    db.query(FederatedSessionLog).filter(FederatedSessionLog.session_id == session_id).delete()
    db.commit()
    
    return {"message": f"All logs for session {session_id} deleted successfully"}

