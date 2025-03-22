from fastapi import APIRouter, Request, Depends
from sqlalchemy.orm import Session
from utility.auth import get_current_user
from utility.db import get_db
from utility.user import get_unnotified_notifications
import json
from datetime import datetime
import asyncio
from sse_starlette import EventSourceResponse

from models.User import User

notifications_router = APIRouter()

@notifications_router.get("/notifications/stream")
async def notifications_stream(
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    async def event_generator():
        while True:
            # Check if client has disconnected
            if await request.is_disconnected():
                break
        
            # Fetch notifications for the current user
            user_notifications = get_unnotified_notifications(user = current_user, db = db)
            
            if(len(user_notifications) > 0):
                data = [n.message for n in user_notifications]

                # Send the data as an SSE event
                yield {
                    "event": "new_notifications",   
                    "data": json.dumps(data),
                }
                

                for notification in user_notifications:
                    notification.notified_at = datetime.now()
                db.commit()

            # Wait for 5 seconds before sending the next update
            await asyncio.sleep(5)

    return EventSourceResponse(event_generator())