from fastapi import WebSocket
import json


class ConnectionManager:
    def __init__(self):
        # Store connections by user ID
        self.active_connections: dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, user_id: str):
        await websocket.accept()
        self.active_connections[user_id] = websocket

    def disconnect(self, user_id: str):
        del self.active_connections[user_id]

    async def send_notification(self, user_id: str, data: dict):
        websocket = self.active_connections.get(user_id)
        if websocket:
            await websocket.send_text(json.dumps(data))

    async def broadcast(self, data: dict):
        for connection in self.active_connections.values():
            if connection:
                await connection.send_text(json.dumps(data))