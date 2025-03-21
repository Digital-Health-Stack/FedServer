from typing import Dict
from fastapi import WebSocket,WebSocketDisconnect
import asyncio
import json


class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        print('Client connected with id:', client_id)
        self.active_connections[client_id] = websocket

        # Implementing a simple ping-pong mechanism to keep the connection alive
        asyncio.create_task(self.ping_pong(websocket, client_id))

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]

    async def send_message(self, message: str, client_id: str):
        websocket = self.active_connections.get(client_id)
        if websocket:
            await websocket.send_text(message)
    
    async def ping_pong(self, websocket: WebSocket, client_id: str):
        try:
            while client_id in self.active_connections:
                await asyncio.sleep(50)  # Wait 10 seconds before sending the next ping
                ping_message = {"type": "ping"}
                await websocket.send_text(json.dumps(ping_message))
                print(f'Sent ping to client {client_id}')
                
                
        except WebSocketDisconnect:
            print(f'Client {client_id} disconnected during ping-pong')
            self.disconnect(client_id)
