class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self.clients = {}
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        client_id = str(uuid.uuid4())
        # Generate unique id for client
        self.active_connections.append(websocket)
        self.clients[client_id] = websocket
        return client_id
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def send_personal_message(self,message: str,client_id: str):
        websocket = self.clients.get(client_id)
        if websocket:
            await websocket.send_text(message)
    
    async def findInterestedClients(self,parameters,client_id):
        new_message = {
            'parameters':parameters,
            'message':'Are you interested in doing Federated Learning'
        }
        message_str = json.dumps(new_message)
        for id,connection in self.clients.items():
            if client_id != id:
                await connection.send_text(message_str)
        
    async def broadcast(self,message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.websocket("/master-ws")
async def websocket_endpoint(websocket: WebSocket):
    client_id = await manager.connect(websocket)
    message = json.dumps({"message": f"Connected with client ID: {client_id} to Master-ws"})
    await websocket.send_text(message)
    try:
        while True:
            data = await websocket.receive_text()
            # await manager.broadcast(f"Message from {client_id}: {data}")
            await manager.findInterestedClients(f"Parameters from {client_id}:{data}",client_id)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.broadcast(f"Client #{client_id} left the chat")
