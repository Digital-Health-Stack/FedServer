import redis.asyncio as redis
from contextlib import asynccontextmanager
import os

# Redis configuration
REDIS_HOST = 'localhost'
REDIS_PORT = 6380
REDIS_PASSWORD = '123456'

@asynccontextmanager
async def get_redis_connection():
    """Context manager for Redis connections to ensure proper cleanup."""
    connection = None
    try:
        connection = redis.Redis(
            host=REDIS_HOST, 
            port=REDIS_PORT, 
            password=REDIS_PASSWORD,
            decode_responses=True
        )
        yield connection
    finally:
        if connection:
            try:
                await connection.close()
            except Exception as e:
                print(f"Error closing Redis connection: {e}")

async def send_notification_for_new_session(message: str):
    """Send notification using a properly managed Redis connection."""
    try:
        async with get_redis_connection() as r:
            await r.publish("new-session", message)
        return True 
    except Exception as e:
        print(f"Error sending notification: {e}")
        return False