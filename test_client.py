import asyncio
import aiohttp
import jwt
import time
import logging

# Configuration
SECRET_KEY = "default_secret"  # Must match your backend's middleware secret
API_URL = "http://localhost:5000"  # Adjust if hosted elsewhere

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("damn_bot_test_client")

def generate_token():
    payload = {
        "user_id": "test_user",
        "iat": int(time.time()),
        "exp": int(time.time()) + 3600  # Valid for 1 hour
    }
    token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
    logger.info(f"Generated JWT token: {token}")
    return token

async def call_status(session):
    logger.info("🔍 Checking /status endpoint...")
    async with session.get(f"{API_URL}/status") as resp:
        logger.info(f"Status code: {resp.status}")
        result = await resp.text()
        logger.info(f"Response: {result}")

async def call_chat(session):
    logger.info("💬 Sending /chat request...")
    
    data = aiohttp.FormData()
    data.add_field("prompt", "Create Me a full  web app for proejct for movie website ")
    data.add_field("web_search", "false")
    data.add_field("voice", "false")
    data.add_field("stream", "false")
    data.add_field("research_mode", "false")
    data.add_field("deep_learning", "false")
    data.add_field("code_execution", "true")
    data.add_field("auto_fix", "true")
    data.add_field("language", "en")
    data.add_field("max_iterations", "3")
    data.add_field("project_id", "test_project_123")

    async with session.post(f"{API_URL}/chat", data=data) as resp:
        logger.info(f"Status code: {resp.status}")
        result = await resp.text()
        logger.info(f"Response: {result}")

async def main():
    token = generate_token()
    headers = {
        "Authorization": f"Bearer {token}"
    }

    async with aiohttp.ClientSession(headers=headers) as session:
        await call_status(session)
        await call_chat(session)

if __name__ == "__main__":
    asyncio.run(main())
