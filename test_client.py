import asyncio
import aiohttp
import jwt
import time
import logging
import json

# Configuration
SECRET_KEY = "default_secret"  # Must match your backend's JWT middleware secret
API_URL = "http://localhost:5000"  # Adjust if hosted elsewhere

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("damn_bot_test_client")


def generate_token():
    payload = {
        "user_id": "test_user",
        "iat": int(time.time()),
        "exp": int(time.time()) + 3600  # Token valid for 1 hour
    }
    token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
    logger.info(f"🔐 Generated JWT token: {token}")
    return token


async def call_health(session):
    logger.info("🔍 Checking /health endpoint...")
    async with session.get(f"{API_URL}/health") as resp:
        logger.info(f"✅ /health status: {resp.status}")
        result = await resp.text()
        logger.info(f"🧠 /health response: {result}")
async def call_chat_with_image(session):
    logger.info("💬 Sending /chat request with image...")

    data = aiohttp.FormData()
    data.add_field("prompt", "Analyze the objects in this image and describe them in detail")
    data.add_field("user_id", "test_user")
    data.add_field("chat_id", "")
    data.add_field("stream", "false")
    data.add_field("web_search", "false")
    data.add_field("voice", "false")
    data.add_field("research_mode", "false")
    data.add_field("deep_learning", "false")  # Let AI analyze deeply
    data.add_field("auto_fix", "true")
    data.add_field("language", "auto")
    data.add_field("auto_detect_media", "true")  # Allow automatic detection of image
    data.add_field("context", json.dumps({}))

    # Read file into memory to avoid 'closed file' errors
    with open("generated_media/image_test_user_huggingface_1754663416.png", "rb") as f:
        file_bytes = f.read()

    data.add_field(
        "file",
        file_bytes,
        filename="img.png",
        content_type="image/png"  # Correct MIME type for PNG
    )

    async with session.post(f"{API_URL}/chat", data=data) as resp:
        logger.info(f"📨 /chat status: {resp.status}")
        result = await resp.text()
        logger.info(f"🧠 /chat response: {result}")


async def call_generate_image(session):
    complex_prompt = (
    "A surreal fantasy scene of a colossal dragon emerging from a glowing crystal cavern, "
    "surrounded by floating islands with waterfalls cascading into the clouds, "
    "combining elements of steampunk machinery and ancient runes, "
    "ultra-photorealistic textures mixed with painterly brush strokes, "
    "dynamic cinematic lighting with god rays, 8K ultra HD, "
    "depth-of-field blur, volumetric fog, and atmospheric particles, "
    "inspired by a fusion of Studio Ghibli and H.R. Giger's biomechanical style"
    )

    
    data = aiohttp.FormData()
    data.add_field("prompt", complex_prompt)
    data.add_field("style", "photorealistic")
    data.add_field("size", "1024x1024")
    data.add_field("user_id", "test_user")
    
    async with session.post(f"{API_URL}/generate/image", data=data) as resp:
        logger.info(f"📨 /generate/image status: {resp.status}")
        result = await resp.json()
        logger.info(f"🧠 /generate/image response: {result}")

async def call_generate_video(session: aiohttp.ClientSession):
    data = aiohttp.FormData()
    data.add_field("prompt", "A futuristic cityscape at night")
    data.add_field("style", "cinematic")
    data.add_field("duration", "10")  # duration as string because Form data is text
    data.add_field("user_id", "test_user")
    
    async with session.post(f"{API_URL}/generate/video", data=data) as resp:
        logger.info(f"📨 /generate/video status: {resp.status}")
        if resp.status == 200:
            result = await resp.json()
            logger.info(f"🧠 /generate/video response: {result}")
            return result
        else:
            text = await resp.text()
            logger.error(f"Failed /generate/video: Status {resp.status} - {text}")
            resp.raise_for_status()
            
async def main():
    token = generate_token()
    headers = {
        "Authorization": f"Bearer {token}",
        # Content-Type will be set automatically by aiohttp when using FormData
    }

    async with aiohttp.ClientSession(headers=headers) as session:
        # await call_health(session)
        await call_generate_image(session)


if __name__ == "__main__":
    asyncio.run(main())
