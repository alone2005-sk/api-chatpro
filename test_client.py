"""
Test client for the AI Backend API
"""

import asyncio
import aiohttp
import json

async def test_api():
    """Test the AI backend API"""
    base_url = "http://localhost:8000/api/v1"
    
    async with aiohttp.ClientSession() as session:
        # Test system status
        print("🔍 Checking system status...")
        async with session.get(f"{base_url}/system/status") as response:
            status = await response.json()
            print(f"Status: {json.dumps(status, indent=2)}")
        
        # Test code generation
        print("\n🧑‍💻 Testing code generation...")
        request_data = {
            "prompt": "Write a Python function to calculate fibonacci numbers",
            "validate_code": True,
            "save_output": True
        }
        
        async with session.post(f"{base_url}/process", json=request_data) as response:
            result = await response.json()
            task_id = result["task_id"]
            print(f"Task created: {task_id}")
        
        # Poll for completion
        while True:
            async with session.get(f"{base_url}/tasks/{task_id}") as response:
                task_status = await response.json()
                print(f"Status: {task_status['status']}, Progress: {task_status.get('progress', 0)}%")
                
                if task_status["status"] in ["completed", "failed"]:
                    print(f"Final result: {json.dumps(task_status, indent=2)}")
                    break
                
                await asyncio.sleep(2)

if __name__ == "__main__":
    asyncio.run(test_api())
