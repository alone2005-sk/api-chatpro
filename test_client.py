
import asyncio
import aiohttp
import jwt
import time
import logging
import json
from typing import Dict, Any, Optional, List
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIAgentTestClient:
    def __init__(self, base_url: str = "http://localhost:5000", secret_key: str = "default_secret"):
        self.base_url = base_url
        self.secret_key = secret_key
        self.access_token = None
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def generate_jwt_token(self, user_id: str = "test_user", additional_claims: Dict = None) -> str:
        """Generate JWT token for authentication"""
        payload = {
            "user_id": user_id,
            "username": "test_user",
            "email": "test@example.com",
            "exp": int(time.time()) + 3600,  # 1 hour expiration
            "iat": int(time.time()),
            "sub": user_id
        }
        
        if additional_claims:
            payload.update(additional_claims)
            
        token = jwt.encode(payload, self.secret_key, algorithm="HS256")
        logger.info(f"Generated JWT token: {token[:50]}...")
        return token
    
    def get_auth_headers(self) -> Dict[str, str]:
        """Get authorization headers with JWT token"""
        if not self.access_token:
            self.access_token = self.generate_jwt_token()
            
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
    
    async def test_health_check(self) -> Dict:
        """Test health check endpoint (no auth required)"""
        logger.info("Testing health check endpoint...")
        try:
            async with self.session.get(f"{self.base_url}/health") as response:
                result = await response.json()
                logger.info(f"Health check status: {response.status}")
                return {"status": response.status, "data": result}
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    async def test_root_endpoint(self) -> Dict:
        """Test root endpoint (no auth required)"""
        logger.info("Testing root endpoint...")
        try:
            async with self.session.get(f"{self.base_url}/") as response:
                result = await response.json()
                logger.info(f"Root endpoint status: {response.status}")
                return {"status": response.status, "data": result}
        except Exception as e:
            logger.error(f"Root endpoint failed: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    async def test_chat_endpoint(self, prompt: str = "Hello, test message") -> Dict:
        """Test chat endpoint with authentication"""
        logger.info("Testing chat endpoint...")
        try:
            # Prepare form data
            data = aiohttp.FormData()
            data.add_field('prompt', prompt)
            data.add_field('user_id', 'test_user')
            data.add_field('stream', 'false')
            data.add_field('web_search', 'false')
            data.add_field('context', '{}')
            
            headers = {"Authorization": f"Bearer {self.access_token or self.generate_jwt_token()}"}
            
            async with self.session.post(
                f"{self.base_url}/chat", 
                data=data, 
                headers=headers
            ) as response:
                result = await response.json()
                logger.info(f"Chat endpoint status: {response.status}")
                return {"status": response.status, "data": result}
                
        except Exception as e:
            logger.error(f"Chat endpoint failed: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    async def test_media_generation(self, prompt: str = "A beautiful sunset") -> Dict:
        """Test media generation endpoint"""
        logger.info("Testing media generation endpoint...")
        try:
            data = aiohttp.FormData()
            data.add_field('prompt', prompt)
            data.add_field('user_id', 'test_user')
            data.add_field('media_type', 'image')
            data.add_field('style', 'realistic')
            
            headers = {"Authorization": f"Bearer {self.access_token or self.generate_jwt_token()}"}
            
            async with self.session.post(
                f"{self.base_url}/media/generate",
                data=data,
                headers=headers
            ) as response:
                result = await response.json()
                logger.info(f"Media generation status: {response.status}")
                return {"status": response.status, "data": result}
                
        except Exception as e:
            logger.error(f"Media generation failed: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    async def test_research_endpoint(self, query: str = "Latest AI developments") -> Dict:
        """Test research endpoint"""
        logger.info("Testing research endpoint...")
        try:
            data = aiohttp.FormData()
            data.add_field('query', query)
            data.add_field('user_id', 'test_user')
            data.add_field('depth', 'moderate')
            data.add_field('sources', '3')
            
            headers = {"Authorization": f"Bearer {self.access_token or self.generate_jwt_token()}"}
            
            async with self.session.post(
                f"{self.base_url}/research",
                data=data,
                headers=headers
            ) as response:
                result = await response.json()
                logger.info(f"Research endpoint status: {response.status}")
                return {"status": response.status, "data": result}
                
        except Exception as e:
            logger.error(f"Research endpoint failed: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    async def test_chat_history(self, user_id: str = "test_user") -> Dict:
        """Test getting user chat history"""
        logger.info("Testing chat history endpoint...")
        try:
            headers = {"Authorization": f"Bearer {self.access_token or self.generate_jwt_token()}"}
            
            async with self.session.get(
                f"{self.base_url}/chats/{user_id}",
                headers=headers
            ) as response:
                result = await response.json()
                logger.info(f"Chat history status: {response.status}")
                return {"status": response.status, "data": result}
                
        except Exception as e:
            logger.error(f"Chat history failed: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    async def test_models_endpoint(self) -> Dict:
        """Test available models endpoint"""
        logger.info("Testing models endpoint...")
        try:
            headers = {"Authorization": f"Bearer {self.access_token or self.generate_jwt_token()}"}
            
            async with self.session.get(
                f"{self.base_url}/models",
                headers=headers
            ) as response:
                result = await response.json()
                logger.info(f"Models endpoint status: {response.status}")
                return {"status": response.status, "data": result}
                
        except Exception as e:
            logger.error(f"Models endpoint failed: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    async def test_stats_endpoint(self) -> Dict:
        """Test stats endpoint"""
        logger.info("Testing stats endpoint...")
        try:
            headers = {"Authorization": f"Bearer {self.access_token or self.generate_jwt_token()}"}
            
            async with self.session.get(
                f"{self.base_url}/stats",
                headers=headers
            ) as response:
                result = await response.json()
                logger.info(f"Stats endpoint status: {response.status}")
                return {"status": response.status, "data": result}
                
        except Exception as e:
            logger.error(f"Stats endpoint failed: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    async def test_cloud_storage_auth(self) -> Dict:
        """Test cloud storage authorization"""
        logger.info("Testing cloud storage authorization...")
        try:
            data = aiohttp.FormData()
            data.add_field('user_id', 'test_user')
            data.add_field('provider', 'google_drive')
            data.add_field('auth_data', '{"access_token": "test_token"}')
            
            headers = {"Authorization": f"Bearer {self.access_token or self.generate_jwt_token()}"}
            
            async with self.session.post(
                f"{self.base_url}/cloud-storage/authorize",
                data=data,
                headers=headers
            ) as response:
                result = await response.json()
                logger.info(f"Cloud storage auth status: {response.status}")
                return {"status": response.status, "data": result}
                
        except Exception as e:
            logger.error(f"Cloud storage auth failed: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    async def test_streaming_chat(self, prompt: str = "Tell me a story") -> Dict:
        """Test streaming chat endpoint"""
        logger.info("Testing streaming chat endpoint...")
        try:
            data = aiohttp.FormData()
            data.add_field('prompt', prompt)
            data.add_field('user_id', 'test_user')
            data.add_field('stream', 'true')
            data.add_field('context', '{}')
            
            headers = {"Authorization": f"Bearer {self.access_token or self.generate_jwt_token()}"}
            
            async with self.session.post(
                f"{self.base_url}/chat",
                data=data,
                headers=headers
            ) as response:
                logger.info(f"Streaming chat status: {response.status}")
                
                if response.status == 200:
                    # Read streaming response
                    content = ""
                    async for chunk in response.content.iter_chunked(1024):
                        chunk_str = chunk.decode('utf-8')
                        content += chunk_str
                        logger.info(f"Received chunk: {chunk_str[:100]}...")
                    
                    return {"status": response.status, "data": {"stream_content": content}}
                else:
                    result = await response.json()
                    return {"status": response.status, "data": result}
                
        except Exception as e:
            logger.error(f"Streaming chat failed: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests and return comprehensive results"""
        logger.info("🚀 Starting comprehensive API tests...")
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "test_results": {}
        }
        
        # Define all tests
        tests = [
            ("health_check", self.test_health_check),
            ("root_endpoint", self.test_root_endpoint),
            ("chat_endpoint", self.test_chat_endpoint),
            ("media_generation", self.test_media_generation),
            ("research_endpoint", self.test_research_endpoint),
            ("chat_history", self.test_chat_history),
            ("models_endpoint", self.test_models_endpoint),
            ("stats_endpoint", self.test_stats_endpoint),
            ("cloud_storage_auth", self.test_cloud_storage_auth),
            ("streaming_chat", self.test_streaming_chat)
        ]
        
        # Generate token once for all tests
        self.access_token = self.generate_jwt_token()
        
        # Run each test
        for test_name, test_func in tests:
            logger.info(f"\n--- Running {test_name} ---")
            result = await test_func()
            
            results["test_results"][test_name] = result
            results["total_tests"] += 1
            
            if result.get("status") in [200, 201]:
                results["passed"] += 1
                logger.info(f"✅ {test_name} PASSED")
            else:
                results["failed"] += 1
                logger.error(f"❌ {test_name} FAILED: {result}")
        
        # Summary
        logger.info(f"\n🎯 Test Summary:")
        logger.info(f"Total Tests: {results['total_tests']}")
        logger.info(f"Passed: {results['passed']}")
        logger.info(f"Failed: {results['failed']}")
        logger.info(f"Success Rate: {(results['passed']/results['total_tests']*100):.1f}%")
        
        return results

async def main():
    """Main function to run all tests"""
    print("🤖 AI Agent API Test Client")
    print("Testing all endpoints with JWT authentication...")
    
    async with AIAgentTestClient() as client:
        results = await client.run_all_tests()
        
        # Save results to file
        with open("test_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📊 Results saved to test_results.json")
        return results

if __name__ == "__main__":
    asyncio.run(main())
