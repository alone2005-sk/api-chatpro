"""
Secure Credential Manager
Handles secure storage and retrieval of user API keys and credentials
"""

import json
import hashlib
import base64
from typing import Dict, List, Optional
from datetime import datetime
from cryptography.fernet import Fernet
import logging

logger = logging.getLogger(__name__)

class CredentialManager:
    def __init__(self, db_connection, encryption_key: Optional[str] = None):
        self.db = db_connection
        
        # In production, use environment variable for encryption key
        if encryption_key:
            self.cipher = Fernet(encryption_key.encode())
        else:
            # Generate a key for demo (use secure key management in production)
            key = Fernet.generate_key()
            self.cipher = Fernet(key)
            
    def _encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        return base64.b64encode(self.cipher.encrypt(data.encode())).decode()
    
    def _decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        return self.cipher.decrypt(base64.b64decode(encrypted_data)).decode()
    
    async def store_api_key(self, user_id: str, service_name: str, api_key: str, metadata: Dict = None) -> Dict:
        """
        Securely store user's API key for a service
        """
        try:
            # Encrypt the API key
            encrypted_key = self._encrypt_data(api_key)
            
            # Create key hash for verification (without storing plain key)
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:16]
            
            # Store in database
            query = """
            INSERT INTO user_api_keys (user_id, service_name, encrypted_key, key_hash, metadata, created_at, status)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (user_id, service_name) 
            DO UPDATE SET encrypted_key = %s, key_hash = %s, metadata = %s, updated_at = %s
            """
            
            metadata_json = json.dumps(metadata or {})
            now = datetime.now()
            
            await self.db.execute(query, (
                user_id, service_name, encrypted_key, key_hash, metadata_json, now, 'active',
                encrypted_key, key_hash, metadata_json, now
            ))
            
            return {
                'success': True, 
                'message': f'API key for {service_name} stored successfully',
                'key_hash': key_hash
            }
            
        except Exception as e:
            logger.error(f"Failed to store API key: {e}")
            return {'success': False, 'error': str(e)}
    
    async def get_api_key(self, user_id: str, service_name: str) -> Optional[str]:
        """
        Retrieve and decrypt user's API key for a service
        """
        try:
            query = """
            SELECT encrypted_key FROM user_api_keys 
            WHERE user_id = %s AND service_name = %s AND status = 'active'
            """
            
            result = await self.db.fetchone(query, (user_id, service_name))
            
            if result:
                encrypted_key = result[0]
                return self._decrypt_data(encrypted_key)
            else:
                return None
                
        except Exception as e:
            logger.error(f"Failed to retrieve API key: {e}")
            return None
    
    async def list_user_services(self, user_id: str) -> List[Dict]:
        """
        List all services with stored API keys for a user
        """
        try:
            query = """
            SELECT service_name, key_hash, metadata, created_at, updated_at, status
            FROM user_api_keys 
            WHERE user_id = %s
            ORDER BY service_name
            """
            
            results = await self.db.fetchall(query, (user_id,))
            
            services = []
            for row in results:
                services.append({
                    'service_name': row[0],
                    'key_hash': row[1],
                    'metadata': json.loads(row[2]) if row[2] else {},
                    'created_at': row[3].isoformat() if row[3] else None,
                    'updated_at': row[4].isoformat() if row[4] else None,
                    'status': row[5]
                })
                
            return services
            
        except Exception as e:
            logger.error(f"Failed to list services: {e}")
            return []
    
    async def delete_api_key(self, user_id: str, service_name: str) -> Dict:
        """
        Delete user's API key for a service
        """
        try:
            query = """
            UPDATE user_api_keys 
            SET status = 'deleted', deleted_at = %s 
            WHERE user_id = %s AND service_name = %s
            """
            
            await self.db.execute(query, (datetime.now(), user_id, service_name))
            
            return {'success': True, 'message': f'API key for {service_name} deleted'}
            
        except Exception as e:
            logger.error(f"Failed to delete API key: {e}")
            return {'success': False, 'error': str(e)}
    
    async def validate_api_key(self, user_id: str, service_name: str) -> Dict:
        """
        Validate if stored API key is still working
        """
        try:
            api_key = await self.get_api_key(user_id, service_name)
            if not api_key:
                return {'valid': False, 'error': 'API key not found'}
            
            # Test the API key based on service
            validation_result = await self._test_api_key(service_name, api_key)
            
            # Update validation status
            query = """
            UPDATE user_api_keys 
            SET last_validated = %s, validation_status = %s 
            WHERE user_id = %s AND service_name = %s
            """
            
            status = 'valid' if validation_result['valid'] else 'invalid'
            await self.db.execute(query, (datetime.now(), status, user_id, service_name))
            
            return validation_result
            
        except Exception as e:
            logger.error(f"API key validation failed: {e}")
            return {'valid': False, 'error': str(e)}
    
    async def _test_api_key(self, service_name: str, api_key: str) -> Dict:
        """
        Test API key validity by making a simple API call
        """
        import aiohttp
        
        test_endpoints = {
            'huggingface': 'https://huggingface.co/api/whoami-v2',
            'groq': 'https://api.groq.com/openai/v1/models',
            'together': 'https://api.together.xyz/v1/models',
            'openai': 'https://api.openai.com/v1/models',
            'anthropic': 'https://api.anthropic.com/v1/messages',
        }
        
        if service_name.lower() not in test_endpoints:
            return {'valid': True, 'message': 'Service validation not implemented'}
        
        try:
            headers = self._get_auth_headers(service_name, api_key)
            endpoint = test_endpoints[service_name.lower()]
            
            async with aiohttp.ClientSession() as session:
                async with session.get(endpoint, headers=headers, timeout=10) as response:
                    if response.status in [200, 401, 403]:
                        valid = response.status == 200
                        return {
                            'valid': valid,
                            'message': 'API key is valid' if valid else 'API key is invalid',
                            'status_code': response.status
                        }
                        
        except Exception as e:
            logger.error(f"API key test failed for {service_name}: {e}")
            
        return {'valid': False, 'error': 'Unable to validate API key'}
    
    def _get_auth_headers(self, service_name: str, api_key: str) -> Dict[str, str]:
        """
        Get appropriate authentication headers for different services
        """
        headers = {'User-Agent': 'AIChatPro/1.0'}
        
        if service_name.lower() in ['huggingface']:
            headers['Authorization'] = f'Bearer {api_key}'
        elif service_name.lower() in ['openai', 'groq', 'together']:
            headers['Authorization'] = f'Bearer {api_key}'
        elif service_name.lower() == 'anthropic':
            headers['x-api-key'] = api_key
            headers['anthropic-version'] = '2023-06-01'
            
        return headers
    
    async def get_service_config(self, user_id: str, service_name: str) -> Optional[Dict]:
        """
        Get complete service configuration including API key and metadata
        """
        try:
            query = """
            SELECT encrypted_key, metadata FROM user_api_keys 
            WHERE user_id = %s AND service_name = %s AND status = 'active'
            """
            
            result = await self.db.fetchone(query, (user_id, service_name))
            
            if result:
                api_key = self._decrypt_data(result[0])
                metadata = json.loads(result[1]) if result[1] else {}
                
                return {
                    'service_name': service_name,
                    'api_key': api_key,
                    'metadata': metadata,
                    'headers': self._get_auth_headers(service_name, api_key)
                }
            else:
                return None
                
        except Exception as e:
            logger.error(f"Failed to get service config: {e}")
            return None
    
    async def bulk_store_keys(self, user_id: str, api_keys: Dict[str, str]) -> Dict:
        """
        Store multiple API keys at once
        """
        results = {'stored': [], 'failed': []}
        
        for service_name, api_key in api_keys.items():
            result = await self.store_api_key(user_id, service_name, api_key)
            
            if result['success']:
                results['stored'].append(service_name)
            else:
                results['failed'].append({
                    'service': service_name,
                    'error': result['error']
                })
        
        return results