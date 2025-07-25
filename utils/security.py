"""
Security Manager - Handles security validation and controls
"""

import re
import hashlib
from typing import Dict, Any, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class SecurityManager:
    def __init__(self):
        # Dangerous patterns in prompts
        self.dangerous_patterns = [
            r'rm\s+-rf\s+/',
            r'sudo\s+rm',
            r'format\s+c:',
            r'del\s+/[sq]',
            r'shutdown\s+',
            r'reboot\s+',
            r'passwd\s+',
            r'chmod\s+777',
            r'eval\s*\(',
            r'exec\s*\(',
            r'__import__\s*\(',
            r'open\s*\(\s*["\'][^"\']*\.(exe|bat|cmd|sh)',
        ]
        
        # Allowed file extensions
        self.safe_extensions = {
            '.txt', '.py', '.js', '.html', '.css', '.json', '.xml', '.yaml', '.yml',
            '.md', '.csv', '.sql', '.dockerfile', '.gitignore', '.cpp', '.c', '.h',
            '.java', '.go', '.rs', '.php', '.rb', '.swift', '.kt', '.scala'
        }
        
        # Dangerous commands
        self.forbidden_commands = {
            'rm', 'rmdir', 'del', 'format', 'fdisk', 'mkfs', 'dd',
            'sudo', 'su', 'chmod', 'chown', 'passwd', 'useradd', 'userdel',
            'reboot', 'shutdown', 'halt', 'poweroff', 'init',
            'kill', 'killall', 'pkill', 'fuser',
            'mount', 'umount', 'fsck', 'crontab'
        }
        
        # Safe commands
        self.allowed_commands = {
            'python', 'python3', 'node', 'npm', 'pip', 'pip3',
            'git', 'ls', 'cat', 'echo', 'pwd', 'whoami', 'date',
            'grep', 'find', 'head', 'tail', 'wc', 'sort', 'uniq'
        }
    
    def validate_request(self, request: Any) -> bool:
        """Validate incoming request for security issues"""
        try:
            # Check prompt for dangerous patterns
            if not self._check_prompt_safety(request.prompt):
                logger.warning(f"Dangerous pattern detected in prompt: {request.prompt[:100]}...")
                return False
            
            # Validate context if present
            if hasattr(request, 'context') and request.context:
                if not self._validate_context(request.context):
                    logger.warning("Unsafe context detected")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Security validation error: {str(e)}")
            return False
    
    def _check_prompt_safety(self, prompt: str) -> bool:
        """Check if prompt contains dangerous patterns"""
        prompt_lower = prompt.lower()
        
        for pattern in self.dangerous_patterns:
            if re.search(pattern, prompt_lower, re.IGNORECASE):
                return False
        
        return True
    
    def _validate_context(self, context: Dict[str, Any]) -> bool:
        """Validate context data"""
        # Check for suspicious file paths
        if 'files' in context:
            for file_path in context['files']:
                if not self._validate_file_path(file_path):
                    return False
        
        # Check for dangerous environment variables
        if 'env' in context:
            dangerous_env_vars = ['PATH', 'LD_LIBRARY_PATH', 'PYTHONPATH']
            for var in dangerous_env_vars:
                if var in context['env']:
                    return False
        
        return True
    
    def _validate_file_path(self, file_path: str) -> bool:
        """Validate file path for security"""
        path = Path(file_path)
        
        # Check extension
        if path.suffix and path.suffix.lower() not in self.safe_extensions:
            return False
        
        # Check for path traversal
        if '..' in str(path) or str(path).startswith('/'):
            return False
        
        # Check for system directories
        dangerous_dirs = ['/etc', '/usr', '/bin', '/sbin', '/root', '/home']
        for dangerous_dir in dangerous_dirs:
            if str(path).startswith(dangerous_dir):
                return False
        
        return True
    
    def validate_command(self, command: str) -> bool:
        """Validate shell command for security"""
        cmd_parts = command.split()
        if not cmd_parts:
            return False
        
        base_command = cmd_parts[0]
        
        # Check if command is forbidden
        if base_command in self.forbidden_commands:
            return False
        
        # Check if command is in allowed list
        if base_command not in self.allowed_commands:
            return False
        
        # Additional checks for specific commands
        if base_command in ['python', 'python3']:
            # Don't allow direct execution of dangerous Python code
            if any(danger in command for danger in ['exec(', 'eval(', '__import__']):
                return False
        
        return True
    
    def sanitize_output(self, output: str) -> str:
        """Sanitize output to remove sensitive information"""
        # Remove potential API keys
        output = re.sub(r'[A-Za-z0-9]{32,}', '[REDACTED]', output)
        
        # Remove file paths that might contain sensitive info
        output = re.sub(r'/home/[^/\s]+', '/home/[USER]', output)
        output = re.sub(r'/Users/[^/\s]+', '/Users/[USER]', output)
        
        return output
    
    def generate_session_token(self) -> str:
        """Generate secure session token"""
        import secrets
        return secrets.token_urlsafe(32)
    
    def hash_sensitive_data(self, data: str) -> str:
        """Hash sensitive data for logging"""
        return hashlib.sha256(data.encode()).hexdigest()[:16]
