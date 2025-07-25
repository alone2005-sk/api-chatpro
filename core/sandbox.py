"""
Code Sandbox - Safe execution environment for code snippets
"""

import asyncio
import tempfile
import subprocess
import docker
import os
import signal
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class CodeSandbox:
    def __init__(self):
        self.docker_client = None
        self.temp_dir = Path(tempfile.mkdtemp(prefix="ai_sandbox_"))
        self.timeout = 30  # seconds
        
        # Security settings
        self.allowed_commands = {
            'python', 'python3', 'node', 'npm', 'pip', 'pip3',
            'git', 'ls', 'cat', 'echo', 'pwd', 'whoami'
        }
        
        self.forbidden_commands = {
            'rm', 'rmdir', 'del', 'format', 'fdisk', 'mkfs',
            'sudo', 'su', 'chmod', 'chown', 'passwd',
            'reboot', 'shutdown', 'halt', 'poweroff'
        }
    
    async def initialize(self):
        """Initialize sandbox environment"""
        try:
            # Try to initialize Docker client
            self.docker_client = docker.from_env()
            logger.info("✅ Docker sandbox initialized")
        except Exception as e:
            logger.warning(f"Docker not available: {str(e)}")
            logger.info("Using process-based sandbox")
    
    async def cleanup(self):
        """Cleanup sandbox resources"""
        try:
            # Clean up temp directory
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            
            if self.docker_client:
                self.docker_client.close()
        except Exception as e:
            logger.error(f"Cleanup error: {str(e)}")
    
    async def validate_code(self, code: str, language: str) -> Dict[str, Any]:
        """Validate code syntax without execution"""
        try:
            if language == "python":
                return await self._validate_python(code)
            elif language == "javascript":
                return await self._validate_javascript(code)
            else:
                return {"status": "unknown", "message": f"Validation not supported for {language}"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    async def execute_python(self, code: str) -> Dict[str, Any]:
        """Execute Python code safely"""
        try:
            # Create temporary file
            temp_file = self.temp_dir / "script.py"
            with open(temp_file, 'w') as f:
                f.write(code)
            
            if self.docker_client:
                return await self._execute_in_docker(str(temp_file), "python")
            else:
                return await self._execute_in_process(["python3", str(temp_file)])
                
        except Exception as e:
            return {"status": "error", "output": "", "error": str(e)}
    
    async def execute_command(self, command: str) -> Dict[str, Any]:
        """Execute shell command safely"""
        try:
            # Parse command
            cmd_parts = command.split()
            if not cmd_parts:
                raise ValueError("Empty command")
            
            base_command = cmd_parts[0]
            
            # Security check
            if base_command in self.forbidden_commands:
                raise PermissionError(f"Command '{base_command}' is not allowed")
            
            if base_command not in self.allowed_commands:
                raise PermissionError(f"Command '{base_command}' is not in allowed list")
            
            return await self._execute_in_process(cmd_parts)
            
        except Exception as e:
            return {"status": "error", "output": "", "error": str(e)}
    
    async def _validate_python(self, code: str) -> Dict[str, Any]:
        """Validate Python syntax"""
        try:
            compile(code, '<string>', 'exec')
            return {"status": "valid", "message": "Python syntax is valid"}
        except SyntaxError as e:
            return {"status": "invalid", "message": f"Syntax error: {str(e)}"}
    
    async def _validate_javascript(self, code: str) -> Dict[str, Any]:
        """Validate JavaScript syntax"""
        try:
            # Create temporary file
            temp_file = self.temp_dir / "script.js"
            with open(temp_file, 'w') as f:
                f.write(code)
            
            # Use Node.js to check syntax
            result = await self._execute_in_process([
                "node", "--check", str(temp_file)
            ])
            
            if result["exit_code"] == 0:
                return {"status": "valid", "message": "JavaScript syntax is valid"}
            else:
                return {"status": "invalid", "message": result["error"]}
                
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    async def _execute_in_docker(self, script_path: str, interpreter: str) -> Dict[str, Any]:
        """Execute code in Docker container"""
        try:
            # Create container
            container = self.docker_client.containers.run(
                "python:3.9-slim",
                f"{interpreter} /app/script.py",
                volumes={str(self.temp_dir): {'bind': '/app', 'mode': 'ro'}},
                working_dir="/app",
                detach=True,
                remove=True,
                mem_limit="128m",
                cpu_quota=50000,  # 50% CPU
                network_disabled=True
            )
            
            # Wait for completion with timeout
            try:
                result = container.wait(timeout=self.timeout)
                logs = container.logs().decode('utf-8')
                
                return {
                    "status": "completed",
                    "exit_code": result["StatusCode"],
                    "output": logs,
                    "error": ""
                }
            except Exception as e:
                container.kill()
                return {
                    "status": "timeout",
                    "exit_code": -1,
                    "output": "",
                    "error": f"Execution timeout: {str(e)}"
                }
                
        except Exception as e:
            return {
                "status": "error",
                "exit_code": -1,
                "output": "",
                "error": str(e)
            }
    
    async def _execute_in_process(self, cmd_parts: list) -> Dict[str, Any]:
        """Execute command in subprocess with timeout"""
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd_parts,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.temp_dir)
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.timeout
                )
                
                return {
                    "status": "completed",
                    "exit_code": process.returncode,
                    "output": stdout.decode('utf-8'),
                    "error": stderr.decode('utf-8')
                }
                
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return {
                    "status": "timeout",
                    "exit_code": -1,
                    "output": "",
                    "error": "Execution timeout"
                }
                
        except Exception as e:
            return {
                "status": "error",
                "exit_code": -1,
                "output": "",
                "error": str(e)
            }
    
    async def get_status(self) -> Dict[str, Any]:
        """Get sandbox status"""
        return {
            "docker_available": self.docker_client is not None,
            "temp_dir": str(self.temp_dir),
            "timeout": self.timeout,
            "allowed_commands": list(self.allowed_commands)
        }
