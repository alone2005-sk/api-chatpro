"""
Secure File Manager - Handles file operations with security controls
"""

import os
import aiofiles
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional
import mimetypes
import logging

logger = logging.getLogger(__name__)

class FileManager:
    def __init__(self):
        self.base_dir = Path("workspace")
        self.upload_dir = Path("uploads")
        self.output_dir = Path("outputs")
        
        # Create directories
        for directory in [self.base_dir, self.upload_dir, self.output_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Security settings
        self.allowed_extensions = {
            '.txt', '.py', '.js', '.html', '.css', '.json', '.xml', '.yaml', '.yml',
            '.md', '.csv', '.sql', '.sh', '.bat', '.dockerfile', '.gitignore',
            '.cpp', '.c', '.h', '.java', '.go', '.rs', '.php', '.rb', '.swift'
        }
        
        self.max_file_size = 10 * 1024 * 1024  # 10MB
        self.forbidden_paths = ['/etc', '/usr', '/bin', '/sbin', '/root', '/home']
    
    def _validate_path(self, file_path: str) -> Path:
        """Validate and sanitize file path"""
        # Convert to Path object and resolve
        path = Path(file_path).resolve()
        
        # Check if path is within allowed directories
        try:
            path.relative_to(self.base_dir.resolve())
        except ValueError:
            # If not in base_dir, check if it's in upload or output dirs
            try:
                path.relative_to(self.upload_dir.resolve())
            except ValueError:
                try:
                    path.relative_to(self.output_dir.resolve())
                except ValueError:
                    raise PermissionError(f"Access denied to path: {file_path}")
        
        # Check for forbidden paths
        for forbidden in self.forbidden_paths:
            if str(path).startswith(forbidden):
                raise PermissionError(f"Access denied to system path: {file_path}")
        
        return path
    
    def _validate_extension(self, file_path: Path) -> bool:
        """Check if file extension is allowed"""
        return file_path.suffix.lower() in self.allowed_extensions
    
    async def read_file(self, file_path: str) -> Dict[str, Any]:
        """Securely read file content"""
        try:
            path = self._validate_path(file_path)
            
            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            if not self._validate_extension(path):
                raise PermissionError(f"File type not allowed: {path.suffix}")
            
            # Check file size
            if path.stat().st_size > self.max_file_size:
                raise PermissionError(f"File too large: {path.stat().st_size} bytes")
            
            async with aiofiles.open(path, 'r', encoding='utf-8') as f:
                content = await f.read()
            
            return {
                "content": content,
                "path": str(path),
                "size": path.stat().st_size,
                "mime_type": mimetypes.guess_type(str(path))[0]
            }
            
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {str(e)}")
            raise
    
    async def write_file(self, file_path: str, content: str) -> Dict[str, Any]:
        """Securely write file content"""
        try:
            path = self._validate_path(file_path)
            
            if not self._validate_extension(path):
                raise PermissionError(f"File type not allowed: {path.suffix}")
            
            # Check content size
            if len(content.encode('utf-8')) > self.max_file_size:
                raise PermissionError("Content too large")
            
            # Create parent directories
            path.parent.mkdir(parents=True, exist_ok=True)
            
            async with aiofiles.open(path, 'w', encoding='utf-8') as f:
                await f.write(content)
            
            return {
                "path": str(path),
                "size": path.stat().st_size,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error writing file {file_path}: {str(e)}")
            raise
    
    async def save_code(self, code: str, language: str, task_id: str) -> Path:
        """Save generated code to output directory"""
        extensions = {
            "python": ".py",
            "javascript": ".js",
            "html": ".html",
            "css": ".css",
            "sql": ".sql",
            "bash": ".sh",
            "java": ".java",
            "cpp": ".cpp",
            "go": ".go",
            "rust": ".rs"
        }
        
        extension = extensions.get(language, ".txt")
        filename = f"{task_id}_generated{extension}"
        file_path = self.output_dir / filename
        
        await self.write_file(str(file_path), code)
        return file_path
    
    async def upload_file(self, file_data: Any) -> Dict[str, Any]:
        """Handle file upload"""
        try:
            # Generate unique filename
            file_hash = hashlib.md5(file_data.filename.encode()).hexdigest()[:8]
            filename = f"{file_hash}_{file_data.filename}"
            file_path = self.upload_dir / filename
            
            # Validate extension
            if not self._validate_extension(file_path):
                raise PermissionError(f"File type not allowed: {file_path.suffix}")
            
            # Save file
            async with aiofiles.open(file_path, 'wb') as f:
                content = await file_data.read()
                if len(content) > self.max_file_size:
                    raise PermissionError("File too large")
                await f.write(content)
            
            return {
                "filename": filename,
                "path": str(file_path),
                "size": file_path.stat().st_size,
                "mime_type": mimetypes.guess_type(str(file_path))[0]
            }
            
        except Exception as e:
            logger.error(f"Error uploading file: {str(e)}")
            raise
    
    async def list_files(self, directory: str = None) -> List[Dict[str, Any]]:
        """List files in directory"""
        try:
            if directory:
                path = self._validate_path(directory)
            else:
                path = self.base_dir
            
            files = []
            for item in path.iterdir():
                if item.is_file() and self._validate_extension(item):
                    files.append({
                        "name": item.name,
                        "path": str(item),
                        "size": item.stat().st_size,
                        "modified": item.stat().st_mtime,
                        "mime_type": mimetypes.guess_type(str(item))[0]
                    })
            
            return files
            
        except Exception as e:
            logger.error(f"Error listing files: {str(e)}")
            raise
    
    async def delete_file(self, file_path: str) -> Dict[str, Any]:
        """Securely delete file"""
        try:
            path = self._validate_path(file_path)
            
            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            path.unlink()
            
            return {
                "path": str(path),
                "deleted": True
            }
            
        except Exception as e:
            logger.error(f"Error deleting file {file_path}: {str(e)}")
            raise
