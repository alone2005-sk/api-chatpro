import asyncio
import json
import logging
import os
import shutil
import subprocess
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import hashlib
import re

from core.config import get_settings
from core.database import DatabaseManager
from core.logger import get_logger
from services.llm_orchestrator import LLMOrchestrator

logger = get_logger(__name__)

class ProjectGenerationService:
    """Service for automatic project generation with professional code"""
    
    def __init__(self, llm_orchestrator: LLMOrchestrator, db_manager: DatabaseManager):
        self.settings = get_settings()
        self.llm_orchestrator = llm_orchestrator
        self.db_manager = db_manager
        
        # Project templates
        self.project_templates = {
            "web_app": {
                "name": "Web Application",
                "description": "Full-stack web application with frontend and backend",
                "tech_stacks": {
                    "react_node": {
                        "frontend": ["react", "typescript", "tailwindcss", "vite"],
                        "backend": ["node.js", "express", "typescript", "prisma"],
                        "database": ["postgresql", "redis"],
                        "tools": ["eslint", "prettier", "jest"]
                    },
                    "vue_python": {
                        "frontend": ["vue3", "typescript", "pinia", "vite"],
                        "backend": ["python", "fastapi", "sqlalchemy", "pydantic"],
                        "database": ["postgresql", "redis"],
                        "tools": ["pytest", "black", "mypy"]
                    }
                },
                "structure": {
                    "frontend": ["src/components", "src/pages", "src/hooks", "src/utils", "public"],
                    "backend": ["api", "models", "services", "middleware", "utils"],
                    "database": ["migrations", "seeds"],
                    "docs": ["README.md", "API.md", "DEPLOYMENT.md"]
                }
            },
            "api_service": {
                "name": "API Service",
                "description": "RESTful API service with database integration",
                "tech_stacks": {
                    "fastapi": {
                        "framework": "fastapi",
                        "database": "postgresql",
                        "auth": "jwt",
                        "docs": "swagger",
                        "testing": "pytest"
                    },
                    "express": {
                        "framework": "express",
                        "database": "mongodb",
                        "auth": "passport",
                        "docs": "swagger",
                        "testing": "jest"
                    }
                },
                "structure": {
                    "api": ["routes", "controllers", "middleware"],
                    "models": ["schemas", "validators"],
                    "services": ["auth", "database", "email"],
                    "utils": ["helpers", "constants"],
                    "tests": ["unit", "integration"],
                    "docs": ["openapi.json"]
                }
            },
            "mobile_app": {
                "name": "Mobile Application",
                "description": "Cross-platform mobile application",
                "tech_stacks": {
                    "react_native": {
                        "framework": "react-native",
                        "navigation": "react-navigation",
                        "state": "redux-toolkit",
                        "ui": "native-base",
                        "testing": "jest"
                    },
                    "flutter": {
                        "framework": "flutter",
                        "state": "bloc",
                        "navigation": "go_router",
                        "ui": "material",
                        "testing": "flutter_test"
                    }
                },
                "structure": {
                    "src": ["screens", "components", "services", "utils", "models"],
                    "assets": ["images", "fonts", "icons"],
                    "tests": ["unit", "widget", "integration"]
                }
            },
            "data_analysis": {
                "name": "Data Analysis Project",
                "description": "Data analysis and visualization project",
                "tech_stacks": {
                    "python": {
                        "analysis": ["pandas", "numpy", "scipy"],
                        "visualization": ["matplotlib", "seaborn", "plotly"],
                        "ml": ["scikit-learn", "tensorflow", "pytorch"],
                        "notebook": ["jupyter", "ipython"]
                    }
                },
                "structure": {
                    "notebooks": ["exploratory", "analysis", "modeling"],
                    "data": ["raw", "processed", "external"],
                    "src": ["preprocessing", "analysis", "visualization", "models"],
                    "reports": ["figures", "reports"]
                }
            }
        }
        
        # Feature modules
        self.feature_modules = {
            "authentication": {
                "description": "User authentication and authorization",
                "files": ["auth.py", "middleware.py", "models.py", "routes.py"],
                "dependencies": ["bcrypt", "jwt", "passlib"],
                "endpoints": ["/login", "/register", "/logout", "/refresh"]
            },
            "user_management": {
                "description": "User profile and management system",
                "files": ["users.py", "profiles.py", "admin.py"],
                "dependencies": ["sqlalchemy", "pydantic"],
                "endpoints": ["/users", "/profile", "/admin/users"]
            },
            "file_upload": {
                "description": "File upload and management",
                "files": ["upload.py", "storage.py", "validation.py"],
                "dependencies": ["pillow", "boto3", "aiofiles"],
                "endpoints": ["/upload", "/files/{id}", "/download/{id}"]
            },
            "payment": {
                "description": "Payment processing integration",
                "files": ["payment.py", "stripe_handler.py", "webhooks.py"],
                "dependencies": ["stripe", "requests"],
                "endpoints": ["/payment/create", "/payment/confirm", "/webhooks/stripe"]
            },
            "notifications": {
                "description": "Notification system",
                "files": ["notifications.py", "email.py", "push.py"],
                "dependencies": ["celery", "redis", "sendgrid"],
                "endpoints": ["/notifications", "/notifications/send"]
            }
        }
        
        # Project detection patterns
        self.detection_patterns = {
            "web_app": [
                r"web.*app", r"website", r"dashboard", r"admin.*panel",
                r"e-commerce", r"blog", r"portfolio", r"landing.*page"
            ],
            "api_service": [
                r"api", r"backend", r"service", r"microservice",
                r"rest.*api", r"graphql", r"server"
            ],
            "mobile_app": [
                r"mobile.*app", r"react.*native", r"flutter",
                r"ios.*app", r"android.*app", r"cross.*platform"
            ],
            "data_analysis": [
                r"data.*analysis", r"data.*science", r"machine.*learning",
                r"visualization", r"analytics", r"dashboard.*data"
            ]
        }
        
        # Active projects
        self.active_projects = {}
    
    async def initialize(self):
        """Initialize the project generation service"""
        try:
            logger.info("Initializing Project Generation Service...")
            
            # Create directories
            Path(self.settings.PROJECT_DATA_DIR).mkdir(exist_ok=True)
            Path(self.settings.TEMP_DIR).mkdir(exist_ok=True)
            
            logger.info("✅ Project Generation Service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Project Generation Service: {str(e)}")
            raise
    
    async def detect_and_create_project(
        self,
        user_message: str,
        chat_id: str,
        user_id: str = "anonymous"
    ) -> Dict[str, Any]:
        """Detect project creation intent and generate project"""
        try:
            logger.info(f"Analyzing message for project intent: {user_message[:100]}...")
            
            # Detect project type
            project_detection = await self._detect_project_type(user_message)
            
            if not project_detection.get("is_project_request"):
                return {
                    "is_project": False,
                    "suggestions": [
                        "Create a React e-commerce website with user authentication",
                        "Build a FastAPI backend service with PostgreSQL database",
                        "Generate a Flutter mobile app for task management",
                        "Create a Python data analysis project with visualizations",
                        "Build a Node.js API with MongoDB integration"
                    ]
                }
            
            # Extract project requirements
            requirements = await self._extract_project_requirements(
                user_message, project_detection["project_type"]
            )
            
            # Create project in database
            project = await self.db_manager.create_project(
                chat_id=chat_id,
                name=requirements.get("name", f"Generated {project_detection['project_type'].title()}"),
                description=user_message,
                project_type=project_detection["project_type"],
                meta={  # Changed from metadata to meta to match database schema
                    "requirements": requirements,
                    "detection_confidence": project_detection.get("confidence", 0.0),
                    "user_id": user_id
                }
            )
            
            # Start project generation in background
            asyncio.create_task(self._generate_project_async(project.id, requirements))
            
            return {
                "is_project": True,
                "project_id": project.id,
                "project_type": project_detection["project_type"],
                "project_name": project.name,
                "requirements": requirements,
                "estimated_time": self._estimate_generation_time(requirements),
                "status": "initializing"
            }
            
        except Exception as e:
            logger.error(f"Project detection and creation failed: {str(e)}")
            return {
                "is_project": False,
                "error": str(e)
            }
    
    async def _detect_project_type(self, message: str) -> Dict[str, Any]:
        """Detect project type from user message"""
        try:
            message_lower = message.lower()
            
            # Pattern matching
            type_scores = {}
            for project_type, patterns in self.detection_patterns.items():
                score = sum(1 for pattern in patterns if re.search(pattern, message_lower))
                if score > 0:
                    type_scores[project_type] = score / len(patterns)
            
            # LLM-based detection for subtle requests
            if not type_scores:
                llm_detection = await self._llm_project_detection(message)
                if llm_detection.get("is_project") and llm_detection.get("confidence", 0) > 0.6:
                    type_scores[llm_detection.get("project_type", "web_app")] = llm_detection.get("confidence", 0.7)
            
            if not type_scores:
                return {"is_project_request": False}
            
            best_type = max(type_scores.items(), key=lambda x: x[1])
            
            return {
                "is_project_request": True,
                "project_type": best_type[0],
                "confidence": best_type[1],
                "all_scores": type_scores
            }
            
        except Exception as e:
            logger.error(f"Project type detection failed: {str(e)}")
            return {"is_project_request": False, "error": str(e)}
    
    async def _llm_project_detection(self, message: str) -> Dict[str, Any]:
        """Use LLM for project detection"""
        try:
            detection_prompt = f"""
            Analyze this message to determine if the user wants to create a software project:
            
            Message: "{message}"
            
            Project types available:
            - web_app: Websites, web applications, dashboards, e-commerce
            - api_service: REST APIs, GraphQL APIs, backend services
            - mobile_app: Mobile applications, React Native, Flutter
            - data_analysis: Data analysis, ML projects, visualizations
            
            Respond in JSON format:
            {{
                "is_project": boolean,
                "project_type": "type",
                "confidence": float (0.0-1.0),
                "reasoning": "explanation"
            }}
            """
            
            response = await self.llm_orchestrator.generate_response(
                prompt=detection_prompt,
                task_type="analytical",
                context={"project_detection": True}
            )
            
            try:
                return json.loads(response.get("text", "{}"))
            except json.JSONDecodeError:
                return {"is_project": False, "confidence": 0.0}
                
        except Exception as e:
            logger.error(f"LLM project detection failed: {str(e)}")
            return {"is_project": False, "confidence": 0.0}
    
    async def _extract_project_requirements(
        self,
        message: str,
        project_type: str
    ) -> Dict[str, Any]:
        """Extract detailed project requirements"""
        try:
            template = self.project_templates.get(project_type, {})
            
            extraction_prompt = f"""
            Extract detailed requirements for this {project_type} project:
            
            User Request: "{message}"
            Available Tech Stacks: {json.dumps(template.get('tech_stacks', {}), indent=2)}
            
            Extract and determine:
            1. Project name (generate if not specified)
            2. Main features and functionality
            3. Best technology stack from available options
            4. Database requirements
            5. Authentication needs
            6. UI/UX preferences
            7. Deployment preferences
            8. Additional features mentioned
            
            Return as JSON with structured requirements.
            """
            
            response = await self.llm_orchestrator.generate_response(
                prompt=extraction_prompt,
                task_type="analytical",
                context={"requirement_extraction": True}
            )
            
            try:
                requirements = json.loads(response.get("text", "{}"))
                
                # Add defaults
                requirements.setdefault("name", f"Generated {project_type.title()}")
                requirements.setdefault("tech_stack", list(template.get("tech_stacks", {}).keys())[0] if template.get("tech_stacks") else "default")
                requirements.setdefault("features", [])
                requirements.setdefault("complexity", "medium")
                
                return requirements
                
            except json.JSONDecodeError:
                return {
                    "name": f"Generated {project_type.title()}",
                    "description": message,
                    "tech_stack": "default",
                    "features": [],
                    "complexity": "medium"
                }
                
        except Exception as e:
            logger.error(f"Requirements extraction failed: {str(e)}")
            return {"name": f"Generated {project_type.title()}", "description": message}
    
    async def _generate_project_async(self, project_id: str, requirements: Dict[str, Any]):
        """Generate project asynchronously with real-time updates"""
        try:
            logger.info(f"Starting project generation for {project_id}")
            self.active_projects[project_id] = asyncio.current_task()
            
            # Update status
            await self.db_manager.update_project(project_id, status="generating", progress=5)
            
            # Phase 1: Create project structure
            await self._update_project_status(project_id, "Creating project structure...", 10)
            project_path = await self._create_project_structure(project_id, requirements)
            
            # Phase 2: Generate core files
            await self._update_project_status(project_id, "Generating core files...", 25)
            core_files = await self._generate_core_files(project_path, requirements)
            
            # Phase 3: Generate feature modules
            await self._update_project_status(project_id, "Implementing features...", 50)
            feature_files = await self._generate_feature_modules(project_path, requirements)
            
            # Phase 4: Generate configuration files
            await self._update_project_status(project_id, "Creating configuration...", 70)
            config_files = await self._generate_config_files(project_path, requirements)
            
            # Phase 5: Generate documentation
            await self._update_project_status(project_id, "Writing documentation...", 85)
            docs = await self._generate_documentation(project_path, requirements)
            
            # Phase 6: Create deployment package
            await self._update_project_status(project_id, "Creating deployment package...", 95)
            zip_path = await self._create_project_zip(project_path, project_id)
            
            # Phase 7: Finalize
            await self._update_project_status(project_id, "Project completed!", 100)
            
            # Update project in database
            await self.db_manager.update_project(
                project_id,
                status="completed",
                progress=100,
                file_path=str(project_path),
                zip_path=str(zip_path),
                completed_at=datetime.utcnow(),
                meta={"completed_at": datetime.utcnow().isoformat()}
            )
            
            # Save generated files to database
            all_files = core_files + feature_files + config_files + docs
            for file_info in all_files:
                await self.db_manager.save_project_file(
                    project_id=project_id,
                    filename=file_info["filename"],
                    file_path=file_info["file_path"],
                    file_type=file_info["file_type"],
                    size=file_info.get("size", 0),
                    content_hash=file_info.get("content_hash", "")
                )
            
            logger.info(f"Project {project_id} generated successfully with {len(all_files)} files")
            
        except Exception as e:
            logger.error(f"Project generation failed for {project_id}: {str(e)}")
            await self.db_manager.update_project(
                project_id,
                status="failed",
                meta={"error": str(e)}
            )
        finally:
            if project_id in self.active_projects:
                del self.active_projects[project_id]
    
    async def _create_project_structure(
        self,
        project_id: str,
        requirements: Dict[str, Any]
    ) -> Path:
        """Create project directory structure"""
        try:
            project_path = Path(self.settings.PROJECT_DATA_DIR) / project_id
            project_path.mkdir(parents=True, exist_ok=True)
            
            project_type = requirements.get("project_type", "web_app")
            template = self.project_templates.get(project_type, {})
            structure = template.get("structure", {})
            
            # Create directory structure
            for category, dirs in structure.items():
                category_path = project_path / category
                category_path.mkdir(exist_ok=True)
                
                if isinstance(dirs, list):
                    for dir_name in dirs:
                        (category_path / dir_name).mkdir(parents=True, exist_ok=True)
                elif isinstance(dirs, dict):
                    for dir_name, subdirs in dirs.items():
                        dir_path = category_path / dir_name
                        dir_path.mkdir(exist_ok=True)
                        
                        if isinstance(subdirs, list):
                            for subdir in subdirs:
                                (dir_path / subdir).mkdir(parents=True, exist_ok=True)
            
            return project_path
            
        except Exception as e:
            logger.error(f"Project structure creation failed: {str(e)}")
            raise
    
    async def _generate_core_files(
        self,
        project_path: Path,
        requirements: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate core project files"""
        try:
            generated_files = []
            project_type = requirements.get("project_type", "web_app")
            tech_stack = requirements.get("tech_stack", "default")
            
            # Generate main application file
            main_file = await self._generate_main_file(project_path, requirements)
            if main_file:
                generated_files.append(main_file)
            
            # Generate package.json or requirements.txt
            deps_file = await self._generate_dependencies_file(project_path, requirements)
            if deps_file:
                generated_files.append(deps_file)
            
            # Generate README.md
            readme_file = await self._generate_readme(project_path, requirements)
            if readme_file:
                generated_files.append(readme_file)
            
            # Generate .gitignore
            gitignore_file = await self._generate_gitignore(project_path, requirements)
            if gitignore_file:
                generated_files.append(gitignore_file)
            
            return generated_files
            
        except Exception as e:
            logger.error(f"Core files generation failed: {str(e)}")
            return []
    
    async def _generate_main_file(
        self,
        project_path: Path,
        requirements: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Generate main application file"""
        try:
            project_type = requirements.get("project_type", "web_app")
            tech_stack = requirements.get("tech_stack", "default")
            
            generation_prompt = f"""
            Generate the main application file for a {project_type} project using {tech_stack}.
            
            Requirements: {json.dumps(requirements, indent=2)}
            
            Create a professional, production-ready main file that includes:
            1. Proper imports and dependencies
            2. Application initialization
            3. Basic routing/structure
            4. Error handling
            5. Configuration setup
            6. Comments and documentation
            
            Return only the code without explanations.
            """
            
            response = await self.llm_orchestrator.generate_response(
                prompt=generation_prompt,
                task_type="code_generation",
                context={"file_generation": True}
            )
            
            # Determine file extension and name
            if "python" in tech_stack.lower() or "fastapi" in tech_stack.lower():
                filename = "main.py"
                file_type = "python"
            elif "node" in tech_stack.lower() or "express" in tech_stack.lower():
                filename = "app.js"
                file_type = "javascript"
            elif "react" in tech_stack.lower():
                filename = "App.tsx"
                file_type = "typescript"
            else:
                filename = "main.py"
                file_type = "python"
            
            file_path = project_path / filename
            content = response.get("text", "")
            
            # Write file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return {
                "filename": filename,
                "file_path": str(file_path),
                "file_type": file_type,
                "size": len(content),
                "content_hash": hashlib.md5(content.encode()).hexdigest()
            }
            
        except Exception as e:
            logger.error(f"Main file generation failed: {str(e)}")
            return None
    
    async def _generate_dependencies_file(
        self,
        project_path: Path,
        requirements: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Generate dependencies file (package.json, requirements.txt, etc.)"""
        try:
            tech_stack = requirements.get("tech_stack", "default")
            project_type = requirements.get("project_type", "web_app")
            
            generation_prompt = f"""
            Generate a dependencies file for a {project_type} project using {tech_stack}.
            
            Requirements: {json.dumps(requirements, indent=2)}
            
            Include all necessary dependencies for:
            1. Core framework
            2. Database integration
            3. Authentication (if needed)
            4. Testing framework
            5. Development tools
            6. Production dependencies
            
            Generate the appropriate file format (package.json, requirements.txt, etc.)
            """
            
            response = await self.llm_orchestrator.generate_response(
                prompt=generation_prompt,
                task_type="code_generation",
                context={"dependencies_generation": True}
            )
            
            # Determine file type
            if "python" in tech_stack.lower():
                filename = "requirements.txt"
                file_type = "text"
            elif "node" in tech_stack.lower() or "react" in tech_stack.lower():
                filename = "package.json"
                file_type = "json"
            elif "flutter" in tech_stack.lower():
                filename = "pubspec.yaml"
                file_type = "yaml"
            else:
                filename = "requirements.txt"
                file_type = "text"
            
            file_path = project_path / filename
            content = response.get("text", "")
            
            # Write file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return {
                "filename": filename,
                "file_path": str(file_path),
                "file_type": file_type,
                "size": len(content),
                "content_hash": hashlib.md5(content.encode()).hexdigest()
            }
            
        except Exception as e:
            logger.error(f"Dependencies file generation failed: {str(e)}")
            return None
    
    async def _generate_feature_modules(
        self,
        project_path: Path,
        requirements: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate feature modules based on requirements"""
        try:
            generated_files = []
            features = requirements.get("features", [])
            
            # Map features to modules
            feature_modules = []
            for feature in features:
                feature_lower = feature.lower()
                for module_name, module_info in self.feature_modules.items():
                    if any(keyword in feature_lower for keyword in module_name.split("_")):
                        feature_modules.append((module_name, module_info))
                        break
            
            # Generate each feature module
            for module_name, module_info in feature_modules:
                module_files = await self._generate_single_feature_module(
                    project_path, module_name, module_info, requirements
                )
                generated_files.extend(module_files)
            
            return generated_files
            
        except Exception as e:
            logger.error(f"Feature modules generation failed: {str(e)}")
            return []
    
    async def _generate_single_feature_module(
        self,
        project_path: Path,
        module_name: str,
        module_info: Dict[str, Any],
        requirements: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate a single feature module"""
        try:
            generated_files = []
            tech_stack = requirements.get("tech_stack", "default")
            
            for filename in module_info.get("files", []):
                generation_prompt = f"""
                Generate {filename} for the {module_name} feature module.
                
                Module Description: {module_info.get('description', '')}
                Tech Stack: {tech_stack}
                Dependencies: {module_info.get('dependencies', [])}
                Endpoints: {module_info.get('endpoints', [])}
                
                Project Requirements: {json.dumps(requirements, indent=2)}
                
                Create professional, production-ready code that includes:
                1. Proper error handling
                2. Input validation
                3. Security best practices
                4. Comprehensive logging
                5. Type hints (if applicable)
                6. Documentation
                7. Unit test considerations
                
                Return only the code without explanations.
                """
                
                response = await self.llm_orchestrator.generate_response(
                    prompt=generation_prompt,
                    task_type="code_generation",
                    context={"feature_generation": True, "module": module_name}
                )
                
                # Determine file path
                if "python" in tech_stack.lower():
                    file_path = project_path / "backend" / "services" / filename
                elif "node" in tech_stack.lower():
                    file_path = project_path / "backend" / "services" / filename.replace(".py", ".js")
                else:
                    file_path = project_path / "src" / "services" / filename
                
                # Ensure directory exists
                file_path.parent.mkdir(parents=True, exist_ok=True)
                
                content = response.get("text", "")
                
                # Write file
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                generated_files.append({
                    "filename": file_path.name,
                    "file_path": str(file_path),
                    "file_type": self._determine_file_type(file_path.name),
                    "size": len(content),
                    "content_hash": hashlib.md5(content.encode()).hexdigest(),
                    "module": module_name
                })
            
            return generated_files
            
        except Exception as e:
            logger.error(f"Single feature module generation failed for {module_name}: {str(e)}")
            return []
    
    async def _generate_readme(
        self,
        project_path: Path,
        requirements: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Generate README.md file"""
        try:
            generation_prompt = f"""
            Generate a professional README.md file for a project with the following requirements:
            
            Requirements: {json.dumps(requirements, indent=2)}
            
            Include:
            1. Project title and description
            2. Installation instructions
            3. Usage examples
            4. Technology stack
            5. Features list
            6. Development setup
            7. Deployment instructions
            8. Contributing guidelines
            
            Use markdown format and return only the content without explanations.
            """
            
            response = await self.llm_orchestrator.generate_response(
                prompt=generation_prompt,
                task_type="documentation",
                context={"readme_generation": True}
            )
            
            file_path = project_path / "README.md"
            content = response.get("text", "")
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return {
                "filename": "README.md",
                "file_path": str(file_path),
                "file_type": "markdown",
                "size": len(content),
                "content_hash": hashlib.md5(content.encode()).hexdigest()
            }
            
        except Exception as e:
            logger.error(f"README generation failed: {str(e)}")
            return None
    
    async def _generate_gitignore(
        self,
        project_path: Path,
        requirements: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Generate .gitignore file"""
        try:
            tech_stack = requirements.get("tech_stack", "default")
            
            generation_prompt = f"""
            Generate a .gitignore file for a project using {tech_stack}.
            
            Requirements: {json.dumps(requirements, indent=2)}
            
            Include ignore patterns for:
            1. Build artifacts
            2. Dependency directories
            3. Environment files
            4. Log files
            5. IDE-specific files
            6. Temporary files
            
            Return only the content without explanations.
            """
            
            response = await self.llm_orchestrator.generate_response(
                prompt=generation_prompt,
                task_type="code_generation",
                context={"gitignore_generation": True}
            )
            
            file_path = project_path / ".gitignore"
            content = response.get("text", "")
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return {
                "filename": ".gitignore",
                "file_path": str(file_path),
                "file_type": "text",
                "size": len(content),
                "content_hash": hashlib.md5(content.encode()).hexdigest()
            }
            
        except Exception as e:
            logger.error(f".gitignore generation failed: {str(e)}")
            return None
    
    async def _generate_config_files(
        self,
        project_path: Path,
        requirements: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate configuration files"""
        try:
            generated_files = []
            tech_stack = requirements.get("tech_stack", "default")
            
            config_files = []
            if "python" in tech_stack.lower():
                config_files = ["Dockerfile", ".env.example", "config.py"]
            elif "node" in tech_stack.lower():
                config_files = ["Dockerfile", ".env.example", "config.js"]
            elif "react" in tech_stack.lower():
                config_files = ["vite.config.ts", ".env.example", "tsconfig.json"]
            elif "flutter" in tech_stack.lower():
                config_files = ["analysis_options.yaml", "pubspec.yaml"]
            
            for filename in config_files:
                generation_prompt = f"""
                Generate {filename} for a project using {tech_stack}.
                
                Requirements: {json.dumps(requirements, indent=2)}
                
                Create a production-ready configuration file that includes:
                1. Environment variables setup
                2. Build configurations
                3. Development/production settings
                4. Security best practices
                5. Comments explaining configuration
                
                Return only the content without explanations.
                """
                
                response = await self.llm_orchestrator.generate_response(
                    prompt=generation_prompt,
                    task_type="code_generation",
                    context={"config_generation": True}
                )
                
                file_path = project_path / filename
                content = response.get("text", "")
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                generated_files.append({
                    "filename": filename,
                    "file_path": str(file_path),
                    "file_type": self._determine_file_type(filename),
                    "size": len(content),
                    "content_hash": hashlib.md5(content.encode()).hexdigest()
                })
            
            return generated_files
            
        except Exception as e:
            logger.error(f"Config files generation failed: {str(e)}")
            return []
    
    async def _generate_documentation(
        self,
        project_path: Path,
        requirements: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate project documentation files"""
        try:
            generated_files = []
            project_type = requirements.get("project_type", "web_app")
            
            doc_files = self.project_templates.get(project_type, {}).get("structure", {}).get("docs", [])
            
            for filename in doc_files:
                if filename == "README.md":  # Skip README as it's generated separately
                    continue
                    
                generation_prompt = f"""
                Generate {filename} documentation for a {project_type} project.
                
                Requirements: {json.dumps(requirements, indent=2)}
                
                Include:
                1. Detailed documentation for the specific aspect (API, deployment, etc.)
                2. Code examples where relevant
                3. Best practices
                4. Clear structure and formatting
                5. Professional tone
                
                Use markdown format unless specified otherwise.
                Return only the content without explanations.
                """
                
                response = await self.llm_orchestrator.generate_response(
                    prompt=generation_prompt,
                    task_type="documentation",
                    context={"doc_generation": True, "filename": filename}
                )
                
                file_path = project_path / "docs" / filename
                file_path.parent.mkdir(exist_ok=True)
                content = response.get("text", "")
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                generated_files.append({
                    "filename": filename,
                    "file_path": str(file_path),
                    "file_type": self._determine_file_type(filename),
                    "size": len(content),
                    "content_hash": hashlib.md5(content.encode()).hexdigest()
                })
            
            return generated_files
            
        except Exception as e:
            logger.error(f"Documentation generation failed: {str(e)}")
            return []
    
    async def _create_project_zip(
        self,
        project_path: Path,
        project_id: str
    ) -> Path:
        """Create a zip file of the project"""
        try:
            zip_path = Path(self.settings.TEMP_DIR) / f"{project_id}.zip"
            
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, _, files in os.walk(project_path):
                    for file in files:
                        file_path = Path(root) / file
                        arcname = str(file_path.relative_to(project_path))
                        zipf.write(file_path, arcname)
            
            return zip_path
            
        except Exception as e:
            logger.error(f"Project zip creation failed: {str(e)}")
            raise
    
    def _determine_file_type(self, filename: str) -> str:
        """Determine file type from filename"""
        ext_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.jsx': 'jsx',
            '.tsx': 'tsx',
            '.html': 'html',
            '.css': 'css',
            '.json': 'json',
            '.md': 'markdown',
            '.yml': 'yaml',
            '.yaml': 'yaml',
            '.txt': 'text'
        }
        
        ext = Path(filename).suffix.lower()
        return ext_map.get(ext, 'text')
    
    def _estimate_generation_time(self, requirements: Dict[str, Any]) -> str:
        """Estimate project generation time"""
        complexity = requirements.get("complexity", "medium")
        features_count = len(requirements.get("features", []))
        
        base_time = {
            "low": 2,
            "medium": 4,
            "high": 8
        }.get(complexity, 4)
        
        feature_time = features_count * 0.5
        total_minutes = base_time + feature_time
        
        return f"{int(total_minutes)}-{int(total_minutes * 1.5)} minutes"
    
    async def _update_project_status(self, project_id: str, message: str, progress: int):
        """Update project status in database"""
        try:
            await self.db_manager.update_project(
                project_id,
                progress=progress,
                meta={"current_step": message, "updated_at": datetime.utcnow().isoformat()}
            )
            
            logger.info(f"Project {project_id}: {message} ({progress}%)")
            
        except Exception as e:
            logger.error(f"Status update failed for project {project_id}: {str(e)}")
    
    async def get_project_status(self, project_id: str) -> Dict[str, Any]:
        """Get current project status"""
        try:
            project = await self.db_manager.get_project(project_id)
            
            if not project:
                return {"error": "Project not found"}
            
            return {
                "project_id": project.id,
                "name": project.name,
                "status": project.status,
                "progress": project.progress,
                "created_at": project.created_at.isoformat(),
                "completed_at": project.completed_at.isoformat() if project.completed_at else None,
                "file_path": project.file_path,
                "zip_path": project.zip_path,
                "meta": project.meta
            }
            
        except Exception as e:
            logger.error(f"Get project status failed: {str(e)}")
            return {"error": str(e)}
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for project generation service"""
        try:
            return {
                "status": "healthy",
                "project_templates": len(self.project_templates),
                "feature_modules": len(self.feature_modules),
                "active_projects": len(self.active_projects),
                "project_data_dir": str(Path(self.settings.PROJECT_DATA_DIR).absolute()),
                "temp_dir": str(Path(self.settings.TEMP_DIR).absolute())
            }
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {"status": "unhealthy", "error": str(e)}
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            # Cancel active project generations
            for project_id in list(self.active_projects.keys()):
                if project_id in self.active_projects:
                    task = self.active_projects[project_id]
                    if not task.done():
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
                    del self.active_projects[project_id]
            
            logger.info("Project Generation Service cleanup completed")
            
        except Exception as e:
            logger.error(f"Project Generation Service cleanup failed: {str(e)}")