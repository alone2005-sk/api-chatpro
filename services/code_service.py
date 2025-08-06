import asyncio
import ast
import re
import json
import subprocess
import tempfile
import os
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from pathlib import Path
import hashlib

# Code analysis and formatting
import black
import isort
from pylint import lint
from pylint.reporters.text import TextReporter
import autopep8

# Security analysis
import bandit
from bandit.core import manager as bandit_manager

# Documentation generation
import pydoc

from core.logger import get_logger
from core.config import get_settings

logger = get_logger(__name__)
settings = get_settings()

class CodeService:
    """Advanced code generation, analysis, and optimization service"""
    import tempfile

class CodeAssistant:
    def __init__(self, settings=None):
        self.settings = settings
        self.temp_dir = tempfile.mkdtemp()

        # Supported language config
        self.supported_languages = {
            'python': self._build_config(
                extensions=['.py'],
                analyzer=self._analyze_python,
                formatter=self._format_python,
                linter=self._lint_python,
                runner=self._run_python,
                templates={
                    'function': self._generate_python_function,
                    'class': self._generate_python_class,
                    'api': self._generate_python_api,
                    'script': self._generate_python_script,
                    'test': self._generate_python_test
                }
            ),
            'javascript': self._build_config(
                extensions=['.js', '.jsx'],
                analyzer=self._analyze_javascript,
                formatter=self._format_javascript,
                linter=self._lint_javascript,
                runner=self._run_javascript,
                templates={
                    'function': self._generate_js_function,
                    'class': self._generate_js_class,
                    'api': self._generate_js_api,
                    'component': self._generate_js_component,
                    'test': self._generate_js_test
                }
            ),
            'typescript': self._build_config(
                extensions=['.ts', '.tsx'],
                analyzer=self._analyze_typescript,
                formatter=self._format_typescript,
                linter=self._lint_typescript,
                runner=self._run_typescript,
                templates={
                    'function': self._generate_ts_function,
                    'class': self._generate_ts_class,
                    'api': self._generate_ts_api,
                    'component': self._generate_ts_component,
                    'test': self._generate_ts_test
                }
            ),
            'html': self._build_config(
                extensions=['.html', '.htm'],
                analyzer=self._analyze_html,
                formatter=self._format_html,
                linter=self._lint_html,
                runner=None,
                templates={
                    'template': self._generate_html_template
                }
            ),
            'css': self._build_config(
                extensions=['.css', '.scss', '.sass'],
                analyzer=self._analyze_css,
                formatter=self._format_css,
                linter=self._lint_css,
                runner=None,
                templates={
                    'style': self._generate_css_template
                }
            ),
            'sql': self._build_config(
                extensions=['.sql'],
                analyzer=self._analyze_sql,
                formatter=self._format_sql,
                linter=self._lint_sql,
                runner=self._run_sql,
                templates={
                    'query': self._generate_sql_query
                }
            )
        }
    # services/code_service.py

from typing import Dict, Callable

class CodeService:
    def __init__(self, settings):
        self.settings = settings
        self.runners: Dict[str, Callable[[str], str]] = {
            'python': self._run_python,
            'javascript': self._run_javascript,
            'bash': self._run_bash,
            'cpp': self._run_cpp,
            'java': self._run_java,
            'c': self._run_c,
            'go': self._run_go,
            'rust': self._run_rust,
            'ruby': self._run_ruby,
        }

    def _run_python(self, code: str) -> str:
        return f"Running Python: {code}"

    def _run_javascript(self, code: str) -> str:
        return f"Running JavaScript: {code}"

    def _run_bash(self, code: str) -> str:
        return f"Running Bash: {code}"

    def _run_cpp(self, code: str) -> str:
        return f"Running C++: {code}"

    def _run_java(self, code: str) -> str:
        return f"Running Java: {code}"

    def _run_c(self, code: str) -> str:
        return f"Running C: {code}"

    def _run_go(self, code: str) -> str:
        return f"Running Go: {code}"

    def _run_rust(self, code: str) -> str:
        return f"Running Rust: {code}"

    def _run_ruby(self, code: str) -> str:
        return f"Running Ruby: {code}"

    def execute(self, language: str, code: str) -> str:
        runner = self.runners.get(language)
        if not runner:
            raise ValueError(f"Unsupported language: {language}")
        return runner(code)

    def _build_config(self, extensions, analyzer, formatter, linter, runner, templates):
        return {
            'extensions': extensions,
            'analyzer': analyzer,
            'formatter': formatter,
            'linter': linter,
            'runner': runner,
            'templates': templates
        }

    def _analyze_javascript(self, code: str) -> dict:
        return {
        "summary": "JavaScript analysis is not implemented yet.",
        "functions": [],
        "variables": [],
        "comments": [],
    }
    def _run_javascript(self, code: str) -> str:
            try:
                result = subprocess.run(
                    ['node', '-e', code],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0:
                    return result.stdout.strip()
                else:
                    return f"Error: {result.stderr.strip()}"
            except Exception as e:
                return f"Execution error: {e}"

    def _lint_javascript(self, code: str) -> str:
            return "// JavaScript Linting not implemented\n" + code

    def _format_javascript(self, code: str) -> str:
        return code  # Currently just returns the original code
    async def generate_code(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Generate code based on requirements"""
        try:
            language = request.get('language', 'python').lower()
            code_type = request.get('type', 'function').lower()
            requirements = request.get('requirements', '')
            style = request.get('style', 'standard')
            
            if language not in self.supported_languages:
                return {'error': f'Unsupported language: {language}'}
            
            # Generate base code
            if language in self.code_templates and code_type in self.code_templates[language]:
                generator = self.code_templates[language][code_type]
                code = await generator(requirements, style)
            else:
                code = await self._generate_generic_code(language, code_type, requirements, style)
            
            if 'error' in code:
                return code
            
            # Analyze generated code
            analysis = await self.analyze_code(code['code'], language)
            
            # Format code
            formatted_code = await self.format_code(code['code'], language)
            
            result = {
                'success': True,
                'language': language,
                'type': code_type,
                'code': formatted_code.get('formatted_code', code['code']),
                'analysis': analysis,
                'metadata': {
                    'generated_at': datetime.utcnow().isoformat(),
                    'style': style,
                    'requirements': requirements,
                    'estimated_complexity': code.get('complexity', 'medium'),
                    'estimated_lines': len(code['code'].splitlines())
                }
            }
            
            # Add documentation if requested
            if request.get('include_docs', False):
                docs = await self._generate_documentation(code['code'], language)
                result['documentation'] = docs
            
            # Add tests if requested
            if request.get('include_tests', False):
                tests = await self._generate_tests(code['code'], language)
                result['tests'] = tests
            
            return result
            
        except Exception as e:
            logger.error(f"Code generation failed: {str(e)}")
            return {'error': f'Code generation failed: {str(e)}'}
    
    async def analyze_code(self, code: str, language: str) -> Dict[str, Any]:
        """Comprehensive code analysis"""
        try:
            if language not in self.supported_languages:
                return {'error': f'Unsupported language for analysis: {language}'}
            
            analyzer = self.supported_languages[language]['analyzer']
            analysis = await analyzer(code)
            
            # Add general metrics
            analysis['general_metrics'] = {
                'total_lines': len(code.splitlines()),
                'non_empty_lines': len([line for line in code.splitlines() if line.strip()]),
                'comment_lines': self._count_comment_lines(code, language),
                'code_to_comment_ratio': self._calculate_code_comment_ratio(code, language),
                'estimated_complexity': self._estimate_complexity(code, language)
            }
            
            # Security analysis
            if language == 'python':
                security_analysis = await self._security_analysis_python(code)
                analysis['security'] = security_analysis
            
            return analysis
            
        except Exception as e:
            logger.error(f"Code analysis failed: {str(e)}")
            return {'error': f'Code analysis failed: {str(e)}'}
    
    async def format_code(self, code: str, language: str, style: str = 'standard') -> Dict[str, Any]:
        """Format code according to language standards"""
        try:
            if language not in self.supported_languages:
                return {'error': f'Unsupported language for formatting: {language}'}
            
            formatter = self.supported_languages[language]['formatter']
            formatted = await formatter(code, style)
            
            return {
                'success': True,
                'original_code': code,
                'formatted_code': formatted['code'],
                'changes_made': formatted.get('changes', []),
                'style': style,
                'formatted_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Code formatting failed: {str(e)}")
            return {'error': f'Code formatting failed: {str(e)}'}
    
    async def lint_code(self, code: str, language: str) -> Dict[str, Any]:
        """Lint code and provide suggestions"""
        try:
            if language not in self.supported_languages:
                return {'error': f'Unsupported language for linting: {language}'}
            
            linter = self.supported_languages[language]['linter']
            if not linter:
                return {'error': f'Linting not available for {language}'}
            
            lint_results = await linter(code)
            
            return {
                'success': True,
                'issues': lint_results.get('issues', []),
                'warnings': lint_results.get('warnings', []),
                'errors': lint_results.get('errors', []),
                'score': lint_results.get('score', 0),
                'suggestions': lint_results.get('suggestions', []),
                'linted_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Code linting failed: {str(e)}")
            return {'error': f'Code linting failed: {str(e)}'}
    
    async def optimize_code(self, code: str, language: str, optimization_type: str = 'performance') -> Dict[str, Any]:
        """Optimize code for performance, readability, or size"""
        try:
            optimizations = []
            optimized_code = code
            
            if language == 'python':
                optimized_code, optimizations = await self._optimize_python(code, optimization_type)
            elif language == 'javascript':
                optimized_code, optimizations = await self._optimize_javascript(code, optimization_type)
            else:
                return {'error': f'Optimization not available for {language}'}
            
            # Analyze improvements
            original_analysis = await self.analyze_code(code, language)
            optimized_analysis = await self.analyze_code(optimized_code, language)
            
            return {
                'success': True,
                'original_code': code,
                'optimized_code': optimized_code,
                'optimizations_applied': optimizations,
                'optimization_type': optimization_type,
                'improvements': {
                    'original_complexity': original_analysis.get('general_metrics', {}).get('estimated_complexity'),
                    'optimized_complexity': optimized_analysis.get('general_metrics', {}).get('estimated_complexity'),
                    'line_reduction': len(code.splitlines()) - len(optimized_code.splitlines())
                },
                'optimized_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Code optimization failed: {str(e)}")
            return {'error': f'Code optimization failed: {str(e)}'}
    
    async def run_code(self, code: str, language: str, inputs: Optional[List[str]] = None) -> Dict[str, Any]:
        """Execute code safely in a sandboxed environment"""
        try:
            if language not in self.supported_languages:
                return {'error': f'Unsupported language for execution: {language}'}
            
            runner = self.supported_languages[language]['runner']
            if not runner:
                return {'error': f'Code execution not available for {language}'}
            
            # Create temporary file
            temp_file = os.path.join(self.temp_dir, f"code_{datetime.now().timestamp()}")
            
            execution_result = await runner(code, temp_file, inputs or [])
            
            return {
                'success': True,
                'output': execution_result.get('output', ''),
                'errors': execution_result.get('errors', ''),
                'return_code': execution_result.get('return_code', 0),
                'execution_time': execution_result.get('execution_time', 0),
                'executed_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Code execution failed: {str(e)}")
            return {'error': f'Code execution failed: {str(e)}'}
    
    # Python-specific methods
    async def _analyze_python(self, code: str) -> Dict[str, Any]:
        """Analyze Python code"""
        try:
            tree = ast.parse(code)
            
            analysis = {
                'functions': [],
                'classes': [],
                'imports': [],
                'variables': [],
                'complexity_metrics': {}
            }
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    analysis['functions'].append({
                        'name': node.name,
                        'args': [arg.arg for arg in node.args.args],
                        'line_number': node.lineno,
                        'docstring': ast.get_docstring(node),
                        'decorators': [d.id if hasattr(d, 'id') else str(d) for d in node.decorator_list]
                    })
                elif isinstance(node, ast.ClassDef):
                    analysis['classes'].append({
                        'name': node.name,
                        'line_number': node.lineno,
                        'docstring': ast.get_docstring(node),
                        'methods': [n.name for n in node.body if isinstance(n, ast.FunctionDef)],
                        'bases': [base.id if hasattr(base, 'id') else str(base) for base in node.bases]
                    })
                elif isinstance(node, ast.Import):
                    analysis['imports'].extend([alias.name for alias in node.names])
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    for alias in node.names:
                        analysis['imports'].append(f"{module}.{alias.name}")
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            analysis['variables'].append(target.id)
            
            # Calculate complexity metrics
            analysis['complexity_metrics'] = {
                'cyclomatic_complexity': self._calculate_cyclomatic_complexity(tree),
                'function_count': len(analysis['functions']),
                'class_count': len(analysis['classes']),
                'import_count': len(set(analysis['imports'])),
                'variable_count': len(set(analysis['variables']))
            }
            
            return analysis
            
        except SyntaxError as e:
            return {'error': f'Python syntax error: {str(e)}'}
        except Exception as e:
            return {'error': f'Python analysis failed: {str(e)}'}
    
    async def _format_python(self, code: str, style: str) -> Dict[str, Any]:
        """Format Python code"""
        try:
            # Use black for formatting
            formatted_code = black.format_str(code, mode=black.FileMode())
            
            # Sort imports
            formatted_code = isort.code(formatted_code)
            
            changes = []
            if code != formatted_code:
                changes.append("Applied Black formatting")
                changes.append("Sorted imports with isort")
            
            return {
                'code': formatted_code,
                'changes': changes
            }
            
        except Exception as e:
            # Fallback to autopep8
            try:
                formatted_code = autopep8.fix_code(code)
                return {
                    'code': formatted_code,
                    'changes': ['Applied autopep8 formatting']
                }
            except Exception as e2:
                return {'error': f'Python formatting failed: {str(e2)}'}
    
    async def _lint_python(self, code: str) -> Dict[str, Any]:
        """Lint Python code using pylint"""
        try:
            # Create temporary file
            temp_file = os.path.join(self.temp_dir, f"lint_{datetime.now().timestamp()}.py")
            with open(temp_file, 'w') as f:
                f.write(code)
            
            # Run pylint
            from io import StringIO
            pylint_output = StringIO()
            reporter = TextReporter(pylint_output)
            
            try:
                lint.Run([temp_file], reporter=reporter, exit=False)
                output = pylint_output.getvalue()
            except SystemExit:
                output = pylint_output.getvalue()
            
            # Parse pylint output
            issues = []
            warnings = []
            errors = []
            score = 0
            
            for line in output.split('\n'):
                if ':' in line and ('error' in line.lower() or 'warning' in line.lower()):
                    if 'error' in line.lower():
                        errors.append(line.strip())
                    else:
                        warnings.append(line.strip())
                    issues.append(line.strip())
                elif 'Your code has been rated at' in line:
                    try:
                        score = float(line.split('rated at ')[1].split('/')[0])
                    except:
                        pass
            
            # Cleanup
            os.remove(temp_file)
            
            return {
                'issues': issues,
                'warnings': warnings,
                'errors': errors,
                'score': score,
                'suggestions': self._generate_python_suggestions(issues)
            }
            
        except Exception as e:
            return {'error': f'Python linting failed: {str(e)}'}
    
    async def _run_python(self, code: str, temp_file: str, inputs: List[str]) -> Dict[str, Any]:
        """Run Python code"""
        try:
            # Write code to file
            python_file = f"{temp_file}.py"
            with open(python_file, 'w') as f:
                f.write(code)
            
            # Prepare input
            input_str = '\n'.join(inputs) if inputs else ''
            
            # Execute with timeout
            start_time = datetime.now()
            result = subprocess.run(
                ['python', python_file],
                input=input_str,
                capture_output=True,
                text=True,
                timeout=30  # 30 second timeout
            )
            end_time = datetime.now()
            
            # Cleanup
            os.remove(python_file)
            
            return {
                'output': result.stdout,
                'errors': result.stderr,
                'return_code': result.returncode,
                'execution_time': (end_time - start_time).total_seconds()
            }
            
        except subprocess.TimeoutExpired:
            return {'error': 'Code execution timed out (30s limit)'}
        except Exception as e:
            return {'error': f'Python execution failed: {str(e)}'}
    
    async def _security_analysis_python(self, code: str) -> Dict[str, Any]:
        """Security analysis for Python code"""
        try:
            # Create temporary file
            temp_file = os.path.join(self.temp_dir, f"security_{datetime.now().timestamp()}.py")
            with open(temp_file, 'w') as f:
                f.write(code)
            
            # Run bandit security analysis
            b_mgr = bandit_manager.BanditManager(bandit.config.BanditConfig(), 'file')
            b_mgr.discover_files([temp_file])
            b_mgr.run_tests()
            
            issues = []
            for issue in b_mgr.get_issue_list():
                issues.append({
                    'severity': issue.severity,
                    'confidence': issue.confidence,
                    'test': issue.test,
                    'text': issue.text,
                    'line_number': issue.lineno
                })
            
            # Cleanup
            os.remove(temp_file)
            
            return {
                'security_issues': issues,
                'risk_level': self._calculate_risk_level(issues),
                'recommendations': self._generate_security_recommendations(issues)
            }
            
        except Exception as e:
            return {'error': f'Security analysis failed: {str(e)}'}
    
    # Code generation methods
    async def _generate_python_function(self, requirements: str, style: str) -> Dict[str, Any]:
        """Generate Python function based on requirements"""
        try:
            # Parse requirements to extract function details
            func_name = self._extract_function_name(requirements) or "generated_function"
            params = self._extract_parameters(requirements)
            return_type = self._extract_return_type(requirements)
            
            # Generate function template
            code = f'''def {func_name}({", ".join(params)}):
    """
    {requirements}
    
    Args:
        {chr(10).join([f"{param}: Description of {param}" for param in params])}
    
    Returns:
        {return_type}: Description of return value
    """
    # TODO: Implement function logic based on requirements
    pass
'''
            
            return {
                'code': code,
                'complexity': 'low',
                'type': 'function'
            }
            
        except Exception as e:
            return {'error': f'Python function generation failed: {str(e)}'}
    
    async def _generate_python_class(self, requirements: str, style: str) -> Dict[str, Any]:
        """Generate Python class based on requirements"""
        try:
            class_name = self._extract_class_name(requirements) or "GeneratedClass"
            methods = self._extract_methods(requirements)
            
            code = f'''class {class_name}:
    """
    {requirements}
    """
    
    def __init__(self):
        """Initialize {class_name}"""
        pass
    
'''
            
            # Add methods
            for method in methods:
                code += f'''    def {method}(self):
        """
        {method.replace('_', ' ').title()} method
        """
        pass
    
'''
            
            return {
                'code': code,
                'complexity': 'medium',
                'type': 'class'
            }
            
        except Exception as e:
            return {'error': f'Python class generation failed: {str(e)}'}
    
    # Helper methods
    def _calculate_cyclomatic_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity of Python code"""
        complexity = 1  # Base complexity
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
            elif isinstance(node, (ast.And, ast.Or)):
                complexity += 1
        
        return complexity
    
    def _count_comment_lines(self, code: str, language: str) -> int:
        """Count comment lines in code"""
        lines = code.splitlines()
        comment_count = 0
        
        if language == 'python':
            for line in lines:
                stripped = line.strip()
                if stripped.startswith('#') or (stripped.startswith('"""') or stripped.startswith("'''")):
                    comment_count += 1
        elif language in ['javascript', 'typescript']:
            for line in lines:
                stripped = line.strip()
                if stripped.startswith('//') or stripped.startswith('/*'):
                    comment_count += 1
        
        return comment_count
    
    def _calculate_code_comment_ratio(self, code: str, language: str) -> float:
        """Calculate ratio of code to comments"""
        total_lines = len([line for line in code.splitlines() if line.strip()])
        comment_lines = self._count_comment_lines(code, language)
        
        if total_lines == 0:
            return 0.0
        
        return round(comment_lines / total_lines, 2)
    
    def _estimate_complexity(self, code: str, language: str) -> str:
        """Estimate code complexity"""
        lines = len(code.splitlines())
        
        if lines < 20:
            return 'low'
        elif lines < 100:
            return 'medium'
        elif lines < 500:
            return 'high'
        else:
            return 'very_high'
    
    def _extract_function_name(self, requirements: str) -> Optional[str]:
        """Extract function name from requirements"""
        # Simple regex to find function names
        match = re.search(r'function\s+(\w+)', requirements, re.IGNORECASE)
        if match:
            return match.group(1)
        
        # Look for "create/make/build X" patterns
        match = re.search(r'(?:create|make|build)\s+(?:a\s+)?(\w+)', requirements, re.IGNORECASE)
        if match:
            return f"create_{match.group(1).lower()}"
        
        return None
    
    def _extract_parameters(self, requirements: str) -> List[str]:
        """Extract parameters from requirements"""
        # Simple parameter extraction
        params = []
        
        # Look for common parameter patterns
        param_patterns = [
            r'with\s+(\w+)',
            r'takes?\s+(\w+)',
            r'accepts?\s+(\w+)',
            r'parameter\s+(\w+)',
            r'argument\s+(\w+)'
        ]
        
        for pattern in param_patterns:
            matches = re.findall(pattern, requirements, re.IGNORECASE)
            params.extend(matches)
        
        return list(set(params)) or ['param1']
    
    def _extract_return_type(self, requirements: str) -> str:
        """Extract return type from requirements"""
        if 'return' in requirements.lower():
            if any(word in requirements.lower() for word in ['list', 'array']):
                return 'List'
            elif any(word in requirements.lower() for word in ['dict', 'dictionary', 'object']):
                return 'Dict'
            elif any(word in requirements.lower() for word in ['string', 'text']):
                return 'str'
            elif any(word in requirements.lower() for word in ['number', 'integer', 'int']):
                return 'int'
            elif any(word in requirements.lower() for word in ['float', 'decimal']):
                return 'float'
            elif any(word in requirements.lower() for word in ['bool', 'boolean']):
                return 'bool'
        
        return 'Any'
    
    def _extract_class_name(self, requirements: str) -> Optional[str]:
        """Extract class name from requirements"""
        match = re.search(r'class\s+(\w+)', requirements, re.IGNORECASE)
        if match:
            return match.group(1)
        
        # Look for "create/make/build X class" patterns
        match = re.search(r'(?:create|make|build)\s+(?:a\s+)?(\w+)\s+class', requirements, re.IGNORECASE)
        if match:
            return match.group(1).title()
        
        return None
    
    def _extract_methods(self, requirements: str) -> List[str]:
        """Extract method names from requirements"""
        methods = []
        
        # Look for method patterns
        method_patterns = [
            r'method\s+(\w+)',
            r'function\s+(\w+)',
            r'can\s+(\w+)',
            r'should\s+(\w+)',
            r'will\s+(\w+)'
        ]
        
        for pattern in method_patterns:
            matches = re.findall(pattern, requirements, re.IGNORECASE)
            methods.extend(matches)
        
        return list(set(methods)) or ['process', 'execute']
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for code service"""
        return {
            'status': 'healthy',
            'supported_languages': list(self.supported_languages.keys()),
            'temp_dir': self.temp_dir,
            'available_features': [
                'code_generation', 'code_analysis', 'code_formatting',
                'code_linting', 'code_optimization', 'code_execution'
            ]
        }
