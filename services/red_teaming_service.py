from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import json
import logging
from enum import Enum
from sqlalchemy.orm import Session

from models.models import Audit
from schemas.llm import (
    RedTeamingCreate,
    RedTeamingResponse,
    RedTeamingProgress,
    RedTeamingVulnerabilities,
    RedTeamingMetrics,
    RedTeamingVisualization,
    RedTeamingPerformance,
    RedTeamingCategoryMetrics
)

logger = logging.getLogger(__name__)

class AttackMode(str, Enum):
    MANUAL = "manual"
    TARGETED = "targeted"
    COMPREHENSIVE = "comprehensive"

class RedTeamingCategory(str, Enum):
    PROMPT_INJECTION = "prompt_injection"
    HARMUL_CONTENT = "harmful_content"
    BIAS = "bias"
    PRIVACY = "privacy"
    SECURITY = "security"
    PERFORMANCE = "performance"

class RedTeamingService:
    def __init__(self, db: Session):
        self.db = db
        self.moonshot_data_dir = Path("moonshot-data")
        self._load_test_cases()

    # Base Audit Methods
    async def create_audit(self, audit_data: Dict[str, Any]) -> Audit:
        """Create a new audit record"""
        try:
            audit = Audit(
                **audit_data,
                status="created",
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            self.db.add(audit)
            await self.db.commit()
            await self.db.refresh(audit)
            return audit
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error creating audit: {str(e)}")
            raise

    async def get_audit(self, audit_id: int) -> Optional[Audit]:
        """Get an audit record by ID"""
        try:
            return await self.db.query(Audit).filter(Audit.id == audit_id).first()
        except Exception as e:
            logger.error(f"Error getting audit: {str(e)}")
            raise

    async def update_audit(
        self,
        audit_id: int,
        update_data: Dict[str, Any]
    ) -> Optional[Audit]:
        """Update an audit record"""
        try:
            audit = await self.get_audit(audit_id)
            if not audit:
                return None

            for field, value in update_data.items():
                setattr(audit, field, value)
            
            audit.updated_at = datetime.utcnow()
            await self.db.commit()
            await self.db.refresh(audit)
            return audit
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error updating audit: {str(e)}")
            raise

    async def update_audit_status(
        self,
        audit_id: int,
        status: str,
        error_message: Optional[str] = None
    ) -> Optional[Audit]:
        """Update audit status and error message if any"""
        try:
            update_data = {
                "status": status,
                "error_message": error_message,
                "updated_at": datetime.utcnow()
            }
            return await self.update_audit(audit_id, update_data)
        except Exception as e:
            logger.error(f"Error updating audit status: {str(e)}")
            raise

    async def update_audit_progress(
        self,
        audit_id: int,
        progress: float,
        current_step: str,
        completed_steps: list,
        remaining_steps: list,
        estimated_time_remaining: Optional[int] = None
    ) -> Optional[Audit]:
        """Update audit progress information"""
        try:
            update_data = {
                "progress": progress,
                "current_step": current_step,
                "completed_steps": completed_steps,
                "remaining_steps": remaining_steps,
                "estimated_time_remaining": estimated_time_remaining,
                "updated_at": datetime.utcnow()
            }
            return await self.update_audit(audit_id, update_data)
        except Exception as e:
            logger.error(f"Error updating audit progress: {str(e)}")
            raise

    async def update_audit_results(
        self,
        audit_id: int,
        results: Dict[str, Any]
    ) -> Optional[Audit]:
        """Update audit results"""
        try:
            update_data = {
                "results": results,
                "updated_at": datetime.utcnow()
            }
            return await self.update_audit(audit_id, update_data)
        except Exception as e:
            logger.error(f"Error updating audit results: {str(e)}")
            raise

    # Red Teaming Specific Methods
    async def create_red_teaming(self, red_teaming: RedTeamingCreate) -> RedTeamingResponse:
        """Create a new red teaming session"""
        try:
            # Create audit record for red teaming
            audit = await self.create_audit({
                "project_id": red_teaming.project_id,
                "user_id": red_teaming.user_id,
                "connector_id": red_teaming.connector_id,
                "audit_type": "red_teaming",
                "parameters": {
                    "attack_mode": red_teaming.attack_mode,
                    "target_categories": red_teaming.target_categories,
                    **(red_teaming.parameters if red_teaming.parameters else {})
                }
            })

            return RedTeamingResponse(
                id=audit.id,
                project_id=audit.project_id,
                user_id=audit.user_id,
                connector_id=audit.connector_id,
                attack_mode=red_teaming.attack_mode,
                target_categories=red_teaming.target_categories,
                parameters=red_teaming.parameters,
                status=audit.status,
                results=audit.results,
                error_message=audit.error_message,
                created_at=audit.created_at,
                updated_at=audit.updated_at
            )
        except Exception as e:
            logger.error(f"Error creating red teaming session: {str(e)}")
            raise

    async def run_attack_mode(
        self,
        attack_mode: AttackMode,
        target_categories: List[RedTeamingCategory],
        parameters: Optional[Dict[str, Any]] = None,
        max_attempts: int = 10
    ) -> Dict[str, Any]:
        """Run an attack mode against the model"""
        try:
            results = {
                "attack_mode": attack_mode,
                "target_categories": [cat.value for cat in target_categories],
                "start_time": datetime.utcnow().isoformat(),
                "attempts": [],
                "vulnerabilities_found": 0
            }

            for category in target_categories:
                for _ in range(max_attempts):
                    # Generate attack prompt based on mode
                    if attack_mode == AttackMode.MANUAL:
                        prompt = await self._generate_attack_prompt(category, parameters)
                    elif attack_mode == AttackMode.TARGETED:
                        prompt = await self._generate_targeted_attack_prompt(category, parameters)
                    elif attack_mode == AttackMode.COMPREHENSIVE:
                        prompt = await self._generate_comprehensive_attack_prompt(category, parameters)
                    else:
                        raise ValueError(f"Unsupported attack mode: {attack_mode}")

                    # Execute attack
                    attack_result = await self._execute_attack(prompt, category)
                    results["attempts"].append(attack_result)

                    # Check if vulnerability was found
                    if attack_result["vulnerability_found"]:
                        results["vulnerabilities_found"] += 1

            results["end_time"] = datetime.utcnow().isoformat()
            results["success_rate"] = results["vulnerabilities_found"] / len(results["attempts"])

            return results
        except Exception as e:
            logger.error(f"Error running attack mode: {str(e)}")
            raise

    async def _generate_attack_prompt(
        self,
        category: RedTeamingCategory,
        parameters: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate a prompt for manual attack mode"""
        try:
            # Load category-specific test cases
            test_cases = getattr(self, f"{category.value}_cases", [])
            if not test_cases:
                raise ValueError(f"No test cases found for category: {category}")

            # Select a test case based on parameters
            test_case = test_cases[0]  # TODO: Implement better selection logic
            return test_case["prompt"]
        except Exception as e:
            logger.error(f"Error generating attack prompt: {str(e)}")
            raise

    async def _generate_targeted_attack_prompt(
        self,
        category: RedTeamingCategory,
        parameters: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate a prompt for targeted attack mode"""
        try:
            # Load category-specific test cases and combine with target parameters
            test_cases = getattr(self, f"{category.value}_cases", [])
            if not test_cases:
                raise ValueError(f"No test cases found for category: {category}")

            # Select and modify test case based on target parameters
            test_case = test_cases[0]  # TODO: Implement better selection logic
            prompt = test_case["prompt"]

            if parameters:
                # Modify prompt based on target parameters
                for key, value in parameters.items():
                    prompt = prompt.replace(f"{{{key}}}", str(value))

            return prompt
        except Exception as e:
            logger.error(f"Error generating targeted attack prompt: {str(e)}")
            raise

    async def _generate_comprehensive_attack_prompt(
        self,
        category: RedTeamingCategory,
        parameters: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate a prompt for comprehensive attack mode"""
        try:
            # Load and combine test cases from multiple categories
            all_test_cases = []
            for cat in RedTeamingCategory:
                test_cases = getattr(self, f"{cat.value}_cases", [])
                all_test_cases.extend(test_cases)

            # Select and combine relevant test cases
            selected_cases = all_test_cases[:3]  # TODO: Implement better selection logic
            combined_prompt = " ".join(case["prompt"] for case in selected_cases)

            if parameters:
                # Modify prompt based on parameters
                for key, value in parameters.items():
                    combined_prompt = combined_prompt.replace(f"{{{key}}}", str(value))

            return combined_prompt
        except Exception as e:
            logger.error(f"Error generating comprehensive attack prompt: {str(e)}")
            raise

    async def _execute_attack(
        self,
        prompt: str,
        category: RedTeamingCategory
    ) -> Dict[str, Any]:
        """Execute an attack and analyze the response"""
        try:
            # TODO: Implement actual model call
            response = "Model response placeholder"

            # Analyze the response for vulnerabilities
            vulnerability_found = await self._analyze_vulnerability(response, category)

            return {
                "prompt": prompt,
                "response": response,
                "category": category.value,
                "vulnerability_found": vulnerability_found,
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Error executing attack: {str(e)}")
            raise

    async def _analyze_vulnerability(
        self,
        response: str,
        category: RedTeamingCategory
    ) -> bool:
        """Analyze if the response contains vulnerabilities"""
        try:
            # Load category-specific vulnerability patterns
            patterns = []
            patterns_file = self.moonshot_data_dir / f"{category.value}_patterns.json"
            if patterns_file.exists():
                with open(patterns_file) as f:
                    patterns = json.load(f)

            # Check for vulnerability patterns in the response
            for pattern in patterns:
                if pattern["pattern"] in response:
                    return True

            return False
        except Exception as e:
            logger.error(f"Error analyzing vulnerability: {str(e)}")
            raise

    async def get_red_teaming_progress(self, audit_id: int) -> RedTeamingProgress:
        """Get the progress of a red teaming session"""
        try:
            audit = await self.get_audit(audit_id)
            if not audit:
                raise ValueError("Red teaming session not found")

            return RedTeamingProgress(
                audit_id=audit.id,
                status=audit.status,
                progress=audit.progress if hasattr(audit, "progress") else 0,
                current_step=audit.current_step if hasattr(audit, "current_step") else "initializing",
                completed_steps=audit.completed_steps if hasattr(audit, "completed_steps") else [],
                remaining_steps=audit.remaining_steps if hasattr(audit, "remaining_steps") else [],
                estimated_time_remaining=audit.estimated_time_remaining if hasattr(audit, "estimated_time_remaining") else None,
                last_updated=audit.updated_at
            )
        except Exception as e:
            logger.error(f"Error getting red teaming progress: {str(e)}")
            raise

    async def get_red_teaming_vulnerabilities(self, audit_id: int) -> RedTeamingVulnerabilities:
        """Get vulnerabilities found in a red teaming session"""
        try:
            audit = await self.get_audit(audit_id)
            if not audit:
                raise ValueError("Red teaming session not found")

            results = audit.results or {}
            attempts = results.get("attempts", [])
            
            vulnerabilities = [
                attempt for attempt in attempts 
                if attempt.get("vulnerability_found", False)
            ]

            severity_distribution = {}
            category_distribution = {}
            
            for vuln in vulnerabilities:
                severity = vuln.get("severity", "unknown")
                category = vuln.get("category", "unknown")
                
                severity_distribution[severity] = severity_distribution.get(severity, 0) + 1
                category_distribution[category] = category_distribution.get(category, 0) + 1

            return RedTeamingVulnerabilities(
                audit_id=audit.id,
                vulnerabilities=vulnerabilities,
                total_count=len(vulnerabilities),
                severity_distribution=severity_distribution,
                category_distribution=category_distribution,
                last_updated=audit.updated_at
            )
        except Exception as e:
            logger.error(f"Error getting red teaming vulnerabilities: {str(e)}")
            raise

    async def get_red_teaming_metrics(self, audit_id: int) -> RedTeamingMetrics:
        """Get metrics for a red teaming session"""
        try:
            audit = await self.get_audit(audit_id)
            if not audit:
                raise ValueError("Red teaming session not found")

            results = audit.results or {}
            attempts = results.get("attempts", [])
            
            successful_attacks = sum(1 for attempt in attempts if attempt.get("vulnerability_found", False))
            failed_attacks = len(attempts) - successful_attacks
            
            response_times = [attempt.get("response_time", 0) for attempt in attempts]
            average_response_time = sum(response_times) / len(response_times) if response_times else 0

            token_usage = results.get("token_usage", {})
            cost_metrics = results.get("cost_metrics", {})

            return RedTeamingMetrics(
                audit_id=audit.id,
                total_attempts=len(attempts),
                successful_attacks=successful_attacks,
                failed_attacks=failed_attacks,
                average_response_time=average_response_time,
                token_usage=token_usage,
                cost_metrics=cost_metrics,
                last_updated=audit.updated_at
            )
        except Exception as e:
            logger.error(f"Error getting red teaming metrics: {str(e)}")
            raise

    async def get_red_teaming_visualization(
        self, 
        audit_id: int, 
        chart_type: str
    ) -> RedTeamingVisualization:
        """Get visualization data for a red teaming session"""
        try:
            audit = await self.get_audit(audit_id)
            if not audit:
                raise ValueError("Red teaming session not found")

            results = audit.results or {}
            
            # Generate visualization data based on chart type
            data = self._generate_visualization_data(chart_type, results)
            
            return RedTeamingVisualization(
                audit_id=audit.id,
                chart_type=chart_type,
                data=data,
                options=self._get_chart_options(chart_type),
                last_updated=audit.updated_at
            )
        except Exception as e:
            logger.error(f"Error getting red teaming visualization: {str(e)}")
            raise

    async def get_red_teaming_performance(self, audit_id: int) -> RedTeamingPerformance:
        """Get performance metrics for a red teaming session"""
        try:
            audit = await self.get_audit(audit_id)
            if not audit:
                raise ValueError("Red teaming session not found")

            results = audit.results or {}
            attempts = results.get("attempts", [])
            
            response_times = [attempt.get("response_time", 0) for attempt in attempts]
            token_usage = [attempt.get("token_usage", 0) for attempt in attempts]
            cost_tracking = [attempt.get("cost", 0) for attempt in attempts]
            success_rates = [1 if attempt.get("vulnerability_found", False) else 0 for attempt in attempts]

            return RedTeamingPerformance(
                audit_id=audit.id,
                response_times=response_times,
                token_usage=token_usage,
                cost_tracking=cost_tracking,
                success_rates=success_rates,
                last_updated=audit.updated_at
            )
        except Exception as e:
            logger.error(f"Error getting red teaming performance: {str(e)}")
            raise

    async def get_red_teaming_category_metrics(
        self, 
        audit_id: int, 
        category: str
    ) -> RedTeamingCategoryMetrics:
        """Get metrics for a specific category in a red teaming session"""
        try:
            audit = await self.get_audit(audit_id)
            if not audit:
                raise ValueError("Red teaming session not found")

            results = audit.results or {}
            attempts = results.get("attempts", [])
            
            category_attempts = [attempt for attempt in attempts if attempt.get("category") == category]
            
            successful_attacks = sum(1 for attempt in category_attempts if attempt.get("vulnerability_found", False))
            success_rate = successful_attacks / len(category_attempts) if category_attempts else 0
            
            response_times = [attempt.get("response_time", 0) for attempt in category_attempts]
            average_response_time = sum(response_times) / len(response_times) if response_times else 0
            
            total_token_usage = sum(attempt.get("token_usage", 0) for attempt in category_attempts)
            total_cost = sum(attempt.get("cost", 0) for attempt in category_attempts)
            
            vulnerabilities = [
                attempt for attempt in category_attempts 
                if attempt.get("vulnerability_found", False)
            ]

            return RedTeamingCategoryMetrics(
                audit_id=audit.id,
                category=category,
                attempts=len(category_attempts),
                success_rate=success_rate,
                average_response_time=average_response_time,
                token_usage=total_token_usage,
                cost=total_cost,
                vulnerabilities=vulnerabilities,
                last_updated=audit.updated_at
            )
        except Exception as e:
            logger.error(f"Error getting red teaming category metrics: {str(e)}")
            raise

    def _generate_visualization_data(self, chart_type: str, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate visualization data based on chart type"""
        # Implementation depends on chart type
        pass

    def _get_chart_options(self, chart_type: str) -> Dict[str, Any]:
        """Get chart options based on chart type"""
        # Implementation depends on chart type
        pass

    async def generate_red_teaming_report(
        self,
        red_teaming_id: int,
        format: str = "pdf"
    ) -> Dict[str, Any]:
        """Generate a report for a red teaming session"""
        try:
            audit = await self.get_audit(red_teaming_id)
            if not audit:
                raise ValueError("Red teaming session not found")

            report = {
                "id": audit.id,
                "project_id": audit.project_id,
                "model_id": audit.model_id,
                "status": audit.status,
                "summary": {
                    "total_attempts": len(audit.results.get("attempts", [])),
                    "vulnerabilities_found": audit.results.get("vulnerabilities_found", 0),
                    "success_rate": audit.results.get("success_rate", 0)
                },
                "categories": {},
                "recommendations": []
            }

            # Aggregate results by category
            for attempt in audit.results.get("attempts", []):
                category = attempt["category"]
                if category not in report["categories"]:
                    report["categories"][category] = {
                        "attempts": 0,
                        "vulnerabilities": 0,
                        "examples": []
                    }

                report["categories"][category]["attempts"] += 1
                if attempt["vulnerability_found"]:
                    report["categories"][category]["vulnerabilities"] += 1
                    report["categories"][category]["examples"].append({
                        "prompt": attempt["prompt"],
                        "response": attempt["response"]
                    })

            # Generate recommendations based on findings
            for category, data in report["categories"].items():
                if data["vulnerabilities"] > 0:
                    report["recommendations"].append({
                        "category": category,
                        "risk_level": "high" if data["vulnerabilities"] / data["attempts"] > 0.5 else "medium",
                        "suggestion": f"Improve {category} defenses based on identified vulnerabilities"
                    })

            return report
        except Exception as e:
            logger.error(f"Error generating red teaming report: {str(e)}")
            raise 