import sys
import pkg_resources
from typing import Dict, List, Any
from core.config import settings

def check_python_version() -> Dict[str, Any]:
    """Check if Python version meets requirements"""
    required_version = (3, 11)
    current_version = sys.version_info[:2]
    is_compatible = current_version >= required_version
    
    return {
        "name": "Python Version",
        "is_compatible": is_compatible,
        "message": f"Python {'.'.join(map(str, current_version))} {'meets' if is_compatible else 'does not meet'} requirement of {'.'.join(map(str, required_version))}"
    }

def check_ml_dependencies() -> List[Dict[str, Any]]:
    """Check if ML dependencies meet requirements"""
    required_packages = {
        'scikit-learn': '1.0.2',
        'numpy': '1.21.0',
        'pandas': '1.3.0',
        'tensorflow': '2.6.0',
        'torch': '1.9.0',
        'transformers': '4.11.3'
    }
    
    checks = []
    for package, required_version in required_packages.items():
        try:
            installed_version = pkg_resources.get_distribution(package).version
            is_compatible = pkg_resources.parse_version(installed_version) >= pkg_resources.parse_version(required_version)
            message = f"{package} {installed_version} {'meets' if is_compatible else 'does not meet'} requirement of {required_version}"
        except pkg_resources.DistributionNotFound:
            is_compatible = False
            message = f"{package} is not installed"
        
        checks.append({
            "name": package,
            "is_compatible": is_compatible,
            "message": message
        })
    
    return checks

def check_all_versions() -> Dict[str, Any]:
    """Check all version requirements"""
    python_check = check_python_version()
    ml_checks = check_ml_dependencies()
    
    # In development mode, consider everything compatible
    if settings.DEV_MODE:
        return {
            "is_compatible": True,
            "checks": [
                {"name": "Python Version", "is_compatible": True, "message": "Development mode: Version check bypassed"},
                *[{"name": check["name"], "is_compatible": True, "message": "Development mode: Version check bypassed"} for check in ml_checks]
            ]
        }
    
    # In production, check all requirements
    all_checks = [python_check] + ml_checks
    is_compatible = all(check["is_compatible"] for check in all_checks)
    
    return {
        "is_compatible": is_compatible,
        "checks": all_checks
    }

if __name__ == "__main__":
    results = check_all_versions()
    print("\nVersion Compatibility Check:")
    print(f"Overall Compatibility: {'✓' if results['is_compatible'] else '✗'}")
    print("\nChecks:")
    for check in results['checks']:
        print(f"- {check['message']}") 