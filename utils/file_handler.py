import os
import json
from pathlib import Path
from fastapi import UploadFile
import shutil
from typing import Dict, Any

async def save_file(file: UploadFile, directory: str) -> Path:
    """
    Save an uploaded file to the specified directory.
    
    Args:
        file: The uploaded file
        directory: The directory to save the file in
        
    Returns:
        Path: The path to the saved file
    """
    # Create directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    
    # Create file path
    file_path = Path(directory) / file.filename
    
    # Save file
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return file_path

def load_json_file(file_path: str) -> Dict[str, Any]:
    """
    Load and parse a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Dict[str, Any]: The parsed JSON data
    """
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json_file(data: Dict[str, Any], file_path: str) -> None:
    """
    Save data to a JSON file.
    
    Args:
        data: The data to save
        file_path: Path where to save the JSON file
    """
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4) 