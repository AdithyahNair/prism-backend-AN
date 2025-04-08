import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from sqlalchemy import text
from models.models import User
from core.auth import get_password_hash
from datetime import datetime
import asyncio
from services.database import AsyncSessionLocal, init_db
import logging

logger = logging.getLogger(__name__)

# Synchronous version for direct calling
def create_test_user_sync():
    """Synchronous wrapper to create a test user"""
    # Initialize database before creating test user
    init_db()
    asyncio.run(create_test_user())

async def create_test_user():
    async with AsyncSessionLocal() as db:
        try:
            # Generate unique email with timestamp
            #timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            timestamp = 1
            username = f"test_user_{timestamp}"
            email = f"test_{timestamp}@example.com"
            
            # Check if test user already exists
            result = await db.execute(
                text(f"SELECT * FROM users WHERE username = '{username}'")
            )
            test_user = result.scalar_one_or_none()
            
            if test_user:
                logger.info("Test user already exists")
                return

            # Create test user
            hashed_password = get_password_hash("test_password")
            test_user = User(
                username=username,
                email=email,
                hashed_password=hashed_password,
                is_active=True
            )
            
            db.add(test_user)
            await db.commit()
            logger.info("Test user created successfully")
            logger.info(f"Username: {username}")
            logger.info("Password: test_password")
        except Exception as e:
            logger.error(f"Error creating test user: {str(e)}")
            await db.rollback()

if __name__ == "__main__":
    # Initialize database first
    init_db()
    asyncio.run(create_test_user()) 