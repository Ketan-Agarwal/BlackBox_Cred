from sqlalchemy.orm import Session
from db import SessionLocal

async def get_db():
    async with SessionLocal() as session:
        yield session
