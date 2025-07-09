"""
Database connection management for PostgreSQL.

This module provides connection pooling and session management
for the PostgreSQL database.
"""

import logging
from contextlib import contextmanager
from typing import Generator, Optional
from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError

from ..configuration import get_config
from .models import Base

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages database connections and sessions."""
    
    def __init__(self):
        self.config = get_config()
        self.engine: Optional[Engine] = None
        self.SessionLocal: Optional[sessionmaker] = None
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize database engine and session factory."""
        try:
            # Use URL if provided, otherwise construct from components
            if self.config.postgres.url:
                database_url = self.config.postgres.url
            else:
                database_url = (
                    f"postgresql://{self.config.postgres.username}:"
                    f"{self.config.postgres.password}@"
                    f"{self.config.postgres.host}:{self.config.postgres.port}/"
                    f"{self.config.postgres.database}"
                )
            
            self.engine = create_engine(
                database_url,
                poolclass=QueuePool,
                pool_size=self.config.postgres.pool_size,
                max_overflow=self.config.postgres.max_overflow,
                pool_pre_ping=True,
                echo=self.config.debug,
            )
            
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )
            
            logger.info("Database connection initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def create_tables(self):
        """Create all database tables."""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except SQLAlchemyError as e:
            logger.error(f"Failed to create tables: {e}")
            raise
    
    def drop_tables(self):
        """Drop all database tables."""
        try:
            Base.metadata.drop_all(bind=self.engine)
            logger.info("Database tables dropped successfully")
        except SQLAlchemyError as e:
            logger.error(f"Failed to drop tables: {e}")
            raise
    
    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Get a database session with automatic cleanup."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def get_engine(self) -> Engine:
        """Get the database engine."""
        return self.engine
    
    def health_check(self) -> bool:
        """Check database connection health."""
        try:
            with self.get_session() as session:
                session.execute("SELECT 1")
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False

# Global database manager instance
db_manager = DatabaseManager()

def get_db_session() -> Generator[Session, None, None]:
    """Dependency for getting database sessions."""
    with db_manager.get_session() as session:
        yield session

def init_database():
    """Initialize database tables."""
    db_manager.create_tables()
