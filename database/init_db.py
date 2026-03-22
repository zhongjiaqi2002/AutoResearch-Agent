"""
Database Initialization and Connection Management
"""
from typing import Generator
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker
from config import SQLALCHEMY_DATABASE_URL

# Create SQLAlchemy engine with connection pool settings
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    pool_pre_ping=True,  # Check connection validity before use
    pool_recycle=3600,  # Recycle connections every hour
    pool_timeout=30,  # Connection timeout in seconds
    max_overflow=10  # Maximum number of overflow connections
)

# Create session factory for database connections
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)


def get_db_session():
    """
    Get a database session instance
    Used for all database operations across tools

    Returns:
        SQLAlchemy database session
    """
    db = SessionLocal()
    try:
        return db
    except Exception:
        db.rollback()
        raise


def get_table_schema() -> str:
    """
    Retrieve complete database schema with tables, columns, primary keys, and foreign keys
    Provides structured schema for Text2SQL and Chat2SQL tools

    Returns:
        Formatted string of full database schema
    """
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    tables_info = []

    for table in tables:
        # Get primary key information
        pk_constraint = inspector.get_pk_constraint(table)
        primary_keys = pk_constraint.get('constrained_columns', [])

        # Get column definitions
        columns = inspector.get_columns(table)
        columns_info = []
        for col in columns:
            col_name = col['name']
            col_type = str(col['type'])
            nullable = "nullable" if col['nullable'] else "not null"
            pk_tag = "PRIMARY KEY" if col_name in primary_keys else ""
            col_desc = f"- {col_name}: {col_type} ({nullable} {pk_tag})".strip(' ')
            columns_info.append(col_desc.replace(' )', ')'))

        # Get foreign key relationships
        foreign_keys = inspector.get_foreign_keys(table)
        fk_info = []
        for fk in foreign_keys:
            from_col = fk['constrained_columns'][0]
            to_table = fk['referred_table']
            to_col = fk['referred_columns'][0]
            fk_info.append(f"- {table}.{from_col} -> {to_table}.{to_col}")

        # Build table description
        table_desc = f"Table: {table}\nColumns:\n" + "\n".join(columns_info)
        if fk_info:
            table_desc += "\nForeign Keys:\n" + "\n".join(fk_info)
        tables_info.append(table_desc)

    # Final schema output
    schema = f"Total {len(tables)} tables in database:\n\n" + "\n\n".join(tables_info)
    return schema


def init_database() -> bool:
    """
    Initialize database connection and verify connectivity
    Used at application startup for health check

    Returns:
        True if connection succeeds, False otherwise
    """
    try:
        db = get_db_session()
        db.execute(text("SELECT 1"))
        db.close()
        return True
    except Exception as e:
        print(f"Database initialization failed: {str(e)}")
        return False
