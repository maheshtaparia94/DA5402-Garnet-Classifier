import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

load_dotenv()

DATABASE_URL = (f"postgresql+psycopg2://"
            f"{os.getenv('PSQL_APP_USER')}:"
            f"{os.getenv('PSQL_APP_PASSWORD')}@"
            f"{os.getenv('PSQL_APP_HOST', 'app-postgres')}/"
            f"{os.getenv('PSQL_APP_DB')}")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    filename = Column(String, nullable=False)
    file_path  = Column(String, nullable=True)
    predicted_class = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    is_low_confidence = Column(Boolean, default=False)
    drift_score = Column(Float, default=0.0)
    is_drifted  = Column(Boolean, default=False)
    ground_truth = Column(String, nullable=True)
    is_wrong = Column(Boolean, nullable=True)
    model_version = Column(String, nullable=False)
    data_version = Column(String, nullable=False)


def create_tables():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()