import os
import logging
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("storage.postgres")

Base = declarative_base()

class Incident(Base):
    __tablename__ = 'incidents'
    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(255), nullable=False)
    service = Column(String(100), nullable=False)
    severity = Column(String(50), nullable=False)
    status = Column(String(50), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    resolved_at = Column(DateTime, nullable=True)

class Deployment(Base):
    __tablename__ = 'deployments'
    id = Column(Integer, primary_key=True, autoincrement=True)
    service = Column(String(100), nullable=False)
    version = Column(String(50), nullable=False)
    status = Column(String(50), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class Alert(Base):
    __tablename__ = 'alerts'
    id = Column(Integer, primary_key=True, autoincrement=True)
    alert_name = Column(String(100), nullable=False)
    severity = Column(String(50), nullable=False)
    status = Column(String(50), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class AuditLog(Base):
    __tablename__ = 'audit_logs'
    id = Column(Integer, primary_key=True, autoincrement=True)
    query = Column(Text, nullable=False)
    blocked_by = Column(String(100), nullable=False)
    reason = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class EvalRun(Base):
    __tablename__ = 'eval_runs'
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    graph_version = Column(String(50), nullable=False)
    faithfulness = Column(Float, nullable=False)
    answer_relevancy = Column(Float, nullable=False)
    context_recall = Column(Float, nullable=False)
    context_precision = Column(Float, nullable=False)

# Connection string setup with fallback to SQLite
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/ops_db")
FALLBACK_SQLITE_URL = "sqlite:///./ops_db.sqlite"

engine = None
SessionLocal = None

try:
    logger.info("Attempting to connect to PostgreSQL...")
    engine = create_engine(DATABASE_URL, connect_args={"connect_timeout": 3})
    # Test connection
    with engine.connect() as conn:
        logger.info("Successfully connected to PostgreSQL.")
except Exception as e:
    logger.warning(f"Failed to connect to PostgreSQL: {e}. Falling back to SQLite.")
    engine = create_engine(FALLBACK_SQLITE_URL, connect_args={"check_same_thread": False})
    logger.info("SQLite engine initialized.")

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables initialized.")
    
    # Seed mock SRE data if tables are empty
    db = SessionLocal()
    try:
        from datetime import datetime, timedelta
        if db.query(Incident).count() == 0:
            logger.info("Seeding mock incidents...")
            incidents = [
                Incident(title="CoreDNS loopback detected", service="coredns", severity="High", status="Resolved", 
                         created_at=datetime.utcnow() - timedelta(days=2), resolved_at=datetime.utcnow() - timedelta(days=1.8)),
                Incident(title="API Server latency exceeds SLA", service="kube-apiserver", severity="Critical", status="Active", 
                         created_at=datetime.utcnow() - timedelta(hours=4), resolved_at=None),
                Incident(title="Postgres database pod OOMKilled", service="postgres", severity="High", status="Resolved", 
                         created_at=datetime.utcnow() - timedelta(days=5), resolved_at=datetime.utcnow() - timedelta(days=4.9)),
                Incident(title="Ingress controller ingress-nginx-controller CrashLoopBackOff", service="ingress-nginx", severity="High", status="Active",
                         created_at=datetime.utcnow() - timedelta(hours=2), resolved_at=None)
            ]
            db.add_all(incidents)
            db.commit()

        if db.query(Deployment).count() == 0:
            logger.info("Seeding mock deployments...")
            deployments = [
                Deployment(service="auth-service", version="v1.2.3", status="Failed", created_at=datetime.utcnow() - timedelta(days=1)),
                Deployment(service="payment-service", version="v2.1.0", status="Success", created_at=datetime.utcnow() - timedelta(days=3)),
                Deployment(service="frontend", version="v3.0.1", status="Success", created_at=datetime.utcnow() - timedelta(hours=12))
            ]
            db.add_all(deployments)
            db.commit()

        if db.query(Alert).count() == 0:
            logger.info("Seeding mock alerts...")
            alerts = [
                Alert(alert_name="KubePodCrashLooping", severity="Warning", status="Firing", created_at=datetime.utcnow() - timedelta(minutes=45)),
                Alert(alert_name="KubeCPUOvercommitted", severity="Critical", status="Resolved", created_at=datetime.utcnow() - timedelta(days=2)),
                Alert(alert_name="NodeDiskPressure", severity="Critical", status="Firing", created_at=datetime.utcnow() - timedelta(hours=1))
            ]
            db.add_all(alerts)
            db.commit()
            
        logger.info("Database seeding completed.")
    except Exception as e:
        logger.error(f"Error seeding database: {e}")
        db.rollback()
    finally:
        db.close()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
