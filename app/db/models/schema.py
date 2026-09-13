import uuid
from datetime import datetime
from sqlalchemy import Column, String, Text, Integer, Float, Boolean, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from app.db.session import Base

def generate_uuid():
    return str(uuid.uuid4())

class Brand(Base):
    __tablename__ = "brands"

    id = Column(String, primary_key=True, default=generate_uuid)
    name = Column(String(100), unique=True, nullable=False)
    handle = Column(String(100), nullable=False)
    category = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    conversations = relationship("Conversation", back_populates="brand")

class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(String, primary_key=True, default=generate_uuid)
    external_id = Column(String(100), unique=True, index=True, nullable=False)
    brand_id = Column(String, ForeignKey("brands.id"), nullable=True)
    resolution_quality = Column(String(20), default="UNKNOWN")  # HIGH, MEDIUM, LOW, UNKNOWN
    created_at = Column(DateTime, default=datetime.utcnow)

    brand = relationship("Brand", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")
    retrieval_doc = relationship("RetrievalDocument", back_populates="conversation", uselist=False)

class Message(Base):
    __tablename__ = "messages"

    id = Column(String, primary_key=True, default=generate_uuid)
    conversation_id = Column(String, ForeignKey("conversations.id"), nullable=False)
    tweet_id = Column(String(100), index=True, nullable=False)
    role = Column(String(20), nullable=False)  # "customer" or "agent"
    text = Column(Text, nullable=False)
    author_id = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    conversation = relationship("Conversation", back_populates="messages")

class Intent(Base):
    __tablename__ = "intents"

    id = Column(String, primary_key=True, default=generate_uuid)
    name = Column(String(100), unique=True, nullable=False)
    description = Column(Text, nullable=False)
    version = Column(String(20), default="v1")
    created_at = Column(DateTime, default=datetime.utcnow)

class RetrievalDocument(Base):
    __tablename__ = "retrieval_documents"

    id = Column(String, primary_key=True, default=generate_uuid)
    conversation_id = Column(String, ForeignKey("conversations.id"), nullable=False)
    problem_text = Column(Text, nullable=False)
    resolution_text = Column(Text, nullable=False)
    quality = Column(String(20), default="HIGH")
    intent = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    conversation = relationship("Conversation", back_populates="retrieval_doc")

class GoldenExample(Base):
    __tablename__ = "golden_examples"

    id = Column(String, primary_key=True, default=generate_uuid)
    external_id = Column(String(100), unique=True, index=True)
    message = Column(Text, nullable=False)
    context = Column(Text, nullable=True)
    intent = Column(String(100), nullable=False)
    expected_action = Column(String(20), nullable=False)  # "auto" or "escalate"
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class EvaluationRun(Base):
    __tablename__ = "evaluation_runs"

    id = Column(String, primary_key=True, default=generate_uuid)
    model_name = Column(String(100), nullable=False)
    dataset_version = Column(String(50), nullable=False)
    metrics = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class EvaluationResult(Base):
    __tablename__ = "evaluation_results"

    id = Column(String, primary_key=True, default=generate_uuid)
    run_id = Column(String, ForeignKey("evaluation_runs.id"), nullable=False)
    golden_example_id = Column(String, nullable=True)
    message = Column(Text, nullable=False)
    gold_intent = Column(String(100), nullable=False)
    pred_intent = Column(String(100), nullable=False)
    gold_action = Column(String(20), nullable=False)
    pred_action = Column(String(20), nullable=False)
    is_safe = Column(Boolean, default=True)
    reply = Column(Text, nullable=True)
