from sqlalchemy import Column, Integer, String, JSON
from database import Base

class CustomerEmbedding(Base):
    __tablename__ = "customer_embeddings"

    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(String(255), unique=True, index=True)
    customer_name = Column(String(255))
    embedding = Column(JSON)
