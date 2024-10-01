import pandas as pd
from sqlalchemy.orm import Session
from database import get_db
from embedding import get_openai_embedding
from models import CustomerEmbedding
from database import Base, engine
import json

# Create the database tables if they do not exist
Base.metadata.create_all(bind=engine)


data = pd.read_csv('retail_data2.csv')


# Function to save an embedding and customer data into the MySQL database
def save_embedding_to_db(db: Session, customer_id: str, customer_name: str, embedding: list):
    """
    Save the embedding and customer data into the MySQL database.

    Parameters:
    - db (Session): SQLAlchemy session object
    - customer_id (str): The unique ID of the customer
    - customer_name (str): Name of the customer (won't be used in embedding)
    - embedding (list): Embedding vector as a list (stored as JSON in the DB)
    - shipping_method (str): The customer's chosen shipping method
    """
    embedding_json = json.dumps(embedding)
    db_embedding = CustomerEmbedding(
        customer_id=customer_id,
        customer_name=customer_name,
        embedding=embedding_json

    )
    db.add(db_embedding)
    db.commit()
    db.refresh(db_embedding)
    return db_embedding


# Function to process and save embeddings to the database
def process_and_save_embeddings(db: Session):
    for index, row in data.iterrows():
        customer_id = row['Transaction_ID']
        customer_name = row['Name']
        city = row['City']
        state = row['State']
        country = row['Country']
        age = row['Age']
        gender = row['Gender']
        income = row['Income']
        total_purchases = row['Total_Purchases']
        total_amount = row['Total_Amount']
        product_category = row['Product_Category']
        product_brand = row['Product_Brand']
        product_type = row['Product_Type']
        shipping_method = row['Shipping_Method']
        products = row['products']

        # Get embedding excluding customer name
        embedding = get_openai_embedding(city, state, country, age, gender, income, total_purchases, total_amount,
                                         product_category, product_brand, product_type, shipping_method, products)

        # Save embedding and customer data to the database
        save_embedding_to_db(db, customer_id, customer_name, embedding)


# Main entry point for the script
if __name__ == "__main__":
    db = next(get_db())  # Get a database session
    process_and_save_embeddings(db)  # Process and save embeddings
