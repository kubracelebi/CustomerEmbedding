import openai
import os
from dotenv import load_dotenv
from sqlalchemy.orm import Session
from models import CustomerEmbedding
import json

# Load environment variables from .env file
load_dotenv()

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")


# Function to get OpenAI embedding for a given customer using selected parameters
def get_openai_embedding(city, state, country, age, gender, income, total_purchases, total_amount, product_category,
                         product_brand, product_type, shipping_method, products, model="text-embedding-3-small"):
    """
    Generate an embedding using OpenAI's embedding API for a customer using selected parameters.

    Parameters:
    - city (str): The customer's city.
    - state (str): The customer's state.
    - country (str): The customer's country.
    - age (int): The customer's age.
    - gender (str): The customer's gender.
    - income (float): The customer's income.
    - total_purchases (int): The total number of purchases made by the customer.
    - total_amount (float): The total amount spent by the customer.
    - product_category (str): The product category purchased by the customer.
    - product_brand (str): The product brand purchased by the customer.
    - product_type (str): The product type purchased by the customer.
    - shipping_method (str): The shipping method chosen by the customer.
    - products (str): The products purchased by the customer.
    - model (str): The OpenAI model to use for embedding (default: text-embedding-3-small).

    Returns:
    - list: The embedding vector.
    """
    text = f"{city} {state} {country} {age} {gender} {income} {total_purchases} {total_amount} {product_category} {product_brand} {product_type} {shipping_method} {products}"
    text = text.replace("\n", " ")  # Normalize text by replacing newlines
    response = openai.embeddings.create(input=[text], model=model)  # Get embedding from OpenAI API
    return response.data[0].embedding


# Function to save an embedding and customer data into the MySQL database
def save_embedding_to_db(db: Session, customer_id: str, customer_name: str, embedding: list):
    """
    Save the embedding and customer data into the MySQL database.

    Parameters:
    - db (Session): The SQLAlchemy session to interact with the database.
    - customer_id (str): The unique customer ID.
    - customer_name (str): The name of the customer.
    - embedding (list): The embedding vector to be saved.
    """
    embedding_json = json.dumps(embedding)  # Convert embedding to JSON format
    db_embedding = CustomerEmbedding(
        customer_id=customer_id,
        customer_name=customer_name,
        embedding=embedding_json

    )
    db.add(db_embedding)  # Add the embedding and customer data to the session
    db.commit()  # Commit the transaction
    db.refresh(db_embedding)  # Refresh the instance with new data from the DB

