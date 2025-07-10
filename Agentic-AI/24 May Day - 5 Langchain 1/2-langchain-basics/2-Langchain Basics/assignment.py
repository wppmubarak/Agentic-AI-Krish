import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI API client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Define Pydantic Models
class ProductInfoRequest(BaseModel):
    product_query: str = Field(..., title="Product Query", description="Query about a product")

class ProductInfoResponse(BaseModel):
    product_name: str
    product_details: str
    tentative_price_inr: int

def create_product_query_prompt(query: str) -> str:
    return f"""Provide product information for: {query}

Respond with valid JSON in this exact format:
{{
    "product_name": "name here",
    "product_details": "details here",
    "tentative_price_inr": 99999
}}"""

def get_product_info(query: str) -> str:
    prompt = create_product_query_prompt(query)
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful product assistant. Always respond with valid JSON."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=200
    )
    
    return response.choices[0].message.content.strip()

def fetch_product_info(query: str) -> ProductInfoResponse:
    raw_response = get_product_info(query)
    
    try:
        data = json.loads(raw_response)
        return ProductInfoResponse(**data)
    except (json.JSONDecodeError, ValueError) as e:
        raise ValueError(f"Failed to parse response: {e}")

# Example usage:
if __name__ == "__main__":
    query = "Tell me about the motorola edge 60 ultra."
    product_info = fetch_product_info(query)
    
    if product_info:
        print(f"Product Name: {product_info.product_name}")
        print(f"Product Details: {product_info.product_details}")
        print(f"Tentative Price in INR: {product_info.tentative_price_inr}")
    else:
        print("Failed to fetch product info.")
