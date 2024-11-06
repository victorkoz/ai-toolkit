from pymongo import MongoClient
import os

MONGO_URL = os.getenv('MONGO_URL')

print('[MONGO_URL]', MONGO_URL)

client = MongoClient(MONGO_URL)

# Optional: Check if the connection was successful
try:
    client.admin.command('ping')  # Ping the server to check the connection
    print("Connected to MongoDB successfully.")
except Exception as e:
    print(f"Could not connect to MongoDB: {e}")

# Export the client for use in other modules
def get_client():
    return client['AIv1']