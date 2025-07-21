# main_db.py
import os
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv

# --- Important: Install required libraries ---
# pip install pymongo python-dotenv

# Load environment variables from a .env file
load_dotenv()

class MongoAuthenticator:
    """
    A class to handle user enrollment and verification using MongoDB.
    Now stores a statistical profile along with the embedding.
    """
    def __init__(self):
        """
        Initializes the database connection.
        """
        uri = os.getenv("MONGO_URI")
        if not uri:
            raise ValueError("MONGO_URI environment variable not set. Please create a .env file.")

        self.client = MongoClient(uri, server_api=ServerApi('1'))
        self.db = self.client['biometric_auth_db']
        self.collection = self.db['users']

        try:
            self.client.admin.command('ping')
            print("[INFO] Successfully connected to MongoDB!")
        except Exception as e:
            print(f"[ERROR] Could not connect to MongoDB: {e}")
            self.client = None

    def enroll_user(self, user_id: str, embedding: list, profile_mean: float, profile_std: float) -> bool:
        """
        Saves or updates a user's profile, including embedding and statistics.

        Args:
            user_id: The unique identifier for the user.
            embedding: The user's biometric embedding.
            profile_mean: The mean of the user's characteristic sensor data.
            profile_std: The standard deviation of the user's sensor data.

        Returns:
            True if the operation was successful, False otherwise.
        """
        if not self.client:
            return False
            
        try:
            self.collection.update_one(
                {'_id': user_id},
                {'$set': {
                    'embedding': embedding,
                    'profile_mean': profile_mean,
                    'profile_std': profile_std
                }},
                upsert=True
            )
            print(f"[INFO] Successfully enrolled/updated user: {user_id}")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to enroll user {user_id}: {e}")
            return False

    def get_user_profile(self, user_id: str) -> dict | None:
        """
        Retrieves a user's stored profile from the database.

        Args:
            user_id: The unique identifier for the user.

        Returns:
            A dictionary containing the user's profile, or None if not found.
        """
        if not self.client:
            return None

        try:
            user_document = self.collection.find_one({'_id': user_id})
            if user_document:
                print(f"[INFO] Found profile for user: {user_id}")
                return user_document
            else:
                print(f"[WARN] No user found with ID: {user_id}")
                return None
        except Exception as e:
            print(f"[ERROR] Failed to retrieve profile for {user_id}: {e}")
            return None

    def close_connection(self):
        """
        Closes the connection to the database.
        """
        if self.client:
            self.client.close()
            print("[INFO] MongoDB connection closed.")
