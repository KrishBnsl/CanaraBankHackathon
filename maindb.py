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
    """
    def __init__(self):
        """
        Initializes the database connection.
        """
        # --- 1. Set up your MongoDB Connection ---
        # For security, the connection string is stored in an environment variable.
        # Create a file named ".env" in your project directory and add the following line:
        # MONGO_URI="your_mongodb_connection_string"
        #
        # You can get this string from your MongoDB Atlas dashboard.
        uri = os.getenv("MONGO_URI")
        if not uri:
            raise ValueError("MONGO_URI environment variable not set. Please create a .env file.")

        # Create a new client and connect to the server
        self.client = MongoClient(uri, server_api=ServerApi('1'))
        
        # Select the database and collection
        self.db = self.client['biometric_auth_db']
        self.collection = self.db['users']

        # Send a ping to confirm a successful connection
        try:
            self.client.admin.command('ping')
            print("[INFO] Successfully connected to MongoDB!")
        except Exception as e:
            print(f"[ERROR] Could not connect to MongoDB: {e}")
            self.client = None

    def enroll_user(self, user_id: str, embedding: list) -> bool:
        """
        Saves or updates a user's embedding in the database.

        Args:
            user_id: The unique identifier for the user (e.g., username or email).
            embedding: The user's biometric embedding as a list of floats.

        Returns:
            True if the operation was successful, False otherwise.
        """
        if not self.client:
            return False
            
        try:
            # Use update_one with upsert=True.
            # This will create a new document if the user_id doesn't exist,
            # or update the existing one if it does.
            self.collection.update_one(
                {'_id': user_id},
                {'$set': {'embedding': embedding}},
                upsert=True
            )
            print(f"[INFO] Successfully enrolled/updated user: {user_id}")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to enroll user {user_id}: {e}")
            return False

    def get_user_embedding(self, user_id: str) -> list | None:
        """
        Retrieves a user's stored embedding from the database.

        Args:
            user_id: The unique identifier for the user.

        Returns:
            The user's embedding as a list, or None if the user is not found.
        """
        if not self.client:
            return None

        try:
            user_document = self.collection.find_one({'_id': user_id})
            if user_document:
                print(f"[INFO] Found embedding for user: {user_id}")
                return user_document.get('embedding')
            else:
                print(f"[WARN] No user found with ID: {user_id}")
                return None
        except Exception as e:
            print(f"[ERROR] Failed to retrieve embedding for {user_id}: {e}")
            return None

    def close_connection(self):
        """
        Closes the connection to the database.
        """
        if self.client:
            self.client.close()
            print("[INFO] MongoDB connection closed.")


# =========================================================================================
# --- Example Usage ---
# =========================================================================================
if __name__ == '__main__':
    # 1. Initialize the authenticator (connects to DB)
    auth = MongoAuthenticator()

    # 2. Example Enrollment
    # This is the data you would get from your model
    user_to_enroll = "kartik"
    enrollment_embedding = [-0.95, 0.20, 1.05, 0.45] # A sample embedding
    
    auth.enroll_user(user_id=user_to_enroll, embedding=enrollment_embedding)

    # 3. Example Verification
    # A user tries to log in. You need to retrieve their stored embedding to compare.
    user_to_verify = "kartik"
    stored_embedding = auth.get_user_embedding(user_id=user_to_verify)

    if stored_embedding:
        # Now you would generate a NEW embedding from the live sensor data
        # and compare it with 'stored_embedding' using cosine similarity.
        print(f"\nRetrieved embedding for verification: {stored_embedding}")
        # Example: if cosine_similarity(new_embedding, stored_embedding) > 0.9:
        #              print("User Authenticated!")
        #          else:
        #              print("Authentication Failed.")
    
    # 4. Close the connection when your application shuts down
    auth.close_connection()