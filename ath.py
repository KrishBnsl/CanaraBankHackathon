import numpy as np
from collections import Counter

def adaptive_threshold_heuristic(
    score_history,
    proportion_limit=0.01,
    periodicity_limit=2
):
    """
    An adapted implementation of the Adaptive Thresholding Heuristic (ATH).

    This function calculates a dynamic threshold for a series of historical scores.
    For login similarity, we are looking for scores that are abnormally low, so
    the 'tail' is 'left' (lower scores are outliers).

    Args:
        score_history (list): A list of historical similarity scores for a user.
        proportion_limit (float): The maximum permitted proportion of scores that can be
                                  considered anomalies[cite: 18].
        periodicity_limit (int): The number of permitted periodic outlier occurrences[cite: 66].

    Returns:
        float: The calculated adaptive threshold.
    """
    if len(score_history) < 10:  # Need sufficient history to be meaningful
        return 0.90  # Return a default, strict threshold if history is short

    # 1. Get unique scores and sort them. For similarity, lower scores are
    #    anomalous, so we sort ascendingly ('left' tail)[cite: 6].
    score_array = np.array(score_history)
    unique_thresholds = sorted(np.unique(score_array))

    final_threshold = unique_thresholds[0]
    previous_threshold = final_threshold

    # 2. Iterate through possible thresholds to find one that breaks the constraints[cite: 101].
    for thresh in unique_thresholds:
        # Identify outliers: scores BELOW the current threshold are anomalous [cite: 14]
        outlier_indices = np.where(score_array < thresh)[0]
        outliers_found = score_array[outlier_indices]

        # 3. Check if the proportion limit is exceeded [cite: 18]
        if len(outliers_found) / len(score_array) > proportion_limit:
            break  # This threshold is too lenient; stop here

        # 4. Check if the periodicity limit is exceeded [cite: 18]
        if len(outlier_indices) > 1:
            # Compute temporal difference between outlier occurrences [cite: 141]
            diffs = np.diff(outlier_indices)
            diff_counts = Counter(diffs)
            
            # If any time difference occurs more than the limit, it's a periodic pattern
            if any(count > periodicity_limit for count in diff_counts.values()):
                break # This pattern is periodic, not anomalous; stop here

        # If we get here, the current threshold is valid. Save it and continue.
        previous_threshold = thresh

    # 5. Return the last valid threshold before the constraints were broken[cite: 102].
    final_threshold = previous_threshold
    return final_threshold

# Helper for cosine similarity
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

class LoginManager:
    def __init__(self):
        # Using a dictionary to simulate a user database
        self.user_db = {}

    def enroll_user(self, user_id, reference_embedding):
        """Stores the user's reference vector and initializes their history."""
        print(f"Enrolling user '{user_id}'.")
        self.user_db[user_id] = {
            "reference_embedding": np.array(reference_embedding),
            "login_score_history": [] # History is crucial for ATH
        }

    def login(self, user_id, new_login_embedding):
        """Simulates a user login, calculates a dynamic threshold, and verifies."""
        if user_id not in self.user_db:
            print("User not found.")
            return

        new_login_embedding = np.array(new_login_embedding)
        user_data = self.user_db[user_id]
        
        # Calculate the similarity for the current attempt
        current_similarity = cosine_similarity(
            new_login_embedding,
            user_data["reference_embedding"]
        )

        print(f"\n--- Attempting login for '{user_id}' ---")
        print(f"Current login similarity: {current_similarity:.4f}")

        # Calculate the adaptive threshold based on PAST history
        adaptive_threshold = adaptive_threshold_heuristic(
            user_data["login_score_history"]
        )
        print(f"Adaptive threshold for this user is: {adaptive_threshold:.4f}")

        # --- The Decision ---
        if current_similarity >= adaptive_threshold:
            print("✅ Login Successful!")
            # Optional: you could be more strict and require successful logins to
            # also be above a minimum global threshold, e.g., > 0.8
        else:
            print("❌ Login Failed: Similarity score is below the adaptive threshold.")

        # --- IMPORTANT ---
        # Add the new score to the user's history for future calculations
        user_data["login_score_history"].append(current_similarity)
        print("Updated user's login history.")
        return current_similarity >= adaptive_threshold


# # --- SIMULATION ---

# # 1. Initialize manager and enroll a user
# manager = LoginManager()
# # This is the "user vector" from your DB
# reference_vec = np.random.rand(128) - 0.5 
# manager.enroll_user("test_user", reference_vec)

# # 2. Simulate a series of mostly valid logins to build history
# print("\n--- Building login history ---")
# for i in range(20):
#     # Simulate a similar vector with slight noise
#     noise = (np.random.rand(128) - 0.5) * 0.1
#     login_vec = reference_vec + noise
#     manager.login("test_user", login_vec)

# # 3. Now, simulate a truly different login attempt (anomaly)
# print("\n--- Simulating an ANOMALOUS login ---")
# anomalous_vec = np.random.rand(128) - 0.5 # A completely different vector
# manager.login("test_user", anomalous_vec)

# # 4. Simulate another valid login
# print("\n--- Simulating another VALID login ---")
# valid_vec = reference_vec + (np.random.rand(128) - 0.5) * 0.05
# manager.login("test_user", valid_vec)