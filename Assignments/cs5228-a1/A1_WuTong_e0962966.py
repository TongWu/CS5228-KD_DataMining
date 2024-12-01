import numpy as np
import pandas as pd
from distributed.utils_test import cluster

from sklearn.metrics.pairwise import euclidean_distances



def clean(df_condos_dirty):
    """
    Handle all "dirty" records in the condos dataframe

    Inputs:
    - df_condos_dirty: Pandas dataframe of dataset containing "dirty" records

    Returns:
    - df_condos_cleaned: Pandas dataframe of dataset without "dirty" records
    """   
    
    # We first create a copy of the dataset and use this one to clean the data.
    df_condos_cleaned = df_condos_dirty.copy()

    #########################################################################################
    ### Your code starts here ###############################################################

    # Remove duplicate records based on 'transaction_id' which is a primary key
    df_condos_cleaned = df_condos_cleaned.drop_duplicates(subset='transaction_id', keep='first')

    # Remove records where 'date_of_sale' is not in the expected format "MMM-YY" (e.g., "mar-19")
    df_condos_cleaned = df_condos_cleaned[df_condos_cleaned['date_of_sale'].str.match(r'^[a-z]{3}-\d{2}$', case=False, na=False)]

    # Remove records where 'postal_district' is not in Singapore standard postal district format (1-28)
    df_condos_cleaned = df_condos_cleaned[df_condos_cleaned['postal_district'].between(1, 28)]

    ### Your code ends here #################################################################
    #########################################################################################
    
    return df_condos_cleaned


def handle_nan(df_condos_nan):
    """
    Handle all nan values in the condos dataframe

    Inputs:
    - df_condos_nan: Pandas dataframe of dataset containing nan values

    Returns:
    - df_condos_no_nan: Pandas dataframe of dataset without nan values
    """       

    # We first create a copy of the dataset and use this one to clean the data.
    df_condos_no_nan = df_condos_nan.copy()

    #########################################################################################
    ### Your code starts here ###############################################################

    # Replace 'nan' values in 'postal_district', 'area_sqft', 'price' with the mean of the column
    numerical_cols = ['postal_district', 'area_sqft', 'price']
    for col in numerical_cols:
        mean_value = df_condos_no_nan[col].mean()
        df_condos_no_nan[col].fillna(mean_value, inplace=True)

    # Replace 'nan' in categorical columns with the mode of the column
    categorical_cols = ['type', 'subzone', 'planning_area', 'date_of_sale', 'floor_level', 'eco_category']
    for col in categorical_cols:
        mode_value = df_condos_no_nan[col].mode().values[0]
        df_condos_no_nan[col].fillna(mode_value, inplace=True)

    # Remove records with missing 'transaction_id'
    df_condos_no_nan = df_condos_no_nan.dropna(subset=['transaction_id'])

    # Replace 'nan' values in 'url' and 'name' with 'Unknown'
    df_condos_no_nan['url'].fillna('Unknown', inplace=True)
    df_condos_no_nan['name'].fillna('Unknown', inplace=True)

    ### Your code ends here #################################################################
    #########################################################################################
    
    return df_condos_no_nan


def extract_facts(df_condos_facts):
    """
    Extract the facts as required from the condos dataset

    Inputs:
    - df_condos_facts: Pandas dataframe of dataset containing the cars dataset

    Returns:
    - Nothing; you can simply us simple print statements that somehow show the result you
      put in the table; the format of the  outputs is not important; see example below.
    """       

    #########################################################################################
    ### Your code starts here ###############################################################

    # (1) earliest transaction date
    earliest_date = df_condos_facts['date_of_sale'].min()

    # (2) number of transactions per type
    type_counts = df_condos_facts['type'].value_counts()

    # (3) number of condos in Redhill costing more than SGD 2,000,000
    redhill_expensive = \
    df_condos_facts[(df_condos_facts['subzone'] == 'redhill') & (df_condos_facts['price'] > 2000000)].shape[0]

    # (4) planning area with the most transactions
    most_transactions_area = df_condos_facts['planning_area'].value_counts().idxmax()
    most_transactions_count = df_condos_facts['planning_area'].value_counts().max()

    # (5) condo with the highest price-to-area ratio in postal district 11
    df_postal_11 = df_condos_facts[df_condos_facts['postal_district'] == 11].copy()  # 使用 .copy() 防止链式赋值
    df_postal_11.loc[:, 'price_per_sqft'] = df_postal_11['price'] / df_postal_11['area_sqft']
    highest_ratio_condo = df_postal_11.loc[df_postal_11['price_per_sqft'].idxmax()]
    highest_ratio_name = highest_ratio_condo['name']
    highest_ratio_value = round(highest_ratio_condo['price_per_sqft'], 2)

    # (6) correlation between resale price and area
    correlation_price_area = df_condos_facts['price'].corr(df_condos_facts['area_sqft'])

    # (7) number of transactions between 50th and 60th floor
    floor_range_transactions = df_condos_facts[df_condos_facts['floor_level'].isin(['50 to 55', '56 to 60'])].shape[0]

    # Print the results
    print(f"(1) Earliest transaction date: {earliest_date}")
    print(f"(2) Transactions per type:\n{type_counts}")
    print(f"(3) Number of condos in Redhill costing more than SGD 2,000,000: {redhill_expensive}")
    print(
        f"(4) Planning area with the most transactions: {most_transactions_area} with {most_transactions_count} transactions")
    print(
        f"(5) Condo with the highest price-to-area ratio in postal district 11: {highest_ratio_name} with {highest_ratio_value} SGD per sqft")
    print(f"(6) Correlation between resale price and area: {correlation_price_area}")
    print(f"(7) Transactions between 50th and 60th floor: {floor_range_transactions}")
    
    ### Your code ends here #################################################################
    #########################################################################################

    
    
    
    
    
    
    
class MyKMeans:
    
    def __init__(self, n_clusters=8, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.cluster_centers_ = None
        self.labels_ = None
        self.n_iter_ = 0
        
        
    def initialize_centroids(self, X):
    
        # Pick the first centroid randomly
        c1 = np.random.choice(X.shape[0], 1)

        # Add first centroids to the list of cluster centers
        self.cluster_centers_ = X[c1]

        # Calculate and add c2, c3, ..., ck (we assume that we always have more unique data points than k!)
        while len(self.cluster_centers_) < self.n_clusters:

            # c is a data point representing the next centroid
            c = None

            #########################################################################################
            ### Your code starts here ###############################################################

            # Distance from each data point to the nearest centroid
            distance = euclidean_distances(X, self.cluster_centers_)
            min_distance = np.min(distance, axis=1)

            # Probability of each data point being selected as the next centroid
            squared_distance = min_distance ** 2
            probabilities = squared_distance / squared_distance.sum()

            # Randomly select the next centroid based on the probabilities
            c_index = np.random.choice(X.shape[0], p=probabilities)
            c = X[c_index]
            
            ### Your code ends here #################################################################
            #########################################################################################                

            # Add next centroid c to the array of already existing centroids
            self.cluster_centers_ = np.concatenate((self.cluster_centers_, [c]), axis=0)

    
    
    def assign_clusters(self, X):
        # Reset all clusters (i.e., the cluster labels)
        self.labels_ = None

        #########################################################################################
        ### Your code starts here ###############################################################    

        # Calculate the distance between each data point and each centroid
        distances = euclidean_distances(X, self.cluster_centers_)
        # Assign each data point to the nearest centroid
        self.labels_ = np.argmin(distances, axis=1)
            
        ### Your code ends here #################################################################
        #########################################################################################

    

    def update_centroids(self, X):

        # Initialize list of new centroids with all zeros
        new_cluster_centers_ = np.zeros_like(self.cluster_centers_)

        for cluster_id in range(self.n_clusters):

            new_centroid = None

            #########################################################################################
            ### Your code starts here ###############################################################

            # Calculate the new centroid for each cluster
            points_in_cluster = X[self.labels_ == cluster_id]

            # If there are no points in the cluster, keep the old centroid
            new_centroid = points_in_cluster.mean(axis=0)

            # If there are points in the cluster, calculate the new centroid
            new_cluster_centers_[cluster_id] = new_centroid

            ### Your code ends here #################################################################
            #########################################################################################

            new_cluster_centers_[cluster_id] = new_centroid  
            
        # Check if old and new centroids are identical; if so, we are done
        done = (self.cluster_centers_ == new_cluster_centers_).all()    
        
        # Update list of centroids
        self.cluster_centers_ = new_cluster_centers_

        # Return TRUE if the centroids have not changed; return FALSE otherwise
        return done
    
    
    def fit(self, X):
        
        self.initialize_centroids(X)

        self.n_iter_ = 0
        for _ in range(self.max_iter):
            
            # Update iteration counter
            self.n_iter_ += 1
            
            # Assign cluster
            self.assign_clusters(X)

            # Update centroids
            done = self.update_centroids(X)

            if done:
                break

        return self    