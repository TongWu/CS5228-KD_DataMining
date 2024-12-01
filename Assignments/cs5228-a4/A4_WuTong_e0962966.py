import numpy as np
import networkx as nx

from networkx.algorithms.shortest_paths import *


class MyLogisticRegression:
        
    def __init__(self):
        self.theta = None
    

    def add_bias(self, X):
        # Create a vector of size |X| (= number of samples) with all values being 1
        ones = np.ones(X.shape[0]).reshape(-1, 1)
        # Return new data matrix with the 1-vector stack "in front of" X
        return np.hstack([ones, X])

    
    def calc_loss(self, y, y_pred):
        # Calculate and return the Cross Entropy loss (binary classification)
        return (-y * np.log(y_pred) - (1 - y) * np.log(1 - y_pred)).mean()

    
    def calc_h(self, X):
        
        h = None
        
        #########################################################################################
        ### Your code starts here ###############################################################        

        z = np.dot(X, self.theta)
        h = 1 / (1 + np.exp(-z))
    
        ### Your code ends here #################################################################
        #########################################################################################
        
        return h
        

    def calc_gradient(self, X, y, h):
        
        grad = None
        
        #########################################################################################
        ### Your code starts here ###############################################################
        grad = 2 * np.dot(X.T, (h - y)) / y.size
        ### Your code ends here #################################################################
        #########################################################################################
        
        return grad

    
    
    
    def fit(self, X, y, lr=0.001, num_iter=100, verbose=False):

        # Add bias term x_0=1 to data
        X = self.add_bias(X)
        
        # weights initialization
        self.theta = np.random.rand(X.shape[1])

        for i in range(num_iter):

            #########################################################################################
            ### Your code starts here ###############################################################      

            h = self.calc_h(X)
            grad = self.calc_gradient(X, y, h)
            self.theta -= lr * grad
            
            ### Your code ends here #################################################################
            #########################################################################################        
            
            # Print loss every 10% of the iterations
            if verbose == True:
                if(i % (num_iter/10) == 0):
                    print('Loss: {:.6f} \t {:.0f}%'.format(self.calc_loss(y, h), (i / (num_iter/100))))

        # Print final loss
        if verbose == True:
            print('Loss: {:.6f} \t 100%'.format(self.calc_loss(y, h)))
    
        return self
    
    
    def predict(self, X, threshold=0.5):
        
        # Add bias term x_0=1 to data
        X = self.add_bias(X)
        
        y_pred = None

        #########################################################################################
        ### Your code starts here ###############################################################

        y_pred = (self.calc_h(X) >= threshold).astype(int)
    
        ### Your code ends here #################################################################
        #########################################################################################
        
        return y_pred
    
    
    
    
    
class NMF:
    
    def __init__(self, M, k=100):
        self.M, self.k = M, k
    
        num_users, num_items = M.shape
        
        self.Z = np.argwhere(M != 0)
        self.W = np.random.rand(num_users, k)
        self.H = np.random.rand(k, num_items)

        
        
    def calc_loss(self):
        
        loss = np.sum(np.square((self.M - np.dot(self.W, self.H)))[self.M != 0])

        return loss    
    
    
    
    def fit(self, learning_rate=0.0001, lambda_reg=0.1, num_iter=100, verbose=False):
        for it in range(num_iter):

            #########################################################################################
            ### Your code starts here ############################################################### 

            for i, j in self.Z:
                error_ij = self.M[i, j] - np.dot(self.W[i, :], self.H[:, j])

                # Update rules with gradient and regularization
                for k in range(self.k):
                    grad_w_ik = -2 * error_ij * self.H[k, j] + 2 * lambda_reg * self.W[i, k]
                    grad_h_kj = -2 * error_ij * self.W[i, k] + 2 * lambda_reg * self.H[k, j]

                    self.W[i, k] -= learning_rate * grad_w_ik
                    self.H[k, j] -= learning_rate * grad_h_kj

            ### Your code ends here #################################################################
            #########################################################################################           

            # Print loss every 10% of the iterations
            if verbose == True:
                if(it % (num_iter/10) == 0):
                    print('Loss: {:.5f} \t {:.0f}%'.format(self.calc_loss(), (it / (num_iter/100))))

        # Print final loss        
        if verbose == True:
            print('Loss: {:.5f} \t 100%'.format(self.calc_loss()))        
        
        
    def predict(self):
        #
        return np.dot(self.W, self.H)
    
    
    
    
    

    



def girvan_newman(G_orig, verbose=False):
    
    # Create a copy so we do not modify the original Graph G
    G = G_orig.copy()
    
    # Compute the components of Graph G (assume G to be undirected in strongly connected)
    components = list(nx.connected_components(G))

    while len(components) < 2:
        
        #########################################################################################
        ### Your code starts here ############################################################### 

        # Calculate edge betweenness centrality for all edges
        edge_betweenness = nx.edge_betweenness_centrality(G)

        # Find the edge with the highest betweenness centrality
        max_centrality = max(edge_betweenness.values())
        edges_to_remove = [edge for edge, centrality in edge_betweenness.items() if centrality == max_centrality]

        # Randomly pick one edge with maximum edge betweenness centrality
        edge_to_remove = edges_to_remove[0]

        # Remove the chosen edge from the graph
        G.remove_edge(*edge_to_remove)

        if verbose:
            print(f"Edge {edge_to_remove} removed (edge betweenness centrality: {max_centrality:.3f})")

        ### Your code ends here #################################################################
        #########################################################################################             
            
        # Get all connected components of the graph again
        components = list(nx.connected_components(G))

    
    # Once we split the graph, we return the components sorted by their sizes (largest first)
    return sorted(components, key=len, reverse=True), G



