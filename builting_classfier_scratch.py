import numpy as np
import statistics
class KNN_classfier():
    def __init__(self,distance_matrix):
        self.distance_matrix=distance_matrix
        
    def get_distance_matrice(self,X_train,test):
        matrix=[]
        if(self.distance_matrix=="eulidean"):
            dist=0
            for i in range(X_train):
                dist+=(X_train[i]+test[i])**2
            eclidean_dist=np.sqrt(dist)
            return eclidean_dist
        else:
            dist=0
            for i in range(X_train):
                dist+=abs(X_train[i]+test[i])
            
            return dist

    def nearest_neighbour(self,X_train,x_test,k):
        distance_list=[]
        for training_data in X_train:
            self.get_distance_matrice(training_data,x_test)
            distance_list.append((training_data,x_test))

        distance_list.sort(key=lambda x:x[1])
        neighbours_list=[]

        for i in range(k):
            neighbours_list.append(distance_list[i][0])

        return neighbours_list
    
    def predict(self,x_train,x_test,k):
        neighbors=self.nearest_neighbour(x_train,x_test,k)
        labels=[]
        for data in neighbors:
            labels.append(data[-1])
        
        predict_class=statistics.mode(labels)

        return predict_class