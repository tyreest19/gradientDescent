import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def hypothesis(x, theta0, theta1):
        return  theta0 + theta1 * x

def costFunction(x, y, theta0, theta1, m):
        loss = 0 
        for i in range(m): # Represents summation
                loss += (hypothesis(x[i], theta0, theta1) - y[i])**2
        loss *= 1 / (2 * m) # Represents 1/2m
        return loss

def gradientDescent(x, y, theta0, theta1, alpha, m, iterations=15000):
        for i in range(iterations):
                gradient0 = 0
                gradient1 = 0
                for j in range(m): # Represents summation
                        gradient0 += hypothesis(x[j], theta0, theta1) - y[j]
                        gradient1 += (hypothesis(x[j], theta0, theta1) - y[j]) * x[j]
                gradient0 *= 1/m
                gradient1 *= 1/m
                temp0 = theta0 - alpha * gradient0
                temp1 = theta1 - alpha * gradient1
                theta0 = temp0
                theta1 = temp1
                error = costFunction(x, y, theta0, theta1, len(y))
                print("Error is:", error)
        return theta0, theta1


def main():
        data = pd.read_csv('anime.csv')
        # print(data.corr())
        # print(data['members'].isnull().values.any()) # Prints False
        # print(data['rating'].isnull().values.any()) # Prints True

        members = [] # Corresponding fan club size for row 
        ratings = [] # Corresponding rating for row

        for row in data.iterrows():
                if not math.isnan(row[1]['rating']): # Checks for Null ratings
                        members.append(row[1]['members'])
                        ratings.append(row[1]['rating'])
                
        members = np.asarray(members)
        ratings = np.asarray(ratings)

        theta0 = 0.00001 # Random guess
        theta1 = 0.00001 # Random guess
        error = 0

        #fig = plt.figure(dpi=100, figsize=(5, 4))
        #plt.scatter(members, ratings)
        #line, = plt.plot(members, hypothesis(members,theta0, theta0))
        #plt.xlabel('Members')
        #plt.ylabel('Ratings')
        #plt.xlim((0, max(members)))
        #plt.ylim((0, max(ratings)))
        #plt.show()

        alpha = 0.000000000001 # Learning Rate
        m = len(ratings) # Size of the dataset
        #print('Our final theta\'s', gradientDescent(members, ratings, theta0, theta1, alpha, m))
        theta0, theta1 = gradientDescent(members, ratings, theta0, theta1, alpha, m, iterations=30000)
        print('theta0:', theta0, 'theta1:', theta1)
        #fig = plt.figure(dpi=100, figsize=(5, 4))
        #plt.scatter(members, ratings)
        #line,= plt.plot(members, hypothesis(members,theta0, theta0))
        #plt.xlabel('Members')
        #plt.ylabel('Ratings')
        #plt.xlim((0, max(members)))
        #plt.ylim((0, max(ratings)))
        #plt.save('finalgraph')

if __name__ == '__main__':
        main()
