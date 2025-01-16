import numpy as np
import pandas as pd

#use pandas to load real_estate_dataset.csv
df = pd.read_csv('real_estate_dataset.csv')

#get the number of samples and features
n_samples, n_features = df.shape

print(f"Number of samples, features: {n_samples, n_features}")

# get the names of the columns
columns = df.columns
print(f"Columns: {columns}")

#save the column names to a file for accesing later as text file
np.savetxt('columns.txt', columns, fmt='%s')

# use Square_Feet, garage_size, Location_Score, Distance_Center as features
x = df[['Square_Feet', 'garage_size', 'Location_Score', 'Distance_Center']]

#use price as the` target
y = df['Price'].values

print(f"shape of x: {x.shape}")
print(f"shape of y: {y.shape}")

#get the number of the samples and features in X
n_samples, n_features = x.shape

# Build a linear model to predict the price from the four features in X
# make an array of coefs of the size of n_features+!, initialized to 1

coefs = np.ones(n_features+1)

#predict the price for the each sample in X
predictions_bydefn = x@coefs[1:] + coefs[0]

#append a column of ones to X
x = np.hstack((np.ones((n_samples, 1)), x))

#predict the price for each sample in X
predictions = x@coefs

#see if all enteries in predictions_bydefn and predictions are the same
is_same = np.allclose(predictions_bydefn, predictions)

print(f"are the predictions the same with x*coefs[1:] + coefs[0] and x@coefs: {is_same}\n")

#calculate the error using  predictions and y
errors = y - predictions

# Calculate the relative errors
rel_errors = errors / y

# calculate the mean of the square of the errors using the loop
loss_loop = 0
for i in range(n_samples):
    loss_loop += errors[i]**2

loss_loop /= n_samples

# calculate the mean of the square of the errors using the matrix operations
loss_matrix = np.trasa(errors)@errors/n_samples

#compare the two methods of calculating the mean of the square of the errors
is_diff = np.allclose(loss_loop,loss_matrix)
print(f"Are the loss by direct and matirx same?{is_diff}\n")

#print the size of errors, and its L2 norm
print(f" Size of the errors: {errors.shape}")
print(f"L2 norm of the errors: {np.linalg.norm(errors)}")
print(f"L2 norm of the relative error: {np.linalg.norm(rel_errors)}")


