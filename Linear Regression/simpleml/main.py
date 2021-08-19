from sklearn.linear_model import LinearRegression
import random

feature_set = []
target_set = []

nRows = 200
randomNLimit = 2000

for i in range(0, nRows):
    x = random.randint(0, randomNLimit)
    y = random.randint(0, randomNLimit)
    z = random.randint(0, randomNLimit)
    
    function = (10*x)+(2*y)+(3*z)

    feature_set.append([x, y, z])
    target_set.append(function)

# Model

model = LinearRegression()
model.fit(feature_set, target_set)

# Testing Data set
test_set = [[8, 4, 7]] # Expected Output = function(8,4,7) = (10*8) + (2*4) + (3*7) =  109
prediction = model.predict(test_set)

test_set_2 = [[9, 2, 2]]  # Expected Output = function(9,2,2) = (10*9) + (2*2) + (3*2) =  100
prediction2 = model.predict(test_set_2)


print('Prediction:' + str(prediction) + '   Co - Efficient: ' + str(model.coef_))
print('Prediction:' + str(prediction2) + '   Co - Efficient: ' + str(model.coef_))



