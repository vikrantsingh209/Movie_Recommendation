#Movie Rating Undertanding using Auto Encoder
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import matplotlib.pyplot as plt

# Importing the libraries
import numpy as np
import pandas as pd



# Importing the dataset
movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

# Preparing the training set and the test set
training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int')
test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')

# Getting the number of users and movies
nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))

# Converting the data into an array with users in lines and movies in columns
def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[:,1][data[:,0] == id_users]
        id_ratings = data[:,2][data[:,0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data
training_set = convert(training_set)
test_set = convert(test_set)

# Converting the data into Torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Creating the architecture of the Neural Network
class SAE(nn.Module):# SAE - Stacked AutoEncoder
    def __init__(self, ):
        super(SAE, self).__init__()
        self.fc1 = nn.Linear(nb_movies, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 20)
        self.fc4 = nn.Linear(20, nb_movies)
        self.activation = nn.Sigmoid()
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x
sae = SAE()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5)

plt_loss = []
plt_epoch = []
# Training the SAE
nb_epoch = 10
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.
    for id_user in range(nb_users):
        input = Variable(training_set[id_user]).unsqueeze(0)
        target = input.clone()
        if torch.sum(target.data > 0) > 0:
            output = sae(input)
            target.require_grad = False
            output[target == 0] = 0
            loss = criterion(output, target)
            mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)


            
            loss.backward()
            train_loss += np.sqrt(loss.data * mean_corrector)
            s += 1.
            optimizer.step()

    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))
    plt_loss.append([train_loss/s])
    plt_epoch.append([epoch])
# Testing the SAE
test_loss = 0
s = 0.
test_flag = 0
for id_user in range(nb_users):
    input = Variable(training_set[id_user]).unsqueeze(0)
    if(test_flag == 0):
        print(input)
    test_flag += 1
    target = Variable(test_set[id_user])
    if torch.sum(target.data > 0) > 0:
        output = sae(input)
        target.require_grad = False
        output[(target == 0).unsqueeze(0)] = 0
        loss = criterion(output, target)
        mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
        test_loss += np.sqrt(loss.data * mean_corrector)
        s += 1.
        
print('test loss: '+str(test_loss/s))

#checking what new user have seen

new_user = pd.read_csv('new_user.inp', sep = '\t', header = None, engine = 'python', encoding = 'latin-1')

movies_watched = new_user.iloc[:,[1,2]].values

all_ratings = np.zeros(1682)

for val,rating in movies_watched:
    all_ratings[val] = rating

all_ratings = list(all_ratings)
all_ratings = [all_ratings]
user_data = torch.FloatTensor(all_ratings)

out_new = sae(user_data)
out_new = out_new.tolist()[0]

suggestions = {"val":[],"index":[],"movie":[]}
s_m_i = 1 #suggested movie index

movies = movies.iloc[:,[0,1]].values
for val in out_new:

    if(val>(3.0) and s_m_i > 0):
        suggestions["val"].append(val)
        suggestions["index"].append(s_m_i)
        suggestions["movie"].append(movies[s_m_i - 1][1])
    s_m_i += 1

file_out = pd.DataFrame(data = suggestions)
file_out.to_csv("./suggestions.csv", sep=',',index=False)

plt.plot(plt_epoch,plt_loss, color = "red")
plt.show()