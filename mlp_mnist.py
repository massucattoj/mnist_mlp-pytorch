"""
    A construção de um modelo de uma Rede Neural MLP para resolver o problema
    das imagens compostas no dataset MNIST é composta de 4 partes:
        1. Carregar as imagens do dataset e dividi-las em treinamento, validacao e teste
        2. Criar a arquitetura da rede neural que sera usada para resolver o problema de classificação
        3. Definir a funcao de perda e otimizacao
        4. Treinar a rede neural criada
        
    @author: ebrithil
    Agradecimentos pelo conhecimento adquirido: Pytorch Udacity
"""

##
# Import libraries
#
import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt

###############################################################################
################          DATA PRE PROCESSING          ########################
###############################################################################
# Load and split into train, validation and test
num_workers = 0      # number of subprocess to use for data loading
batch_size = 20      # number of samples that will be load per batch
validation = 0.2     # size of validation set
transform = transforms.ToTensor() # converter os dados para o formato FloatTensor

# MNIST Dataset
train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='data', train=False, download=True, transform=transform)

# Obtendo os indices de treinamento que serao usados para validacao
num_train = len(train_data)         # Numero de amostras no conjunto de treinamento
indices = list(range(num_train))    # Lista de indices
np.random.shuffle(indices)          # Embaralhar lista de indices
split = int(np.floor(validation * num_train))
train_idx, valid_idx = indices[split:], indices[:split] # As primeiras amostras para validacao e o restante para treinamento

# Define um conjunto de amostras aleatorias para os batches de treino e validacao
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# Prepare dataloaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size, sampler=train_sampler, num_workers=num_workers)
validation_loader = torch.utils.data.DataLoader(train_data, batch_size, sampler=valid_sampler, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size, num_workers)


##
# Visualize a batch sample of training data
# -> Obtain one batch of training images
dataiter = iter(train_loader)
images, labels = dataiter.next()
images = images.numpy()

# -> Plotar as imagens do batch com seu label correspondente
fig = plt.figure(figsize=(25,4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(images[idx]), cmap='gray')
    
    # Print o label correto de cada imagem
    # .item() gets the value contained in a Tensor
    ax.set_title(str(labels[idx].item()))    

# Ver a imagem a partir dos pixels
img = np.squeeze(images[1])
fig = plt.figure(figsize = (12,12)) 
ax = fig.add_subplot(111)
ax.imshow(img, cmap='gray')
width, height = img.shape
thresh = img.max()/2.5
for x in range(width):
    for y in range(height):
        val = round(img[x][y],2) if img[x][y] !=0 else 0
        ax.annotate(str(val), xy=(y,x),
                    horizontalalignment='center',
                    verticalalignment='center',
                    color='white' if img[x][y]<thresh else 'black')    
    
# End of pre processing
# -----------------------------------------------------------------------------
        
        
###############################################################################
##############    DEFINE THE NEURAL NETWORK ARCHITECTURE    ###################
###############################################################################
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        
        super(Net, self).__init__()     
        # Numero de neuronios da camada escondida
        hidden_1 = 512
        hidden_2 = 512
        
        self.fc1 = nn.Linear(28*28, hidden_1) # 28*28 = Flatten image (Tratando-se de uma MLP a entrada precisa ser um array)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.fc3 = nn.Linear(hidden_2, 10) # 10 = number of output categories
        self.dropout = nn.Dropout(0.2) # Dropout of 20%
        
    def forward(self, x):
        
        # Flatten image
        x = x.view(-1, 28*28) # parametro -1 a funcao mesmo escolhe o melhor tamanho
        x = F.relu(self.fc1(x)) # Add a hiden layer with ReLU activation
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x) # OBS: Como é a ultima camada nao há necessidade de dropout!
        
        return x
    
# Initialize the Neural Network and print the Architecture
model = Net()
print(model)
        
# End of defining NN architecture
# -----------------------------------------------------------------------------


"""
    Por se tratar de um problema de classificação umas das melhores funções de perda para 
    problemas do genero é utilizar a Cross-Entropy Loss.
"""        
# Especificar as funcoes de Perda e Otimização.
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
# Use GPU if it's available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device);

###############################################################################
#############    TRANING THE NEURAL NETWORK ARCHITECTURE    ###################
###############################################################################
"""
    - Definir o numero de epocas
    - Preparar o modelo para treinamento
    - Criar um loop atraves das epocas definidas
"""
epochs = 60             # Number of epochs
valid_loss_min = np.Inf # Set initial "min" to infinity // Initialize tracker  for minimum validation loss


for epoch in range(epochs):
    
    # Monitorar o custo da Loss Function
    training_loss = 0.0
    validation_loss = 0.0 
    
    ######################
    # TRAINING THE MODEL #
    ######################
    model.train()
    for data, target in train_loader:
        
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()               # Clear the gradient of all optimized variables
        output = model(data)                # Forward pass: Computar valores previstos passando as entradas para o modelo
        loss = criterion(output, target)    # Calculando o valor da perda
        loss.backward()                     # Backward pass: compute gradient of the loss with respect to model parameters
        optimizer.step()                    # Perform a single optimization step (parameter update)
        training_loss += loss.item()*data.size(0) # Update running training loss
    
    ######################
    # TRAINING THE MODEL #
    ######################
    model.eval()
    for data, target in validation_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        validation_loss += loss.item()*data.size(0)
    
    # Print training statistics 
    # Calculate average loss over an epoch
    training_loss = training_loss/len(train_loader.dataset)
    validation_loss = validation_loss/len(validation_loader.dataset)
    
    print('Epoch: {} \tTraining Loss: {:.6f} \Validation Loss: {:.6f}'.format(epoch+1, training_loss, validation_loss))
    
    # Save model if validation loss has decreased
    if validation_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        valid_loss_min,
        validation_loss))
        torch.save(model.state_dict(), 'model.pt')
        valid_loss_min = validation_loss
        
# End of training model
# -----------------------------------------------------------------------------
        
        
###############################################################################
#############    TESTING THE NEURAL NETWORK ARCHITECTURE    ###################
###############################################################################   

# Load the best model, with the Lowest Validation Loss
model.load_state_dict(torch.load('model.pt'))

"""
    -> Inicializar as listas para monitorar loss e accuracy
    -> Setar o modelo para modo de teste
"""
test_loss = 0.0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

model.eval() # Prepare model for evaluation

for data, target in test_loader:
        
    data, target = data.to(device), target.to(device)
    output = model(data) # Forward pass
    loss = criterion(output, target)
    test_loss += loss.item()*data.size(0)
    
    _, pred = torch.max(output, 1)                           # Convert output probabilities to predicted class
    correct = np.squeeze(pred.eq(target.data.view_as(pred))) # Compare predictions to true label
    
    #  Calculate test accuracy for each object class    
    for i in range(batch_size):
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1
    

# Calculate and print avg test loss
test_loss = test_loss/len(test_loader.dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))

for i in range(10):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            str(i), 100 * class_correct[i] / class_total[i], np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (class_total[i]))
        
print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (100. * np.sum(class_correct) / np.sum(class_total),
                                                      np.sum(class_correct), np.sum(class_total)))
    
    
        
        