from importlib.metadata import requires
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import random
def fixed_points(net, activity_dict, hidden_size, trial_length, task, order, input_raw=None):
    # Freeze for parameters in the recurrent network
    for param in net.parameters():
        param.requires_grad = False

    batch_size = 128

    # Inputs should be the 0-coherence mean input during stimulus period
    # This will be task-specific
    
    if input_raw == None:
        if task == ('MyTaskCopy-v0' or 'MyTaskSim-v0'):
            input_raw = [1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0]
        else:
            input_raw = [1, 0.5, 0.5]
        # input = []
    #     for i in range(batch_size):
    #         a = random.randint(0,180)
    #         input_tmp = [1, np.sin(a), np.cos(a)]
    #         input.extend(input_tmp)
    # input = np.array(input).reshape(3, batch_size).transpose()
    input = np.tile(input_raw, (batch_size, 1))
    input = torch.tensor(input, dtype=torch.float32)
 
    # Here hidden activity is the variable to be optimized
    # Initialized randomly for search in parallel (activity all positive)
    if order == 0:
        hidden = np.zeros((batch_size, hidden_size))
        k = 0
        for i in range(batch_size):
            for j in range(trial_length):
                    hidden[k] = activity_dict[i][j]
                    k += 1
                    if k == batch_size:
                        break
            if k == batch_size:
                break
        hidden = torch.tensor(hidden, requires_grad=True, dtype=torch.float32)
    else:
        hidden = torch.tensor(np.random.rand(batch_size, hidden_size)*3,
                            requires_grad=True, dtype=torch.float32) # random
    # Use Adam optimizer
    optimizer = optim.Adam([hidden], lr=0.001) #对hidden做optimization而不是nn.params
    criterion = nn.MSELoss() # argmin|Fx|^2

    running_loss = 0
    epochs = 20000
    # i = 0
    for i in range(epochs):
    # while True:
        optimizer.zero_grad()   # zero the gradient buffers
        
        # Take the one-step recurrent function from the trained network

        new_h = net.rnn.recurrence(input, hidden) 
        loss = criterion(new_h, hidden)
        # new_h = net.rnn.recurrence(input, (hidden,hidden)) #EIRNN中new_h是个tuple，但后续需要tensor
        # loss = criterion(new_h[1], hidden)

        # if loss < 1e-03:
        #     print('Step {}, Loss {:0.5f}'.format(i+1, running_loss))
        #     break
        
        loss.backward()
        optimizer.step()    # Does the update

        running_loss += loss.item()
        if i % 10000 == 9999:
            running_loss /= 10000
            # if i+1 == 10000:
            print('Step {}, Loss {:0.5f}'.format(i+1, running_loss))
            running_loss = 0

            # hidden_ = hidden.detach().numpy()
            # i_fp = np.argsort(hidden_[:, 0])[int(hidden_.shape[0]/2)]
            # fp = torch.from_numpy(hidden_[i_fp])
            # fp.requires_grad = True
            # input = torch.tensor(input_raw, dtype = torch.float32)
            # deltah = net.rnn.recurrence(input, fp) - fp
            # print(deltah)

        # i += 1
    fixedpoints = hidden.detach().numpy() # batch_size * num_neuron; activity: trial_length * num_neuron

    # index of fixed point to focus on
    # choose one close to center by sorting PC1
    i_fp = np.argsort(fixedpoints[:, 0])[int(fixedpoints.shape[0]/2)]
    
    fp = torch.from_numpy(fixedpoints[i_fp])
    fp.requires_grad = True

    # Inputs should be the 0-coherence mean input during stimulus period
    # This will be task-specific
    input = torch.tensor(input_raw, dtype = torch.float32)
    deltah = net.rnn.recurrence(input, fp) - fp
    # print(deltah)
    # activity = np.concatenate(list(activity_dict[i] for i in range(500)), axis=0)
    # pca = PCA(n_components=2)
    # pca.fit(activity)
    # speed = np.squeeze(pca.transform(np.expand_dims(deltah.detach().numpy(),axis=0)))
    # print(speed)

    jacT = torch.zeros(hidden_size, hidden_size)
    for i in range(hidden_size):                                                                                                                     
        output = torch.zeros(hidden_size)                                                                                                          
        output[i] = 1.                                                                                                                     
        jacT[:,i] = torch.autograd.grad(deltah, fp, grad_outputs=output, retain_graph=True)[0]
        
    jac = jacT.detach().numpy().T

    eigval, eigvec = np.linalg.eig(jac)

    # plt.figure()
    # plt.scatter(np.real(eigval), np.imag(eigval))
    # plt.plot([0, 0], [-1, 1], '--')
    # plt.xlabel('Real')
    # plt.ylabel('Imaginary')

    # vec = np.real(eigvec[:, np.argmax(eigval)]) 
    # end_pts = np.array([+vec, -vec]) * 2
    # end_pts = pca.transform(fp.detach().numpy() + end_pts)
    # fixedpoints = fixedpoints.transpose(1,0)

    return fixedpoints