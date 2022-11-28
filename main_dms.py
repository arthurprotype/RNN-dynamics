#coding=UTF-8 
from cProfile import label
import sys
import time
from turtle import color
# sys.path.insert(0, "c:\\Users\\77485\\Desktop\\Vscode\\rnn")
from network import RNNNet
import neurogym as ngym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA,TruncatedSVD
from scipy import stats
import seaborn as sns 
import csv
from mpl_toolkits import mplot3d
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as MSE
from regression_projection import generalize_q_twovar, generalize_q_threevar
from condition_average import condition_average_twovar, condition_average_threevar, condition_average_fourvar, condition_average_pdm, condition_average_dms
from plot import plot_twovar, plot_threevar, plot_fourvar, plot_pdm, plot_dms, plot_var, plot_pca_threevar, plot_pca_twovar ,plot_firing_rate
from fixed_points import fixed_points

import pickle as pl

# from PIL import Image

# Environment

task = 'PerceptualDecisionMaking-v0'
plot_ = plot_pdm
condition_average = condition_average_pdm

# task = 'DelayMatchSample-v0'
# plot_ = plot_dms
# condition_average = condition_average_dms

print(task)

hidden_size = 256
lr = 0.0001

seq_len = 100
running_loss = 0
num_trial = 500

kwargs = {'dt': 100}
# Make supervised dataset
dataset = ngym.Dataset(task, env_kwargs=kwargs, batch_size=16,
                    seq_len=seq_len)

# A sample environment from dataset
env = dataset.env

# Network input and output size

timing = env.timing
marker_list = []
trial_length = 0
for i in timing.keys():
    marker_list.append(trial_length)
    trial_length += int(timing[i] / env.dt)

input_size = env.observation_space.shape[0]
output_size = env.action_space.n
loss_list = []

# Instantiate the network and print information
init_train = True
init_randn = False

import datetime
import os

file_name_0 = 'DMS' if task == 'DelayMatchSample-v0' else 'PDM'
file_name_1 = ' init_train' if init_train == True else ' init_nottrain'
file_name_2 = ' init_randn ' if init_randn == True else ' init_zeros '

position = 0.5
tau = 100
dt = 100
file_name_5 = 'tau='+str(tau)+' dt='+str(dt)+' ' 
epochs = 10000 if file_name_0 == 'DMS' else 5000
# epochs = 10000 if position <= 1.1 else 20000
file_name_4 = 'hidden_kaiming '
if 'kaiming' in file_name_4:
    position = 1
kai_bool = True if 'kaiming' in file_name_4 else False
file_name_3 = 'pos='+str(position)+' delay='+str(int(timing[i] / env.dt))+' '

mkfile_time = datetime.datetime.strftime(datetime.datetime.now(), '%m-%d-%H:%M') #这里是引用时间
mkfile_time = file_name_0 + file_name_1 + file_name_2 + file_name_3 + file_name_4 + file_name_5 + mkfile_time #注意这里在下面是用作文件夹名字的，里面也用了列表使每个文件夹的名字不一样
figure_save_path = mkfile_time #给文件夹赋予名字

for run_batch_num in range(1,6):
    print("batch_order: "+str(run_batch_num))
    # lr = 0.0001 * run_batch_num
    # print('lr: '+str(lr))
    net = RNNNet(input_size=input_size, hidden_size=hidden_size,
                    output_size=output_size, dt=int(dt), init_train=init_train, init_randn=init_randn, position=position, kaiming=kai_bool, tau=tau)

    hidden_weight_0 = list(net.parameters())[2].detach().numpy()
    hidden_eigval_0, hidden_eigvec_0 = np.linalg.eig(hidden_weight_0)

    # Use Adam optimizer
    optimizer = optim.Adam(net.parameters(), lr=lr)#, weight_decay=0.01) ##定义param为初值，可以对初值做训练
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.MSELoss()

    for i in range(epochs):
        inputs, labels = dataset()
        
        inputs = torch.from_numpy(inputs).type(torch.float) #(100,16,9)
        labels = torch.from_numpy(labels.flatten()).type(torch.long) #(100,16) -> (1600)
        
        # in your training loop:
        optimizer.zero_grad()   # zero the gradient buffers
        output_raw, rnn_activity, speed = net(inputs) 
        output = output_raw.view(-1, output_size) #(100,16,3) -> (1600,3) 
        # output_ = output.detach().numpy()
        # print(output_.shape)
        # print(labels.shape)
        
        loss = criterion(output, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), max_norm=1, norm_type=2)
        optimizer.step()    # Does the update
        running_loss += loss.item()
        
        if i % 1000 == 999:
            running_loss /= 1000
            print('Step {}, Loss {:0.5f}'.format(i+1, running_loss))
            loss_log = running_loss
            # acc_list.append(np.mean(train_correct[i-50:i+1]))
            running_loss = 0

    env.reset(no_step=True)

    # check一下net的eigenvalue

    if init_train == True:
        init_weight = list(net.parameters())[0].detach().numpy()
        init_u, init_eigval, init_v = np.linalg.svd(init_weight)
        hidden_weight = list(net.parameters())[3].detach().numpy()
        print(np.sort(init_eigval))
    else:
        hidden_weight = list(net.parameters())[2].detach().numpy()
    hidden_eigval, hidden_eigvec = np.linalg.eig(hidden_weight)
    
    fig0 = plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.scatter(np.real(hidden_eigval_0), np.imag(hidden_eigval_0),color='red',s=10)
    plt.scatter(np.real(hidden_eigval), np.imag(hidden_eigval),s=10)
    plt.plot([position, position], [-0.7, 0.7], '--') # y轴  
    plt.plot([position-0.5, position+0.5], [0, 0], '--') # x轴
    plt.xlabel('Real')
    plt.ylabel('Imaginary')

    activity_dict = {}
    activity_dict_1 = {}
    activity_dict_2 = {}
    activity_dict_3 = {}
    activity_dict_4 = {}
    activity_dict_5 = {}
    speed_dict = {}
    trial_infos = {}
    inputs_list = []
    input_type_list = []
    for i in range(num_trial):
        env.new_trial()
        ob, gt = env.ob, env.gt
        inputs = torch.from_numpy(ob[:, np.newaxis, :]).type(torch.float) # 维度 ob(22,1,3) gt(22,) input&output(100,16,3) hidden(100,16,64)
        output, rnn_activity, speed = net(inputs)
        # Compute performance
        output = torch.softmax(output, -1)
        output = output.detach().numpy()
        choice = np.argmax(output[-1, 0, :])
        correct = choice == gt[-1]

        # Log trial info
        trial_info = env.trial
        trial_info.update({'correct': correct, 'choice': choice})
        trial_infos[i] = trial_info

        if file_name_0 == 'DMS':
            if trial_info['stim_theta'] == 0 and trial_info['ground_truth'] == 0 and len(inputs_list) == 0:
                inputs_list.append(inputs)
                input_type = 'stim: '+str(round(trial_info['stim_theta'],2))+' gt: '+str(trial_info['ground_truth'])
                input_type_list.append(input_type)
                print(input_type)

            if trial_info['stim_theta'] == 0 and trial_info['ground_truth'] != 0 and len(inputs_list) == 1:
                inputs_list.append(inputs)
                input_type = 'stim: '+str(round(trial_info['stim_theta'],2))+' gt: '+str(trial_info['ground_truth'])
                input_type_list.append(input_type)
                print(input_type)

            if trial_info['stim_theta'] != 0 and trial_info['ground_truth'] == 0 and len(inputs_list) == 2:
                inputs_list.append(inputs)
                input_type = 'stim: '+str(round(trial_info['stim_theta'],2))+' gt: '+str(trial_info['ground_truth'])
                input_type_list.append(input_type)
                print(input_type)

            if trial_info['stim_theta'] != 0 and trial_info['ground_truth'] != 0 and len(inputs_list) == 3:
                inputs_list.append(inputs)
                input_type = 'stim: '+str(round(trial_info['stim_theta'],2))+' gt: '+str(trial_info['ground_truth'])
                input_type_list.append(input_type)
                print(input_type)

        else:
            if trial_info['stim_theta'] == 0 and len(inputs_list) == 0:
                inputs_list.append(inputs)
                input_type = 'gt: '+str(trial_info['ground_truth'])
                input_type_list.append(input_type)
                print(input_type)

            if trial_info['stim_theta'] != 0 and len(inputs_list) == 1:
                inputs_list.append(inputs)
                input_type = 'gt: '+str(trial_info['ground_truth'])
                input_type_list.append(input_type)
                print(input_type)

        # Log stimulus period activity
        rnn_activity_0 = rnn_activity[:, 0, :].detach().numpy() #取all period的activity分析
        activity_dict[i] = rnn_activity_0

        # speed = speed[:, 0, :].detach().numpy() #取all period的activity分析
        # speed_dict[i] = speed

        rnn_activity_1 = rnn_activity[:marker_list[1], 0, :].detach().numpy() #只取sample1 delay1 period的activity分析
        activity_dict_1[i] = rnn_activity_1

        rnn_activity_2 = rnn_activity[marker_list[1]:marker_list[2], 0, :].detach().numpy() #只取sample1 delay1 period的activity分析
        activity_dict_2[i] = rnn_activity_2

        if file_name_0 == 'DMS':
            rnn_activity_3 = rnn_activity[marker_list[2]:marker_list[3], 0, :].detach().numpy() #只取sample1 delay1 period的activity分析
            activity_dict_3[i] = rnn_activity_3

            rnn_activity_4 = rnn_activity[marker_list[3]:marker_list[4], 0, :].detach().numpy() #只取sample1 delay1 period的activity分析
            activity_dict_4[i] = rnn_activity_4

            rnn_activity_5 = rnn_activity[marker_list[4]:, 0, :].detach().numpy() #只取sample1 delay1 period的activity分析
            activity_dict_5[i] = rnn_activity_5

        else:
            activity_dict_3[i] = rnn_activity_2

            rnn_activity_4 = rnn_activity[marker_list[3]:, 0, :].detach().numpy() #只取sample1 delay1 period的activity分析
            activity_dict_4[i] = rnn_activity_4

    acc_mean = np.mean([val['correct'] for val in trial_infos.values()])
    print('Average performance', acc_mean)

    activity_mean_list = np.mean([val for val in activity_dict.values()],axis=2)
    activity_mean_list_ = np.mean(activity_mean_list,axis=0)

    plt.subplot(1,2,2)
    plot_firing_rate(activity_mean_list, activity_mean_list_, 'ground_truth', 'stim_theta', trial_infos, marker_list, task)
    
    if acc_mean < 0.7:
        continue

    # with open("./0824_accuracy_rnn_nocue_epoch10000.csv","a+") as csvfile:
    #    # if not np.mean(acc_1_list) == 1 and np.mean(acc_2_list) == 1:
    #     writer = csv.writer(csvfile)
    #     writer.writerow([acc_mean, acc_mean_1, acc_mean_2])

    fixedpoints_1 = fixed_points(net, activity_dict_1, hidden_size, marker_list[1]-marker_list[0], task, 1)
    fixedpoints_2 = fixed_points(net, activity_dict_2, hidden_size, marker_list[2]-marker_list[1], task, 1)
    if file_name_0 == 'DMS':
        fixedpoints_3 = fixed_points(net, activity_dict_3, hidden_size, marker_list[3]-marker_list[2], task, 1)
        fixedpoints_4 = fixed_points(net, activity_dict_4, hidden_size, marker_list[4]-marker_list[3], task, 1)
        fixedpoints_5 = fixed_points(net, activity_dict_5, hidden_size, trial_length-marker_list[4], task, 1)
        fixed_points_list = [fixedpoints_1,fixedpoints_2,fixedpoints_3,fixedpoints_4,fixedpoints_5]
    else:
        fixedpoints_3 = fixedpoints_2
        fixedpoints_4 = fixed_points(net, activity_dict_4, hidden_size, trial_length-marker_list[3], task, 1)
        fixed_points_list = [fixedpoints_1,fixedpoints_2,fixedpoints_3,fixedpoints_4]
    
    # var_list_0 = ['ground_truth', 'stim_theta']
    # q = generalize_q_twovar(trial_length, num_trial, trial_infos, activity_dict, hidden_size, var_list_0)

    # var_list = ['ground_truth', 'stim_theta']
    
    activity_mean_list, activity_var_list, activity_list = condition_average(trial_length, num_trial, trial_infos, activity_dict, hidden_size)

    fig = plt.figure()
    eigval_norm_list = []
    for i in range(trial_length):
        activity = np.concatenate(list(activity_dict[j][i] for j in range(num_trial)), axis=0).reshape(-1,256)
        eigval, eigvec = np.linalg.eig(np.cov(activity.T))
        eigval_norm = np.real(eigval)/np.sum(np.real(eigval))
        # print(np.sum(np.square(eigval_norm)))
        # print('ED:')
        # print(1/np.sum(np.square(eigval_norm)))
        eigval_norm_list.append(1/np.sum(np.square(eigval_norm)))
    plt.plot(np.arange(trial_length), eigval_norm_list)
    

    # pca_list = []
    activity = np.concatenate(list(activity_dict_1[i] for i in range(num_trial)), axis=0)
    pca1 = PCA(n_components=2)
    pca1.fit(activity)
    print(pca1.explained_variance_ratio_)
    # print(activity.shape)
    # print(np.cov(activity.T))
    # print(np.cov(activity.T).shape)
    eigval, eigvec = np.linalg.eig(np.cov(activity.T))
    # print(eigval)
    # print(np.array(eigval).shape)
    eigval_norm = np.real(eigval)/np.sum(np.real(eigval))
    # print('ED:')
    # print(1/np.sum(np.square(eigval_norm)))
    # eigval = pca1.explained_variance_
    # print(eigval)

    activity = np.concatenate(list(activity_dict_2[i] for i in range(num_trial)), axis=0)
    pca2 = PCA(n_components=2)
    pca2.fit(activity)
    print(pca2.explained_variance_ratio_)
    eigval, eigvec = np.linalg.eig(np.cov(activity.T))
    # eigval_norm = np.real(eigval)/np.sum(np.real(eigval))
    # print('ED:')
    # print(1/np.sum(np.square(eigval_norm)))

    activity = np.concatenate(list(activity_dict_3[i] for i in range(num_trial)), axis=0)
    pca3 = PCA(n_components=2)
    pca3.fit(activity)
    print(pca3.explained_variance_ratio_)
    eigval, eigvec = np.linalg.eig(np.cov(activity.T))
    # eigval_norm = np.real(eigval)/np.sum(np.real(eigval))
    # print('ED:')
    # print(1/np.sum(np.square(eigval_norm)))

    activity = np.concatenate(list(activity_dict_4[i] for i in range(num_trial)), axis=0)
    pca4 = PCA(n_components=2)
    pca4.fit(activity)
    print(pca4.explained_variance_ratio_)
    eigval, eigvec = np.linalg.eig(np.cov(activity.T))
    # eigval_norm = np.real(eigval)/np.sum(np.real(eigval))
    # print('ED:')
    # print(1/np.sum(np.square(eigval_norm)))

    pca4_1 = PCA(n_components=3)
    pca4_1.fit(activity)
    print(pca4_1.explained_variance_ratio_)

    if file_name_0 == 'DMS':
        activity = np.concatenate(list(activity_dict_5[i] for i in range(num_trial)), axis=0)
        pca5 = PCA(n_components=2)
        pca5.fit(activity)
        print(pca5.explained_variance_ratio_)
        eigval, eigvec = np.linalg.eig(np.cov(activity.T))
        # eigval_norm = np.real(eigval)/np.sum(np.real(eigval))
        # print('ED:')
        # print(1/np.sum(np.square(eigval_norm)))

    # weight_input = list(net.parameters())[1].detach().numpy()
    # bias_input = list(net.parameters())[2].detach().numpy()
    # weight_hidden = list(net.parameters())[3].detach().numpy()
    # bias_hidden = list(net.parameters())[4].detach().numpy()

    # plt.show()
    recurrence = net.rnn.recurrence

    # inputs_0 = inputs_list[0].detach().numpy()
    # inputs_1 = inputs_list[1].detach().numpy()

    # speed_norm_list = []
    order = 1
    gap = 4
    seg_list = [marker_list[1]-1,marker_list[2]-1,marker_list[3]-1, \
        marker_list[4]-1,int((marker_list[4]+trial_length)/2)] if file_name_0 == 'DMS'  \
            else [marker_list[1]-1,marker_list[2]-1, \
                marker_list[3]-1,int((marker_list[3]+trial_length)/2)]#,28,29,32,38] if file_name_0 == 'DMS' else [0,2,11,21,26] 
    fig4 = plt.figure(figsize=(6*(len(seg_list))+1,6*(len(inputs_list))))
    grid = plt.GridSpec(len(inputs_list), len(seg_list)*gap+1)
    # axes = fig4.subplots(2,(len(seg_list)))

    # if file_name_0 == 'DMS':
    #     up_lim = 15
    #     seq = 0.25
    # else:
    up_lim = 40
    seq = 0.5
    down_lim = -up_lim
    
    speed_norm = np.arange(int((up_lim-down_lim)*(up_lim-down_lim)/(seq*seq))).reshape(int((up_lim-down_lim)/seq),int((up_lim-down_lim)/seq))
    speed_x = np.arange(int(((up_lim-down_lim)*(up_lim-down_lim))/(seq*seq))).reshape(int((up_lim-down_lim)/seq),int((up_lim-down_lim)/seq))
    speed_y = np.arange(int(((up_lim-down_lim)*(up_lim-down_lim))/(seq*seq))).reshape(int((up_lim-down_lim)/seq),int((up_lim-down_lim)/seq))
    x = np.arange(down_lim,up_lim,seq) 
    y = np.arange(down_lim,up_lim,seq) 
    X, Y = np.meshgrid(x, y)

    # for inputs_order in range(len(inputs_list)):
    #     inputs = inputs_list[inputs_order].detach().numpy()
    #     # print('stim_theta: '+str(inputs[seg_list[1],0,:]))
    #     # print('test_theta: '+str(inputs[seg_list[4],0,:]))

    # seg_list = [30,31,32,33,34]
    # pca = pca2

    weight_output = list(net.parameters())[5].detach().numpy() 

    svd = TruncatedSVD(2)

    for input_order in range(len(inputs_list)):
        inputs = inputs_list[input_order].detach().numpy()
        for seg_order in range(len(seg_list)):
            if file_name_0 == 'DMS':
                # pca = pca1 if seg_order == 0 else pca2 if seg_order == 1 else pca3 if seg_order == 2 \
                #       else pca4 if seg_order == 3 else pca5

                pca = pca4 if seg_order < 3 else pca5

                title = 'Fix' if seg_order == 0 else 'Stim' if seg_order == 1 else 'Memory' if seg_order == 2 \
                    else 'Probe' if seg_order == 3 else 'Decision'
            else:
                pca = pca1 if seg_order == 0 else pca2 if seg_order == 1 else pca3 if seg_order == 2 else pca4
                title = 'Fix' if seg_order == 0 else 'Decision' if seg_order == 3 else 'Stim' if seg_order == 1 else 'Memory'

            # pca = pca1
            fixedpoints = fixed_points_list[seg_order]

            if seg_order < len(seg_list)-1:
                ax = plt.subplot(grid[input_order,seg_order*gap:seg_order*gap+gap])
            else: 
                ax = plt.subplot(grid[input_order,seg_order*gap:])

            if input_order < len(inputs_list)-1:
                ax.set_xticks([])
            if seg_order > 0:
                ax.set_yticks([])

            time_len = seg_list[seg_order]
            
            if file_name_0 == 'DMS':
                
                activity_projected_1_1 = svd.fit_transform(activity_list[0][:,time_len,:])
                activity_projected_1_2 = svd.fit_transform(activity_list[1][:,time_len,:])
                activity_projected_2_1 = svd.fit_transform(activity_list[2][:,time_len,:])
                activity_projected_2_2 = svd.fit_transform(activity_list[3][:,time_len,:])

                # activity_projected_1_1 = pca.transform(activity_list[0][:,time_len,:])
                # activity_projected_1_2 = pca.transform(activity_list[1][:,time_len,:])
                # activity_projected_2_1 = pca.transform(activity_list[2][:,time_len,:])
                # activity_projected_2_2 = pca.transform(activity_list[3][:,time_len,:]) # trial_num * trial_len * phase_num

                # if seg_order == len(seg_list) - 1:
                #     output_1 = np.dot(weight_output, activity_list[0][:,time_len,:].T)
                #     output_2 = np.dot(weight_output, activity_list[1][:,time_len,:].T)
                #     output_3 = np.dot(weight_output, activity_list[2][:,time_len,:].T)
                #     output_4 = np.dot(weight_output, activity_list[3][:,time_len,:].T)
                
                #     for i in range(output_1.shape[1]):
                #         print('output_1')
                #         print(np.argmax(output_1[:,i]))

                #     for i in range(output_2.shape[1]):
                #         print('output_2')
                #         print(np.argmax(output_2[:,i]))

                #     for i in range(output_3.shape[1]):
                #         print('output_3')
                #         print(np.argmax(output_3[:,i]))

                #     for i in range(output_4.shape[1]):
                #         print('output_4')
                #         print(np.argmax(output_4[:,i]))
                    # print('output_1: '+str(output_1))
                    # print('output_2: '+str(output_2))

                ax.scatter(activity_projected_1_1[:,0],activity_projected_1_1[:,1],color='red',s=10,label='gt=1 stim=0')
                ax.scatter(activity_projected_1_2[:,0],activity_projected_1_2[:,1],color='blue',s=10,label='gt=1 stim=180')
                ax.scatter(activity_projected_2_1[:,0],activity_projected_2_1[:,1],color='yellow',s=10,label='gt=0 stim=0')
                ax.scatter(activity_projected_2_2[:,0],activity_projected_2_2[:,1],color='green',s=10,label='gt=0 stim=180')

                # fixedpoints_projected = pca.transform(fixedpoints)
                # fixedpoints_projected = svd.fit_transform(fixedpoints)
                # ax.scatter(fixedpoints_projected[:, 0], fixedpoints_projected[:, 1], color='purple')

            else:
                # activity_projected_1 = pca.transform(activity_list[0][:,time_len,:])
                # activity_projected_2 = pca.transform(activity_list[1][:,time_len,:]) # trial_num * trial_len * phase_num
                
                activity_projected_1 = svd.fit_transform(activity_list[0][:,time_len,:])
                activity_projected_2 = svd.fit_transform(activity_list[1][:,time_len,:])
                # if seg_order == len(seg_list) - 1:
                #     output_1 = np.dot(weight_output, activity_list[0][:,time_len,:].T)
                #     output_2 = np.dot(weight_output, activity_list[1][:,time_len,:].T)
                
                #     for i in range(output_1.shape[1]):
                #         print(np.argmax(output_1[:,i]))

                #     for i in range(output_2.shape[1]):
                #         print(np.argmax(output_2[:,i]))
                #     print('output_1: '+str(output_1))
                #     print('output_2: '+str(output_2))

            fixedpoints_projected = pca.transform(fixedpoints)
            fixedpoints_projected = svd.fit_transform(fixedpoints)

            ax.scatter(fixedpoints_projected[:, 0], fixedpoints_projected[:, 1], color='purple')

                # i_fp = np.argsort(fixedpoints[:, 0])[int(fixedpoints.shape[0]/2)] #从小到大排序返回对应的index序列
                # input = torch.tensor([1, 0.5, 0.5], dtype = torch.float32)
                # speed_ = recurrence(input,torch.tensor(fixedpoints[i_fp],dtype=torch.float32)).detach().numpy() - fixedpoints[i_fp]
                # print(speed_)
                # speed = np.squeeze(pca.transform(np.expand_dims(speed_,axis=0)))
                
                # print(np.squeeze(pca1.transform(np.expand_dims(speed_,axis=0))))
                # print(speed)
                # print(np.squeeze(pca4.transform(np.expand_dims(speed_,axis=0))))
                
                # speed_projected_1 = pca.transform(speed_list[0][:,time_len,:])
                # speed_projected_2 = pca.transform(speed_list[1][:,time_len,:]) # trial_num * trial_len * phase_num

                # ax.scatter(activity_projected_1[:,0],activity_projected_1[:,1],color='red',s=10,label='gt=0')
                # ax.scatter(activity_projected_2[:,0],activity_projected_2[:,1],color='blue',s=10,label='gt=1')

            if seg_order == len(seg_list) - 1:
                    output_projected = pca.transform(weight_output)
                    output_projected = svd.fit_transform(weight_output)
                    # print(weight_output)
                    print(output_projected)
                    ax.arrow(0,0,output_projected[0,0],output_projected[0,1],color='red',width=0.2)
                    ax.arrow(0,0,output_projected[1,0],output_projected[1,1],color='green',width=0.2)
                    ax.arrow(0,0,output_projected[2,0],output_projected[2,1],color='blue',width=0.2)

            for grid_num_x in np.arange(down_lim,up_lim,seq):
                for grid_num_y in np.arange(down_lim,up_lim,seq):
                    h_0 = np.array([grid_num_x, grid_num_y])
                    h = pca.inverse_transform(h_0)
            
                    # print(h_0)
                    # print(np.squeeze(pca.transform(np.expand_dims(h,axis=0))))
                
                    # pre_act = np.dot(weight_hidden, h) + np.dot(weight_input, inputs[time_len,0,:]) + bias_hidden #+ bias_input
                    # speed = (- h + torch.relu(torch.tensor(pre_act)).numpy()) * dt/tau
                    # speed = np.squeeze(pca.transform(np.expand_dims(speed,axis=0)))

                    speed_ = recurrence(torch.tensor(inputs[time_len,0,:],dtype=torch.float32),torch.tensor(h,dtype=torch.float32)).detach().numpy() - h
                    # speed = np.squeeze(pca.transform(np.expand_dims(speed_,axis=0)))
                    speed = np.squeeze(svd.fit_transform(np.expand_dims(speed_,axis=0)))
                    print(speed.shape)
                    # if grid_num_x % 10 == 0 and grid_num_y % 10 == 0:
                    #     print('speed_high')
                    #     print(np.linalg.norm(speed_))
                    #     print('speed_low')
                    #     print(np.linalg.norm(speed))

                    # if np.linalg.norm(speed) < 2:
                    #     # print(speed_)
                    #     print('speed_low')
                    #     print(np.linalg.norm(speed_))
                    #     print(np.linalg.norm(speed))

                    # if np.linalg.norm(speed_) < 8:
                    #     # print(speed_)
                    #     print('speed_high')
                    #     print(np.linalg.norm(speed_))
                    #     print(np.linalg.norm(speed))

                    #     print(np.squeeze(pca1.transform(np.expand_dims(speed_,axis=0))))
                    #     print(np.squeeze(pca3.transform(np.expand_dims(speed_,axis=0))))

                    speed_x[int((grid_num_x-down_lim)/seq)][int((grid_num_y-down_lim)/seq)] = speed[0]
                    speed_y[int((grid_num_x-down_lim)/seq)][int((grid_num_y-down_lim)/seq)] = speed[1]
                    # speed_norm[int((grid_num_x-down_lim)/seq)][int((grid_num_y-down_lim)/seq)] = np.log2(np.linalg.norm(speed_)) ## 高维速度
                    speed_norm[int((grid_num_x-down_lim)/seq)][int((grid_num_y-down_lim)/seq)] = np.log(np.linalg.norm(speed)) ## 低维速度
                    
                    if speed[0] == 0 and speed[1] == 0:
                        ax.scatter(grid_num_x,grid_num_y,color='black',marker='o')
            
            speed_x = speed_x.T
            speed_y = speed_y.T
            speed_norm = speed_norm.T

            # ax.streamplot(X,Y,speed_x,speed_y,density=0.6,color='w')
            im = ax.pcolormesh(X,Y,speed_norm,alpha=0.6)
            
            ax.set_xlim(down_lim,up_lim)
            ax.set_ylim(down_lim,up_lim)

            if seg_order == 0:
                ax.set_ylabel(input_type_list[input_order]+' PC 2')
            if input_order == len(inputs_list)-1:
                ax.set_xlabel('PC 1')
            if input_order == 0:
                ax.set_title(title+' Time: '+str(time_len))
            # order +=1

        clb = plt.colorbar(im)
        clb.set_label(label='log2speed')

    # fig5 = plt.figure(figsize=(12,12))    
    # for i in range(len(seg_list)):
    #     # ax = fig4.add_subplot(2,len(seg_list),i+1)
    #     # ax = axes[i]
    #     ax = plt.subplot(1,len(seg_list),i+1)
    #     # im = ax.imshow(speed_norm_list[i], cmap=plt.cm.hot_r)
    #     im = ax.pcolormesh(x,y,speed_norm_list[i],alpha=0.7)
    #     ax.set_xlabel('PC 1')
    #     ax.set_ylabel('PC 2')
        # ax.set_title('Time: '+str(time_len))
    
    plt.show()
  
    # 指定图片保存路径
    if not os.path.exists(figure_save_path):
        os.makedirs(figure_save_path) # 如果不存在目录figure_save_path，则创建

    fig0.savefig(os.path.join(figure_save_path , str(run_batch_num)+'_0'))
    fig4.savefig(os.path.join(figure_save_path , str(run_batch_num)+'_4'))
    # fig5.savefig(os.path.join(figure_save_path , str(run_batch_num)+'_5'))

    # fig4 = plt.figure(figsize=(12,12))
    # order = 1
    # for time_len in [0,8,28,33]:
    #     plt.subplot(2,2,order)
    #     order += 1

    #     activity_projected_1_1 = q @ activity_list_1[0][:,:,time_len]
    #     activity_projected_1_1 = q @ activity_list_1[0][:,:,time_len]
    #     activity_projected_1_1 = q @ activity_list_1[0][:,:,time_len]
    #     activity_projected_1_1 = q @ activity_list_1[0][:,:,time_len]# phase_num * trial_num * trial_len

    #     plt.scatter(activity_projected_1_1[0],activity_projected_1_1[1],color='red',s=10,label='gt=1 stim=0')
    #     plt.scatter(activity_projected_1_2[0],activity_projected_1_2[1],color='blue',s=10,label='gt=1 stim=180')
    #     plt.scatter(activity_projected_2_1[0],activity_projected_2_1[1],color='yellow',s=10,label='gt=0 stim=0')
    #     plt.scatter(activity_projected_2_2[0],activity_projected_2_2[1],color='green',s=10,label='gt=0 stim=180')
    #     plt.xlabel(var_list[0])
    #     plt.ylabel(var_list[1])
    #     # plt.xlim(-20,20)
    #     # plt.ylim(-20,20)
    #     plt.title('Time: '+str(time_len))
    # fig4.savefig(os.path.join(figure_save_path , str(run_batch_num)+'_5'))

    # fig4 = plot_(q, activity_mean_list, activity_dict, trial_infos, fixedpoints_1, marker_list)
    # fig4.suptitle(task+' acc:'+str(round(acc_mean, 4)))
    # fig4.savefig(os.path.join(figure_save_path , str(run_batch_num)+'_4'))

    # fig5 = plot_var(activity_var_list, marker_list, var_list, task)
    # # fig5.suptitle(task+' acc1:'+str(round(acc_mean_1, 4))+' acc2:'+str(round(acc_mean_2, 4)))

    # fig5_1 = plot_var(activity_var_list_1, marker_list, var_list, task)
    # # fig5_1.suptitle(task+' acc1:'+str(round(acc_mean_1, 4))+' acc2:'+str(round(acc_mean_2, 4)))

    # trial_length_1 = marker_list[5] - marker_list[1]
    # marker_list = [marker_list[i] - 5 for i in range(1,5)]
    # q = generalize_q_threevar(trial_length_1, num_trial, trial_infos, activity_dict_1, hidden_size, var_list_0)
    # activity_mean_list, activity_var_list, activity_var_list_1 = condition_average_threevar(trial_length_1, num_trial, trial_infos, activity_dict_1, hidden_size, var_list)
    # fig6 = plot_threevar(q, activity_mean_list, activity_dict_1, trial_infos, fixedpoints_1, marker_list, var_list)
    # fig6.suptitle('period1 '+task+' acc:'+str(round(acc_mean, 4))))

    # fig7 = plot_var(activity_var_list, marker_list, var_list, task)
    # fig7.suptitle('period1 '+task)

    # fig7_1 = plot_var(activity_var_list_1, marker_list, var_list, task)
    # fig7_1.suptitle('period1 '+task)

    # trial_length_2 = trial_length - trial_length_1 - 5
    # marker_list = [0,5,10]
    # q = generalize_q_threevar(trial_length_2, num_trial, trial_infos, activity_dict_2, hidden_size, var_list_0)
    # activity_mean_list, activity_var_list, activity_var_list_1 = condition_average_threevar(trial_length_2, num_trial, trial_infos, activity_dict_2, hidden_size, var_list)
    # fig8 = plot_threevar(q, activity_mean_list, activity_dict_2, trial_infos, fixedpoints_1, marker_list, var_list)
    # fig8.suptitle('period2 '+task+' acc1:'+str(round(acc_mean_1, 4))+' acc2:'+str(round(acc_mean_2, 4)))

    # fig9 = plot_var(activity_var_list, marker_list, var_list, task)
    # fig9.suptitle('period2 '+task)

    # fig9_1 = plot_var(activity_var_list_1, marker_list, var_list, task)
    # fig9_1.suptitle('period2 '+task)

    # plt.show()

## heatmap

# var_list = ['ground_truth', 'stim1_theta', 'stim2_theta']
# trial_length = 40
# activity_mean_list, activity_var_list = condition_average_threevar(trial_length, num_trial, trial_infos, activity_dict, hidden_size, var_list)
# # for l in range(len(activity_mean_list)):
# #     coef_array = np.zeros((trial_length,trial_length))
# #     activity_mean_list[l] = activity_mean_list[l].transpose(1,0)
# #     plt.figure(figsize=(5,5))
# #     plt.title(l)
# #     for i in range(trial_length):
# #         for j in range(i,trial_length):
# #             tmp = np.c_[activity_mean_list[l][i], activity_mean_list[l][j]].transpose(1,0) 
# #             c_tmp = np.corrcoef(tmp)[0,1]
# #             coef_array[i,j] = coef_array[j,i] = c_tmp
# #     sns.heatmap(coef_array,cmap='hot')

# plt.show()
# fig.suptitle('acc:'+str(acc_mean))

# path = './output/pic/20220501/dms_epoch7000_integral.png'
# fig.savefig(path)
# plt.show()
