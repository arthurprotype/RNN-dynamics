#coding=UTF-8 
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

def generalize_q_twovar(trial_length, num_trial, trial_infos, activity_dict, hidden_size, var_list):
    var1, var2 = var_list
    num_variable = 2
    for i in range(hidden_size):
        for j in range(trial_length):
            rate_array = np.array([activity_dict[k][j,i] for k in range(num_trial) if trial_infos[k]['correct'] == 1])
            variable_array = np.array([[trial_infos[k][var1],trial_infos[k][var2]] for k in range(num_trial) if trial_infos[k]['correct'] == 1])
       
            # l = 0
            # for k in range(num_trial):
            #     trial_info = trial_infos[k]
            #     if trial_info['correct'] == 1:
            #         firing_rate = activity_dict[k][j,i]
            #         if l == 0:
            #             rate_array = np.array(firing_rate)
            #             variable_array = np.array([trial_info[var1],trial_info[var2]]) 
            #             l += 1
            #         else:
            #             rate_array = np.append(rate_array, firing_rate)
            #             variable_array = np.append(variable_array, [trial_info[var1],trial_info[var2]])
                    
            variable_array = variable_array.reshape(-1, num_variable)

            model = LinearRegression()
            model = model.fit(variable_array, rate_array)
            param = model.coef_
            # rate_array_pred = model.predict(variable_array)
            parameter_time_array = np.array(param) if j == 0 else np.append(parameter_time_array, param)

        parameter_array = np.array(parameter_time_array) if i == 0 else np.append(parameter_array,parameter_time_array)

    parameter_array = parameter_array.reshape(hidden_size, -1) #不对时间取average，因为后面要乘D，所以只升至2维

    # Concatenate activity for PCA
    activity = np.concatenate(list(activity_dict[i] for i in range(num_trial)), axis=0)

    # Compute PCA and visualize
    dim_pca = 8
    pca = PCA(n_components=dim_pca)
    pca.fit(activity)

    D_array = np.zeros((hidden_size, hidden_size))
    for i in pca.components_:
        tmp = np.outer(i,i) # 向量外积
        D_array = D_array + tmp
    parameter_pca_array = D_array @ parameter_array
    parameter_pca_array = parameter_pca_array.reshape(hidden_size, trial_length, num_variable).transpose(2, 1, 0)

    # norm_j = [np.linalg.norm(parameter_pca_array[i,j]) for i in range(parameter_pca_array.shape[0]) for j in range(parameter_pca_array.shape[1])]
    
    # order_list = [np.argmax(np.linalg.norm(parameter_pca_array[i,:])) for i in range(parameter_pca_array.shape[0])]
    # print('order_list: '+str(order_list))

    norm_max = 0
    norm_max_order = 0
    order_list = []

    for i in range(parameter_pca_array.shape[0]):
        for j in range(parameter_pca_array.shape[1]):
            norm_j = np.linalg.norm(parameter_pca_array[i,j])
            if norm_j > norm_max:
                norm_max = norm_j
                norm_max_order = j 
        order_list.append(norm_max_order)
        print(norm_max_order)

    parameter_max_array = np.array([parameter_pca_array[i][order_list[i]] for i in range(parameter_pca_array.shape[0])])
    # for i in range(parameter_pca_array.shape[0]):
    #     order_max = order_list[i]
    #     parameter_max_array = np.array(parameter_pca_array[i][order_max]) if i == 0 else np.append(parameter_max_array, parameter_pca_array[i][order_max])

    parameter_max_array = parameter_max_array.reshape(num_variable, hidden_size).transpose(1, 0)
    q,r = np.linalg.qr(parameter_max_array)
    q = np.transpose(q)

    return q

def generalize_q_threevar(trial_length, num_trial, trial_infos, activity_dict, hidden_size, var_list):
    var1, var2, var3 = var_list
    num_variable = 3
    for i in range(hidden_size):
        for j in range(trial_length):
            l = 0
            for k in range(num_trial):
                trial_info = trial_infos[k]
                if trial_info['correct'] == 1:
                    firing_rate = activity_dict[k][j,i]
                    if l == 0:
                        rate_array = np.array(firing_rate)
                        variable_array = np.array([trial_info[var1],trial_info[var2],trial_info[var3]]) #stim1 stim2 stim gt cue
                    else:
                        rate_array = np.append(rate_array, firing_rate)
                        variable_array = np.append(variable_array, [trial_info[var1],trial_info[var2],trial_info[var3]])
                    l += 1
            variable_array = variable_array.reshape(-1, num_variable)

            model = LinearRegression()
            model = model.fit(variable_array, rate_array)
            param = model.coef_
            # rate_array_pred = model.predict(variable_array)
            parameter_time_array = np.array(param) if j == 0 else np.append(parameter_time_array, param)

        parameter_array = np.array(parameter_time_array) if i == 0 else np.append(parameter_array,parameter_time_array)

    parameter_array = parameter_array.reshape(hidden_size, -1) #不对时间取average，因为后面要乘D，所以只升至2维

    # Concatenate activity for PCA
    activity = np.concatenate(list(activity_dict[i] for i in range(num_trial)), axis=0)

    # Compute PCA and visualize
    dim_pca = 8
    pca = PCA(n_components=dim_pca)
    pca.fit(activity)

    D_array = np.zeros((hidden_size, hidden_size))
    for i in pca.components_:
        tmp = np.outer(i,i) # 向量外积
        D_array = D_array + tmp
    parameter_pca_array = D_array @ parameter_array
    parameter_pca_array = parameter_pca_array.reshape(hidden_size, trial_length, num_variable).transpose(2, 1, 0)

    norm_max = 0
    norm_max_order = 0
    order_list = []
    for i in range(parameter_pca_array.shape[0]):
        for j in range(parameter_pca_array.shape[1]):
            norm_j = np.linalg.norm(parameter_pca_array[i,j])
            if norm_j > norm_max:
                norm_max = norm_j
                norm_max_order = j 
        order_list.append(norm_max_order)
        print("the order of max norm: " + str(norm_max_order))

    for i in range(parameter_pca_array.shape[0]):
        order_max = order_list[i]
        parameter_max_array = np.array(parameter_pca_array[i][order_max]) if i == 0 else np.append(parameter_max_array, parameter_pca_array[i][order_max])

    parameter_max_array = parameter_max_array.reshape(num_variable, hidden_size).transpose(1, 0)
    q,r = np.linalg.qr(parameter_max_array)
    q = np.transpose(q)

    # print(q.shape)

    # l_x=np.sqrt(x.dot(x))
    # l_y=np.sqrt(y.dot(y))
    # dian=x.dot(y)
 
    # angle_hu=np.arccos(dian/(l_x*l_y))
    # angle_d=angle_hu*180/np.pi

    return q
