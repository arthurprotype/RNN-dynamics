import numpy as np


def condition_average_twovar(trial_length, num_trial, trial_infos, activity_dict, hidden_size, var_list):
    var1, var2 = var_list
    m = n = 0
    m_1 = m_2 = n_1 = n_2 = 0
    for i in range(num_trial):
        activity = activity_dict[i]
        trial = trial_infos[i]

        if trial['correct'] == 1:
            if trial[var1] == 0:
                activity_array_1 = np.array(activity) if m == 0 else np.append(activity_array_1,activity)
                m += 1
                if trial[var2] == 0: 
                    activity_array_1_1 = np.array(activity) if m_1 == 0 else np.append(activity_array_1_1,activity)
                    m_1 += 1
                else:
                    activity_array_1_2 = np.array(activity) if m_2 == 0 else np.append(activity_array_1_2,activity)
                    m_2 += 1

            else:
                activity_array_2 = np.array(activity) if n == 0 else np.append(activity_array_2,activity)
                n += 1
                if trial[var2] == 0: 
                    activity_array_2_1 = np.array(activity) if n_1 == 0 else np.append(activity_array_2_1,activity)
                    n_1 += 1
                else:
                    activity_array_2_2 = np.array(activity) if n_2 == 0 else np.append(activity_array_2_2,activity)
                    n_2 += 1

    
    activity_array_1_1 = activity_array_1_1.reshape(m_1, trial_length, hidden_size).transpose(0, 2, 1)
    activity_array_1_2 = activity_array_1_2.reshape(m_2, trial_length, hidden_size).transpose(0, 2, 1)
    activity_array_2_1 = activity_array_2_1.reshape(n_1, trial_length, hidden_size).transpose(0, 2, 1)
    activity_array_2_2 = activity_array_2_2.reshape(n_2, trial_length, hidden_size).transpose(0, 2, 1)
    
    activity_mean_list = []
    activity_mean_list.append(np.mean(activity_array_1_1, axis=0))
    activity_mean_list.append(np.mean(activity_array_1_2, axis=0))
    activity_mean_list.append(np.mean(activity_array_2_1, axis=0))
    activity_mean_list.append(np.mean(activity_array_2_2, axis=0))

    activity_var_list = []
    activity_var_list.append(np.mean(np.var(activity_array_1_1,axis=0),axis=0))
    activity_var_list.append(np.mean(np.var(activity_array_1_2,axis=0),axis=0))
    activity_var_list.append(np.mean(np.var(activity_array_2_1,axis=0),axis=0))
    activity_var_list.append(np.mean(np.var(activity_array_2_2,axis=0),axis=0))
    
    return activity_mean_list, activity_var_list

def condition_average_threevar(trial_length, num_trial, trial_infos, activity_dict, hidden_size, var_list):
    var1, var2, var3 = var_list
    m_1 = m_2 = m_3 = m_4 = n_1 = n_2 = n_3 = n_4 = 0
    for i in range(num_trial):
        activity = activity_dict[i]
        trial = trial_infos[i]

        if trial['correct'] == 1:
            if trial[var1] == 0:
                if trial[var2] == 0: 
                    if trial[var3] == 0:
                        activity_array_1_1 = np.array(activity) if m_1 == 0 else np.append(activity_array_1_1,activity)
                        m_1 += 1
                    else:
                        activity_array_1_2 = np.array(activity) if m_2 == 0 else np.append(activity_array_1_2,activity)
                        m_2 += 1
                else:
                   if trial[var3] == 0:
                        activity_array_1_3 = np.array(activity) if m_3 == 0 else np.append(activity_array_1_3,activity)
                        m_3 += 1
                   else:
                        activity_array_1_4 = np.array(activity) if m_4 == 0 else np.append(activity_array_1_4,activity)
                        m_4 += 1

            else:
                if trial[var2] == 0: 
                    if trial[var3] == 0:
                        activity_array_2_1 = np.array(activity) if n_1 == 0 else np.append(activity_array_2_1,activity)
                        n_1 += 1
                    else:
                        activity_array_2_2 = np.array(activity) if n_2 == 0 else np.append(activity_array_2_2,activity)
                        n_2 += 1
                else:
                   if trial[var3] == 0:
                        activity_array_2_3 = np.array(activity) if n_3 == 0 else np.append(activity_array_2_3,activity)
                        n_3 += 1
                   else:
                        activity_array_2_4 = np.array(activity) if n_4 == 0 else np.append(activity_array_2_4,activity)
                        n_4 += 1
    
    activity_array_1_1 = activity_array_1_1.reshape(m_1, trial_length, hidden_size).transpose(0, 2, 1)
    activity_array_1_2 = activity_array_1_2.reshape(m_2, trial_length, hidden_size).transpose(0, 2, 1)
    activity_array_1_3 = activity_array_1_3.reshape(m_3, trial_length, hidden_size).transpose(0, 2, 1)
    activity_array_1_4 = activity_array_1_4.reshape(m_4, trial_length, hidden_size).transpose(0, 2, 1)   
    activity_array_2_1 = activity_array_2_1.reshape(n_1, trial_length, hidden_size).transpose(0, 2, 1)
    activity_array_2_2 = activity_array_2_2.reshape(n_2, trial_length, hidden_size).transpose(0, 2, 1)
    activity_array_2_3 = activity_array_2_3.reshape(n_3, trial_length, hidden_size).transpose(0, 2, 1)
    activity_array_2_4 = activity_array_2_4.reshape(n_4, trial_length, hidden_size).transpose(0, 2, 1)

    activity_mean_list = []
    activity_mean_list.append(np.mean(activity_array_1_1, axis=0))
    activity_mean_list.append(np.mean(activity_array_1_2, axis=0))
    activity_mean_list.append(np.mean(activity_array_1_3, axis=0))
    activity_mean_list.append(np.mean(activity_array_1_4, axis=0))
    activity_mean_list.append(np.mean(activity_array_2_1, axis=0))
    activity_mean_list.append(np.mean(activity_array_2_2, axis=0))
    activity_mean_list.append(np.mean(activity_array_2_3, axis=0))
    activity_mean_list.append(np.mean(activity_array_2_4, axis=0))

    activity_var_list = []
    activity_var_list.append(np.mean(np.var(activity_array_1_1,axis=0),axis=0))
    activity_var_list.append(np.mean(np.var(activity_array_1_2,axis=0),axis=0))
    activity_var_list.append(np.mean(np.var(activity_array_1_3,axis=0),axis=0))
    activity_var_list.append(np.mean(np.var(activity_array_1_4,axis=0),axis=0))
    activity_var_list.append(np.mean(np.var(activity_array_2_1,axis=0),axis=0))
    activity_var_list.append(np.mean(np.var(activity_array_2_2,axis=0),axis=0))
    activity_var_list.append(np.mean(np.var(activity_array_2_3,axis=0),axis=0))
    activity_var_list.append(np.mean(np.var(activity_array_2_4,axis=0),axis=0))

    activity_var_list_1 = []
    activity_var_list_1.append(np.var(np.mean(activity_array_1_1,axis=1),axis=0))
    activity_var_list_1.append(np.var(np.mean(activity_array_1_2,axis=1),axis=0))
    activity_var_list_1.append(np.var(np.mean(activity_array_1_3,axis=1),axis=0))
    activity_var_list_1.append(np.var(np.mean(activity_array_1_4,axis=1),axis=0))
    activity_var_list_1.append(np.var(np.mean(activity_array_2_1,axis=1),axis=0))
    activity_var_list_1.append(np.var(np.mean(activity_array_2_2,axis=1),axis=0))
    activity_var_list_1.append(np.var(np.mean(activity_array_2_3,axis=1),axis=0))
    activity_var_list_1.append(np.var(np.mean(activity_array_2_4,axis=1),axis=0))

    return activity_mean_list, activity_var_list, activity_var_list_1

def condition_average_fourvar(trial_length, num_trial, trial_infos, activity_dict, hidden_size, var_list):
    var1, var2, var3, var4 = var_list
    m_1 = m_2 = m_3 = m_4 = m_5 = m_6 = m_7 = m_8 = n_1 = n_2 = n_3 = n_4 = n_5 = n_6 = n_7 = n_8 = 0
    for i in range(num_trial):
        activity = activity_dict[i]
        trial = trial_infos[i]

        if trial['correct'] == 1:
            if trial[var1] == 0:
                if trial[var2] == 0:
                    if trial[var3] == 0: 
                        if trial[var4] == 0:
                            activity_array_1_1 = np.array(activity) if m_1 == 0 else np.append(activity_array_1_1,activity)
                            m_1 += 1
                        else:
                            activity_array_1_2 = np.array(activity) if m_2 == 0 else np.append(activity_array_1_2,activity)
                            m_2 += 1
                    else:
                        if trial[var4] == 0:
                                activity_array_1_3 = np.array(activity) if m_3 == 0 else np.append(activity_array_1_3,activity)
                                m_3 += 1
                        else:
                                activity_array_1_4 = np.array(activity) if m_4 == 0 else np.append(activity_array_1_4,activity)
                                m_4 += 1
                else:
                    if trial[var3] == 0: 
                        if trial[var4] == 0:
                            activity_array_1_5 = np.array(activity) if m_5 == 0 else np.append(activity_array_1_5,activity)
                            m_5 += 1
                        else:
                            activity_array_1_6 = np.array(activity) if m_6 == 0 else np.append(activity_array_1_6,activity)
                            m_6 += 1
                    else:
                        if trial[var4] == 0:
                                activity_array_1_7 = np.array(activity) if m_7 == 0 else np.append(activity_array_1_7,activity)
                                m_7 += 1
                        else:
                                activity_array_1_8 = np.array(activity) if m_8 == 0 else np.append(activity_array_1_8,activity)
                                m_8 += 1

            else:
                if trial[var2] == 0:
                    if trial[var3] == 0: 
                        if trial[var4] == 0:
                            activity_array_2_1 = np.array(activity) if n_1 == 0 else np.append(activity_array_2_1,activity)
                            n_1 += 1
                        else:
                            activity_array_2_2 = np.array(activity) if n_2 == 0 else np.append(activity_array_2_2,activity)
                            n_2 += 1
                    else:
                        if trial[var4] == 0:
                                activity_array_2_3 = np.array(activity) if n_3 == 0 else np.append(activity_array_2_3,activity)
                                n_3 += 1
                        else:
                                activity_array_2_4 = np.array(activity) if n_4 == 0 else np.append(activity_array_2_4,activity)
                                n_4 += 1
                else:
                    if trial[var3] == 0: 
                        if trial[var4] == 0:
                            activity_array_2_5 = np.array(activity) if n_5 == 0 else np.append(activity_array_2_5,activity)
                            n_5 += 1
                        else:
                            activity_array_2_6 = np.array(activity) if n_6 == 0 else np.append(activity_array_2_6,activity)
                            n_6 += 1
                    else:
                        if trial[var4] == 0:
                            activity_array_2_7 = np.array(activity) if n_7 == 0 else np.append(activity_array_2_7,activity)
                            n_7 += 1
                        else:
                            activity_array_2_8 = np.array(activity) if n_8 == 0 else np.append(activity_array_2_8,activity)
                            n_8 += 1

    
    activity_array_1_1 = activity_array_1_1.reshape(m_1, trial_length, hidden_size).transpose(0, 2, 1)
    activity_array_1_2 = activity_array_1_2.reshape(m_2, trial_length, hidden_size).transpose(0, 2, 1)
    activity_array_1_3 = activity_array_1_3.reshape(m_3, trial_length, hidden_size).transpose(0, 2, 1)
    activity_array_1_4 = activity_array_1_4.reshape(m_4, trial_length, hidden_size).transpose(0, 2, 1) 
    activity_array_1_5 = activity_array_1_5.reshape(m_5, trial_length, hidden_size).transpose(0, 2, 1)
    activity_array_1_6 = activity_array_1_6.reshape(m_6, trial_length, hidden_size).transpose(0, 2, 1)
    activity_array_1_7 = activity_array_1_7.reshape(m_7, trial_length, hidden_size).transpose(0, 2, 1)
    activity_array_1_8 = activity_array_1_8.reshape(m_8, trial_length, hidden_size).transpose(0, 2, 1)   
    activity_array_2_1 = activity_array_2_1.reshape(n_1, trial_length, hidden_size).transpose(0, 2, 1)
    activity_array_2_2 = activity_array_2_2.reshape(n_2, trial_length, hidden_size).transpose(0, 2, 1)
    activity_array_2_3 = activity_array_2_3.reshape(n_3, trial_length, hidden_size).transpose(0, 2, 1)
    activity_array_2_4 = activity_array_2_4.reshape(n_4, trial_length, hidden_size).transpose(0, 2, 1)
    activity_array_2_5 = activity_array_2_5.reshape(n_5, trial_length, hidden_size).transpose(0, 2, 1)
    activity_array_2_6 = activity_array_2_6.reshape(n_6, trial_length, hidden_size).transpose(0, 2, 1)
    activity_array_2_7 = activity_array_2_7.reshape(n_7, trial_length, hidden_size).transpose(0, 2, 1)
    activity_array_2_8 = activity_array_2_8.reshape(n_8, trial_length, hidden_size).transpose(0, 2, 1)  

    activity_mean_list = []
    activity_mean_list.append(np.mean(activity_array_1_1, axis=0))
    activity_mean_list.append(np.mean(activity_array_1_2, axis=0))
    activity_mean_list.append(np.mean(activity_array_1_3, axis=0))
    activity_mean_list.append(np.mean(activity_array_1_4, axis=0))
    activity_mean_list.append(np.mean(activity_array_1_5, axis=0))
    activity_mean_list.append(np.mean(activity_array_1_6, axis=0))
    activity_mean_list.append(np.mean(activity_array_1_7, axis=0))
    activity_mean_list.append(np.mean(activity_array_1_8, axis=0))
    activity_mean_list.append(np.mean(activity_array_2_1, axis=0))
    activity_mean_list.append(np.mean(activity_array_2_2, axis=0))
    activity_mean_list.append(np.mean(activity_array_2_3, axis=0))
    activity_mean_list.append(np.mean(activity_array_2_4, axis=0))
    activity_mean_list.append(np.mean(activity_array_2_5, axis=0))
    activity_mean_list.append(np.mean(activity_array_2_6, axis=0))
    activity_mean_list.append(np.mean(activity_array_2_7, axis=0))
    activity_mean_list.append(np.mean(activity_array_2_8, axis=0))

    activity_var_list = []
    activity_var_list.append(np.mean(np.var(activity_array_1_1,axis=0),axis=0))
    activity_var_list.append(np.mean(np.var(activity_array_1_2,axis=0),axis=0))
    activity_var_list.append(np.mean(np.var(activity_array_1_3,axis=0),axis=0))
    activity_var_list.append(np.mean(np.var(activity_array_1_4,axis=0),axis=0))
    activity_var_list.append(np.mean(np.var(activity_array_1_5,axis=0),axis=0))
    activity_var_list.append(np.mean(np.var(activity_array_1_6,axis=0),axis=0))
    activity_var_list.append(np.mean(np.var(activity_array_1_7,axis=0),axis=0))
    activity_var_list.append(np.mean(np.var(activity_array_1_8,axis=0),axis=0))
    activity_var_list.append(np.mean(np.var(activity_array_2_1,axis=0),axis=0))
    activity_var_list.append(np.mean(np.var(activity_array_2_2,axis=0),axis=0))
    activity_var_list.append(np.mean(np.var(activity_array_2_3,axis=0),axis=0))
    activity_var_list.append(np.mean(np.var(activity_array_2_4,axis=0),axis=0))
    activity_var_list.append(np.mean(np.var(activity_array_2_5,axis=0),axis=0))
    activity_var_list.append(np.mean(np.var(activity_array_2_6,axis=0),axis=0))
    activity_var_list.append(np.mean(np.var(activity_array_2_7,axis=0),axis=0))
    activity_var_list.append(np.mean(np.var(activity_array_2_8,axis=0),axis=0))

    activity_var_list_1 = []
    activity_var_list_1.append(np.var(np.mean(activity_array_1_1,axis=1),axis=0))
    activity_var_list_1.append(np.var(np.mean(activity_array_1_2,axis=1),axis=0))
    activity_var_list_1.append(np.var(np.mean(activity_array_1_3,axis=1),axis=0))
    activity_var_list_1.append(np.var(np.mean(activity_array_1_4,axis=1),axis=0))
    activity_var_list_1.append(np.var(np.mean(activity_array_1_5,axis=1),axis=0))
    activity_var_list_1.append(np.var(np.mean(activity_array_1_6,axis=1),axis=0))
    activity_var_list_1.append(np.var(np.mean(activity_array_1_7,axis=1),axis=0))
    activity_var_list_1.append(np.var(np.mean(activity_array_1_8,axis=1),axis=0))
    activity_var_list_1.append(np.var(np.mean(activity_array_2_1,axis=1),axis=0))
    activity_var_list_1.append(np.var(np.mean(activity_array_2_2,axis=1),axis=0))
    activity_var_list_1.append(np.var(np.mean(activity_array_2_3,axis=1),axis=0))
    activity_var_list_1.append(np.var(np.mean(activity_array_2_4,axis=1),axis=0))
    activity_var_list_1.append(np.var(np.mean(activity_array_2_5,axis=1),axis=0))
    activity_var_list_1.append(np.var(np.mean(activity_array_2_6,axis=1),axis=0))
    activity_var_list_1.append(np.var(np.mean(activity_array_2_7,axis=1),axis=0))
    activity_var_list_1.append(np.var(np.mean(activity_array_2_8,axis=1),axis=0))

    return activity_mean_list, activity_var_list, activity_var_list_1

def condition_average_pdm(trial_length, num_trial, trial_infos, activity_dict, hidden_size):
    activity_array_1 = np.array([activity_dict[i] for i in range(num_trial) if trial_infos[i]['correct'] == 1 and trial_infos[i]['ground_truth'] == 0])
    activity_array_2 = np.array([activity_dict[i] for i in range(num_trial) if trial_infos[i]['correct'] == 1 and trial_infos[i]['ground_truth'] != 0])

    # speed_array_1 = np.array([speed_dict[i] for i in range(num_trial) if trial_infos[i]['correct'] == 1 and trial_infos[i]['ground_truth'] == 0])
    # speed_array_2 = np.array([speed_dict[i] for i in range(num_trial) if trial_infos[i]['correct'] == 1 and trial_infos[i]['ground_truth'] != 0])
    
    # speed_array_1 = speed_array_1.reshape(-1, trial_length, hidden_size)
    # speed_array_2 = speed_array_2.reshape(-1, trial_length, hidden_size) 

    # speed_list = [speed_array_1, speed_array_2]
    
    activity_array_1 = activity_array_1.reshape(-1, trial_length, hidden_size)
    activity_array_2 = activity_array_2.reshape(-1, trial_length, hidden_size) 

    activity_list = [activity_array_1, activity_array_2]
    
    activity_array_1 = activity_array_1.transpose(0, 2, 1)
    activity_array_2 = activity_array_2.transpose(0, 2, 1)

    activity_mean_list = [] 
    activity_mean_list.append(np.mean(activity_array_1, axis=0))
    activity_mean_list.append(np.mean(activity_array_2, axis=0))

    activity_var_list = []
    activity_var_list.append(np.mean(np.var(activity_array_1,axis=0),axis=0))
    activity_var_list.append(np.mean(np.var(activity_array_2,axis=0),axis=0))

    return activity_mean_list, activity_var_list, activity_list

def condition_average_dms(trial_length, num_trial, trial_infos, activity_dict, hidden_size):

    activity_array_1_1 = np.array([activity_dict[i] for i in range(num_trial) if trial_infos[i]['correct'] == 1 and trial_infos[i]['ground_truth'] == 1 and trial_infos[i]['stim_theta'] == 0])
    activity_array_1_2 = np.array([activity_dict[i] for i in range(num_trial) if trial_infos[i]['correct'] == 1 and trial_infos[i]['ground_truth'] == 1 and trial_infos[i]['stim_theta'] != 0])
    activity_array_2_1 = np.array([activity_dict[i] for i in range(num_trial) if trial_infos[i]['correct'] == 1 and trial_infos[i]['ground_truth'] != 1 and trial_infos[i]['stim_theta'] == 0])
    activity_array_2_2 = np.array([activity_dict[i] for i in range(num_trial) if trial_infos[i]['correct'] == 1 and trial_infos[i]['ground_truth'] != 1 and trial_infos[i]['stim_theta'] != 0])

    activity_array_1_1 = activity_array_1_1.reshape(-1, trial_length, hidden_size)
    activity_array_1_2 = activity_array_1_2.reshape(-1, trial_length, hidden_size)
    activity_array_2_1 = activity_array_2_1.reshape(-1, trial_length, hidden_size)
    activity_array_2_2 = activity_array_2_2.reshape(-1, trial_length, hidden_size)

    activity_list = [activity_array_1_1, activity_array_1_2, activity_array_2_1, activity_array_2_2]

    activity_array_1_1 = activity_array_1_1.transpose(0, 2, 1)
    activity_array_1_2 = activity_array_1_2.transpose(0, 2, 1)
    activity_array_2_1 = activity_array_2_1.transpose(0, 2, 1)
    activity_array_2_2 = activity_array_2_2.transpose(0, 2, 1)

    activity_mean_list = []
    activity_mean_list.append(np.mean(activity_array_1_1, axis=0))
    activity_mean_list.append(np.mean(activity_array_1_2, axis=0))
    activity_mean_list.append(np.mean(activity_array_2_1, axis=0))
    activity_mean_list.append(np.mean(activity_array_2_2, axis=0))

    # activity_array_1 = activity_array_1.reshape(m, trial_length, hidden_size).transpose(0, 2, 1)
    # activity_array_2 = activity_array_2.reshape(n, trial_length, hidden_size).transpose(0, 2, 1)    
    # activity_mean_list.append(np.mean(activity_array_1, axis=0))
    # activity_mean_list.append(np.mean(activity_array_2, axis=0)) # num_neuron * trial_length

    activity_var_list = []
    activity_var_list.append(np.mean(np.var(activity_array_1_1,axis=0),axis=0))
    activity_var_list.append(np.mean(np.var(activity_array_1_2,axis=0),axis=0))
    activity_var_list.append(np.mean(np.var(activity_array_2_1,axis=0),axis=0))
    activity_var_list.append(np.mean(np.var(activity_array_2_2,axis=0),axis=0))
    
    return activity_mean_list, activity_var_list, activity_list


# def condition_average(trial_length, num_trial, trial_infos, activity_dict, cue):
#     m = n = 0
#     m_1 = m_2 = m_3 = m_4 = n_1 = n_2 = n_3 = n_4 = 0
#     for i in range(num_trial):
#         activity = activity_dict[i]
#         trial = trial_infos[i]

#         if trial['correct'] == 1:
#             if trial['ground_truth'] == 1 and trial[cue] == 1:
#                 activity_array_1 = np.array(activity) if m == 0 else np.append(activity_array_1,activity)
#                 m += 1
#                 if trial['stim2_theta'] == 0: 
#                     activity_array_1_1 = np.array(activity) if m_1 == 0 else np.append(activity_array_1_1,activity)
#                     m_1 += 1
#                 else:
#                     activity_array_1_2 = np.array(activity) if m_2 == 0 else np.append(activity_array_1_2,activity)
#                     m_2 += 1

#                 if trial['stim1_theta'] == 0: 
#                     activity_array_1_3 = np.array(activity) if m_3 == 0 else np.append(activity_array_1_3,activity)
#                     m_3 += 1
#                 else:
#                     activity_array_1_4 = np.array(activity) if m_4 == 0 else np.append(activity_array_1_4,activity)
#                     m_4 += 1

#             elif trial['ground_truth'] == 0 and trial[cue] == 1:
#                 activity_array_2 = np.array(activity) if n == 0 else np.append(activity_array_2,activity)
#                 n += 1
#                 if trial['stim2_theta'] == 0: 
#                     activity_array_2_1 = np.array(activity) if n_1 == 0 else np.append(activity_array_2_1,activity)
#                     n_1 += 1
#                 else:
#                     activity_array_2_2 = np.array(activity) if n_2 == 0 else np.append(activity_array_2_2,activity)
#                     n_2 += 1

#                 if trial['stim1_theta'] == 0: 
#                     activity_array_2_3 = np.array(activity) if n_3 == 0 else np.append(activity_array_2_3,activity)
#                     n_3 += 1
#                 else:
#                     activity_array_2_4 = np.array(activity) if n_4 == 0 else np.append(activity_array_2_4,activity)
#                     n_4 += 1

#     # activity_array_1 = activity_array_1.reshape(m, trial_length, 256).transpose(0, 2, 1)
#     # activity_mean_1 = np.mean(activity_array_1, axis=0)
#     # activity_projected_1 = q @ activity_mean_1

#     # activity_array_2 = activity_array_2.reshape(n, trial_length, 256).transpose(0, 2, 1)
#     # activity_mean_2 = np.mean(activity_array_2, axis=0)
#     # activity_projected_2 = q @ activity_mean_2

#     # plt.figure()
#     # plt.xlabel('ground_truth')
#     # plt.ylabel('stim_theta')
#     # plt.plot(activity_projected_1[0],activity_projected_1[1],'+-',color='red',label='gt=1 stim=180')
#     # plt.plot(activity_projected_2[0],activity_projected_2[1],'+-',color='blue',label='gt=0 stim=0')
#     # plt.legend()
#     # path = './output/pic/20220422/gt_stim_epoch5000_pdm.png'
#     # plt.savefig(path)

#     activity_mean_list = []
#     activity_array_1_1 = activity_array_1_1.reshape(m_1, trial_length, 256).transpose(0, 2, 1)
#     activity_array_1_2 = activity_array_1_2.reshape(m_2, trial_length, 256).transpose(0, 2, 1)
#     activity_array_2_1 = activity_array_2_1.reshape(n_1, trial_length, 256).transpose(0, 2, 1)
#     activity_array_2_2 = activity_array_2_2.reshape(n_2, trial_length, 256).transpose(0, 2, 1)
#     activity_mean_list.append(np.mean(activity_array_1_1, axis=0))
#     activity_mean_list.append(np.mean(activity_array_1_2, axis=0))
#     activity_mean_list.append(np.mean(activity_array_2_1, axis=0))
#     activity_mean_list.append(np.mean(activity_array_2_2, axis=0))

#     activity_array_1_3 = activity_array_1_3.reshape(m_3, trial_length, 256).transpose(0, 2, 1)
#     activity_array_1_4 = activity_array_1_4.reshape(m_4, trial_length, 256).transpose(0, 2, 1)
#     activity_array_2_3 = activity_array_2_3.reshape(n_3, trial_length, 256).transpose(0, 2, 1)
#     activity_array_2_4 = activity_array_2_4.reshape(n_4, trial_length, 256).transpose(0, 2, 1)
#     activity_mean_list.append(np.mean(activity_array_1_3, axis=0))
#     activity_mean_list.append(np.mean(activity_array_1_4, axis=0))
#     activity_mean_list.append(np.mean(activity_array_2_3, axis=0))
#     activity_mean_list.append(np.mean(activity_array_2_4, axis=0))

#     activity_array_1 = activity_array_1.reshape(m, trial_length, 256).transpose(0, 2, 1)
#     activity_array_2 = activity_array_2.reshape(n, trial_length, 256).transpose(0, 2, 1)    
#     activity_mean_list.append(np.mean(activity_array_1, axis=0))
#     activity_mean_list.append(np.mean(activity_array_2, axis=0)) # num_neuron * trial_length

#     return activity_mean_list
