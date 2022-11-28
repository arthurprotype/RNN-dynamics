from contextlib import redirect_stderr
from turtle import color
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.decomposition import PCA

from fixed_points import fixed_points

matplotlib.rcParams.update({'font.size': 10})

def plot_threevar(q, activity_mean_list, activity_dict, trial_infos, fixedpoints, marker_list, var_list, var_list_1 = None):
    
    var1, var2, var3 = var_list
    # if var_list_1 == None:
        # var_list_1 = var_list

    activity_projected_1_1 = q @ activity_mean_list[0]
    activity_projected_1_2 = q @ activity_mean_list[1]
    activity_projected_1_3 = q @ activity_mean_list[2]
    activity_projected_1_4 = q @ activity_mean_list[3]
    activity_projected_2_1 = q @ activity_mean_list[4]
    activity_projected_2_2 = q @ activity_mean_list[5]
    activity_projected_2_3 = q @ activity_mean_list[6]
    activity_projected_2_4 = q @ activity_mean_list[7]
    fixedpoints_projected = q @ fixedpoints

    fig = plt.figure(figsize=(5, 5))
    ax = plt.axes(projection='3d')
    ax.set_xlabel(var1)
    ax.set_ylabel(var2)
    ax.set_zlabel(var3)
    if var_list_1 == None:
        ax.plot(activity_projected_1_1[0],activity_projected_1_1[1],activity_projected_1_1[2],'-',color='red',label=var1+'=0 '+var2+'=0 '+var3+'=0')
        ax.plot(activity_projected_1_2[0],activity_projected_1_2[1],activity_projected_1_2[2],'-',color='yellow',label=var1+'=0 '+var2+'=0 '+var3+'!=0')
        ax.plot(activity_projected_1_3[0],activity_projected_1_3[1],activity_projected_1_3[2],'-',color='blue',label=var1+'=0 '+var2+'!=0 '+var3+'=0')
        ax.plot(activity_projected_1_4[0],activity_projected_1_4[1],activity_projected_1_4[2],'-',color='green',label=var1+'=0 '+var2+'!=0 '+var3+'!=0')
        ax.plot(activity_projected_2_1[0],activity_projected_2_1[1],activity_projected_2_1[2],'--',color='red',label=var1+'!=0 '+var2+'=0 '+var3+'=0')
        ax.plot(activity_projected_2_2[0],activity_projected_2_2[1],activity_projected_2_2[2],'--',color='yellow',label=var1+'!=0 '+var2+'=0 '+var3+'!=0')
        ax.plot(activity_projected_2_3[0],activity_projected_2_3[1],activity_projected_2_3[2],'--',color='blue',label=var1+'!=0 '+var2+'!=0 '+var3+'=0')
        ax.plot(activity_projected_2_4[0],activity_projected_2_4[1],activity_projected_2_4[2],'--',color='green',label=var1+'!=0 '+var2+'!=0 '+var3+'!=0')
        
        for i in marker_list:
            ax.plot(activity_projected_1_1[0,i],activity_projected_1_1[1,i],activity_projected_1_1[2,i],'*',color='red')
            ax.plot(activity_projected_1_2[0,i],activity_projected_1_2[1,i],activity_projected_1_2[2,i],'*',color='yellow')
            ax.plot(activity_projected_1_3[0,i],activity_projected_1_3[1,i],activity_projected_1_3[2,i],'*',color='blue')
            ax.plot(activity_projected_1_4[0,i],activity_projected_1_4[1,i],activity_projected_1_4[2,i],'*',color='green')
            ax.plot(activity_projected_2_1[0,i],activity_projected_2_1[1,i],activity_projected_2_1[2,i],'*',color='red')
            ax.plot(activity_projected_2_2[0,i],activity_projected_2_2[1,i],activity_projected_2_2[2,i],'*',color='yellow')
            ax.plot(activity_projected_2_3[0,i],activity_projected_2_3[1,i],activity_projected_2_3[2,i],'*',color='blue')
            ax.plot(activity_projected_2_4[0,i],activity_projected_2_4[1,i],activity_projected_2_4[2,i],'*',color='green')
        
    for i in range(10):
        trial = trial_infos[i]
        if trial['correct'] == 1:
            activity_pc = q @ activity_dict[i].transpose(1,0)
            # activity_pc = np.mean(activity_pc,axis=1)
            if trial[var2] == 0:
                color = 'red' if trial[var3] == 0 else 'yellow'
            else:
                color = 'blue' if trial[var3] == 0 else 'green'
            
            linestyle = '-' if trial[var1] == 0 else '--'
            ax.plot(activity_pc[0], activity_pc[1], activity_pc[2],
                    color=color,linestyle=linestyle,alpha=0.1)

    # ax.plot(fixedpoints_projected[0],fixedpoints_projected[1],fixedpoints_projected[2],'x')
    # ax.legend(loc=1)
    ax.set_title(var1+'-'+var2+'-'+var3)

    return fig

def plot_fourvar(q, activity_mean_list, activity_dict, trial_infos, fixedpoints, marker_list, var_list, var_list_1 = None):
    
    var1, var2, var3, var4 = var_list
    # if var_list_1 == None:
        # var_list_1 = var_list

    activity_projected_1_1 = q @ activity_mean_list[0]
    activity_projected_1_2 = q @ activity_mean_list[1]
    activity_projected_1_3 = q @ activity_mean_list[2]
    activity_projected_1_4 = q @ activity_mean_list[3]
    activity_projected_1_5 = q @ activity_mean_list[4]
    activity_projected_1_6 = q @ activity_mean_list[5]
    activity_projected_1_7 = q @ activity_mean_list[6]
    activity_projected_1_8 = q @ activity_mean_list[7]
    activity_projected_2_1 = q @ activity_mean_list[8]
    activity_projected_2_2 = q @ activity_mean_list[9]
    activity_projected_2_3 = q @ activity_mean_list[10]
    activity_projected_2_4 = q @ activity_mean_list[11]
    activity_projected_2_5 = q @ activity_mean_list[12]
    activity_projected_2_6 = q @ activity_mean_list[13]
    activity_projected_2_7 = q @ activity_mean_list[14]
    activity_projected_2_8 = q @ activity_mean_list[15]
    fixedpoints_projected = q @ fixedpoints

    fig = plt.figure(figsize=(5, 5))
    ax = plt.axes(projection='3d')
    ax.set_xlabel(var1)
    ax.set_ylabel(var3)
    ax.set_zlabel(var4)
    if var_list_1 == None:
        ax.plot(activity_projected_1_1[0],activity_projected_1_1[1],activity_projected_1_1[2],'-',color='red',label=var1+'=0 '+var2+'=0 '+var3+'=0 '+var4+'=0')
        ax.plot(activity_projected_1_2[0],activity_projected_1_2[1],activity_projected_1_2[2],'-',color='yellow',label=var1+'=0 '+var2+'=0 '+var3+'=0 '+var4+'!=0')
        ax.plot(activity_projected_1_3[0],activity_projected_1_3[1],activity_projected_1_3[2],'-',color='blue',label=var1+'=0 '+var2+'=0 '+var3+'!=0 '+var4+'=0')
        ax.plot(activity_projected_1_4[0],activity_projected_1_4[1],activity_projected_1_4[2],'-',color='green',label=var1+'=0 '+var2+'=0 '+var3+'!=0 '+var4+'!=0')
        ax.plot(activity_projected_1_5[0],activity_projected_1_5[1],activity_projected_1_5[2],'--',marker='x',color='red',label=var1+'=0 '+var2+'!=0 '+var3+'=0 '+var4+'=0')
        ax.plot(activity_projected_1_6[0],activity_projected_1_6[1],activity_projected_1_6[2],'--',marker='x',color='yellow',label=var1+'=0 '+var2+'!=0 '+var3+'=0 '+var4+'!=0')
        ax.plot(activity_projected_1_7[0],activity_projected_1_7[1],activity_projected_1_7[2],'--',marker='x',color='blue',label=var1+'=0 '+var2+'!=0 '+var3+'!=0 '+var4+'=0')
        ax.plot(activity_projected_1_8[0],activity_projected_1_8[1],activity_projected_1_8[2],'--',marker='x',color='green',label=var1+'=0 '+var2+'!=0 '+var3+'!=0 '+var4+'!=0')

        ax.plot(activity_projected_2_1[0],activity_projected_2_1[1],activity_projected_2_1[2],'-',color='red',label=var1+'!=0 '+var2+'=0 '+var3+'=0 '+var4+'=0')
        ax.plot(activity_projected_2_2[0],activity_projected_2_2[1],activity_projected_2_2[2],'-',color='yellow',label=var1+'!=0 '+var2+'=0 '+var3+'=0 '+var4+'!=0')
        ax.plot(activity_projected_2_3[0],activity_projected_2_3[1],activity_projected_2_3[2],'-',color='blue',label=var1+'!=0 '+var2+'=0 '+var3+'!=0 '+var4+'=0')
        ax.plot(activity_projected_2_4[0],activity_projected_2_4[1],activity_projected_2_4[2],'-',color='green',label=var1+'!=0 '+var2+'=0 '+var3+'!=0 '+var4+'!=0')
        ax.plot(activity_projected_2_5[0],activity_projected_2_5[1],activity_projected_2_5[2],'--',marker='x',color='red',label=var1+'!=0 '+var2+'!=0 '+var3+'=0 '+var4+'=0')
        ax.plot(activity_projected_2_6[0],activity_projected_2_6[1],activity_projected_2_6[2],'--',marker='x',color='yellow',label=var1+'!=0 '+var2+'!=0 '+var3+'=0 '+var4+'!=0')
        ax.plot(activity_projected_2_7[0],activity_projected_2_7[1],activity_projected_2_7[2],'--',marker='x',color='blue',label=var1+'!=0 '+var2+'!=0 '+var3+'!=0 '+var4+'=0')
        ax.plot(activity_projected_2_8[0],activity_projected_2_8[1],activity_projected_2_8[2],'--',marker='x',color='green',label=var1+'!=0 '+var2+'!=0 '+var3+'!=0 '+var4+'!=0')
        
        for i in marker_list:
            ax.plot(activity_projected_1_1[0,i],activity_projected_1_1[1,i],activity_projected_1_1[2,i],'*',color='red')
            ax.plot(activity_projected_1_2[0,i],activity_projected_1_2[1,i],activity_projected_1_2[2,i],'*',color='yellow')
            ax.plot(activity_projected_1_3[0,i],activity_projected_1_3[1,i],activity_projected_1_3[2,i],'*',color='blue')
            ax.plot(activity_projected_1_4[0,i],activity_projected_1_4[1,i],activity_projected_1_4[2,i],'*',color='green')
            ax.plot(activity_projected_1_5[0,i],activity_projected_1_5[1,i],activity_projected_1_5[2,i],'*',color='red')
            ax.plot(activity_projected_1_6[0,i],activity_projected_1_6[1,i],activity_projected_1_6[2,i],'*',color='yellow')
            ax.plot(activity_projected_1_7[0,i],activity_projected_1_7[1,i],activity_projected_1_7[2,i],'*',color='blue')
            ax.plot(activity_projected_1_8[0,i],activity_projected_1_8[1,i],activity_projected_1_8[2,i],'*',color='green')
            ax.plot(activity_projected_2_1[0,i],activity_projected_2_1[1,i],activity_projected_2_1[2,i],'*',color='red')
            ax.plot(activity_projected_2_2[0,i],activity_projected_2_2[1,i],activity_projected_2_2[2,i],'*',color='yellow')
            ax.plot(activity_projected_2_3[0,i],activity_projected_2_3[1,i],activity_projected_2_3[2,i],'*',color='blue')
            ax.plot(activity_projected_2_4[0,i],activity_projected_2_4[1,i],activity_projected_2_4[2,i],'*',color='green')
            ax.plot(activity_projected_2_5[0,i],activity_projected_2_5[1,i],activity_projected_2_5[2,i],'*',color='red')
            ax.plot(activity_projected_2_6[0,i],activity_projected_2_6[1,i],activity_projected_2_6[2,i],'*',color='yellow')
            ax.plot(activity_projected_2_7[0,i],activity_projected_2_7[1,i],activity_projected_2_7[2,i],'*',color='blue')
            ax.plot(activity_projected_2_8[0,i],activity_projected_2_8[1,i],activity_projected_2_8[2,i],'*',color='green')
        
    # for i in range(10):
    #     trial = trial_infos[i]
    #     if trial['correct'] == 1:
    #         activity_pc = q @ activity_dict[i].transpose(1,0)
    #         # activity_pc = np.mean(activity_pc,axis=1)
    #         if trial[var2] == 0:
    #             color = 'red' if trial[var3] == 0 else 'yellow'
    #         else:
    #             color = 'blue' if trial[var3] == 0 else 'green'
            
    #         linestyle = '-' if trial[var1] == 0 else '--'
    #         ax.plot(activity_pc[0], activity_pc[1], activity_pc[2],
    #                 color=color,linestyle=linestyle,alpha=0.1)

    # ax.plot(fixedpoints_projected[0],fixedpoints_projected[1],fixedpoints_projected[2],'x')
    ax.legend(loc=1)
    ax.set_title(var1+'-'+var2+'-'+var3)

    return fig

def plot_var(activity_var_list, marker_list, var_list, task):

    fig = plt.figure(figsize=(5,5))
    plt.xlabel('Time')
    plt.ylabel('Variance')
    if task == 'PerceptualDecisionMaking-v0':
        var1, var2 = var_list
        plt.plot(activity_var_list[0],color='red',marker='*',markevery=marker_list,label=var1+'=0')
        plt.plot(activity_var_list[1],color='blue',marker='*',markevery=marker_list,label=var1+'!=0')  
    elif task == 'DelayMacthSample-v0':
        var1, var2 = var_list
        plt.plot(activity_var_list[2],'-',color='blue',marker='*',markevery=marker_list,label=var1+'=2 '+var2+'=0')
        plt.plot(activity_var_list[3],'--',color='blue',marker='*',markevery=marker_list,label=var1+'=2 '+var2+'!=0')
        plt.plot(activity_var_list[0],'-',color='red',marker='*',markevery=marker_list,label=var1+'=1 '+var2+'=0')
        plt.plot(activity_var_list[1],'--',color='red',marker='*',markevery=marker_list,label=var1+'=1 '+var2+'!=0')
    else:
        if len(var_list) == 2:
            var1, var2 = var_list
            plt.plot(activity_var_list[0],'-',color='red',marker='*',markevery=marker_list,label=var1+'=0 '+var2+'=0')
            plt.plot(activity_var_list[1],'--',color='red',marker='*',markevery=marker_list,label=var1+'=0 '+var2+'!=0')
            plt.plot(activity_var_list[2],'-',color='blue',marker='*',markevery=marker_list,label=var1+'!=0 '+var2+'=0')
            plt.plot(activity_var_list[3],'--',color='blue',marker='*',markevery=marker_list,label=var1+'!=0 '+var2+'!=0')
        elif len(var_list) == 3:
            var1, var2, var3 = var_list
            plt.plot(activity_var_list[0],'-',color='red',label=var1+'=0 '+var2+'=0 '+var3+'=0')
            plt.plot(activity_var_list[1],'-',color='yellow',label=var1+'=0 '+var2+'=0 '+var3+'!=0')
            plt.plot(activity_var_list[2],'-',color='blue',label=var1+'=0 '+var2+'!=0 '+var3+'=0')
            plt.plot(activity_var_list[3],'-',color='green',label=var1+'=0 '+var2+'!=0 '+var3+'!=0')
            plt.plot(activity_var_list[4],'--',color='red',label=var1+'!=0 '+var2+'=0 '+var3+'=0')
            plt.plot(activity_var_list[5],'--',color='yellow',label=var1+'!=0 '+var2+'=0 '+var3+'!=0')
            plt.plot(activity_var_list[6],'--',color='blue',label=var1+'!=0 '+var2+'!=0 '+var3+'=0')
            plt.plot(activity_var_list[7],'--',color='green',label=var1+'!=0 '+var2+'!=0 '+var3+'!=0')
            
            for i in range(len(marker_list)):
                plt.axvline(x=marker_list[i],linestyle='--',color='blue')
        else:
            var1, var2, var3, var4 = var_list
            plt.plot(activity_var_list[0],'-',color='red',label=var1+'=0 '+var2+'=0 '+var3+'=0 '+var4+'=0')
            plt.plot(activity_var_list[1],'-',color='yellow',label=var1+'=0 '+var2+'=0 '+var3+'=0 '+var4+'!=0')
            plt.plot(activity_var_list[2],'-',color='blue',label=var1+'=0 '+var2+'=0 '+var3+'!=0 '+var4+'=0')
            plt.plot(activity_var_list[3],'-',color='green',label=var1+'=0 '+var2+'=0 '+var3+'!=0 '+var4+'!=0')
            plt.plot(activity_var_list[4],'-',marker='x',color='red',label=var1+'=0 '+var2+'!=0 '+var3+'=0 '+var4+'=0')
            plt.plot(activity_var_list[5],'-',marker='x',color='yellow',label=var1+'=0 '+var2+'!=0 '+var3+'=0 '+var4+'!=0')
            plt.plot(activity_var_list[6],'-',marker='x',color='blue',label=var1+'=0 '+var2+'!=0 '+var3+'!=0 '+var4+'=0')
            plt.plot(activity_var_list[7],'-',marker='x',color='green',label=var1+'=0 '+var2+'!=0 '+var3+'!=0 '+var4+'!=0')
            plt.plot(activity_var_list[8],'--',color='red',label=var1+'!=0 '+var2+'=0 '+var3+'=0 '+var4+'=0')
            plt.plot(activity_var_list[9],'--',color='yellow',label=var1+'!=0 '+var2+'=0 '+var3+'=0 '+var4+'!=0')
            plt.plot(activity_var_list[10],'--',color='blue',label=var1+'!=0 '+var2+'=0 '+var3+'!=0 '+var4+'=0')
            plt.plot(activity_var_list[11],'--',color='green',label=var1+'!=0 '+var2+'=0 '+var3+'!=0 '+var4+'!=0')
            plt.plot(activity_var_list[12],'--',marker='x',color='red',label=var1+'!=0 '+var2+'!=0 '+var3+'=0 '+var4+'=0')
            plt.plot(activity_var_list[13],'--',marker='x',color='yellow',label=var1+'!=0 '+var2+'!=0 '+var3+'=0 '+var4+'!=0')
            plt.plot(activity_var_list[14],'--',marker='x',color='blue',label=var1+'!=0 '+var2+'!=0 '+var3+'!=0 '+var4+'=0')
            plt.plot(activity_var_list[15],'--',marker='x',color='green',label=var1+'!=0 '+var2+'!=0 '+var3+'!=0 '+var4+'!=0')
            
            for i in range(len(marker_list)):
                plt.axvline(x=marker_list[i],linestyle='--',color='blue')

    # plt.legend(loc=1)

    return fig

def plot_firing_rate(activity_mean_list, activity_mean_list_, var1, var2, trial_infos, marker_list, task):
    # fig = plt.figure()
    bool11 = bool12 = bool21 = bool22 = False
    for i in range(len(activity_mean_list)):
        # color = 'red' if trial_infos[i]['ground_truth'] == 0 else 'blue'
        trial_info = trial_infos[i]
        if trial_info[var1] == 0:
            if trial_info[var2] == 0:
                if bool11 == False:
                    plt.plot(activity_mean_list[i], color = 'red', label=var1+'=0 '+var2+'=0')
                    bool11 = True
                else:
                    plt.plot(activity_mean_list[i], color = 'red')
            else:
                if bool12 == False:
                    plt.plot(activity_mean_list[i], color = 'yellow', label=var1+'=0 '+var2+'!=0')
                    bool12 = True
                else:
                    plt.plot(activity_mean_list[i], color = 'yellow')
        else:
            if trial_info[var2] == 0:
                if bool21 == False:
                    plt.plot(activity_mean_list[i], color = 'blue', label=var1+'!=0 '+var2+'=0')
                    bool21 = True
                else:
                    plt.plot(activity_mean_list[i], color = 'blue')
            else:
                if bool22 == False:
                    plt.plot(activity_mean_list[i], color = 'green', label=var1+'!=0 '+var2+'!=0')
                    bool22 = True
                else:
                    plt.plot(activity_mean_list[i], color = 'green')

    plt.plot(activity_mean_list_, linestyle = '-', linewidth = 3, color = 'black')
    for i in range(len(marker_list)):
        color = 'red' if i in [1,3] else 'blue'
        plt.axvline(x=marker_list[i],linestyle='--',color=color)
    plt.xlabel('Time')
    plt.ylabel('Firing rate')
    plt.legend(loc=1)
    plt.title(task+' '+var1+'-'+var2)

    # return fig


def plot_pca_twovar(activity_dict, fixedpoints, fixedpoints_2, end_pts, fp, pca, trial_infos, task):
    
    fig = plt.figure(figsize=(5,5))
    for i in range(50):
        activity_pc = pca.transform(activity_dict[i])
        # activity_pc = np.mean(activity_pc,axis=0)
        trial = trial_infos[i]
        if task == 'DelayMatchSample-v0':
            color = 'red' if trial['stim_ch1'] == 0 else 'blue'
        else:
            color = 'red' if trial['ground_truth'] == 0 else 'blue'
        # plt.plot(activity_pc[0], activity_pc[1], '--',
        #          color=color, alpha=1)
        plt.plot(activity_pc[:, 0], activity_pc[:, 1], '--',
                color=color, alpha=0.1)
        plt.plot(activity_pc[0,0],activity_pc[0,1],'*',color=color)

    # Fixed points are shown in cross
    fixedpoints = fixedpoints.transpose(1,0)
    fixedpoints_pc = pca.transform(fixedpoints)
    plt.plot(fixedpoints_pc[:, 0], fixedpoints_pc[:, 1], 'x',label='trajectory')

    fixedpoints_2 = fixedpoints_2.transpose(1,0)
    fixedpoints_pc = pca.transform(fixedpoints_2)
    plt.plot(fixedpoints_pc[:, 0], fixedpoints_pc[:, 1], 'o', markerfacecolor='none', alpha=0.7, label='random')
    
    if task == 'PerceptualDecisionMaking-v0':
        end_pts = pca.transform(fp.detach().numpy() + end_pts)
        plt.plot(end_pts[:, 0], end_pts[:, 1],color='purple')

    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    
    return fig

def plot_pca_threevar(activity_dict, fixedpoints, fixedpoints_2, pca, trial_infos, task):
    
    fig = plt.figure(figsize=(5,5))
    ax = plt.axes(projection='3d')
    for i in range(50):
        activity_pc = pca.transform(activity_dict[i])
        # activity_pc = np.mean(activity_pc,axis=0)
        trial = trial_infos[i]
        if task == 'DelayMatchSample-v0':
            color = 'red' if trial['stim_ch1'] == 0 else 'blue'
        else:
            color = 'red' if trial['ground_truth'] == 0 else 'blue'
        # plt.plot(activity_pc[0], activity_pc[1], '--',
        #          color=color, alpha=1)
        ax.plot(activity_pc[:, 0], activity_pc[:, 1], activity_pc[:, 2], '--',
                color=color, alpha=0.1)
        ax.plot(activity_pc[0,0],activity_pc[0,1],activity_pc[0,2],'*',color=color)

    # Fixed points are shown in cross
    fixedpoints = fixedpoints.transpose(1,0)
    fixedpoints_pc = pca.transform(fixedpoints)
    ax.plot(fixedpoints_pc[:, 0], fixedpoints_pc[:, 1], fixedpoints_pc[:, 2], 'x')

    fixedpoints_2 = fixedpoints_2.transpose(1,0)
    fixedpoints_pc = pca.transform(fixedpoints_2)
    ax.plot(fixedpoints_pc[:, 0], fixedpoints_pc[:, 1], fixedpoints_pc[:, 2], 'o')

    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.set_zlabel('PC 3')

    return fig

def plot_twovar(q, activity_mean_list, activity_dict, trial_infos, fixedpoints, marker_list, var_list, var_list_1 = None):
    
    var1, var2 = var_list
    if var_list_1 == None:
        var_list_1 = var_list
    var1_1, var2_1 = var_list_1

    activity_projected_1_1 = q @ activity_mean_list[0]
    activity_projected_1_2 = q @ activity_mean_list[1]
    activity_projected_2_1 = q @ activity_mean_list[2]
    activity_projected_2_2 = q @ activity_mean_list[3]
    fixedpoints_projected = q @ fixedpoints

    # fixedpoints_projected_2 = q @ fixedpoints_2

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.set_xlabel(var1)
    ax.set_ylabel(var2)
    ax.plot(activity_projected_1_1[0],activity_projected_1_1[1],'-',color='red',label=var1_1+'=0 '+var2_1+'=0')
    ax.plot(activity_projected_1_2[0],activity_projected_1_2[1],'--',color='red',label=var1_1+'=0 '+var2_1+'!=0')
    ax.plot(activity_projected_2_1[0],activity_projected_2_1[1],'-',color='blue',label=var1_1+'!=0 '+var2_1+'=0')
    ax.plot(activity_projected_2_2[0],activity_projected_2_2[1],'--',color='blue',label=var1_1+'!=0 '+var2_1+'!=0')
    for i in marker_list:
        if i == 0:
            ax.plot(activity_projected_1_1[0,i],activity_projected_1_1[1,i],'o',color='red')
            ax.plot(activity_projected_1_2[0,i],activity_projected_1_2[1,i],'o',color='red')
            ax.plot(activity_projected_2_1[0,i],activity_projected_2_1[1,i],'o',color='blue')
            ax.plot(activity_projected_2_2[0,i],activity_projected_2_2[1,i],'o',color='blue')
        else:
            ax.plot(activity_projected_1_1[0,i],activity_projected_1_1[1,i],'*',color='red')
            ax.plot(activity_projected_1_2[0,i],activity_projected_1_2[1,i],'*',color='red')
            ax.plot(activity_projected_2_1[0,i],activity_projected_2_1[1,i],'*',color='blue')
            ax.plot(activity_projected_2_2[0,i],activity_projected_2_2[1,i],'*',color='blue')

    for i in range(50):
        trial = trial_infos[i]
        if trial['correct'] == 1:
            activity_pc = q @ activity_dict[i].transpose(1,0)
            activity_pc = np.mean(activity_pc,axis=1)
            color = 'red' if trial[var1_1] == 0 else 'blue'
            linestyle = '-' if trial[var2_1] == 0 else '--'
            ax.plot(activity_pc[0], activity_pc[1],
                    color=color,linestyle=linestyle,alpha=0.1)
                    #marker='o',markevery=markevery_list)
    # ax.plot(fixedpoints_projected[0],fixedpoints_projected[1],'x')

    # ax.plot(fixedpoints_projected_2[0],fixedpoints_projected_2[1],'o')
    ax.legend(loc=1)
    ax.set_title(var1+'-'+var2)

    return fig
    
def plot_pdm(q, activity_mean_list, activity_dict, trial_infos, fixedpoints, marker_list):

    activity_projected_1 = q @ activity_mean_list[0]
    activity_projected_2 = q @ activity_mean_list[1]
    fixedpoints_projected = q @ fixedpoints
    # fixedpoints_projected_2 = q @ fixedpoints_2
    # end_pts = q @ (fp.detach().numpy() + end_pts).transpose(1,0)

    # fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    fig,(ax,ax1) = plt.subplots(1,2,figsize=(10,5),dpi=600,sharey=True)
    ax.set_xlabel('ground_truth')
    ax.set_ylabel('stim_theta')
    ax.plot(activity_projected_1[0],activity_projected_1[1],'-',color='red',label='gt=0')
    ax.plot(activity_projected_2[0],activity_projected_2[1],'-',color='blue',label='gt=1')
    for i in marker_list:
        ax.plot(activity_projected_1[0,i],activity_projected_1[1,i],'*',color='red')
        ax.plot(activity_projected_2[0,i],activity_projected_2[1,i],'*',color='blue')

    for i in range(50):
        trial = trial_infos[i]
        if trial['correct'] == 1:
            activity_pc = q @ activity_dict[i].transpose(1,0)
            color = 'red' if trial['ground_truth'] == 0 else 'blue'
            ax.plot(activity_pc[0], activity_pc[1],
                    color=color,linestyle='--',alpha=0.1)
    ax.plot(fixedpoints_projected[0],fixedpoints_projected[1],'x',label='fixed points')
    # ax.plot(fixedpoints_projected_2[0],fixedpoints_projected_2[1],'o',markerfacecolor='none',alpha=0.7,label='random')
    # ax.plot(end_pts[0], end_pts[1],color='purple',label='attractor')
    ax.legend(loc=1)

    ax.set_title('gt-stim')

    # fig1, ax1 = plt.subplots(1, 1, figsize=(5, 5))
    ax1.set_xlabel('ground_truth')
    ax1.set_ylabel('stim_theta')
    ax1.plot(activity_projected_1[0],activity_projected_1[1],'-',color='red',label='gt=0')
    ax1.plot(activity_projected_2[0],activity_projected_2[1],'-',color='blue',label='gt=1')
    for i in marker_list:
        ax1.plot(activity_projected_1[0,i],activity_projected_1[1,i],'*',color='red')
        ax1.plot(activity_projected_2[0,i],activity_projected_2[1,i],'*',color='blue')

    for i in range(50):
        trial = trial_infos[i]
        if trial['correct'] == 1:
            activity_pc = q @ activity_dict[i].transpose(1,0)
            color = 'red' if trial['ground_truth'] == 0 else 'blue'
            ax1.plot(activity_pc[0], activity_pc[1],
                    color=color,linestyle='--',alpha=0.1)
    ax1.legend(loc=1)

    return fig
    
def plot_dms(q, activity_mean_list, activity_dict, trial_infos, fixedpoints, marker_list):
    activity_projected_1_1 = q @ activity_mean_list[0]
    activity_projected_1_2 = q @ activity_mean_list[1]
    activity_projected_2_1 = q @ activity_mean_list[2]
    activity_projected_2_2 = q @ activity_mean_list[3]

    activity_projected_1 = q @ activity_mean_list[4]
    activity_projected_2 = q @ activity_mean_list[5]

    fixedpoints_projected = q @ fixedpoints
    # fixedpoints_projected_2 = q @ fixedpoints_2

    # end_pts = q @ (fp.detach().numpy() + end_pts).transpose(1,0)


    fig,(ax,ax1) = plt.subplots(1,2,figsize=(10,5),dpi=600,sharey=True)

    ax.set_xlabel('ground_truth')
    ax.set_ylabel('stim_theta')
    ax.plot(activity_projected_1_1[0],activity_projected_1_1[1],'-',color='red',label='gt=1 stim=0')
    ax.plot(activity_projected_1_2[0],activity_projected_1_2[1],'--',color='red',label='gt=1 stim=180')
    ax.plot(activity_projected_2_1[0],activity_projected_2_1[1],'-',color='blue',label='gt=0 stim=0')
    ax.plot(activity_projected_2_2[0],activity_projected_2_2[1],'--',color='blue',label='gt=0 stim=180')
    for i in marker_list:
        ax.plot(activity_projected_1_1[0,i],activity_projected_1_1[1,i],'*',color='red')
        ax.plot(activity_projected_1_2[0,i],activity_projected_1_2[1,i],'*',color='red')
        ax.plot(activity_projected_2_1[0,i],activity_projected_2_1[1,i],'*',color='blue')
        ax.plot(activity_projected_2_2[0,i],activity_projected_2_2[1,i],'*',color='blue')
        
    # ax.legend()
    # ax.set_title('gt-stim_stim')

    # ax = fig.add_subplot(122)
    # ax.set_xlabel('ground_truth')
    # ax.plot(activity_projected_1[0],activity_projected_1[1],'-',color='red',label='gt=1')
    # ax.plot(activity_projected_2[0],activity_projected_2[1],'-',color='blue',label='gt=0')
    # for i in marker_list:
    #     ax.plot(activity_projected_1[0,i],activity_projected_1[1,i],'*',color='red')
    #     ax.plot(activity_projected_2[0,i],activity_projected_2[1,i],'*',color='blue')

    # for i in range(50):
    #     trial = trial_infos[i]
    #     if trial['correct'] == 1:
    #         activity_pc = q @ activity_dict[i].transpose(1,0)
    #         color = 'red' if trial['ground_truth'] == 0 else 'blue'
    #         ax.plot(activity_pc[0], activity_pc[1],
    #                 color=color,linestyle='--',alpha=0.1)
    ax.legend(loc=1)
    ax.set_title('gt-stim')

    # fig1, ax1 = plt.subplots(1, 1, figsize=(5, 5))
    # ax1 = plt.subplot(122)
    ax1.set_xlabel('ground_truth')
    ax1.set_ylabel('stim_theta')
    ax1.plot(activity_projected_1_1[0],activity_projected_1_1[1],'-',color='red',label='gt=1 stim=0')
    ax1.plot(activity_projected_1_2[0],activity_projected_1_2[1],'--',color='red',label='gt=1 stim=180')
    ax1.plot(activity_projected_2_1[0],activity_projected_2_1[1],'-',color='blue',label='gt=0 stim=0')
    ax1.plot(activity_projected_2_2[0],activity_projected_2_2[1],'--',color='blue',label='gt=0 stim=180')
    for i in marker_list:
        ax1.plot(activity_projected_1_1[0,i],activity_projected_1_1[1,i],'*',color='red')
        ax1.plot(activity_projected_1_2[0,i],activity_projected_1_2[1,i],'*',color='red')
        ax1.plot(activity_projected_2_1[0,i],activity_projected_2_1[1,i],'*',color='blue')
        ax1.plot(activity_projected_2_2[0,i],activity_projected_2_2[1,i],'*',color='blue')

    for i in range(50):
        trial = trial_infos[i]
        if trial['correct'] == 1:
            activity_pc = q @ activity_dict[i].transpose(1,0)
            color = 'red' if trial['ground_truth'] == 0 else 'blue'
            ax1.plot(activity_pc[0], activity_pc[1],
                    color=color,linestyle='--',alpha=0.1)

    ax1.plot(fixedpoints_projected[0],fixedpoints_projected[1],'x',label='fixed points')
    # ax.plot(fixedpoints_projected_2[0],fixedpoints_projected_2[1],'o',markerfacecolor='none',alpha=0.7,label='random')
    
    # ax1.plot(end_pts[0], end_pts[1],color='purple',label='attractor')
    ax1.legend(loc=1)
    ax.set_title('gt-stim')
     
    return fig
  


