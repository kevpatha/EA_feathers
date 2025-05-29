
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

import scipy.stats as stats

import seaborn as sns


from matplotlib.lines import Line2D

def run(load_data = False):
    if load_data:
        load_and_save_data()
 
    Base_df = pd.read_csv('Base_wing_data_full_set_vel_adjX.csv',sep = ',',
                              usecols=[0,1,2,3,4,8,9,10])
    Base_df['CD'] *= -1
    Base_df['Cm'] *= -1
    Base_df['LD'] = Base_df['CL']/Base_df['CD']
    Base_df['iter_d'] = (np.select([Base_df['direction'].eq('PN'),
                                    Base_df['direction'].eq('NP')],
                                   [Base_df['iter'],
                                    Base_df['iter'] +4],
                                   Base_df['iter']))
   
    
    print('Base Loaded')
    Plate_df = pd.read_csv('Plate_wing_data_full_set_vel_adjX.csv',sep = ',',
                              usecols=[0,1,2,3,4,8,9,10])
    Plate_df['CD'] *= -1
    Plate_df['Cm'] *= -1
    Plate_df['LD'] = Plate_df['CL']/Plate_df['CD']
    Plate_df['iter_d'] = (np.select([Plate_df['direction'].eq('PN'),
                                    Plate_df['direction'].eq('NP')],
                                   [Plate_df['iter'],
                                    Plate_df['iter'] +4],
                                   Plate_df['iter']))
 
   
    print('Plate Loaded')
    F9v0_df = pd.read_csv('F9v0_wing_data_full_set_vel_adjX.csv',sep = ',',
                              usecols=[0,1,2,3,4,8,9,10])
    F9v0_df['CD'] *= -1
    F9v0_df['Cm'] *= -1
    F9v0_df['LD'] = F9v0_df['CL']/F9v0_df['CD']
    F9v0_df['iter_d'] = (np.select([F9v0_df['direction'].eq('PN'),
                                    F9v0_df['direction'].eq('NP')],
                                   [F9v0_df['iter'],
                                    F9v0_df['iter'] +4],
                                   F9v0_df['iter']))
    

    print('F9v0 Loaded')
    
    F9v15_df = pd.read_csv('F9v15_wing_data_full_set_vel_adjX.csv',sep = ',',
                              usecols=[0,1,2,3,4,8,9,10])
    F9v15_df['CD'] *= -1
    F9v15_df['Cm'] *= -1
    F9v15_df['LD'] = F9v15_df['CL']/F9v15_df['CD']
    F9v15_df['iter_d'] = (np.select([F9v15_df['direction'].eq('PN'),
                                    F9v15_df['direction'].eq('NP')],
                                   [F9v15_df['iter'],
                                    F9v15_df['iter'] +4],
                                   F9v15_df['iter']))
 

    print('F9v15 Loaded')
    
    F9v2_df = pd.read_csv('F9v2_wing_data_full_set_vel_adjX.csv',sep = ',',
                              usecols=[0,1,2,3,4,8,9,10])
    F9v2_df['CD'] *= -1
    F9v2_df['Cm'] *= -1
    F9v2_df['LD'] = F9v2_df['CL']/F9v2_df['CD']
    F9v2_df['iter_d'] = (np.select([F9v2_df['direction'].eq('PN'),
                                    F9v2_df['direction'].eq('NP')],
                                   [F9v2_df['iter'],
                                    F9v2_df['iter'] +4],
                                   F9v2_df['iter']))
  

    print('F9v2 Loaded')
    
    F9v3_df = pd.read_csv('F9v3_wing_data_full_set_vel_adjX.csv',sep = ',',
                              usecols=[0,1,2,3,4,8,9,10])
    F9v3_df['CD'] *= -1
    F9v3_df['Cm'] *= -1
    F9v3_df['LD'] = F9v3_df['CL']/F9v3_df['CD']
    F9v3_df['iter_d'] = (np.select([F9v3_df['direction'].eq('PN'),
                                    F9v3_df['direction'].eq('NP')],
                                   [F9v3_df['iter'],
                                    F9v3_df['iter'] +4],
                                   F9v3_df['iter']))
   

    print('F9v3 Loaded')
    
    coeff_df=pd.concat([Base_df,Plate_df,F9v0_df,F9v15_df,F9v2_df,F9v3_df], 
                                  ignore_index=True)
    
    coeff_df['LD']=coeff_df['CL']/coeff_df['CD']
    coeff_df['vel']=coeff_df['rpm']*0.055+0.525
    
    coeff_grouped = coeff_df.groupby(['rpm','config'])
    
    figCL = plt.figure(figsize=(5,3), dpi=300)
    clax=sns.boxplot(coeff_df,x ='CL', y='config', hue='vel')
    sns.move_legend(clax, "upper left", bbox_to_anchor=(1, 1))
    figCD = plt.figure(figsize=(5,3), dpi=300)
    sns.boxplot(coeff_df,x ='CD', y='config', hue='vel', legend=False)
  
    sns.boxplot(coeff_df.loc[coeff_df['rpm']==200],x ='CL', y='config')
    sns.boxplot(coeff_df.loc[coeff_df['rpm']==300],x ='CL', y='config')
    sns.boxplot(coeff_df.loc[coeff_df['rpm']==100],x ='CD', y='config')
    sns.boxplot(coeff_df.loc[coeff_df['rpm']==200],x ='CD', y='config')
    sns.boxplot(coeff_df.loc[coeff_df['rpm']==300],x ='CD', y='config')
    
    coeff_df = coeff_df.loc[coeff_df['direction']=='NP']
    
    print(coeff_df.head())
    CL_df = combine_iters_CL(coeff_df)
    CD_df = combine_iters_CD(coeff_df)
    LD_df = combine_iters_LD(coeff_df)
    Cm_df = combine_iters_Cm(coeff_df)
    
    CL_df.to_csv('CL_condensed_data.csv',sep = ',', index=False)
    CD_df.to_csv('CD_condensed_data.csv',sep = ',', index=False)
    LD_df.to_csv('LD_condensed_data.csv',sep = ',', index=False)
    Cm_df.to_csv('Cm_condensed_data.csv',sep = ',', index=False)
    ## Static Plots
    b_color = '#221F20'
    p_color = '#D5D5D7'
    f9v0_color = '#4771b2'
    f9v3_color = '#c6793a'
    
    CL_vs_AoA(CL_df)
    CD_vs_AoA(CD_df)
    LD_vs_AoA(LD_df)
    Cm_vs_AoA(Cm_df)
    CL_vs_Cm(CL_df, Cm_df)
    
    CL_df_dirs = combine_but_dir_it(coeff_df,'CL')
    CL_maxs_dirs = CL_df_dirs.groupby(['config','rpm', 'iter_d'])['CL'].idxmax(axis=0)
    CL_maxs_dirs_df = CL_df_dirs.iloc[CL_maxs_dirs]
    print(CL_maxs_dirs_df)
    
    CL_df_up = up_sweep_analysis(coeff_df,'CL')
    LD_df_up = up_sweep_analysis(coeff_df,'LD')
    CD_df_up = up_sweep_analysis(coeff_df, 'CD')
    Cm_df_up = up_sweep_analysis(coeff_df,'Cm')
    
    CL_vs_AoA_up(CL_df_up)
    CD_vs_AoA_up(CD_df_up)
    Cm_vs_AoA_up(Cm_df_up)
    
    mixed_plots(LD_df, Cm_df, CL_df)
    
    feather_plots(CL_df_up, LD_df_up, Cm_df_up)
    print('feather plots')
    
    
    #maxCL stat analysis
    
    cl_vel = pd.DataFrame()
    
    print("Base max_cl stat analysis")
    cl_vel['base'], mx_stat_df_base = max_stats_analys(Base_df,
                                CL_maxs_dirs_df.loc[CL_maxs_dirs_df['config'] == 'base'],
                                coeff='CL')
    print("Plate max_cl stat analysis")
    cl_vel['plate'], mx_stat_df_plate = max_stats_analys(Plate_df, 
                                CL_maxs_dirs_df.loc[CL_maxs_dirs_df['config'] == 'plate'],
                                coeff='CL')
    print("f9v0 max_cl stat analysis")
    cl_vel['f9_v0'], mx_stat_df_f9v0 = max_stats_analys(F9v0_df, 
                                    CL_maxs_dirs_df.loc[CL_maxs_dirs_df['config'] == 'f9_v0'],
                                    coeff='CL')
    print("f9v3 max_cl stat analysis")
    cl_vel['f9_v3'], mx_stat_df_f9v3 =  max_stats_analys(F9v3_df, 
                                    CL_maxs_dirs_df.loc[CL_maxs_dirs_df['config'] == 'f9_v3'],
                                    coeff='CL')
    
    mx_stat_df = pd.concat([mx_stat_df_base, mx_stat_df_plate, mx_stat_df_f9v0, mx_stat_df_f9v3],
              ignore_index=True)

    print(cl_vel)
    fig, ax = plt.subplots(2,2,figsize=(4,3), width_ratios=[2,1], dpi=300)
    print('Max CL stat analysis at each velocity')
    max_plots(mx_stat_df, 'CL', ax[0,:], trans=False, OLS=False)
    mx_stat_df.to_csv('mx_stat_df.csv')
    
    
    ax[0,0].spines['bottom'].set_visible(False)
    ax[0,0].get_xaxis().set_visible(False)
    ax[0,1].get_xaxis().set_visible(False)
    
    ax[0,0].set_ylim([0.7,1.3])
    ax[0,0].get_yaxis().set_ticks([0.7,1.0,1.3])
    ax[0,0].set_ylabel('$C_{L max}$')
    ax[0,1].set_ylim([-0.01,0.02])
    ax[0,1].set_ylabel('$dC_{L max} /dU$')
    

    print('CL slopes')
    get_slopes_X(Base_df, Plate_df, F9v0_df, F9v3_df, ax[1,:])
    all_df = pd.concat([Base_df, Plate_df, F9v0_df, F9v3_df], ignore_index=True)
    all_df['vel'] = all_df['rpm']*0.055+0.525

    
    ax[1,0].set_ylim([0.06,0.075])
    ax[1,0].set_ylabel('$C_{L \\alpha}$')
    
    ax[1,1].spines['bottom'].set_visible(False)
    ax[1,1].set_ylim([-0.001,0.001])
    ax[1,1].set_ylabel('$dC_{L \\alpha} /dU$')
    ax[1,1].get_xaxis().set_visible(False)
    
    custom_lines1 = [Line2D([0], [0], color= b_color, lw=1),
                Line2D([0], [0], color= p_color, lw=1),
                Line2D([0], [0], color= f9v0_color, lw=1),
                Line2D([0], [0], color= f9v3_color, lw=1)]
    
    legend1 = fig.legend(custom_lines1,['Baseline','Plate','F9_v0','F9_v3'],
                          ncol= 4, loc=8, bbox_to_anchor=(0.55,-0.04), frameon=False)
    
   
    fig.tight_layout()
    
    fig.savefig('slope_plots', dpi=300)
    
    ### L/D max
    
    LD_df_dirs = combine_but_dir_it(coeff_df,'LD')
    LD_maxs_dirs = LD_df_dirs.groupby(['config','rpm', 'iter_d'])['LD'].idxmax(axis=0)
    LD_maxs_dirs_df = LD_df_dirs.iloc[LD_maxs_dirs]
    print('LD_maxs_dirs_df: ', LD_maxs_dirs_df)
    
    LD_vel = pd.DataFrame()
    
    print("Base max_LD stat analysis")
    LD_vel['base'], LD_stat_df_base = max_stats_analys(Base_df,
                                LD_maxs_dirs_df.loc[LD_maxs_dirs_df['config'] == 'base'],
                                coeff='LD')
    print("Plate max_LD stat analysis")
    LD_vel['plate'], LD_stat_df_plate = max_stats_analys(Plate_df, 
                                LD_maxs_dirs_df.loc[LD_maxs_dirs_df['config'] == 'plate'],
                                coeff='LD')
    print("f9v0 max_LD stat analysis")
    LD_vel['f9_v0'], LD_stat_df_f9v0 = max_stats_analys(F9v0_df, 
                                    LD_maxs_dirs_df.loc[LD_maxs_dirs_df['config'] == 'f9_v0'],
                                    coeff='LD')
    print("f9v3 max_LD stat analysis")
    LD_vel['f9_v3'], LD_stat_df_f9v3 =  max_stats_analys(F9v3_df, 
                                    LD_maxs_dirs_df.loc[LD_maxs_dirs_df['config'] == 'f9_v3'],
                                    coeff='LD')
    
    LD_stat_df = pd.concat([LD_stat_df_base, LD_stat_df_plate, LD_stat_df_f9v0, LD_stat_df_f9v3],
              ignore_index=True)
    

    
    print('Max LD stat analysis at each velocity')
  
    fig, ax = plt.subplots(2,2,figsize=(4,4), width_ratios=[2,1], dpi=300)
    max_plots(LD_stat_df, 'LD', ax[0,:])
    
    LD_stat_df.to_csv('LD_stat_df.csv')

    rpms = [200, 250, 300]

    ax[0,0].set_ylim([4,8])
    ax[0,0].set_ylabel('$L/D_{max}$')
    vels = ['11.5', '14.3', '17']
    ax[0,0].set_xticks(rpms, vels)
    ax[0,1].get_xaxis().set_visible(False)
    ax[0,1].set_ylim([-0.1,0.2])
    ax[0,1].set_ylabel('$dL/D_{max} /dU$')
    fig.tight_layout()
  
   
    
#%%

def assort_max_df(df):
    lb = (df.iloc[0]-df.iloc[1]).to_numpy().reshape((1,-1))
    ub = (df.iloc[2] - df.iloc[0]).to_numpy().reshape((1,-1))
    
    lub = np.append(lb, ub, axis = 0)
    rpms = ['200','250','300']
    df_lub = pd.DataFrame(lub, columns=rpms)
    
    return pd.concat([df,df_lub], ignore_index=True, axis=0)

def assort_si_dfs(df1, df2, df3, df4):
    print(df1,df2,df3,df4)
    columns = ['slope', 's_lb', 's_ub', 'intercept', 'i_lb', 'i_ub']
    df = pd.DataFrame(np.array([df1,df2,df3,df4]), columns=columns)
    df = df.set_index(pd.Series(['base','plate','f9_v0','f9_v3']))
    print(df)
    
    return df

def max_plots(df, coeff, ax, trans = False, OLS = False):
    b_color = '#221F20'
    p_color = '#D5D5D7'
    f9v0_color = '#4771b2'
    f9v3_color = '#c6793a'
    
    rpms = [200,250,300]
    
    f9v3_p, f9v0_p, base_p, plate_p = indvel_stat_analys(df,coeff)
    
    if OLS:
        f9v3_si, f9v0_si, base_si, plate_si = max_inter_stats_analysOLS(df, coeff)
    else:
        f9v3_si, f9v0_si, base_si, plate_si = max_inter_stats_analys(df, coeff)
    print('TESTINGGGGGG')
    f9v3_p = assort_max_df(f9v3_p)
    f9v0_p = assort_max_df(f9v0_p)
    base_p = assort_max_df(base_p)
    plate_p = assort_max_df(plate_p)
    
    
    print('PLOTTING MAXs')
    print(base_p)
    print(plate_p)
    print(f9v0_p)
    print(f9v3_p)
    
    ax[0].errorbar(rpms, base_p.iloc[0].to_numpy(), 
                   yerr= [base_p.iloc[3],base_p.iloc[4]],
                fmt="o", markersize=2 ,elinewidth=1, capsize = 2, capthick=0.5,
                color=b_color)
    ax[0].errorbar(rpms, plate_p.iloc[0],
                   yerr= [plate_p.iloc[3],plate_p.iloc[4]],
                fmt="o", markersize=2, elinewidth=1, capsize = 2, capthick=0.5,
                color=p_color)
    ax[0].errorbar(rpms, f9v0_p.iloc[0],
                   yerr= [f9v0_p.iloc[3],f9v0_p.iloc[4]],
                fmt="o", markersize=2, elinewidth=1, capsize = 2, capthick=0.5,
                color=f9v0_color)
    ax[0].errorbar(rpms, f9v3_p.iloc[0], 
                   yerr= [f9v3_p.iloc[3],f9v3_p.iloc[4]],
                fmt="o", markersize=2, elinewidth=1, capsize = 2, capthick=0.5,
                color=f9v3_color)
    
    vel_df = assort_si_dfs(base_si, plate_si, f9v0_si, f9v3_si)
    
    maxs_plot_lme(ax, vel_df)
    
    ylims = [0.9,1.6]

    ax[0].spines[['right','top']].set_visible(False)
    ax[0].set_ylabel('$C_{L max}$')
    ax[0].set_ylim(ylims)
    ax[0].set_xticks([200,250,300])

def mixed_plots(LD, Cm, CL):

    b_color = '#221F20'
    p_color = '#D5D5D7'
    f9v0_color = '#4771b2'
    f9v3_color = '#c6793a'
    
    
    #PLot the Cl vs Alpha curves and analyse slope and such
    CL = CL.loc[CL['AoA']<12]
    Cm = Cm.loc[Cm['AoA']<12]
    
    LD_300 = pd.pivot_table(LD.loc[LD['rpm'] == 300],
                   index='AoA', columns = 'config', values='LD')
    
    LDvar_300 = pd.pivot_table(LD.loc[LD['rpm'] == 300],
                   index='AoA', columns = 'config', values='LDvar')
    
    CL_300 = pd.pivot_table(CL.loc[CL['rpm'] == 300],
                   index='AoA', columns = 'config', values='CL')

    CLvar_300 = pd.pivot_table(CL.loc[CL['rpm'] == 300],
                   index='AoA', columns = 'config', values='CLvar')
    
    Cm_300 = pd.pivot_table(Cm.loc[Cm['rpm'] == 300],
                   index='AoA', columns = 'config', values='Cm')
    
    Cmvar_300 = pd.pivot_table(Cm.loc[Cm['rpm'] == 300],
                   index='AoA', columns = 'config', values='Cmvar')
    
    fig, ax = plt.subplots(2,1, figsize=(3,4), dpi=300)
    

   
    ax[0].errorbar(LD_300.index, LD_300['base'], yerr=1.96*LDvar_300['base']**0.5,
                  linewidth=1, elinewidth=0.5, capsize = 1, capthick=0.5,
                  color=b_color)
    ax[0].errorbar(LD_300.index, LD_300['plate'], yerr=1.96*LDvar_300['plate']**0.5,
                   linewidth=1, elinewidth=0.5, capsize=1, capthick=0.5,
                   color=p_color)
    ax[0].errorbar(LD_300.index, LD_300['f9_v0'], yerr=1.96*LDvar_300['f9_v0']**0.5,
                   linewidth=1, elinewidth=0.5, capsize=1, capthick=0.5,
                   color=f9v0_color)
    ax[0].errorbar(LD_300.index, LD_300['f9_v3'], yerr=1.96*LDvar_300['f9_v3']**0.5,
                   linewidth=1, elinewidth=0.5, capsize=1, capthick=0.5,
                   color=f9v3_color)
    
    ax[1].errorbar(CL_300['base'], Cm_300['base'], xerr=1.96*CLvar_300['base']**0.5, yerr=1.96*Cmvar_300['base']**0.5,
                   linewidth=1, elinewidth=0.5, capsize = 1, capthick=0.5,
                   color=b_color)
    ax[1].errorbar(CL_300['plate'], Cm_300['plate'], xerr=1.96*CLvar_300['plate']**0.5, yerr=1.96*Cmvar_300['plate']**0.5,
                   linewidth=1, elinewidth=0.5, capsize=1, capthick=0.5,
                   color=p_color)
    ax[1].errorbar(CL_300['f9_v0'], Cm_300['f9_v0'], xerr=1.96*CLvar_300['f9_v0']**0.5, yerr=1.96*Cmvar_300['f9_v0']**0.5,
                   linewidth=1, elinewidth=0.5, capsize=1, capthick=0.5,
                   color=f9v0_color)
    ax[1].errorbar(CL_300['f9_v3'], Cm_300['f9_v3'], xerr=1.96*CLvar_300['f9_v3']**0.5 ,yerr=1.96*Cmvar_300['f9_v3']**0.5,
                   linewidth=1, elinewidth=0.5, capsize=1, capthick=0.5,
                   color=f9v3_color)
    
    ax[1].vlines(x=0,ymin=-0.1,ymax=0, linestyles=':', color='k')
    
    xlims = [-6,20]
    
    
    ax[0].spines[['right','top']].set_visible(False)
    ax[0].set_ylabel('$L/D$')
    ax[0].set_xlim(xlims)
    ax[0].set_xlabel('Angle of Attack (deg)')
    ax[0].set_ylim([-5,7.5])
    ax[0].get_yaxis().set_ticks([-2.5,0,2.5,5,7.5])
    ax[0].get_xaxis().set_ticks([0,10,20])
    ax[1].spines[['right','top']].set_visible(False)
    ax[1].set_ylabel('$C_{m}$')
    ax[1].set_xlabel('$C_{L}$')
    ax[1].set_xlim([-0.4,1.2])
    ax[1].get_xaxis().set_ticks([0,0.4,0.8,1.2])
    ax[1].set_ylim([-0.1, 0])
    ax[1].get_yaxis().set_ticks([ -0.1, -0.05, 0])
    

    fig.tight_layout()
    fig.savefig('LD_CLvCm', dpi=300)
    
def mixed_plotsX(LD, Cm):

    b_color = '#221F20'
    p_color = '#D5D5D7'
    f9v0_color = '#4771b2'
    f9v3_color = '#c6793a'
    
    
    #PLot the Cl vs Alpha curves and analyse slope and such

    
    LD_300 = pd.pivot_table(LD.loc[LD['rpm'] == 300],
                   index='AoA', columns = 'config', values='LD')
    
    LDvar_300 = pd.pivot_table(LD.loc[LD['rpm'] == 300],
                   index='AoA', columns = 'config', values='LDvar')
    
    Cm_300 = pd.pivot_table(Cm.loc[Cm['rpm'] == 300],
                   index='AoA', columns = 'config', values='Cm')
    
    Cmvar_300 = pd.pivot_table(Cm.loc[Cm['rpm'] == 300],
                   index='AoA', columns = 'config', values='Cmvar')
    
    fig, ax = plt.subplots(2,1, figsize=(3,4), dpi=300)
    
    
   
    ax[0].errorbar(LD_300.index, LD_300['base'], yerr=1.96*LDvar_300['base']**0.5,
                  linewidth=1, elinewidth=0.5, capsize = 1, capthick=0.5,
                  color=b_color)
    ax[0].errorbar(LD_300.index, LD_300['plate'], yerr=1.96*LDvar_300['plate']**0.5,
                   linewidth=1, elinewidth=0.5, capsize=1, capthick=0.5,
                   color=p_color)
    ax[0].errorbar(LD_300.index, LD_300['f9_v0'], yerr=1.96*LDvar_300['f9_v0']**0.5,
                   linewidth=1, elinewidth=0.5, capsize=1, capthick=0.5,
                   color=f9v0_color)
    ax[0].errorbar(LD_300.index, LD_300['f9_v3'], yerr=1.96*LDvar_300['f9_v3']**0.5,
                   linewidth=1, elinewidth=0.5, capsize=1, capthick=0.5,
                   color=f9v3_color)
    
    ax[1].errorbar(Cm_300.index, Cm_300['base'], yerr=1.96*Cmvar_300['base']**0.5,
                  linewidth=1, elinewidth=0.5, capsize = 1, capthick=0.5,
                  color=b_color)
    ax[1].errorbar(Cm_300.index, Cm_300['plate'], yerr=1.96*Cmvar_300['plate']**0.5,
                   linewidth=1, elinewidth=0.5, capsize=1, capthick=0.5,
                   color=p_color)
    ax[1].errorbar(Cm_300.index, Cm_300['f9_v0'], yerr=1.96*Cmvar_300['f9_v0']**0.5,
                   linewidth=1, elinewidth=0.5, capsize=1, capthick=0.5,
                   color=f9v0_color)
    ax[1].errorbar(Cm_300.index, Cm_300['f9_v3'], yerr=1.96*Cmvar_300['f9_v3']**0.5,
                   linewidth=1, elinewidth=0.5, capsize=1, capthick=0.5,
                   color=f9v3_color)
    
    xlims = [-6,20]
    
    
    ax[0].spines[['right','top','bottom']].set_visible(False)
    ax[0].set_ylabel('$L/D$')
    ax[0].set_xlim(xlims)
    ax[0].set_ylim([-5,7.5])
    ax[0].get_xaxis().set_ticks([])
    ax[1].spines[['right','top']].set_visible(False)
    ax[1].set_ylabel('$C_{m}$')
    ax[1].set_xlim(xlims)
    ax[1].set_ylim([-0.225, 0])
    ax[1].get_yaxis().set_ticks([-0.2, -0.15, -0.1, -0.05, 0])
    ax[1].set_xlabel('Angle of Attack (deg)')
    
    custom_lines1 = [Line2D([0], [0], color= b_color, lw=1),
                Line2D([0], [0], color= p_color, lw=1),
                Line2D([0], [0], color= f9v0_color, lw=1),
                Line2D([0], [0], color= f9v3_color, lw=1)]
    

    
    fig.tight_layout()
    fig.savefig('LD_Cm', dpi=300)
def feather_plots(CL, LD, Cm):

    b_color = '#221F20'
    p_color = '#D5D5D7'
    f9v0_color = '#4771b2'
    f9v15_color = '#7e4794'
    f9v2_color = '#a5d5d8'
    f9v3_color = '#c6793a'
    
    
    #PLot the Cl vs Alpha curves and analyse slope and such

    CL_300 = pd.pivot_table(CL.loc[CL['rpm'] == 300],
                   index='AoA', columns = 'config', values='CL')
    
    CLvar_300 = pd.pivot_table(CL.loc[CL['rpm'] == 300],
                   index='AoA', columns = 'config', values='CLvar')
    
    LD_300 = pd.pivot_table(LD.loc[LD['rpm'] == 300],
                   index='AoA', columns = 'config', values='LD')
    
    LDvar_300 = pd.pivot_table(LD.loc[LD['rpm'] == 300],
                   index='AoA', columns = 'config', values='LDvar')
    
    Cm_300 = pd.pivot_table(Cm.loc[Cm['rpm'] == 300],
                   index='AoA', columns = 'config', values='Cm')
    
    Cmvar_300 = pd.pivot_table(Cm.loc[Cm['rpm'] == 300],
                   index='AoA', columns = 'config', values='Cmvar')
    
    fig, ax = plt.subplots(2,1, figsize=(2,3), dpi=300)
    
    v3_v0_diff = CL_300['f9_v3'] - CL_300['f9_v0']
    v3_v0_diffvar = CLvar_300['f9_v3'] + CLvar_300['f9_v0']
    
    v15_v0_diff = CL_300['f9_v15'] - CL_300['f9_v0']
    v15_v0_diffvar = CLvar_300['f9_v15'] + CLvar_300['f9_v0']
    
    v2_v0_diff = CL_300['f9_v2'] - CL_300['f9_v0']
    v2_v0_diffvar = CLvar_300['f9_v2'] + CLvar_300['f9_v0']
    
    v3_v0_diff_norm = v3_v0_diff/v3_v0_diff
    v3_v0_diffvar_norm = v3_v0_diffvar/v3_v0_diff
    
    v15_v0_diff_norm = v15_v0_diff/v3_v0_diff
    v15_v0_diffvar_norm = v15_v0_diffvar/v3_v0_diff
    
    v2_v0_diff_norm = v2_v0_diff/v3_v0_diff
    v2_v0_diffvar_norm = v2_v0_diffvar/v3_v0_diff
    
    v0_var_norm = CLvar_300['f9_v0']/v3_v0_diff
    v0_norm = CL_300['f9_v0']-CL_300['f9_v0']
    
   
    ax[0].errorbar(CL_300.index, CL_300['f9_v0'], yerr=1.96*CLvar_300['f9_v0']**0.5,
                   linewidth=1, elinewidth=0.5, capsize=1, capthick=0.5,
                   color=f9v0_color)
    ax[0].errorbar(CL_300.index, CL_300['f9_v15'], yerr=1.96*CLvar_300['f9_v15']**0.5,
                   linewidth=1, elinewidth=0.5, capsize=1, capthick=0.5,
                   color=f9v15_color)
    ax[0].errorbar(CL_300.index, CL_300['f9_v2'], yerr=1.96*CLvar_300['f9_v2']**0.5,
                   linewidth=1, elinewidth=0.5, capsize=1, capthick=0.5,
                   color=f9v2_color)
    ax[0].errorbar(CL_300.index, CL_300['f9_v3'], yerr=1.96*CLvar_300['f9_v3']**0.5,
                   linewidth=1, elinewidth=0.5, capsize=1, capthick=0.5,
                   color=f9v3_color)

    
    ax[1].plot(CL_300.index, v0_norm, linewidth=1, color=f9v0_color)
    ax[1].plot(CL_300.index, v15_v0_diff_norm, linewidth=1, color=f9v15_color)
    ax[1].plot(CL_300.index, v2_v0_diff_norm, linewidth=1, color=f9v2_color)
    ax[1].plot(CL_300.index, v3_v0_diff_norm, linewidth=1, color=f9v3_color)
    ax[1].fill_between(CL_300.index, v0_norm-1.96*v0_var_norm**0.5,
                       v0_norm+1.96*v0_var_norm**0.5, alpha=0.3, color=f9v0_color)
    ax[1].fill_between(CL_300.index, v15_v0_diff_norm-1.96*v15_v0_diffvar_norm**0.5,
                       v15_v0_diff_norm+1.96*v15_v0_diffvar_norm**0.5, alpha=0.3, color=f9v15_color)
    ax[1].fill_between(CL_300.index, v2_v0_diff_norm-1.96*v2_v0_diffvar_norm**0.5,
                       v2_v0_diff_norm+1.96*v2_v0_diffvar_norm**0.5, alpha=0.3, color=f9v2_color)
    ax[1].fill_between(CL_300.index, v3_v0_diff_norm-1.96*v3_v0_diffvar_norm**0.5,
                       v3_v0_diff_norm+1.96*v3_v0_diffvar_norm**0.5, alpha=0.3, color=f9v3_color)
    

    xlims = [-6,15]
    ylims1 = [-0.25,1.25]
    
    ax[0].spines[['right','top','bottom']].set_visible(False)
    ax[0].set_ylabel('$C_{L}$')
    ax[0].set_xlim(xlims)
    ax[0].get_xaxis().set_ticks([])
    ax[1].spines[['right','top']].set_visible(False)
    ax[1].set_ylabel('$C_{L} diff$')
    ax[1].set_xlabel('Angle of Attack (deg)')
    ax[1].set_xlim(xlims)
    ax[1].set_ylim(ylims1)
    ax[1].get_xaxis().set_ticks([-5,0,5,10,15])
    
  
    
    custom_lines1 = [Line2D([0], [0], color= b_color, lw=1),
                     Line2D([0], [0], color= p_color, lw=1),
                     Line2D([0], [0], color= f9v0_color, lw=1),
                     Line2D([0], [0], color= f9v15_color, lw=1),
                     Line2D([0], [0], color= f9v2_color, lw=1),
                     Line2D([0], [0], color= f9v3_color, lw=1),
                     Line2D([0], [0], color='k', marker='^', markersize=5, lw=0),
                     Line2D([0], [0], color='k',marker='v', markersize=5,lw=0),]
    
 
    legend1 = fig.legend(custom_lines1,['Baseline','Plate','f9_v0','f9_v15','f9_v2','f9_v3', 'Up-sweep',
                                        'Down-sweep'],
                         ncol= 2, loc=8, bbox_to_anchor=(0.55,-0.5), frameon=False)
    
    fig.tight_layout()
    fig.savefig('feather_vary', dpi=300)
#%%
def get_slopes(b_df, p_df, f9v0_df, f9v3_df, ax):
    rpms = np.array([200,250,300])
    vels = rpms*0.055+0.525
    
    b_df = b_df[b_df['AoA'] < 10]
    p_df = p_df[p_df['AoA'] < 10]
    f9v0_df = f9v0_df[f9v0_df['AoA'] < 10]
    f9v3_df = f9v3_df[f9v3_df['AoA'] < 10]
    
    
    b_slopes = np.zeros((3,3))
    p_slopes = np.zeros((3,3))
    f9v0_slopes = np.zeros((3,3))
    f9v3_slopes = np.zeros((3,3))
    
    
    for ind, r in enumerate(rpms):
        Base_df_r = b_df[b_df['rpm'] == r]
        Plate_df_r = p_df[p_df['rpm'] == r]
        F9v0_df_r = f9v0_df[f9v0_df['rpm'] == r]
        F9v3_df_r = f9v3_df[f9v3_df['rpm'] == r]
        
        b_slopes[:,ind] = single_wing_rpm_OLS(Base_df_r)
        p_slopes[:,ind] = single_wing_rpm_OLS(Plate_df_r)
        f9v0_slopes[:,ind] = single_wing_rpm_OLS(F9v0_df_r)
        f9v3_slopes[:,ind] = single_wing_rpm_OLS(F9v3_df_r)

    b_color = '#221F20'
    p_color = '#D5D5D7'
    f9v0_color = '#4771b2'
    f9v3_color = '#c6793a'
    
    ax.errorbar(rpms,b_slopes[0,:],yerr=b_slopes[-2:,:], 
                fmt="^", markersize=2, elinewidth=0.5, capsize = 1, capthick=0.5, 
                color=b_color)
    ax.errorbar(rpms,p_slopes[0,:],yerr=p_slopes[-2:,:],
                fmt="^", markersize=2, elinewidth=0.5, capsize = 1, capthick=0.5,
                color=p_color)
    ax.errorbar(rpms,f9v0_slopes[0,:],yerr=f9v0_slopes[-2:,:],
                fmt="^", markersize=2, elinewidth=0.5, capsize = 1, capthick=0.5,
                color=f9v0_color)
    ax.errorbar(rpms,f9v3_slopes[0,:],yerr=f9v3_slopes[-2:,:], 
                fmt="^", markersize=2, elinewidth=0.5, capsize = 1, capthick=0.5,
                color=f9v3_color)
    
    
    ax.spines[['right','top']].set_visible(False)
    ax.set_ylabel('$C_{L \\alpha} Slope$')
    vels = ['11.5', '14.25', '17']
    ax.set_xticks(rpms, vels)
    ax.set_xlabel('Flow Velocity (m/s)')
    


def get_slopes_X(b_df, p_df, f9v0_df, f9v3_df, ax, coeff='CL'):
    b_color = '#221F20'
    p_color = '#D5D5D7'
    f9v0_color = '#4771b2'
    f9v3_color = '#c6793a'
    
    rpms = np.array([200,250,300])
    vels = rpms*0.055+0.525
    
    b_df = b_df[b_df['AoA'] < 10]
    p_df = p_df[p_df['AoA'] < 10]
    f9v0_df = f9v0_df[f9v0_df['AoA'] < 10]
    f9v3_df = f9v3_df[f9v3_df['AoA'] < 10]
    all_df = pd.concat([b_df, p_df, f9v0_df, f9v3_df], ignore_index=True)
    all_df['vel'] = all_df['rpm']*0.055+0.525
    
    
    if coeff == 'Cm':
        all_df = all_df[all_df['AoA'] >= 0]
        all_df.to_csv('CmAoA_data_df.csv')
    else:
        all_df.to_csv('CLAoA_data_df.csv')
    
    f9v3_si, f9v0_si, base_si, plate_si = slope_inter_stats_analys(all_df, coeff)
    f9v3_s, f9v0_s, base_s, plate_s = indvel_slope_stat_analysX(all_df, coeff)
    
    
    cols = ['rpm','config','slope','s_ulim', 's_llim', 'color']
    slopes_df = pd.DataFrame(columns=cols)
   
    for r in [200,250,300]:        
       temp = {cols[0]: [r,r,r,r],
               cols[1]: ['base','plate','f9_v0','f9_v3'],
               cols[2]: [base_s[str(r)][0] ,plate_s[str(r)][0] ,f9v0_s[str(r)][0], f9v3_s[str(r)][0]],
               cols[3]: [base_s[str(r)][2]-base_s[str(r)][0] ,plate_s[str(r)][2]-plate_s[str(r)][0] ,
                         f9v0_s[str(r)][2]-f9v0_s[str(r)][0],f9v3_s[str(r)][2]-f9v3_s[str(r)][0]],
               cols[4]: [base_s[str(r)][0]-base_s[str(r)][1], plate_s[str(r)][0]-plate_s[str(r)][1],
                         f9v0_s[str(r)][0]-f9v0_s[str(r)][1], f9v3_s[str(r)][0]-f9v3_s[str(r)][1]],
               cols[5]: [b_color, p_color, f9v0_color, f9v3_color]}
       
       temp_df = pd.DataFrame(temp)
       slopes_df = pd.concat([slopes_df, temp_df], ignore_index=True)
            
    
    print(slopes_df)
    
    slopes_df['rpm'] = pd.to_numeric(slopes_df['rpm'])
    slopes_df['slope'] = pd.to_numeric(slopes_df['slope'])
    slopes_df['s_ulim'] = pd.to_numeric(slopes_df['s_ulim'])
    slopes_df['s_llim'] = pd.to_numeric(slopes_df['s_llim'])
    base = slopes_df.loc[slopes_df['config'] == 'base']
    plate = slopes_df.loc[slopes_df['config'] == 'plate']
    f9_v0 = slopes_df.loc[slopes_df['config'] == 'f9_v0']
    f9_v3 = slopes_df.loc[slopes_df['config'] == 'f9_v3']
    

    ax[0].errorbar(base['rpm'], base['slope'], yerr=[base['s_llim'],
                                                               base['s_ulim']],
            fmt='o', markersize=2, elinewidth=1, capsize = 2,capthick=0.5, color=b_color)
    ax[0].errorbar(plate['rpm'], plate['slope'], yerr=[plate['s_llim'],
                                                               plate['s_ulim']],
            fmt='o',markersize=2, elinewidth=1, capsize = 2,capthick=0.5, color=p_color)
    ax[0].errorbar(f9_v0['rpm'], f9_v0['slope'], yerr=[f9_v0['s_llim'],
                                                               f9_v0['s_ulim']],
            fmt='o',markersize=2, elinewidth=1, capsize = 2,capthick=0.5, color=f9v0_color)
    ax[0].errorbar(f9_v3['rpm'], f9_v3['slope'], yerr=[f9_v3['s_llim'],
                                                               f9_v3['s_ulim']],
            fmt='o', markersize=2, elinewidth=1, capsize = 2,capthick=0.5, color=f9v3_color)

    ax[0].spines[['right','top']].set_visible(False)
    ax[0].set_ylabel('$C_{L \\alpha}$')
    vels = ['11.5', '14.3', '17.0']
    ax[0].set_xticks(rpms, vels)
    ax[0].set_xlabel('Flow Velocity (m/s)')
 
    b_slopes = slopes_df.loc[slopes_df['config']=='base']
    p_slopes = slopes_df.loc[slopes_df['config']=='plate']
    f9v0_slopes = slopes_df.loc[slopes_df['config']=='f9_v0']
    f9v3_slopes = slopes_df.loc[slopes_df['config']=='f9_v3']
    
    slope_vel = pd.DataFrame()
    
    print("Base slope stat analysis")
    slope_vel['base'] = base_si
    print("Plate slope stat analysis")
    slope_vel['plate'] = plate_si
    print("f9_v0 slope stat analysis")
    slope_vel['f9_v0'] = f9v0_si
    print("f9_v3 slope stat analysis")
    slope_vel['f9_v3'] =  f9v3_si
    
    
    slope_vel = slope_vel.transpose()
    slope_vel = slope_vel.rename(columns = {0:'slope',1:'s_lb',2:'s_ub', 3:'intercept',
                                      4:'i_lb', 5:'i_ub'})
    print(slope_vel)
    slopes_plot_lme(ax, slope_vel)

def maxs_plot_lme(ax, lin):
    
    b_color = '#221F20'
    p_color = '#D5D5D7'
    f9v0_color = '#4771b2'
    f9v3_color = '#c6793a'
   
    
    rpms = np.array([200,250,300])
    vels = rpms*0.055+0.525 
    
    b_lin = pd.DataFrame()
    
    b_lin['m'] = lin.loc['base','slope']*vels + lin.loc['base','intercept']
    b_lin['lb'] = lin.loc['base','s_ub']*vels + lin.loc['base','i_lb']
    b_lin['ub'] = lin.loc['base','s_lb']*vels + lin.loc['base','i_ub']
    
    ax[0].plot(rpms, b_lin['m'],linewidth=1 ,color=b_color)
    ax[0].fill_between(rpms, b_lin['lb'], b_lin['ub'], alpha=0.2, color=b_color)
    
    p_lin = pd.DataFrame()
    
    p_lin['m'] = lin.loc['plate','slope']*vels + lin.loc['plate','intercept']
    p_lin['lb'] = lin.loc['plate','s_ub']*vels + lin.loc['plate','i_lb']
    p_lin['ub'] = lin.loc['plate','s_lb']*vels + lin.loc['plate','i_ub']
    
    ax[0].plot(rpms, p_lin['m'],linewidth=1 ,color=p_color)
    ax[0].fill_between(rpms, p_lin['lb'], p_lin['ub'], alpha=0.2, color=p_color)
    
    f9v0_lin = pd.DataFrame()
    
    f9v0_lin['m'] = lin.loc['f9_v0','slope']*vels + lin.loc['f9_v0','intercept']
    f9v0_lin['lb'] = lin.loc['f9_v0','s_ub']*vels + lin.loc['f9_v0','i_lb']
    f9v0_lin['ub'] = lin.loc['f9_v0','s_lb']*vels + lin.loc['f9_v0','i_ub']
    
    ax[0].plot(rpms, f9v0_lin['m'],linewidth=1 ,color=f9v0_color)
    ax[0].fill_between(rpms, f9v0_lin['lb'], f9v0_lin['ub'], alpha=0.2, color=f9v0_color)
    
    f9v3_lin = pd.DataFrame()
    
    f9v3_lin['m'] = lin.loc['f9_v3','slope']*vels + lin.loc['f9_v3','intercept']
    f9v3_lin['lb'] = lin.loc['f9_v3','s_ub']*vels + lin.loc['f9_v3','i_lb']
    f9v3_lin['ub'] = lin.loc['f9_v3','s_lb']*vels + lin.loc['f9_v3','i_ub']
    
    ax[0].plot(rpms, f9v3_lin['m'],linewidth=1 ,color=f9v3_color)
    ax[0].fill_between(rpms, f9v3_lin['lb'], f9v3_lin['ub'], alpha=0.2, color=f9v3_color)
    
    ylims = [0.9,1.6]
    
    ax[0].spines[['right','top']].set_visible(False)
    ax[0].set_ylabel('$C_{L max}$')
    ax[0].set_ylim(ylims)
    #ax[0].set_xticks([200,250,300])
    rpm = [200, 250, 300]
    vels = ['11.5', '14.3', '17']
    ax[0].set_xticks(rpms, vels)
    ax[0].set_xlabel('Flow Velocity (m/s)')
    
   
    width = 0.15
    inds = [0, width, width*2, width*3]
    
    m_colors = [b_color, p_color, f9v0_color, f9v3_color]
    
    lin['s_uerr'] = lin['s_ub']-lin['slope']
    lin['s_lerr'] = lin['slope']-lin['s_lb']
    lin['color'] = m_colors
    print(lin)
    
    ax[1].errorbar(inds[0], lin.loc['base','slope'], yerr=lin.loc['base','s_lerr'],
                   fmt='o',markersize=2, elinewidth=1, capsize = 2,capthick=0.5,
                   color=lin.loc['base','color'] )
    ax[1].errorbar(inds[1], lin.loc['plate','slope'], yerr=lin.loc['plate','s_lerr'],
                   fmt='o',markersize=2, elinewidth=1, capsize = 2,capthick=0.5,
                   color=lin.loc['plate','color'] )
    ax[1].errorbar(inds[2], lin.loc['f9_v0','slope'], yerr=lin.loc['f9_v0','s_lerr'],
                   fmt='o',markersize=2, elinewidth=1, capsize = 2,capthick=0.5,
                   color=lin.loc['f9_v0','color'] )
    ax[1].errorbar(inds[3], lin.loc['f9_v3','slope'], yerr=lin.loc['f9_v3','s_lerr'],
                   fmt='o',markersize=2, elinewidth=1, capsize = 2,capthick=0.5,
                   color=lin.loc['f9_v3','color'] )
    
    ax[1].spines[['right','top','bottom']].set_visible(False)
    ax[1].set_ylabel('$dC_{L max}/dU$')
    ax[1].set_ylim([-0.02,0.02])
    ax[1].set_xlim([-width,width*4])

def slopes_plot_lme(ax, lin):
    
    b_color = '#221F20'
    p_color = '#D5D5D7'
    f9v0_color = '#4771b2'
    f9v3_color = '#c6793a'
   
    rpms = np.array([200,250,300])
    vels = rpms*0.055+0.525
    
    b_lin = pd.DataFrame()
    
    b_lin['m'] = lin.loc['base','slope']*vels + lin.loc['base','intercept']
    b_lin['lb'] = lin.loc['base','s_ub']*vels + lin.loc['base','i_lb']
    b_lin['ub'] = lin.loc['base','s_lb']*vels + lin.loc['base','i_ub']
    
    ax[0].plot(rpms, b_lin['m'],linewidth=1 ,color=b_color)
    ax[0].fill_between(rpms, b_lin['lb'], b_lin['ub'], alpha=0.2, color=b_color)
    
    p_lin = pd.DataFrame()
    
    p_lin['m'] = lin.loc['plate','slope']*vels + lin.loc['plate','intercept']
    p_lin['lb'] = lin.loc['plate','s_ub']*vels + lin.loc['plate','i_lb']
    p_lin['ub'] = lin.loc['plate','s_lb']*vels + lin.loc['plate','i_ub']
    
    ax[0].plot(rpms, p_lin['m'],linewidth=1 ,color=p_color)
    ax[0].fill_between(rpms, p_lin['lb'], p_lin['ub'], alpha=0.2, color=p_color)
    
    f9v0_lin = pd.DataFrame()
    
    f9v0_lin['m'] = lin.loc['f9_v0','slope']*vels + lin.loc['f9_v0','intercept']
    f9v0_lin['lb'] = lin.loc['f9_v0','s_ub']*vels + lin.loc['f9_v0','i_lb']
    f9v0_lin['ub'] = lin.loc['f9_v0','s_lb']*vels + lin.loc['f9_v0','i_ub']
    
    ax[0].plot(rpms, f9v0_lin['m'],linewidth=1 ,color=f9v0_color)
    ax[0].fill_between(rpms, f9v0_lin['lb'], f9v0_lin['ub'], alpha=0.2, color=f9v0_color)
    
    f9v3_lin = pd.DataFrame()
    
    f9v3_lin['m'] = lin.loc['f9_v3','slope']*vels + lin.loc['f9_v3','intercept']
    f9v3_lin['lb'] = lin.loc['f9_v3','s_ub']*vels + lin.loc['f9_v3','i_lb']
    f9v3_lin['ub'] = lin.loc['f9_v3','s_lb']*vels + lin.loc['f9_v3','i_ub']
    
    ax[0].plot(rpms, f9v3_lin['m'],linewidth=1 ,color=f9v3_color)
    ax[0].fill_between(rpms, f9v3_lin['lb'], f9v3_lin['ub'], alpha=0.2, color=f9v3_color)
    
    width = 0.15
    inds = [0, width, width*2, width*3]
    
    m_colors = [b_color, p_color, f9v0_color, f9v3_color]
    
    lin['s_uerr'] = lin['s_ub']-lin['slope']
    lin['s_lerr'] = lin['slope']-lin['s_lb']
    lin['color'] = m_colors
    
    ax[1].errorbar(inds[0], lin.loc['base','slope'], yerr=lin.loc['base','s_lerr'],
                   fmt='o',markersize=2, elinewidth=1, capsize = 2, capthick=0.5,
                   color=lin.loc['base','color'] )
    ax[1].errorbar(inds[1], lin.loc['plate','slope'], yerr=lin.loc['plate','s_lerr'],
                   fmt='o',markersize=2, elinewidth=1, capsize = 2,capthick=0.5,
                   color=lin.loc['plate','color'] )
    ax[1].errorbar(inds[2], lin.loc['f9_v0','slope'], yerr=lin.loc['f9_v0','s_lerr'],
                   fmt='o',markersize=2, elinewidth=1, capsize = 2,capthick=0.5,
                   color=lin.loc['f9_v0','color'] )
    ax[1].errorbar(inds[3], lin.loc['f9_v3','slope'], yerr=lin.loc['f9_v3','s_lerr'],
                   fmt='o',markersize=2, elinewidth=1, capsize = 2,capthick=0.5,
                   color=lin.loc['f9_v3','color'] )
    
    ax[1].spines[['right','top','bottom']].set_visible(False)
    ax[1].set_ylabel('$dC_{L \\alpha} /dU$')
    ax[1].set_ylim([-0.0025,0.0005])
    ax[1].set_xlim([-width,width*4])
    

def single_wing_rpm_OLS(df_):
    df = df_.copy()
    X = sm.add_constant(df['AoA'])
    md = sm.OLS(df['CL'], X)#,vc_formula = {"AoA" : "0 + AoA"}) #, re_formula="~taps")
    mdf = md.fit()
  
    conf = mdf.conf_int().loc['AoA'].tolist()
    #print(mdf.params['AoA'], conf)
    return mdf.params["AoA"],mdf.params["AoA"] - conf[0], conf[1] - mdf.params["AoA"]


#%%
def indvel_stat_analys(df, coeff):
    f9v3 = pd.DataFrame()
    f9v0 = pd.DataFrame()
    base = pd.DataFrame()
    plate = pd.DataFrame()
    for r in [200,250,300]:
        df_vel = df.loc[df['rpm'] == r]
        oo = np.ones(df_vel.shape[0])
        #print(df_vel)
        vc_form = {'g1': "0 + C(iter)", 'g2': "0 + C(direction)"}
        df_vel_b = df_vel.loc[df_vel['config']=='base']
        md_b = smf.mixedlm(coeff+' ~ 1' , 
                         df_vel_b, groups=df_vel_b['iter_d']) #, vc_formula=vc_form, 
                         #re_formula='~ C(config, Treatment(reference="f9_v3"))')#df_vel['iter','direction'])
        mdf_b = md_b.fit()
        print('RPM: ', r, ' summary', mdf_b.summary())
        temp_b_sl = mdf_b.params['Intercept']
        temp_b_ci = mdf_b.conf_int().loc['Intercept'].tolist()
        base[str(r)] = [temp_b_sl, temp_b_ci[0], temp_b_ci[1]]
        
        fig = plt.figure(figsize = (2,2), dpi=300)
        ax = sns.distplot(mdf_b.resid, hist = False, kde_kws = {"shade" : True, "lw": 1}, fit = stats.norm)
        ax.set_title(coeff+'max base analysis'+ str(np.round(r*0.055+0.525,1))+' m/s  RMSE: ' +
                     str(np.sqrt(np.mean(mdf_b.resid*mdf_b.resid)).round(5)) +
                     ' Shapiro: ' + str(stats.shapiro(mdf_b.resid)[0]))
        
        df_vel_p = df_vel.loc[df_vel['config']=='plate']
        md_p = smf.mixedlm(coeff+' ~ 1' , 
                         df_vel_p, groups=df_vel_p['iter_d']) #, vc_formula=vc_form, 
                         #re_formula='~ C(config, Treatment(reference="f9_v3"))')#df_vel['iter','direction'])
        mdf_p = md_p.fit()
        print('RPM: ', r, ' summary', mdf_p.summary())
        temp_p_sl = mdf_p.params['Intercept']
        temp_p_ci = mdf_p.conf_int().loc['Intercept'].tolist()
        plate[str(r)] = [temp_p_sl, temp_p_ci[0], temp_p_ci[1]]
        
        fig = plt.figure(figsize = (2,2), dpi=300)
        ax = sns.distplot(mdf_p.resid, hist = False, kde_kws = {"shade" : True, "lw": 1}, fit = stats.norm)
        ax.set_title(coeff+'max plate analysis'+ str(np.round(r*0.055+0.525,1))+' m/s  RMSE: ' +
                     str(np.sqrt(np.mean(mdf_p.resid*mdf_p.resid)).round(5)) +
                     ' Shapiro: ' + str(stats.shapiro(mdf_p.resid)[0]))
        
        df_vel_v0 = df_vel.loc[df_vel['config']=='f9_v0']
        md_v0 = smf.mixedlm(coeff+' ~ 1' , 
                         df_vel_v0, groups=df_vel_v0['iter_d']) #, vc_formula=vc_form, 
                         #re_formula='~ C(config, Treatment(reference="f9_v3"))')#df_vel['iter','direction'])
        mdf_v0 = md_v0.fit()
        print('RPM: ', r, ' summary', mdf_v0.summary())
        temp_v0_sl = mdf_v0.params['Intercept']
        temp_v0_ci = mdf_v0.conf_int().loc['Intercept'].tolist()
        f9v0[str(r)] = [temp_v0_sl, temp_v0_ci[0], temp_v0_ci[1]]
        
        fig = plt.figure(figsize = (2,2), dpi=300)
        ax = sns.distplot(mdf_v0.resid, hist = False, kde_kws = {"shade" : True, "lw": 1}, fit = stats.norm)
        ax.set_title(coeff+'max v0 analysis '+ str(np.round(r*0.055+0.525,1))+' m/s  RMSE: ' + 
                     str(np.sqrt(np.mean(mdf_v0.resid*mdf_v0.resid)).round(5)) +
                     ' Shapiro: ' + str(stats.shapiro(mdf_v0.resid)[0]))
        
        df_vel_v3 = df_vel.loc[df_vel['config']=='f9_v3']
        md_v3 = smf.mixedlm(coeff+' ~ 1' , 
                         df_vel_v3, groups=df_vel_v3['iter_d']) #, vc_formula=vc_form, 
                         #re_formula='~ C(config, Treatment(reference="f9_v3"))')#df_vel['iter','direction'])
        mdf_v3 = md_v3.fit()
        print('RPM: ', r, ' summary', mdf_v3.summary())
        temp_v3_sl = mdf_v3.params['Intercept']
        temp_v3_ci = mdf_v3.conf_int().loc['Intercept'].tolist()
        f9v3[str(r)] = [temp_v3_sl, temp_v3_ci[0], temp_v3_ci[1]]
        
        fig = plt.figure(figsize = (2,2), dpi=300)
        ax = sns.distplot(mdf_v3.resid, hist = False, kde_kws = {"shade" : True, "lw": 1}, fit = stats.norm)
        ax.set_title(coeff+'max v3 analysis'+ str(np.round(r*0.055+0.525,1))+' m/s  RMSE: ' +
                     str(np.sqrt(np.mean(mdf_v3.resid*mdf_v3.resid)).round(5)) +
                     ' Shapiro: ' + str(stats.shapiro(mdf_v3.resid)[0]))
    
    
    return f9v3, f9v0, base, plate 


def indvel_slope_stat_analysX(df, coeff):
     f9v3 = pd.DataFrame()
     f9v0 = pd.DataFrame()
     base = pd.DataFrame()
     plate = pd.DataFrame()
     for r in [200,250,300]:
         df_vel = df.loc[df['rpm'] == r]
         #print(df_vel)
         md = smf.mixedlm(coeff+' ~ AoA + C(config, Treatment(reference="f9_v3"))+\
                          AoA:C(config, Treatment(reference="f9_v3"))', df_vel, groups=df_vel['iter_d'])
         mdf = md.fit()
         
         comp = mdf.params['AoA']
         comp_v3_ci = mdf.conf_int().loc['AoA'].tolist()
         f9v3[str(r)] = [comp, comp_v3_ci[0], comp_v3_ci[1]]
         temp_v0_sl = comp+mdf.params['AoA:C(config, Treatment(reference="f9_v3"))'+
                                   '[T.f9_v0]']
         temp_v0_ci = mdf.conf_int().loc['AoA:C(config, Treatment(reference="f9_v3"))'+
                                   '[T.f9_v0]'].tolist()
         f9v0[str(r)] = [temp_v0_sl, temp_v0_ci[0] + comp_v3_ci[0], temp_v0_ci[1] + comp_v3_ci[1]]
         
         temp_b_sl = comp+mdf.params['AoA:C(config, Treatment(reference="f9_v3"))'+
                                   '[T.base]']
         temp_b_ci = mdf.conf_int().loc['AoA:C(config, Treatment(reference="f9_v3"))'+
                                   '[T.base]'].tolist()
         base[str(r)] = [temp_b_sl, temp_b_ci[0] + comp_v3_ci[0], temp_b_ci[1] + comp_v3_ci[1]]
         
         temp_p_sl = comp+mdf.params['AoA:C(config, Treatment(reference="f9_v3"))'+
                                   '[T.plate]']
         temp_p_ci = mdf.conf_int().loc['AoA:C(config, Treatment(reference="f9_v3"))'+
                                   '[T.plate]'].tolist()
         plate[str(r)] = [temp_p_sl, temp_p_ci[0] + comp_v3_ci[0], temp_p_ci[1] + comp_v3_ci[1]]
         
         print('RPM: ', r, ' summary', mdf.summary())
     
         _ = plt.figure(figsize = (2,2), dpi=300)
         ax = sns.distplot(mdf.resid, hist = False, kde_kws = {"shade" : True, "lw": 1}, fit = stats.norm)
         ax.set_title(coeff+' slope analysis '+ str(r*0.055+0.525) + 'm/s, RMSE: '+
                      str(np.sqrt(np.mean(mdf.resid*mdf.resid)).round(5)) +
                      ' Shapiro: ' + str(stats.shapiro(mdf.resid)[0]))
        

     
     return f9v3, f9v0, base, plate       

def drop_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    
    IQR = Q3-Q1
    
    lb = Q1 - 1.5*IQR
    ub = Q3 + 1.5*IQR
    
    filtered_df = df.loc[(df[column] >= lb) & (df[column] <= ub)]
    return filtered_df

def max_inter_stats_analysOLS(df, coeff):
    
    vc_form = {'g1': "0 + C(iter)", 'g2': "0 + C(direction)"}

    df_b = df.loc[df['config']=='base']
    X = sm.add_constant(df_b['vel'])
    md_b = sm.OLS(df_b[coeff], X)


    mdf_b = md_b.fit()
    print('max_slope_base_summary', mdf_b.summary())
    temp_b_s = mdf_b.params['vel']
    temp_b_s_ci = mdf_b.conf_int().loc['vel'].tolist()
    temp_b_i = mdf_b.params['const']
    temp_b_i_ci = mdf_b.conf_int().loc['const'].tolist()
    base = [temp_b_s, temp_b_s_ci[0], temp_b_s_ci[1],
                    temp_b_i, temp_b_i_ci[0], temp_b_i_ci[1]]
    
    fig = plt.figure(figsize = (2,2), dpi=300)
    ax = sns.distplot(mdf_b.resid, hist = False, kde_kws = {"shade" : True, "lw": 1}, fit = stats.norm)
    ax.set_title('vel-slope max base anal: '+ coeff)
    
    df_p = df.loc[df['config']=='plate']

    X = sm.add_constant(df_p['vel'])
    md_p = sm.OLS(df_p[coeff], X)
    mdf_p = md_p.fit()
    
    print('max slope plate summary', mdf_p.summary())
    temp_p_s = mdf_p.params['vel']
    temp_p_s_ci = mdf_p.conf_int().loc['vel'].tolist()
    temp_p_i = mdf_p.params['const']
    temp_p_i_ci = mdf_p.conf_int().loc['const'].tolist()
    plate = [temp_p_s, temp_p_s_ci[0], temp_p_s_ci[1],
                    temp_p_i, temp_p_i_ci[0], temp_p_i_ci[1]]
    
    fig = plt.figure(figsize = (2,2), dpi=300)
    ax = sns.distplot(mdf_p.resid, hist = False, kde_kws = {"shade" : True, "lw": 1}, fit = stats.norm)
    ax.set_title('vel-slope max plate anal: '+ coeff)
    
    df_v0 = df.loc[df['config']=='f9_v0']

    X = sm.add_constant(df_v0['vel'])
    md_v0 = sm.OLS(df_v0[coeff], X)
    mdf_v0 = md_v0.fit()
    print('max slope f9v0 summary', mdf_v0.summary())
    temp_v0_s = mdf_v0.params['vel']
    temp_v0_s_ci = mdf_v0.conf_int().loc['vel'].tolist()
    temp_v0_i = mdf_v0.params['const']
    temp_v0_i_ci = mdf_v0.conf_int().loc['const'].tolist()
    f9v0 = [temp_v0_s, temp_v0_s_ci[0], temp_v0_s_ci[1],
                    temp_v0_i, temp_v0_i_ci[0], temp_v0_i_ci[1]]
    
    fig = plt.figure(figsize = (2,2), dpi=300)
    ax = sns.distplot(mdf_v0.resid, hist = False, kde_kws = {"shade" : True, "lw": 1}, fit = stats.norm)
    ax.set_title('vel-slope max f9v0 anal: '+ coeff)
    
    df_v3 = df.loc[df['config']=='f9_v3']
    X = sm.add_constant(df_v3['vel'])
    md_v3 = sm.OLS(df_v3[coeff], X)
    mdf_v3 = md_v3.fit()
    print('max slope f9v3 summary', mdf_v3.summary())
    temp_v3_s = mdf_v3.params['vel']
    temp_v3_s_ci = mdf_v3.conf_int().loc['vel'].tolist()
    temp_v3_i = mdf_v3.params['const']
    temp_v3_i_ci = mdf_v3.conf_int().loc['const'].tolist()
    f9v3 = [temp_v3_s, temp_v3_s_ci[0], temp_v3_s_ci[1],
                    temp_v3_i, temp_v3_i_ci[0], temp_v3_i_ci[1]]
    
    _ = plt.figure(figsize = (2,2), dpi=300)
    ax = sns.distplot(mdf_v3.resid, hist = False, kde_kws = {"shade" : True, "lw": 1}, fit = stats.norm)
    ax.set_title('vel-slope max f9v3 anal: '+ coeff)
    
    
    return f9v3, f9v0, base, plate



def max_inter_stats_analys(df, coeff, trans=False):
    f9v3 = pd.DataFrame()
    f9v0 = pd.DataFrame()
    base = pd.DataFrame()
    plate = pd.DataFrame()
    vc_form = {'g1': "0 + C(iter)", 'g2': "0 + C(direction)"} 
     
    md = smf.mixedlm(coeff+' ~ C(config, Treatment(reference="f9_v3")) +\
                     vel + vel:C(config, Treatment(reference="f9_v3"))',
                     df, groups=df['iter_d'], vc_formula=vc_form,
                     re_formula='~ C(config, Treatment(reference="f9_v3"))')
    
    mdf = md.fit()
    
    print('Slope summ: ', mdf.summary())

    resid = mdf.resid
    RSME = np.sqrt(np.mean(resid*resid))
    fig = plt.figure(figsize = (2,2), dpi=300)
    ax = sns.distplot(resid, hist = False, kde_kws = {"shade" : True, "lw": 1}, fit = stats.norm)
    ax.set_title(coeff + ' slope inter anal. RSME: ' + str(RSME))

    comp_s = mdf.params['vel']
    v3_s_ci = mdf.conf_int().loc['vel'].tolist()
    comp_i = mdf.params['Intercept']
    v3_i_ci = mdf.conf_int().loc['Intercept']
    f9v3 = [comp_s, v3_s_ci[0], v3_s_ci[1], comp_i, v3_i_ci[0], v3_i_ci[1]]
    
    print('f9_v3 Slope std: ', str(mdf.bse.loc['vel'].tolist()))
    v0_sl = comp_s+mdf.params['vel:C(config, Treatment(reference="f9_v3"))'+
                              '[T.f9_v0]']
    v0_s_ci = mdf.conf_int().loc['vel:C(config, Treatment(reference="f9_v3"))'+
                              '[T.f9_v0]'].tolist()
    v0_i = comp_i+mdf.params['C(config, Treatment(reference="f9_v3"))'+
                              '[T.f9_v0]']
    v0_i_ci = mdf.conf_int().loc['C(config, Treatment(reference="f9_v3"))'+
                              '[T.f9_v0]'].tolist()
    print('f9_v0 Slope std: ', str(mdf.bse.loc['vel:C(config, Treatment(reference="f9_v3"))'+
                              '[T.f9_v0]'].tolist()))
    f9v0 = [v0_sl, v0_s_ci[0] + comp_s, v0_s_ci[1] + comp_s,
            v0_i, v0_i_ci[0] + comp_i, v0_i_ci[1] + comp_i]
    
    b_sl = comp_s+mdf.params['vel:C(config, Treatment(reference="f9_v3"))'+
                              '[T.base]']
    b_s_ci = mdf.conf_int().loc['vel:C(config, Treatment(reference="f9_v3"))'+
                              '[T.base]'].tolist()
    
    b_i = comp_i+mdf.params['C(config, Treatment(reference="f9_v3"))'+
                              '[T.base]']
    b_i_ci = mdf.conf_int().loc['C(config, Treatment(reference="f9_v3"))'+
                              '[T.base]'].tolist()
    base = [b_sl, b_s_ci[0] + comp_s, b_s_ci[1] + comp_s, 
            b_i, b_i_ci[0] + comp_i, b_i_ci[1] + comp_i]
    print('base Slope std: ', str(mdf.bse.loc['vel:C(config, Treatment(reference="f9_v3"))'+
                              '[T.base]'].tolist()))
    
    
    p_sl = comp_s+mdf.params['vel:C(config, Treatment(reference="f9_v3"))'+
                              '[T.plate]']
    p_s_ci = mdf.conf_int().loc['vel:C(config, Treatment(reference="f9_v3"))'+
                              '[T.plate]'].tolist()
    p_i = comp_i+mdf.params['C(config, Treatment(reference="f9_v3"))'+
                              '[T.plate]']
    p_i_ci = mdf.conf_int().loc['C(config, Treatment(reference="f9_v3"))'+
                              '[T.plate]'].tolist()
    plate = [p_sl, p_s_ci[0] + comp_s, p_s_ci[1] + comp_s, 
             p_i, p_i_ci[0] + comp_i, p_i_ci[1] + comp_i]
    print('plate Slope std: ', str(mdf.bse.loc['vel:C(config, Treatment(reference="f9_v3"))'+
                              '[T.plate]'].tolist()))
    
    print(' summary', mdf.summary())
    
    fig = plt.figure(figsize = (2,2), dpi=300)
    ax = sns.distplot(mdf.resid, hist = False, kde_kws = {"shade" : True, "lw": 1}, fit = stats.norm)
    ax.set_title(coeff+'max vel slope anal: ' +
                 str(np.sqrt(np.mean(mdf.resid*mdf.resid)).round(5)) +
                 ' Shapiro: ' + str(stats.shapiro(mdf.resid)[0]))
    labels = ["Statistic", "p-value"]

    norm_res = stats.shapiro(mdf.resid)
    print('NORMALITY SHAPIRO TEST: ', norm_res)
    fig_new = plt.figure(figsize=(2,2), dpi=300)
    stats.probplot(mdf.resid, dist='norm', plot=plt)
    
    RSME = np.sqrt(np.mean(mdf.resid*mdf.resid))
    print('RSME: ', RSME)
   
    for key, val in dict(zip(labels, norm_res)).items():
        print(key, val)
    
    return f9v3, f9v0, base, plate

def slope_inter_stats_analys(df, coeff):
    f9v3 = pd.DataFrame()
    f9v0 = pd.DataFrame()
    base = pd.DataFrame()
    plate = pd.DataFrame()
    vc_form = {'g1': "0 + C(iter)", 'g2': "0 + C(direction)"}
    oo = np.ones(df.shape[0])
    md = smf.mixedlm(coeff+' ~ AoA + C(config, Treatment(reference="f9_v3")) +\
                     vel + vel:C(config, Treatment(reference="f9_v3")) +\
                     vel:AoA + AoA:C(config, Treatment(reference="f9_v3")) +\
                     AoA:vel:C(config, Treatment(reference="f9_v3"))',
                     df, groups=df['iter_d']) #, vc_formula=vc_form)
    
    mdf = md.fit()
    print('Slope summ: ', mdf.summary())
    fig = plt.figure(figsize = (2,2), dpi=300)
    ax = sns.distplot(mdf.resid, hist = False, kde_kws = {"shade" : True, "lw": 1}, fit = stats.norm)
    ax.set_title('slope inter anal')
    
    comp_s = mdf.params['vel:AoA']
    v3_s_ci = mdf.conf_int().loc['vel:AoA'].tolist()
    print('f9_v3 Slope std: ', str(mdf.bse.loc['vel:AoA'].tolist()))
    comp_i = mdf.params['AoA']
    v3_i_ci = mdf.conf_int().loc['AoA']
    f9v3 = [comp_s, v3_s_ci[0], v3_s_ci[1], comp_i, v3_i_ci[0], v3_i_ci[1]]
    v0_sl = comp_s+mdf.params['AoA:vel:C(config, Treatment(reference="f9_v3"))'+
                              '[T.f9_v0]']
    v0_s_ci = mdf.conf_int().loc['AoA:vel:C(config, Treatment(reference="f9_v3"))'+
                              '[T.f9_v0]'].tolist()
    print('f9_v0 Slope std: ', str(mdf.bse.loc['AoA:vel:C(config, Treatment(reference="f9_v3"))'+
                              '[T.f9_v0]'].tolist()))
    v0_i = comp_i+mdf.params['AoA:C(config, Treatment(reference="f9_v3"))'+
                              '[T.f9_v0]']
    v0_i_ci = mdf.conf_int().loc['AoA:C(config, Treatment(reference="f9_v3"))'+
                              '[T.f9_v0]'].tolist()
    f9v0 = [v0_sl, v0_s_ci[0] + comp_s, v0_s_ci[1] + comp_s,
            v0_i, v0_i_ci[0] + comp_i, v0_i_ci[0] + comp_i]
    
    b_sl = comp_s+mdf.params['AoA:vel:C(config, Treatment(reference="f9_v3"))'+
                              '[T.base]']
    b_s_ci = mdf.conf_int().loc['AoA:vel:C(config, Treatment(reference="f9_v3"))'+
                              '[T.base]'].tolist()
    print('base Slope std: ', str(mdf.bse.loc['AoA:vel:C(config, Treatment(reference="f9_v3"))'+
                              '[T.base]'].tolist()))
    
    b_i = comp_i+mdf.params['AoA:C(config, Treatment(reference="f9_v3"))'+
                              '[T.base]']
    b_i_ci = mdf.conf_int().loc['AoA:C(config, Treatment(reference="f9_v3"))'+
                              '[T.base]'].tolist()
    base = [b_sl, b_s_ci[0] + comp_s, b_s_ci[1] + comp_s, 
            b_i, b_i_ci[0] + comp_i, b_i_ci[1] + comp_i]
    
    p_sl = comp_s+mdf.params['AoA:vel:C(config, Treatment(reference="f9_v3"))'+
                              '[T.plate]']
    p_s_ci = mdf.conf_int().loc['AoA:vel:C(config, Treatment(reference="f9_v3"))'+
                              '[T.plate]'].tolist()
    print('plate Slope std: ', str(mdf.bse.loc['AoA:vel:C(config, Treatment(reference="f9_v3"))'+
                              '[T.plate]'].tolist()))
    p_i = comp_i+mdf.params['AoA:C(config, Treatment(reference="f9_v3"))'+
                              '[T.plate]']
    p_i_ci = mdf.conf_int().loc['AoA:C(config, Treatment(reference="f9_v3"))'+
                              '[T.plate]'].tolist()
    plate = [p_sl, p_s_ci[0] + comp_s, p_s_ci[1] + comp_s, 
             p_i, p_i_ci[0] + comp_i, p_i_ci[1] + comp_i]
    
    print(' summary', mdf.summary())

    fig = plt.figure(figsize = (2,2), dpi=300)
    ax = sns.distplot(mdf.resid, hist = False, kde_kws = {"shade" : True, "lw": 1}, fit = stats.norm)
    ax.set_title(coeff+' vel slope anal: '+
                 str(np.sqrt(np.mean(mdf.resid*mdf.resid)).round(5)) +
                 ' Shapiro: ' + str(stats.shapiro(mdf.resid)[0]))
    labels = ["Statistic", "p-value"]

    norm_res = stats.shapiro(mdf.resid)

    for key, val in dict(zip(labels, norm_res)).items():
        print(key, val)
    return f9v3, f9v0, base, plate
def max_stats_analys(l_df, maxs_df, coeff='CL'):
    
    df = pd.DataFrame()
    for r in [200,250,300]:
        for it in range(8):
            alph = maxs_df.loc[(maxs_df['rpm']==r) & (maxs_df['iter_d']==it), ['AoA']].to_numpy()
            _df = l_df.loc[(l_df['rpm'] == r) & (l_df['AoA'] == alph[0,0]) &
                             (l_df['iter_d'] == it)]
            
        
            df = pd.concat([df,_df])
    df['vel'] = df['rpm']*0.055+0.525
        
    #if coeff=='LD':
        #print(df)
    md = smf.mixedlm(coeff+" ~ vel", 
                     df, groups=df["iter"]) #, re_formula="~taps")
    mdf = md.fit() #method=["lbfgs"])
    #print(mdf.summary())
    #print('Max_CL vs Velocity slope: ', mdf.params['vel'])
    conf_v = mdf.conf_int().loc['vel'].tolist()
    conf_i = mdf.conf_int().loc['Intercept'].tolist()
    return [mdf.params['vel'],conf_v[0], conf_v[1], mdf.params['Intercept'], conf_i[0], conf_i[1]], df 


#%%
def combine_iters_CL(df):    
    CL_group =   df.groupby(['config','rpm','AoA'])['CL']  
    CL_df_mean = CL_group.mean()
    
    #Carry through unscertainty
    CLvar_df = CL_group.var().reset_index()['CL']
    CL_df = CL_df_mean.reset_index()
    CL_df['CLvar'] = CLvar_df
    
    return CL_df

def combine_iters_CD(df):    
    CD_group =   df.groupby(['config','rpm','AoA'])['CD']  
    CD_df_mean = CD_group.mean()
    
    #Carry through unscertainty
    CDvar_df = CD_group.var().reset_index()['CD']
    CD_df = CD_df_mean.reset_index()
    CD_df['CDvar'] = CDvar_df
    
    return CD_df

def combine_iters_LD(df):    
    LD_group =   df.groupby(['config','rpm','AoA'])['LD']  
    LD_df_mean = LD_group.mean()
    
    #Carry through unscertainty
    LDvar_df = LD_group.var().reset_index()['LD']
    LD_df = LD_df_mean.reset_index()
    LD_df['LDvar'] = LDvar_df
    
    return LD_df

def combine_iters_Cm(df):    
    Cm_group =   df.groupby(['config','rpm','AoA'])['Cm']  
    Cm_df_mean = Cm_group.mean()
    
    #Carry through unscertainty
    Cmvar_df = Cm_group.var().reset_index()['Cm']
    Cm_df = Cm_df_mean.reset_index()
    Cm_df['Cmvar'] = Cmvar_df
    
    return Cm_df


def up_sweep_analysis(df, coeff):
    
    df = df[df['direction'] == 'NP']

    CL_group =   df.groupby(['config','rpm','AoA'])[[coeff]]  
    CL_df_mean = CL_group.mean()
    CL_mean_df = CL_df_mean.reset_index()
    CL_df_var = CL_group.var()
    CL_var_df = CL_df_var.reset_index()
    
    CL_up_sweep_mean = CL_mean_df
    CL_up_sweep_var = CL_var_df

    CLvar_df = CL_up_sweep_var[coeff]
    
    CL_up_sweep_df = CL_up_sweep_mean.copy()
    CL_up_sweep_df[coeff+'var'] = CLvar_df
    return CL_up_sweep_df


def combine_but_dir_it(df, coeff):
    
    CL_group =   df.groupby(['config','rpm','AoA', 'iter_d'])[[coeff]]  
    CL_df_mean = CL_group.mean()
    CL_mean_df = CL_df_mean.reset_index()    

    CLvar_df = CL_group.var().reset_index()[coeff]
    
    CL_df = CL_mean_df.copy()
    CL_df[coeff+'var'] = CLvar_df
    return CL_df
#%%
def CL_vs_AoA_up(df):
    b_color = '#221F20'
    p_color = '#D5D5D7'
    f9v0_color = '#4771b2'
    f9v3_color = '#c6793a'
    
    
    #PLot the Cl vs Alpha curves and analyse slope and such

    CL_200 = pd.pivot_table(df.loc[df['rpm'] == 200],
                   index='AoA', columns = 'config', values='CL')
    CL_250 = pd.pivot_table(df.loc[df['rpm'] == 250],
                   index='AoA', columns = 'config', values='CL')
    CL_300 = pd.pivot_table(df.loc[df['rpm'] == 300],
                   index='AoA', columns = 'config', values='CL')
    
    CL_200.to_csv('CL_alph_200rpm_up_vel_adj.csv')
    CL_250.to_csv('CL_alph_250rpm_up_vel_adj.csv')
    CL_300.to_csv('CL_alph_300rpm_up_vel_adj.csv')
    
def CD_vs_AoA_up(df):
    b_color = '#221F20'
    p_color = '#D5D5D7'
    f9v0_color = '#4771b2'
    f9v3_color = '#c6793a'
    
    
    #PLot the Cl vs Alpha curves and analyse slope and such

    CD_200 = pd.pivot_table(df.loc[df['rpm'] == 200],
                   index='AoA', columns = 'config', values='CD')
    CD_250 = pd.pivot_table(df.loc[df['rpm'] == 250],
                   index='AoA', columns = 'config', values='CD')
    CD_300 = pd.pivot_table(df.loc[df['rpm'] == 300],
                   index='AoA', columns = 'config', values='CD')
    
    CD_200.to_csv('CD_alph_200rpm_up_vel_adj.csv')
    CD_250.to_csv('CD_alph_250rpm_up_vel_adj.csv')
    CD_300.to_csv('CD_alph_300rpm_up_vel_adj.csv')
    
def Cm_vs_AoA_up(df):
    b_color = '#221F20'
    p_color = '#D5D5D7'
    f9v0_color = '#4771b2'
    f9v3_color = '#c6793a'
    
    
    #PLot the Cm vs Alpha curves and analyse slope and such

    Cm_200 = pd.pivot_table(df.loc[df['rpm'] == 200],
                   index='AoA', columns = 'config', values='Cm')
    Cm_250 = pd.pivot_table(df.loc[df['rpm'] == 250],
                   index='AoA', columns = 'config', values='Cm')
    Cm_300 = pd.pivot_table(df.loc[df['rpm'] == 300],
                   index='AoA', columns = 'config', values='Cm')
    
    Cm_200.to_csv('Cm_alph_200rpm_up_vel_adj.csv')
    Cm_250.to_csv('Cm_alph_250rpm_up_vel_adj.csv')
    Cm_300.to_csv('Cm_alph_300rpm_up_vel_adj.csv')
    
def CL_vs_AoA(df):
    b_color = '#221F20'
    p_color = '#D5D5D7'
    f9v0_color = '#4771b2'
    f9v3_color = '#c6793a'
    
    print('Average CL')
    print(df.loc[df['AoA'] < 10]['CL'].mean())
    
    #PLot the Cl vs Alpha curves and analyse slope and such

    CL_200 = pd.pivot_table(df.loc[df['rpm'] == 200],
                   index='AoA', columns = 'config', values='CL')
    CL_250 = pd.pivot_table(df.loc[df['rpm'] == 250],
                   index='AoA', columns = 'config', values='CL')
    CL_300 = pd.pivot_table(df.loc[df['rpm'] == 300],
                   index='AoA', columns = 'config', values='CL')
    
    #CL_200.to_csv('CL_alph_200rpm.csv')
    #CL_250.to_csv('CL_alph_250rpm.csv')
    #CL_300.to_csv('CL_alph_300rpm.csv')

    CLvar_200 = pd.pivot_table(df.loc[df['rpm'] == 200],
                   index='AoA', columns = 'config', values='CLvar')
    CLvar_250 = pd.pivot_table(df.loc[df['rpm'] == 250],
                   index='AoA', columns = 'config', values='CLvar')
    CLvar_300 = pd.pivot_table(df.loc[df['rpm'] == 300],
                   index='AoA', columns = 'config', values='CLvar')
    
    fig, ax = plt.subplots(3,1, figsize=(2,5), dpi=300)
    
    
    ax[0].errorbar(CL_200.index, CL_200['base'], yerr=1.96*CLvar_200['base']**0.5,
                   linewidth=1, elinewidth=0.5, capsize = 1, capthick=0.5,
                   color=b_color)
    ax[0].errorbar(CL_200.index, CL_200['plate'], yerr=1.96*CLvar_200['plate']**0.5,
                   linewidth=1, elinewidth=0.5, capsize=1, capthick=0.5,
                   color=p_color)
    ax[0].errorbar(CL_200.index, CL_200['f9_v0'], yerr=1.96*CLvar_200['f9_v0']**0.5,
                   linewidth=1, elinewidth=0.5, capsize=1, capthick=0.5,
                   color=f9v0_color)
    ax[0].errorbar(CL_200.index, CL_200['f9_v3'], yerr=1.96*CLvar_200['f9_v3']**0.5,
                   linewidth=1, elinewidth=0.5, capsize=1, capthick=0.5,
                   color=f9v3_color)
    
    ax[1].errorbar(CL_250.index, CL_250['base'], yerr=1.96*CLvar_250['base']**0.5,
                   linewidth=1, elinewidth=0.5, capsize = 1, capthick=0.5,
                   color=b_color)
    ax[1].errorbar(CL_250.index, CL_250['plate'], yerr=1.96*CLvar_250['plate']**0.5,
                   linewidth=1, elinewidth=0.5, capsize=1, capthick=0.5,
                   color=p_color)
    ax[1].errorbar(CL_250.index, CL_250['f9_v0'], yerr=1.96*CLvar_250['f9_v0']**0.5,
                   linewidth=1, elinewidth=0.5, capsize=1, capthick=0.5,
                   color=f9v0_color)
    ax[1].errorbar(CL_250.index, CL_250['f9_v3'], yerr=1.96*CLvar_250['f9_v3']**0.5,
                   linewidth=1, elinewidth=0.5, capsize=1, capthick=0.5,
                   color=f9v3_color)
    
    ax[2].errorbar(CL_300.index, CL_300['base'], yerr=1.96*CLvar_300['base']**0.5,
                  linewidth=1, elinewidth=0.5, capsize = 1, capthick=0.5,
                  color=b_color)
    ax[2].errorbar(CL_300.index, CL_300['plate'], yerr=1.96*CLvar_300['plate']**0.5,
                   linewidth=1, elinewidth=0.5, capsize=1, capthick=0.5,
                   color=p_color)
    ax[2].errorbar(CL_300.index, CL_300['f9_v0'], yerr=1.96*CLvar_300['f9_v0']**0.5,
                   linewidth=1, elinewidth=0.5, capsize=1, capthick=0.5,
                   color=f9v0_color)
    ax[2].errorbar(CL_300.index, CL_300['f9_v3'], yerr=1.96*CLvar_300['f9_v3']**0.5,
                   linewidth=1, elinewidth=0.5, capsize=1, capthick=0.5,
                   color=f9v3_color)
    
    xlims = [-6,20]
    ylims = [-0.5,1.5]
    
    ax[0].spines[['right','top','bottom']].set_visible(False)
    ax[0].set_ylabel('$C_{L}$')
    ax[0].set_xlim(xlims)
    ax[0].set_ylim(ylims)
    ax[0].get_xaxis().set_ticks([])
    ax[0].get_yaxis().set_ticks([-0.5,0,0.5,1,1.5])
    ax[0].set_title('11.5 m/s')
    ax[1].spines[['right','top','bottom']].set_visible(False)
    ax[1].set_ylabel('$C_{L}$')
    ax[1].set_xlim(xlims)
    ax[1].set_ylim(ylims)
    ax[1].get_xaxis().set_ticks([])
    ax[1].get_yaxis().set_ticks([-0.5,0,0.5,1,1.5])
    ax[1].set_title('14.3 m/s')
    ax[2].spines[['right','top']].set_visible(False)
    ax[2].set_ylabel('$C_{L}$')
    ax[2].set_xlim(xlims)
    ax[2].set_ylim(ylims)
    ax[2].get_yaxis().set_ticks([-0.5,0,0.5,1,1.5])
    ax[2].set_title('17 m/s')
    ax[2].set_xlabel('Angle of Attack (deg)')
    
    custom_lines1 = [Line2D([0], [0], color= b_color, lw=1),
                Line2D([0], [0], color= p_color, lw=1),
                Line2D([0], [0], color= f9v0_color, lw=1),
                Line2D([0], [0], color= f9v3_color, lw=1)]
    
    legend1 = fig.legend(custom_lines1,['Baseline','Plate','f9_v0','f9_v3'],
                         ncol= 4, loc=8, bbox_to_anchor=(0.55,-0.03), frameon=False)
    
    fig.tight_layout()
    fig.savefig('CL_alpha', dpi=300)
    
def CD_vs_AoA(df):
    b_color = '#221F20'
    p_color = '#D5D5D7'
    f9v0_color = '#4771b2'
    f9v3_color = '#c6793a'
    
    
    #PLot the CD vs Alpha curves and analyse slope and such

    CD_200 = pd.pivot_table(df.loc[df['rpm'] == 200],
                   index='AoA', columns = 'config', values='CD')
    CD_250 = pd.pivot_table(df.loc[df['rpm'] == 250],
                   index='AoA', columns = 'config', values='CD')
    CD_300 = pd.pivot_table(df.loc[df['rpm'] == 300],
                   index='AoA', columns = 'config', values='CD')
    

    CDvar_200 = pd.pivot_table(df.loc[df['rpm'] == 200],
                   index='AoA', columns = 'config', values='CDvar')
    CDvar_250 = pd.pivot_table(df.loc[df['rpm'] == 250],
                   index='AoA', columns = 'config', values='CDvar')
    CDvar_300 = pd.pivot_table(df.loc[df['rpm'] == 300],
                   index='AoA', columns = 'config', values='CDvar')
    
    fig, ax = plt.subplots(3,1, figsize=(2,5), dpi=300)
    
    
    ax[0].errorbar(CD_200.index, CD_200['base'], yerr=CDvar_200['base']**0.5,
                   linewidth=1, elinewidth=0.5, capsize = 1, capthick=0.5,
                   color=b_color)
    ax[0].errorbar(CD_200.index, CD_200['plate'], yerr=CDvar_200['plate']**0.5,
                   linewidth=1, elinewidth=0.5, capsize=1, capthick=0.5,
                   color=p_color)
    ax[0].errorbar(CD_200.index, CD_200['f9_v0'], yerr=CDvar_200['f9_v0']**0.5,
                   linewidth=1, elinewidth=0.5, capsize=1, capthick=0.5,
                   color=f9v0_color)
    ax[0].errorbar(CD_200.index, CD_200['f9_v3'], yerr=CDvar_200['f9_v3']**0.5,
                   linewidth=1, elinewidth=0.5, capsize=1, capthick=0.5,
                   color=f9v3_color)
    
    ax[1].errorbar(CD_250.index, CD_250['base'], yerr=CDvar_250['base']**0.5,
                   linewidth=1, elinewidth=0.5, capsize = 1, capthick=0.5,
                   color=b_color)
    ax[1].errorbar(CD_250.index, CD_250['plate'], yerr=CDvar_250['plate']**0.5,
                   linewidth=1, elinewidth=0.5, capsize=1, capthick=0.5,
                   color=p_color)
    ax[1].errorbar(CD_250.index, CD_250['f9_v0'], yerr=CDvar_250['f9_v0']**0.5,
                   linewidth=1, elinewidth=0.5, capsize=1, capthick=0.5,
                   color=f9v0_color)
    ax[1].errorbar(CD_250.index, CD_250['f9_v3'], yerr=CDvar_250['f9_v3']**0.5,
                   linewidth=1, elinewidth=0.5, capsize=1, capthick=0.5,
                   color=f9v3_color)
    
    ax[2].errorbar(CD_300.index, CD_300['base'], yerr=CDvar_300['base']**0.5,
                  linewidth=1, elinewidth=0.5, capsize = 1, capthick=0.5,
                  color=b_color)
    ax[2].errorbar(CD_300.index, CD_300['plate'], yerr=CDvar_300['plate']**0.5,
                   linewidth=1, elinewidth=0.5, capsize=1, capthick=0.5,
                   color=p_color)
    ax[2].errorbar(CD_300.index, CD_300['f9_v0'], yerr=CDvar_300['f9_v0']**0.5,
                   linewidth=1, elinewidth=0.5, capsize=1, capthick=0.5,
                   color=f9v0_color)
    ax[2].errorbar(CD_300.index, CD_300['f9_v3'], yerr=CDvar_300['f9_v3']**0.5,
                   linewidth=1, elinewidth=0.5, capsize=1, capthick=0.5,
                   color=f9v3_color)
    
    print('CD 300: ', CD_300)
    
    xlims = [-6,20]
    
    
    ax[0].spines[['right','top','bottom']].set_visible(False)
    ax[0].set_ylabel('$C_{D}$')
    ax[0].set_xlim(xlims)
    ax[0].get_xaxis().set_ticks([])
    ax[0].set_title('11.5 m/s')
    ax[1].spines[['right','top','bottom']].set_visible(False)
    ax[1].set_ylabel('$C_{D}$')
    ax[1].set_xlim(xlims)
    ax[1].get_xaxis().set_ticks([])
    ax[1].set_title('14.3 m/s')
    ax[2].spines[['right','top']].set_visible(False)
    ax[2].set_ylabel('$C_{D}$')
    ax[2].set_xlim(xlims)
    ax[2].set_title('17.0 m/s')
    ax[2].set_xlabel('Angle of Attack (deg)')
    
    custom_lines1 = [Line2D([0], [0], color= b_color, lw=1),
                Line2D([0], [0], color= p_color, lw=1),
                Line2D([0], [0], color= f9v0_color, lw=1),
                Line2D([0], [0], color= f9v3_color, lw=1)]
    
    legend1 = fig.legend(custom_lines1,['Baseline','Plate','f9_v0','f9_v3'],
                         ncol= 4, loc=8, bbox_to_anchor=(0.55,-0.03), frameon=False)
    
    fig.tight_layout()
    
def LD_vs_AoA(df):
    b_color = '#221F20'
    p_color = '#D5D5D7'
    f9v0_color = '#4771b2'
    f9v3_color = '#c6793a'
    
   
    #PLot the L/D vs Alpha curves and analyse slope and such

    LD_200 = pd.pivot_table(df.loc[df['rpm'] == 200],
                   index='AoA', columns = 'config', values='LD')
    LD_250 = pd.pivot_table(df.loc[df['rpm'] == 250],
                   index='AoA', columns = 'config', values='LD')
    LD_300 = pd.pivot_table(df.loc[df['rpm'] == 300],
                   index='AoA', columns = 'config', values='LD')
    

    LDvar_200 = pd.pivot_table(df.loc[df['rpm'] == 200],
                   index='AoA', columns = 'config', values='LDvar')
    LDvar_250 = pd.pivot_table(df.loc[df['rpm'] == 250],
                   index='AoA', columns = 'config', values='LDvar')
    LDvar_300 = pd.pivot_table(df.loc[df['rpm'] == 300],
                   index='AoA', columns = 'config', values='LDvar')
    
    fig, ax = plt.subplots(3,1, figsize=(2,5), dpi=300)
    
    
    ax[0].errorbar(LD_200.index, LD_200['base'], yerr=LDvar_200['base']**0.5,
                   linewidth=1, elinewidth=0.5, capsize = 1, capthick=0.5,
                   color=b_color)
    ax[0].errorbar(LD_200.index, LD_200['plate'], yerr=LDvar_200['plate']**0.5,
                   linewidth=1, elinewidth=0.5, capsize=1, capthick=0.5,
                   color=p_color)
    ax[0].errorbar(LD_200.index, LD_200['f9_v0'], yerr=LDvar_200['f9_v0']**0.5,
                   linewidth=1, elinewidth=0.5, capsize=1, capthick=0.5,
                   color=f9v0_color)
    ax[0].errorbar(LD_200.index, LD_200['f9_v3'], yerr=LDvar_200['f9_v3']**0.5,
                   linewidth=1, elinewidth=0.5, capsize=1, capthick=0.5,
                   color=f9v3_color)
    
    ax[1].errorbar(LD_250.index, LD_250['base'], yerr=LDvar_250['base']**0.5,
                   linewidth=1, elinewidth=0.5, capsize = 1, capthick=0.5,
                   color=b_color)
    ax[1].errorbar(LD_250.index, LD_250['plate'], yerr=LDvar_250['plate']**0.5,
                   linewidth=1, elinewidth=0.5, capsize=1, capthick=0.5,
                   color=p_color)
    ax[1].errorbar(LD_250.index, LD_250['f9_v0'], yerr=LDvar_250['f9_v0']**0.5,
                   linewidth=1, elinewidth=0.5, capsize=1, capthick=0.5,
                   color=f9v0_color)
    ax[1].errorbar(LD_250.index, LD_250['f9_v3'], yerr=LDvar_250['f9_v3']**0.5,
                   linewidth=1, elinewidth=0.5, capsize=1, capthick=0.5,
                   color=f9v3_color)
    
    ax[2].errorbar(LD_300.index, LD_300['base'], yerr=LDvar_300['base']**0.5,
                  linewidth=1, elinewidth=0.5, capsize = 1, capthick=0.5,
                  color=b_color)
    ax[2].errorbar(LD_300.index, LD_300['plate'], yerr=LDvar_300['plate']**0.5,
                   linewidth=1, elinewidth=0.5, capsize=1, capthick=0.5,
                   color=p_color)
    ax[2].errorbar(LD_300.index, LD_300['f9_v0'], yerr=LDvar_300['f9_v0']**0.5,
                   linewidth=1, elinewidth=0.5, capsize=1, capthick=0.5,
                   color=f9v0_color)
    ax[2].errorbar(LD_300.index, LD_300['f9_v3'], yerr=LDvar_300['f9_v3']**0.5,
                   linewidth=1, elinewidth=0.5, capsize=1, capthick=0.5,
                   color=f9v3_color)
    
    xlims = [-6,20]
    
    
    ax[0].spines[['right','top','bottom']].set_visible(False)
    ax[0].set_ylabel('$L/D$')
    ax[0].set_xlim(xlims)
    ax[0].get_xaxis().set_ticks([])
    ax[0].set_title('11.5 m/s')
    ax[1].spines[['right','top','bottom']].set_visible(False)
    ax[1].set_ylabel('$L/D$')
    ax[1].set_xlim(xlims)
    ax[1].get_xaxis().set_ticks([])
    ax[1].set_title('14.3 m/s')
    ax[2].spines[['right','top']].set_visible(False)
    ax[2].set_ylabel('$L/D$')
    ax[2].set_xlim(xlims)
    ax[2].set_title('17.0 m/s')
    ax[2].set_xlabel('Angle of Attack (deg)')
    
    custom_lines1 = [Line2D([0], [0], color= b_color, lw=1),
                Line2D([0], [0], color= p_color, lw=1),
                Line2D([0], [0], color= f9v0_color, lw=1),
                Line2D([0], [0], color= f9v3_color, lw=1)]
    
    legend1 = fig.legend(custom_lines1,['Baseline','Plate','f9_v0','f9_v3'],
                         ncol= 4, loc=8, bbox_to_anchor=(0.55,-0.03), frameon=False)
    
    fig.tight_layout()
    
def Cm_vs_AoA(df):
    b_color = '#221F20'
    p_color = '#D5D5D7'
    f9v0_color = '#4771b2'
    f9v3_color = '#c6793a'
    
    print(df.loc[(df['AoA'] < 10) & (df['AoA'] > 0)]['Cm'].mean())
    #PLot the Cm vs Alpha curves and analyse slope and such

    Cm_200 = pd.pivot_table(df.loc[df['rpm'] == 200],
                   index='AoA', columns = 'config', values='Cm')
    Cm_250 = pd.pivot_table(df.loc[df['rpm'] == 250],
                   index='AoA', columns = 'config', values='Cm')
    Cm_300 = pd.pivot_table(df.loc[df['rpm'] == 300],
                   index='AoA', columns = 'config', values='Cm')
    
    #Cm_200.to_csv('Cm_alph_200rpm.csv')
    #Cm_250.to_csv('Cm_alph_250rpm.csv')
    #Cm_300.to_csv('Cm_alph_300rpm.csv')

    Cmvar_200 = pd.pivot_table(df.loc[df['rpm'] == 200],
                   index='AoA', columns = 'config', values='Cmvar')
    Cmvar_250 = pd.pivot_table(df.loc[df['rpm'] == 250],
                   index='AoA', columns = 'config', values='Cmvar')
    Cmvar_300 = pd.pivot_table(df.loc[df['rpm'] == 300],
                   index='AoA', columns = 'config', values='Cmvar')
    
    fig, ax = plt.subplots(3,1, figsize=(2,5), dpi=300)
    
    
    ax[0].errorbar(Cm_200.index, Cm_200['base'], yerr=1.96*Cmvar_200['base']**0.5,
                   linewidth=1, elinewidth=0.5, capsize = 1, capthick=0.5,
                   color=b_color)
    ax[0].errorbar(Cm_200.index, Cm_200['plate'], yerr=1.96*Cmvar_200['plate']**0.5,
                   linewidth=1, elinewidth=0.5, capsize=1, capthick=0.5,
                   color=p_color)
    ax[0].errorbar(Cm_200.index, Cm_200['f9_v0'], yerr=1.96*Cmvar_200['f9_v0']**0.5,
                   linewidth=1, elinewidth=0.5, capsize=1, capthick=0.5,
                   color=f9v0_color)
    ax[0].errorbar(Cm_200.index, Cm_200['f9_v3'], yerr=1.96*Cmvar_200['f9_v3']**0.5,
                   linewidth=1, elinewidth=0.5, capsize=1, capthick=0.5,
                   color=f9v3_color)
    
    ax[1].errorbar(Cm_250.index, Cm_250['base'], yerr=1.96*Cmvar_250['base']**0.5,
                   linewidth=1, elinewidth=0.5, capsize = 1, capthick=0.5,
                   color=b_color)
    ax[1].errorbar(Cm_250.index, Cm_250['plate'], yerr=1.96*Cmvar_250['plate']**0.5,
                   linewidth=1, elinewidth=0.5, capsize=1, capthick=0.5,
                   color=p_color)
    ax[1].errorbar(Cm_250.index, Cm_250['f9_v0'], yerr=1.96*Cmvar_250['f9_v0']**0.5,
                   linewidth=1, elinewidth=0.5, capsize=1, capthick=0.5,
                   color=f9v0_color)
    ax[1].errorbar(Cm_250.index, Cm_250['f9_v3'], yerr=1.96*Cmvar_250['f9_v3']**0.5,
                   linewidth=1, elinewidth=0.5, capsize=1, capthick=0.5,
                   color=f9v3_color)
    
    ax[2].errorbar(Cm_300.index, Cm_300['base'], yerr=1.96*Cmvar_300['base']**0.5,
                  linewidth=1, elinewidth=0.5, capsize = 1, capthick=0.5,
                  color=b_color)
    ax[2].errorbar(Cm_300.index, Cm_300['plate'], yerr=1.96*Cmvar_300['plate']**0.5,
                   linewidth=1, elinewidth=0.5, capsize=1, capthick=0.5,
                   color=p_color)
    ax[2].errorbar(Cm_300.index, Cm_300['f9_v0'], yerr=1.96*Cmvar_300['f9_v0']**0.5,
                   linewidth=1, elinewidth=0.5, capsize=1, capthick=0.5,
                   color=f9v0_color)
    ax[2].errorbar(Cm_300.index, Cm_300['f9_v3'], yerr=1.96*Cmvar_300['f9_v3']**0.5,
                   linewidth=1, elinewidth=0.5, capsize=1, capthick=0.5,
                   color=f9v3_color)
    
    xlims = [-6,20]
    
    
    ax[0].spines[['right','top','bottom']].set_visible(False)
    ax[0].set_ylabel('$C_{m}$')
    ax[0].set_xlim(xlims)
    ax[0].get_xaxis().set_ticks([])
    ax[0].set_title('11.3 m/s')
    ax[1].spines[['right','top','bottom']].set_visible(False)
    ax[1].set_ylabel('$C_{m}$')
    ax[1].set_xlim(xlims)
    ax[1].get_xaxis().set_ticks([])
    ax[1].set_title('14.3 m/s')
    ax[2].spines[['right','top']].set_visible(False)
    ax[2].set_ylabel('$C_{m}$')
    ax[2].set_xlim(xlims)
    ax[2].set_title('17 m/s')
    ax[2].set_xlabel('Angle of Attack (deg)')
    
    custom_lines1 = [Line2D([0], [0], color= b_color, lw=1),
                Line2D([0], [0], color= p_color, lw=1),
                Line2D([0], [0], color= f9v0_color, lw=1),
                Line2D([0], [0], color= f9v3_color, lw=1)]
    
    legend1 = fig.legend(custom_lines1,['Baseline','Plate','f9_v0','f9_v3'],
                         ncol= 4, loc=8, bbox_to_anchor=(0.55,-0.03), frameon=False)
    
    fig.tight_layout()
    
def CL_vs_Cm(CL_df, Cm_df):
    b_color = '#221F20'
    p_color = '#D5D5D7'
    f9v0_color = '#4771b2'
    f9v3_color = '#c6793a'
    
    CL_df = CL_df.loc[CL_df['AoA']<12]
    Cm_df = Cm_df.loc[Cm_df['AoA']<12]
    


    CL_200 = pd.pivot_table(CL_df.loc[CL_df['rpm'] == 200],
                   index='AoA', columns = 'config', values='CL')
    CL_250 = pd.pivot_table(CL_df.loc[CL_df['rpm'] == 250],
                   index='AoA', columns = 'config', values='CL')
    CL_300 = pd.pivot_table(CL_df.loc[CL_df['rpm'] == 300],
                   index='AoA', columns = 'config', values='CL')
    
    #Cm_200.to_csv('Cm_alph_200rpm.csv')
    #Cm_250.to_csv('Cm_alph_250rpm.csv')
    #Cm_300.to_csv('Cm_alph_300rpm.csv')

    CLvar_200 = pd.pivot_table(CL_df.loc[CL_df['rpm'] == 200],
                   index='AoA', columns = 'config', values='CLvar')
    CLvar_250 = pd.pivot_table(CL_df.loc[CL_df['rpm'] == 250],
                   index='AoA', columns = 'config', values='CLvar')
    CLvar_300 = pd.pivot_table(CL_df.loc[CL_df['rpm'] == 300],
                   index='AoA', columns = 'config', values='CLvar')
    
   

    Cm_200 = pd.pivot_table(Cm_df.loc[Cm_df['rpm'] == 200],
                   index='AoA', columns = 'config', values='Cm')
    Cm_250 = pd.pivot_table(Cm_df.loc[Cm_df['rpm'] == 250],
                   index='AoA', columns = 'config', values='Cm')
    Cm_300 = pd.pivot_table(Cm_df.loc[Cm_df['rpm'] == 300],
                   index='AoA', columns = 'config', values='Cm')
    
    #Cm_200.to_csv('Cm_alph_200rpm.csv')
    #Cm_250.to_csv('Cm_alph_250rpm.csv')
    #Cm_300.to_csv('Cm_alph_300rpm.csv')

    Cmvar_200 = pd.pivot_table(Cm_df.loc[Cm_df['rpm'] == 200],
                   index='AoA', columns = 'config', values='Cmvar')
    Cmvar_250 = pd.pivot_table(Cm_df.loc[Cm_df['rpm'] == 250],
                   index='AoA', columns = 'config', values='Cmvar')
    Cmvar_300 = pd.pivot_table(Cm_df.loc[Cm_df['rpm'] == 300],
                   index='AoA', columns = 'config', values='Cmvar')
    
    fig, ax = plt.subplots(3,1, figsize=(2,5), dpi=300)
    
    
    ax[0].errorbar(CL_200['base'], Cm_200['base'], xerr=1.96*CLvar_200['base']**0.5, yerr=1.96*Cmvar_200['base']**0.5,
                   linewidth=1, elinewidth=0.5, capsize = 1, capthick=0.5,
                   color=b_color)
    ax[0].errorbar(CL_200['plate'], Cm_200['plate'], xerr=1.96*CLvar_200['plate']**0.5, yerr=1.96*Cmvar_200['plate']**0.5,
                   linewidth=1, elinewidth=0.5, capsize=1, capthick=0.5,
                   color=p_color)
    ax[0].errorbar(CL_200['f9_v0'], Cm_200['f9_v0'], xerr=1.96*CLvar_200['f9_v0']**0.5, yerr=1.96*Cmvar_200['f9_v0']**0.5,
                   linewidth=1, elinewidth=0.5, capsize=1, capthick=0.5,
                   color=f9v0_color)
    ax[0].errorbar(CL_200['f9_v3'], Cm_200['f9_v3'], xerr=1.96*CLvar_200['f9_v3']**0.5 ,yerr=1.96*Cmvar_200['f9_v3']**0.5,
                   linewidth=1, elinewidth=0.5, capsize=1, capthick=0.5,
                   color=f9v3_color)
    
    ax[1].errorbar(CL_250['base'], Cm_250['base'], xerr=1.96*CLvar_250['base']**0.5, yerr=1.96*Cmvar_250['base']**0.5,
                   linewidth=1, elinewidth=0.5, capsize = 1, capthick=0.5,
                   color=b_color)
    ax[1].errorbar(CL_250['plate'], Cm_250['plate'], xerr=1.96*CLvar_250['plate']**0.5, yerr=1.96*Cmvar_250['plate']**0.5,
                   linewidth=1, elinewidth=0.5, capsize=1, capthick=0.5,
                   color=p_color)
    ax[1].errorbar(CL_250['f9_v0'], Cm_250['f9_v0'], xerr=1.96*CLvar_250['f9_v0']**0.5, yerr=1.96*Cmvar_250['f9_v0']**0.5,
                   linewidth=1, elinewidth=0.5, capsize=1, capthick=0.5,
                   color=f9v0_color)
    ax[1].errorbar(CL_250['f9_v3'], Cm_250['f9_v3'], xerr=1.96*CLvar_250['f9_v3']**0.5 ,yerr=1.96*Cmvar_250['f9_v3']**0.5,
                   linewidth=1, elinewidth=0.5, capsize=1, capthick=0.5,
                   color=f9v3_color)
    
    ax[2].errorbar(CL_300['base'], Cm_300['base'], xerr=1.96*CLvar_300['base']**0.5, yerr=1.96*Cmvar_300['base']**0.5,
                   linewidth=1, elinewidth=0.5, capsize = 1, capthick=0.5,
                   color=b_color)
    ax[2].errorbar(CL_300['plate'], Cm_300['plate'], xerr=1.96*CLvar_300['plate']**0.5, yerr=1.96*Cmvar_300['plate']**0.5,
                   linewidth=1, elinewidth=0.5, capsize=1, capthick=0.5,
                   color=p_color)
    ax[2].errorbar(CL_300['f9_v0'], Cm_300['f9_v0'], xerr=1.96*CLvar_300['f9_v0']**0.5, yerr=1.96*Cmvar_300['f9_v0']**0.5,
                   linewidth=1, elinewidth=0.5, capsize=1, capthick=0.5,
                   color=f9v0_color)
    ax[2].errorbar(CL_300['f9_v3'], Cm_300['f9_v3'], xerr=1.96*CLvar_300['f9_v3']**0.5 ,yerr=1.96*Cmvar_300['f9_v3']**0.5,
                   linewidth=1, elinewidth=0.5, capsize=1, capthick=0.5,
                   color=f9v3_color)
    
    xlims = [-0.4,1]
    
    
    ax[0].spines[['right','top','bottom']].set_visible(False)
    ax[0].set_ylabel('$C_{m}$')
    ax[0].set_xlim(xlims)
    ax[0].get_xaxis().set_ticks([])
    ax[0].set_title('11.3 m/s')
    ax[1].spines[['right','top','bottom']].set_visible(False)
    ax[1].set_ylabel('$C_{m}$')
    ax[1].set_xlim(xlims)
    ax[1].get_xaxis().set_ticks([])
    ax[1].set_title('14.3 m/s')
    ax[2].spines[['right','top']].set_visible(False)
    ax[2].set_ylabel('$C_{m}$')
    ax[2].set_xlim(xlims)
    ax[2].set_title('17 m/s')
    ax[2].set_xlabel('$C_{L}$')
    
    custom_lines1 = [Line2D([0], [0], color= b_color, lw=1),
                Line2D([0], [0], color= p_color, lw=1),
                Line2D([0], [0], color= f9v0_color, lw=1),
                Line2D([0], [0], color= f9v3_color, lw=1)]
    
    legend1 = fig.legend(custom_lines1,['Baseline','Plate','f9_v0','f9_v3'],
                         ncol= 4, loc=8, bbox_to_anchor=(0.55,-0.03), frameon=False)
    
    fig.tight_layout()
#%%
def load_and_save_data():
    
    clean_cal1 = np.loadtxt('Feather_Wing_Experiments/Calibration/CleanCal.txt')
    clean_cal1 = np.transpose(clean_cal1)
    
    clean_cal2 = np.loadtxt('Feather_Wing_Experiments/Calibration/Cal_matrix_07.txt')
    clean_cal2 = np.transpose(clean_cal2)
    
    df_cols = ['config','rpm','iter','AoA','L','D','m','CL','CD',
               'Cm']
    Aero_data = pd.DataFrame(columns=df_cols)
    
    base_rpms = [200, 250, 300]
    plate_rpms = [200, 250, 300]
    f9_v0_rpms = [200, 250,300]
    f9_v150_rpms = [300]
    f9_v200_rpms = [300]
    f9_v300_rpms = [200, 250, 300]
    
    Aero_data = load_and_save_data_one_config('Feather_Wing_Experiments/Force_Data/Baseline/rpm',
                                  'base', clean_cal1, clean_cal2, Aero_data, df_cols, base_rpms)
    print('Base Done')
    Aero_data.to_csv('Base_wing_data_full_set_adj.csv',sep = ',', index=False)
    Aero_data = pd.DataFrame(columns=df_cols)
    
    Aero_data = load_and_save_data_one_config('Feather_Wing_Experiments/Force_Data/Plate/rpm',
                                  'plate', clean_cal1, clean_cal2, Aero_data, df_cols, plate_rpms,
                                  c_path='_plate')
    print('Plate Done')
    Aero_data.to_csv('Plate_wing_data_full_set_adj.csv',sep = ',', index=False)
    Aero_data = pd.DataFrame(columns=df_cols)
    
    Aero_data = load_and_save_data_one_config('Feather_Wing_Experiments/Force_Data/Feather9/rpm',
                                  'f9_v0', clean_cal1, clean_cal2, Aero_data, df_cols, f9_v0_rpms,
                                  c_path='_feather9_volt0')
    print('F9 Volt0 Done')
    Aero_data.to_csv('F9v0_wing_data_full_set_adj.csv',sep = ',', index=False)
    Aero_data = pd.DataFrame(columns=df_cols)
    
    Aero_data = load_and_save_data_one_config('Feather_Wing_Experiments/Force_Data/Feather9/rpm',
                                  'f9_v15', clean_cal1, clean_cal2, Aero_data, df_cols, f9_v150_rpms,
                                  c_path='_feather9_volt150')
    print('F9 Volt150 Done')
    Aero_data.to_csv('F9v15_wing_data_full_set.csv',sep = ',', index=False)
    Aero_data = pd.DataFrame(columns=df_cols)
    
    Aero_data = load_and_save_data_one_config('Feather_Wing_Experiments/Force_Data/Feather9/rpm',
                                  'f9_v2', clean_cal1, clean_cal2, Aero_data, df_cols, f9_v200_rpms,
                                  c_path='_feather9_volt200')
    print('F9 Volt200 Done')
    
    Aero_data.to_csv('F9v2_wing_data_full_set_adj.csv',sep = ',', index=False)
    Aero_data = pd.DataFrame(columns=df_cols)
    
    Aero_data = load_and_save_data_one_config('Feather_Wing_Experiments/Force_Data/Feather9/rpm',
                                  'f9_v3', clean_cal1, clean_cal2, Aero_data, df_cols, f9_v300_rpms,
                                  c_path='_feather9_volt300')
    print('F9 Volt300 Done')
    Aero_data.to_csv('F9v3_wing_data_full_set_adj.csv',sep = ',', index=False)
    return Aero_data
#%%
def load_and_save_data_one_config(file_path, config_str, clean_cal1, clean_cal2, 
                                  Aero_data, df_cols, rpms, c_path=''):
    alphas = np.arange(-5, 21)
    iters = [0,1,2,3]
    if config_str == 'f9_v2':
        iters = [0,1]
    
    deg2rad = np.pi/180
    data_len = 4000

    c_list = np.array([config_str for _ in range(data_len)]).reshape((-1,1))
    get_esa = False
    for rpm in rpms:
        rpm_str = str(rpm)
        rpm_list = np.ones((data_len,1))*rpm
        vel = rpm*0.055+0.525
        if config_str == 'base':
            chord = 0.125
            qS = 0.5*1.225*vel*vel*0.4572*chord
        elif config_str == 'plate':
            chord = 0.132
            qS = 0.5*1.225*vel*vel*0.4572*chord
        elif 'f9' in config_str:
            chord = 0.127
            qS = 0.5*1.225*vel*vel*0.4572*chord
            get_esa = True
        else:
            chord=0.127
            qS = 0.5*1.225*vel*vel*0.4572*chord
            
        if rpm == 300:
            clean_cal = clean_cal2
        elif config_str == 'f9_v15':
            clean_cal = clean_cal2
        else:
            clean_cal = clean_cal1
            
        for i in iters:
            iter_str = str(i)
            iter_list = np.array([iter_str for _ in range(data_len)]).reshape((-1,1))
            for a in alphas:
                a_list = np.ones((data_len,1))*a
                rot_matrix = -np.array([[np.cos(a*deg2rad), -np.sin(a*deg2rad)],
                                       [np.sin(a*deg2rad), np.cos(a*deg2rad)]])
                
                alph_str = str(a)
                
                
                
                ## BASELINE DATA
                
                # Tare
                tare = pd.read_csv(file_path+rpm_str+'/rpm'+rpm_str+c_path+
                                      '_angle'+alph_str+'_iter'+iter_str+
                                      '_Tare.csv',skiprows=22,delimiter=',')
                tare.drop(['X_Value', 'Comment'], axis=1, inplace=True)
                
                tare_df = rotate_and_combine(tare, clean_cal, rot_matrix)
                tare_mean = tare_df.mean(axis=0)
                
                
                # Testing
                
                test = pd.read_csv(file_path+rpm_str+'/rpm'+rpm_str+c_path+
                                      '_angle'+alph_str+'_iter'+iter_str+
                                      '.csv',skiprows=22, delimiter=',')
                test.drop(['X_Value', 'Comment'], axis=1, inplace=True)
                
                test_df = rotate_and_combine(test, clean_cal, rot_matrix)
                
                df = test_df - tare_df
                
                
         

                
                ### Drop first 1000 rows
                df.drop(df.index[:1000], inplace=True)
     
                #esa_df.drop(esa_df.index[:1000], inplace=True)
                
                ## Put data into large df
  
                L = np.array(df[0]).reshape((-1,1))
                D = np.array(df[1]).reshape((-1,1))
                m = np.array(df[5]).reshape((-1,1))
                
                data = np.concatenate((c_list,rpm_list,iter_list,
                                          a_list,L, D, m, 
                                          L/qS,D/qS,m/qS/chord),
                                          axis=1)
                
                
                    
                Aero_data=pd.concat([Aero_data,pd.DataFrame(data,columns = df_cols)], 
                                              ignore_index=True)
                
                Aero_data = drop_outliers_iqr(Aero_data, 'CL')
                Aero_data = drop_outliers_iqr(Aero_data, 'CD')
                Aero_data = drop_outliers_iqr(Aero_data, 'Cm')
             
            print(config_str,', rpm: ', rpm_str, ', iter: ', iter_str, ' DONE')
            
    
    print(Aero_data.head())
    return Aero_data
#%%
def rotate_and_combine(load_cell_df, clean_cal, rot_matrix):
    forces = load_cell_df.dot(clean_cal)
    LD = forces[[0,1]].dot(rot_matrix)
    Aero_df = pd.concat([LD, forces[[5]]],axis=1)
        
    return Aero_df
    
#%%


if __name__=="__main__":
    run(load_data = False)
    print('Done')
