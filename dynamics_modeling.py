# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 15:34:41 2024

@author: kevin.p.haughn
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tools.eval_measures import rmse
import sympy as sym
from sympy import nsolve, Symbol 

from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec

from wing_pos_model import flyer_state_resp

def run(load_lins=False):
    
    if load_lins:
        load_and_get_lins()
    
    F9v0_CL_lin= pd.read_csv('F9v0_CL_lin.csv')
    F9v0_CD_lin= pd.read_csv('F9v0_CD_lin.csv')
    F9v0_CL_se= pd.read_csv('F9v0_CL_se.csv')
    F9v0_CD_se= pd.read_csv('F9v0_CD_se.csv')
    
    F9v3_CL_lin= pd.read_csv('F9v3_CL_lin.csv')
    F9v3_CD_lin= pd.read_csv('F9v3_CD_lin.csv')
    F9v3_CL_se= pd.read_csv('F9v3_CL_se.csv')
    F9v3_CD_se= pd.read_csv('F9v3_CD_se.csv')

    f9v0_color = '#4771b2'
    f9v3_color = '#c6793a'
    
    aT_v3_10 = aT_solve(11.525, 0.5, F9v3_CL_lin, F9v3_CD_lin)
    print('AoA: ', aT_v3_10[0]*180/np.pi)
    print('Thrust: ', aT_v3_10[1])

    
 # note X = [u, dAoA, dq, dtheta] 
 # note X_state = [Velocity, Angle of attack, q, theta]

    chord=0.127
    S = 0.4572*2*chord
    d2r = np.pi/180
    A0 = 3*d2r
    
    U0 = 11.525
    d2r = np.pi/180
    q0 = 0
    th0 = A0 # gamma = 0 for steady level flight

    mass = 0.7
    aT_v0_10 = aT_solve(11.525, mass, F9v0_CL_lin, F9v0_CD_lin)
    aT_v3_10 = aT_solve(11.525, mass, F9v3_CL_lin, F9v3_CD_lin)
    
    aT_v0_12 = aT_solve(14.275, mass, F9v0_CL_lin, F9v0_CD_lin)
    aT_v3_12 = aT_solve(14.275, mass, F9v3_CL_lin, F9v3_CD_lin)
    
    aT_v0_15 = aT_solve(17.025, mass, F9v0_CL_lin, F9v0_CD_lin)
    aT_v3_15 = aT_solve(17.025, mass, F9v3_CL_lin, F9v3_CD_lin)
    
    fig, ax = plt.subplots(6,1, figsize=(3,6.5), dpi=300)
    
    
    fig1 = plt.figure(layout='constrained', figsize=(4,3.5), dpi=300)
    gs = GridSpec(2,2, figure=fig1)
    
    ax1 = fig1.add_subplot(gs[0,:])
    ax2 = fig1.add_subplot(gs[1,0])
    ax3 = fig1.add_subplot(gs[1,1])
    
    run_pitch_sim(11.525, aT_v0_10[0], F9v0_CL_lin, F9v0_CD_lin, ax,ax1, ax2, ax3, '-', f9v0_color, m=mass,
                  CL_std=F9v0_CL_se,CD_std=F9v0_CD_se, ESA=False, opaque=0.3)
    run_pitch_sim(14.275, aT_v0_12[0], F9v0_CL_lin, F9v0_CD_lin, ax,ax1, ax2, ax3, '-', f9v0_color,m=mass,
                  CL_std=F9v0_CL_se,CD_std=F9v0_CD_se,ESA=False, opaque=0.6)
    run_pitch_sim(17.025, aT_v0_15[0], F9v0_CL_lin, F9v0_CD_lin, ax,ax1, ax2, ax3, '-', f9v0_color,m=mass,
                  CL_std=F9v0_CL_se,CD_std=F9v0_CD_se,ESA=False, opaque=1)
    run_pitch_sim(11.525, aT_v3_10[0], F9v3_CL_lin, F9v3_CD_lin, ax,ax1, ax2, ax3, '-', f9v3_color,m=mass,
                  CL_std=F9v3_CL_se,CD_std=F9v3_CD_se,ESA=True, opaque=0.3)
    run_pitch_sim(14.275, aT_v3_12[0], F9v3_CL_lin, F9v3_CD_lin, ax,ax1, ax2, ax3, '-', f9v3_color,m=mass,
                  CL_std=F9v3_CL_se,CD_std=F9v3_CD_se,ESA=True, opaque=0.6)
    run_pitch_sim(17.025, aT_v3_15[0], F9v3_CL_lin, F9v3_CD_lin, ax,ax1, ax2, ax3, '-', f9v3_color,m=mass,
                  CL_std=F9v3_CL_se,CD_std=F9v3_CD_se,ESA=True, opaque=1)
    
    
    fig.tight_layout()
    fig1.tight_layout()
    fig1.savefig('flight_plots', dpi=300)
    
def aT_solve(v, m, linCl, linCd):
    a = Symbol('a')
    T = Symbol('T')
    
    w = m*9.81
    
    chord=0.127
    S = 0.4572*2*chord
    qS = 0.5*1.225*v*v*S
    d2r = np.pi/180
    
    f1 = T*sym.cos(a) - qS*(linCd.loc[0] + linCd.loc[1]*a + linCd.loc[2]*v + linCd.loc[3]*a*v + linCd.loc[4]*a*a + linCd.loc[5]*v*v)
    f2 = w - T*sym.sin(a) - qS*(linCl.loc[0] + linCl.loc[1]*a + linCl.loc[2]*v + linCl.loc[3]*a*v + linCl.loc[4]*a*a + linCl.loc[5]*v*v)
    
    aT = nsolve((f1,f2), (a, T), (1, 5))
    aT = aT.evalf()
    
    aT = np.array([aT[0], aT[1]], dtype = np.float64)
    print('Testing Trim for: ', aT)
    test_trim(v,m,linCl, linCd, aT)
  
    return aT
def test_trim(v,m,linCl, linCd, at_mat):
    a = at_mat[0]
    T = at_mat[1]
    
    w = m*9.81
    
    chord=0.127
    S = 0.4572*2*chord
    qS = 0.5*1.225*v*v*S
    d2r = np.pi/180
    
    f1 = T*sym.cos(a) - qS*(linCd.loc[0] + linCd.loc[1]*a + linCd.loc[2]*v + linCd.loc[3]*a*v + linCd.loc[4]*a*a + linCd.loc[5]*v*v)
    f2 = w - T*sym.sin(a) - qS*(linCl.loc[0] + linCl.loc[1]*a + linCl.loc[2]*v + linCl.loc[3]*a*v + linCl.loc[4]*a*a + linCl.loc[5]*v*v)
    
    print(f1, f2)
    
    
def run_pitch_sim(U0, A0, CL_model, CD_model, ax, ax1,ax2,ax3, ls, color, m=False, thrust = False,
                  CL_std=0, CD_std=0, ESA=True, opaque=1):
    
    d2r = np.pi/180
    chord=0.127
    S = 0.4572*2*chord
    th0 = A0
    # Setting Trim Condition
    
    
    qS = 0.5*1.225*U0*U0*S
    X_state_t = np.array([U0, A0, 0, th0])
    
    CL_t = lin_(CL_model, X_state_t)
    CD_t = lin_(CD_model, X_state_t)
    CL_tmax = lin_(CL_model, X_state_t) + 1.96*CL_std['0'][0]
    CD_tmax = lin_(CD_model, X_state_t) + 1.96*CD_std['0'][0]
    CL_tmin = lin_(CL_model, X_state_t) - 1.96*CL_std['0'][0]
    CD_tmin = lin_(CD_model, X_state_t) - 1.96*CD_std['0'][0]
    

    mass = m
    mass_max = m
    mass_min = m

    T = qS*(CD_t*np.cos(A0)-CL_t*np.sin(A0))+ 9.81*mass*np.sin(th0)
    T_max = qS*(CD_tmax*np.cos(A0)-CL_tmax*np.sin(A0))+ 9.81*mass*np.sin(th0)
    T_min = qS*(CD_tmin*np.cos(A0)-CL_tmin*np.sin(A0))+ 9.81*mass*np.sin(th0)
    
    q0 = 0
    th0 = A0 # gamma = 0 for steady level flight
    # X_state: [U, AoA, q, th]
    X_state = np.array([U0, A0, 0, th0])
    
    CL_ = lin_(CL_model, X_state)
    CD_ = lin_(CD_model, X_state)
    CL_max = lin_(CL_model, X_state) + 1.96*CL_std['0'][0]
    CD_max = lin_(CD_model, X_state) + 1.96*CD_std['0'][0]
    CL_min = lin_(CL_model, X_state) - 1.96*CL_std['0'][0]
    CD_min = lin_(CD_model, X_state) - 1.96*CD_std['0'][0]
    
    qS = 0.5*1.225*U0*U0*S
    
    
    #if m:
     #   mass = m
      #  T = thrust
    
    
    t_total = 7
    dt = 0.05
    t = np.arange(0,t_total,dt)
    
    # Call wing positioning class
    
    wing = flyer_state_resp(U0, A0, q0, th0, mass, dt)
    wing_max = flyer_state_resp(U0, A0, q0, th0, mass_max, dt)
    wing_min = flyer_state_resp(U0, A0, q0, th0, mass_min, dt)
    
    # Body State: [x, z, u, w, a, q, th, gamma])
    state_b = wing.body
    # Wind/control State: [xc, zc, uc, wc, a, q, th, gamma])
    state_c = wing.wind
    # Earth State: [xe, alt, ue, we, a, q, th, gamma])
    state_e = wing.earth
    
    
    state_b_set = np.zeros((len(t),len(state_b)))
    state_c_set = np.zeros((len(t),len(state_c)))
    state_e_set = np.zeros((len(t),len(state_e)))
    
    state_b_set[0,:] = state_b
    state_c_set[0,:] = state_c
    state_e_set[0,:] = state_e
    _L = qS*CL_
    _D = qS*CD_
    dT = 0
    
    # Body State: [x, z, u, w, a, q, th, gamma])
    state_b_max = wing_max.body
    # Wind/control State: [xc, zc, uc, wc, a, q, th, gamma])
    state_c_max = wing_max.wind
    # Earth State: [xe, alt, ue, we, a, q, th, gamma])
    state_e_max = wing_max.earth
    
    
    state_b_set_max = np.zeros((len(t),len(state_b)))
    state_c_set_max = np.zeros((len(t),len(state_c)))
    state_e_set_max = np.zeros((len(t),len(state_e)))
    
    state_b_set_max[0,:] = state_b_max
    state_c_set_max[0,:] = state_c_max
    state_e_set_max[0,:] = state_e_max
    
    _L_max = qS*CL_
    _D_max = qS*CD_
    
    # Body State: [x, z, u, w, a, q, th, gamma])
    state_b_min = wing_min.body
    # Wind/control State: [xc, zc, uc, wc, a, q, th, gamma])
    state_c_min = wing_min.wind
    # Earth State: [xe, alt, ue, we, a, q, th, gamma])
    state_e_min = wing_min.earth
    
    
    state_b_set_min = np.zeros((len(t),len(state_b)))
    state_c_set_min = np.zeros((len(t),len(state_c)))
    state_e_set_min = np.zeros((len(t),len(state_e)))
    
    state_b_set_min[0,:] = state_b_min
    state_c_set_min[0,:] = state_c_min
    state_e_set_min[0,:] = state_e_min
    
    _L_min = qS*CL_
    _D_min = qS*CD_
    
    start_ind = 0
    a_max = 10*d2r
    chk = 0
    q_stop = False
    alphas = A0*np.ones((start_ind,))
    alphas_ = np.linspace(A0,a_max, len(t)-start_ind)
    alphas = np.append(alphas, alphas_)
    alphasX = a_max*np.ones((len(t),))
    
    err_ = 0
    ierr = 0
    a_ = wing.a
    for i in range(len(t)):
        ii = i+1
        
        if ii == len(t):
            break
        
        X_state = np.array([state_c[2], state_c[4], state_c[5], state_c[6]])
        #print(X_state)
        #print(F9v0_CL_lin)
        CL_ = lin_(CL_model, X_state)
        CD_ = lin_(CD_model, X_state)
        CL_max = lin_(CL_model, X_state) + 1.96*CL_std['0'][0]
        CD_max = lin_(CD_model, X_state) + 1.96*CD_std['0'][0]
        CL_min = lin_(CL_model, X_state) - 1.96*CL_std['0'][0]
        CD_min = lin_(CD_model, X_state) - 1.96*CD_std['0'][0]
        
        u = X_state[0]
        qS = 0.5*1.225*u*u*S
        
        L_ = qS*CL_
        D_ = qS*CD_
        L_max = qS*CL_max
        D_max = qS*CD_max
        L_min = qS*CL_min
        D_min= qS*CD_min
            
        a_dot_max = 0.01*2*u/chord
        a_max = 15*d2r
        max_dq = 360*d2r
        pGain = 40#20 #30
        iGain = 0.5#180#4#0
        dGain = 0.1#0.75#0
        
        
        q_count = 10 
        Tinc = 1.5
        count = 40
        
        #a_targ = alphas[i]
        a_targ = 10*d2r
        if i < start_ind:
            dq = 0*d2r
        elif q_stop:
            dq = 0
        elif wing.a < a_max:
            err = a_targ-wing.a
            derr = -(wing.a - a_)/dt
            ierr += err*dt
            #dq = max(min(1,(err*pGain + ierr/iGain + derr*dGain)),-1)*max_dq
            dq = max(min(max_dq,(err*pGain + (pGain/iGain)*ierr + pGain*dGain*derr)),-max_dq)
            #dq = 100*d2r
        else:
            dq = 0
     
        if wing.gamma > 2*np.pi/4:
            break
        dL = L_ - _L
        dD = D_ - _D
        
        a_ = wing.a
        wing.step_pitch(L_, D_, T, dq)
        wing_max.step_pitch(L_max, D_max, T, dq)
        wing_min.step_pitch(L_min, D_min, T, dq)
        
        state_b = wing.body
        state_c = wing.wind
        state_e = wing.earth
        
        state_b_set[ii,:] = state_b
        state_c_set[ii,:] = state_c
        state_e_set[ii,:] = state_e
        
        state_b_max = wing_max.body
        state_c_max = wing_max.wind
        state_e_max = wing_max.earth
        
        state_b_set_max[ii,:] = state_b_max
        state_c_set_max[ii,:] = state_c_max
        state_e_set_max[ii,:] = state_e_max
        
        state_b_min = wing_min.body
        state_c_min = wing_min.wind
        state_e_min = wing_min.earth
        
        state_b_set_min[ii,:] = state_b_min
        state_c_set_min[ii,:] = state_c_min
        state_e_set_min[ii,:] = state_e_min
        
        _L = L_
        _D = D_
    #opaque = 1
    ax[0].plot(t[:i],state_e_set[:i,1], ls= ls, color=color, alpha=opaque)
    ax[1].plot(t,alphasX/d2r, color='r')
    ax[1].plot(t[:i],state_c_set[:i,4]/d2r, ls= ls, color=color,alpha=opaque)
    ax[2].plot(t[:i],state_c_set[:i,5]/d2r, ls= ls, color=color,alpha=opaque)
    ax[3].plot(t[:i],state_c_set[:i,2], ls= ls, color=color,alpha=opaque)
    ax[4].plot(t[:i],state_c_set[:i,8]/d2r, ls= ls, color=color,alpha=opaque)
    ax[5].plot(t[:i],state_c_set[:i,9]/d2r, ls= ls, color=color,alpha=opaque)
    #ax[6].plot(t[:i],state_e_set[:i,7]/d2r, ls= ls, color=color,alpha=opaque)
    #ax[7].plot(state_e_set[:i,0],state_e_set[:i,1], ls= ls, color=color,alpha=opaque)
    
    xlims = [0,t_total]
    
    ax[0].spines[['right','top','bottom']].set_visible(False)
    ax[0].set_ylabel('altitude (m)')
    ax[0].set_xlim(xlims)
    ax[0].set_ylim([-0.5,30])
    ax[0].get_yaxis().set_ticks([0,15,30])
    ax[0].get_xaxis().set_ticks([])
    ax[1].spines[['right','top','bottom']].set_visible(False)
    ax[1].set_ylabel('AoA (deg)')
    ax[1].set_xlim(xlims)
    ax[1].set_ylim([0,12.2])
    ax[1].get_xaxis().set_ticks([])
    ax[1].get_yaxis().set_ticks([0,6,12])
    ax[2].spines[['right','top','bottom']].set_visible(False)
    ax[2].set_ylabel('a_dot (deg/s)')
    ax[2].get_xaxis().set_ticks([])
    ax[2].set_xlim(xlims)
    ax[2].set_ylim([-5,25])
    ax[2].get_yaxis().set_ticks([0,15,30])
    ax[3].spines[['right','top','bottom']].set_visible(False)
    ax[3].set_ylabel('Uc (m/s)')
    ax[3].get_xaxis().set_ticks([])
    ax[3].set_xlim(xlims)
    ax[3].set_ylim([10,20])
    ax[3].get_yaxis().set_ticks([10,15,20])
    ax[4].spines[['right','top','bottom']].set_visible(False)
    ax[4].set_ylabel('gamma (deg)')
    ax[4].get_xaxis().set_ticks([])
    ax[4].set_xlim(xlims)
    ax[4].set_ylim([-1,100])
    ax[4].get_yaxis().set_ticks([0,50,100])
    ax[5].spines[['right','top']].set_visible(False)
    ax[5].set_ylabel('gamma_dot (deg/s)')
    ax[5].set_xlim(xlims)
    ax[5].set_ylim([0,100])
    ax[5].get_yaxis().set_ticks([0,50,100])
    ax[5].set_xlabel('Time(sec)')

    
    
    ax1.plot(state_e_set[:i,0],state_e_set[:i,1], ls= ls, color=color,alpha=opaque)

    ax1.set_ylabel('Height (m)')
    ax1.set_xlabel('X-position (m)')
    ax1.spines[['right','top']].set_visible(False)
    ax1.set_xlim([0,70])
    ax1.set_ylim([-0.5,30])
    ax1.get_xaxis().set_ticks([0,35,70])
    ax1.get_yaxis().set_ticks([0,15,30])
    
    if ESA:
        width = 0.25
    else:
        width = -0.25
    inds = [0, width, width*2, width*3]
    vels = [11.525,14.275,17.025]
    
    
    
    y = np.array([[np.max(state_e_set[:,0])]])
    ymax = np.max(state_e_set_min[:,0]) - y[0]
    ymin = y[0] - np.max(state_e_set_max[:,0]) 
    yerr = np.array([ymin,ymax])
    
    
    ax2.errorbar(U0+width, np.max(state_e_set[:,0]),
                      yerr=yerr,
                   fmt='o',markersize=1, elinewidth=1, capsize = 2, capthick=0.5,
                   color=color, alpha=1 )
    
    print('x-max: ')
    print(np.max(state_e_set[:,0]))
       
    ax2.spines[['right','top']].set_visible(False)
    ax2.set_ylabel('$X_{max}$ (m)')
    ax2.get_xaxis().set_ticks([])
    ax2.set_ylim([10,75])
    ax2.set_xlim([10,18])
    ax2.get_yaxis().set_ticks([10,75])
    ax2.set_xlabel('Velocity (m/s)')
    ax2.get_xaxis().set_ticks([11.5,14.3,17])
     
       
    y = np.array([[np.max(state_c_set[:i,9])/d2r]])
    ymax = np.max(state_c_set_max[:i,9])/d2r - y[0]
    ymin = y[0] - np.max(state_c_set_min[:i,9])/d2r 
    yerr = np.array([ymin,ymax])
    
    ax3.errorbar(U0+width, np.max(state_c_set[:i,9])/d2r,
                      yerr=yerr,
                   fmt='o',markersize=1, elinewidth=1, capsize = 2, capthick=0.5,
                   color=color, alpha=1 )
    
    print('flight path angle')
    print(np.max(state_c_set[:i,9])/d2r)
    
    ax3.spines[['right','top']].set_visible(False)
    ax3.set_ylabel('$\gamma\u0307_{max}$ (deg/s)')
    ax3.set_xlabel('Velocity (m/s)')
    ax3.get_xaxis().set_ticks([11.5,14.3,17])
    ax3.set_ylim([50,90])
    ax3.set_xlim([10,18])
    ax3.get_yaxis().set_ticks([50,90])
    


def lin_(c, X):
    state = np.array([[1, X[1], X[0], X[0]*X[1], X[1]**2, X[0]**2]])
    out = np.dot(state, c)
    return out[0,0]

    
def load_and_get_lins():
    
    Base_df = pd.read_csv('Base_wing_data_full_set_vel_adjX.csv',sep = ',',
                              usecols=[0,1,2,3,4,8,9,10])
    Base_df['CD'] *= -1
    Base_df['Cm'] *= -1
    Base_df['vel'] = Base_df['rpm']*0.055+0.525
    print('Base Loaded')
    Plate_df = pd.read_csv('Plate_wing_data_full_set_vel_adjX.csv',sep = ',',
                              usecols=[0,1,2,3,4,8,9,10])
    Plate_df['CD'] *= -1
    Plate_df['Cm'] *= -1
    Plate_df['vel'] = Plate_df['rpm']*0.055+0.525
    print('Plate Loaded')
    F9v0_df = pd.read_csv('F9v0_wing_data_full_set_vel_adjX.csv',sep = ',',
                              usecols=[0,1,2,3,4,8,9,10])
    F9v0_df['CD'] *= -1
    F9v0_df['Cm'] *= -1
    F9v0_df['vel'] = F9v0_df['rpm']*0.055+0.525
    print('F9v0 Loaded')
    
    F9v3_df = pd.read_csv('F9v3_wing_data_full_set_vel_adjX.csv',sep = ',',
                              usecols=[0,1,2,3,4,8,9,10])
    F9v3_df['CD'] *= -1
    F9v3_df['Cm'] *= -1
    F9v3_df['vel'] = F9v3_df['rpm']*0.055+0.525
    
    print('F9v3 Loaded')
    
    coeff_df=pd.concat([Base_df,Plate_df,F9v0_df,F9v3_df], 
                                  ignore_index=True)
    
    
    print(coeff_df.head())
    CL_df = combine_iters_CL(coeff_df)
    CD_df = combine_iters_CD(coeff_df)

    
    F9v0_df = F9v0_df.loc[F9v0_df['AoA'] < 13]
    F9v3_df = F9v3_df.loc[F9v3_df['AoA'] < 13]
    F9v0_df['AoAr'] = F9v0_df['AoA']*np.pi/180
    F9v3_df['AoAr'] = F9v3_df['AoA']*np.pi/180
    Base_df = Base_df.loc[Base_df['AoA'] < 13]
    Base_df['AoAr'] = Base_df['AoA']*np.pi/180
    

    F9v0_CL_lins, F9v0_CL_se = lin_reg(F9v0_df, 'CL')
    F9v0_CD_lins, F9v0_CD_se = lin_reg(F9v0_df, 'CD')

    
    F9v3_CL_lins, F9v3_CL_se = lin_reg(F9v3_df, 'CL')
    F9v3_CD_lins, F9v3_CD_se = lin_reg(F9v3_df, 'CD')
    


    
    F9v0_CL_lins.to_csv('F9v0_CL_lin.csv',sep = ',', index=False)
    F9v0_CD_lins.to_csv('F9v0_CD_lin.csv',sep = ',', index=False)
    F9v0_CL_se.to_csv('F9v0_CL_se.csv', sep = ',', index=False)
    F9v0_CD_se.to_csv('F9v0_CD_se.csv', sep = ',', index=False)

                        
    F9v3_CL_lins.to_csv('F9v3_CL_lin.csv',sep = ',', index=False)
    F9v3_CD_lins.to_csv('F9v3_CD_lin.csv',sep = ',', index=False)
    F9v3_CL_se.to_csv('F9v3_CL_se.csv', sep = ',', index=False)   
    F9v3_CD_se.to_csv('F9v3_CD_se.csv', sep = ',', index=False)    

    
    
    CL_vs_AoA(CL_df, F9v0_CL_lins, F9v3_CL_lins)
    CD_vs_AoA(CD_df, F9v0_CD_lins, F9v3_CD_lins)
    
    # Pitching linreg
    F9v0_dyn_df = pd.read_csv('F9v0_dyn_pitch_data_full_set.csv',sep = ',',
                              usecols=[0,2,3,4,8,9,10,11,14])
    F9v0_dyn_df = F9v0_dyn_df.loc[F9v0_dyn_df['AoA']<13]
    #F9v0_dyn_df = F9v0_dyn_df.loc[abs(F9v0_dyn_df['q'])>0]
    F9v0_dyn_df['AoAr'] = F9v0_dyn_df['AoA']*np.pi/180
    
    F9v0_CL_ = np.zeros((len(F9v0_dyn_df),))
    F9v0_CD_ = np.zeros((len(F9v0_dyn_df),))

    F9v0_AoA_vel = F9v0_dyn_df[['AoAr','vel']]
    
    for i in range(len(F9v0_AoA_vel)):
        vec = F9v0_AoA_vel.iloc[i,:]
        a = vec[0]
        u = vec[1]
        st = np.array([1,a,u,u*a, a**2, u**2])
        F9v0_CL_[i] = np.dot(F9v0_CL_lins, st)
        F9v0_CD_[i] = np.dot(F9v0_CD_lins, st)

    
    F9v0_dyn_df['_CL'] = F9v0_CL_
    F9v0_dyn_df['_CD'] = F9v0_CD_

    F9v0_dyn_df['CL_'] = F9v0_dyn_df['CL']-F9v0_CL_
    F9v0_dyn_df['CD_'] = F9v0_dyn_df['CD']-F9v0_CD_


    print('F9v0 Loaded')
    
    F9v3_dyn_df = pd.read_csv('F9v3_dyn_pitch_data_full_set.csv',sep = ',',
                              usecols=[0,2,3,4,8,9,10,11,14])
    F9v3_dyn_df = F9v3_dyn_df.loc[F9v3_dyn_df['AoA']<13]
    F9v3_dyn_df['AoAr'] = F9v3_dyn_df['AoA']*np.pi/180
    
    F9v3_CL_ = np.zeros((len(F9v3_dyn_df),))
    F9v3_CD_ = np.zeros((len(F9v3_dyn_df),))

    F9v3_AoA_vel = F9v3_dyn_df[['AoAr','vel']]
    for i in range(len(F9v3_AoA_vel)):
        vec = F9v3_AoA_vel.iloc[i,:]
        a = vec[0]
        u = vec[1]
        st = np.array([1,a,u,u*a, a**2, u**2])
        F9v3_CL_[i] = np.dot(F9v3_CL_lins, st)
        F9v3_CD_[i] = np.dot(F9v3_CD_lins, st)

        
    F9v3_dyn_df['_CL'] = F9v3_CL_
    F9v3_dyn_df['_CD'] = F9v3_CD_
      
    F9v3_dyn_df['CL_'] = F9v3_dyn_df['CL'] - F9v3_CL_
    F9v3_dyn_df['CD_'] = F9v3_dyn_df['CD'] - F9v3_CD_

    print(F9v3_dyn_df.head())
    print('F9v3 Loaded')
    

#%%
def lin_reg(df, variable):
    X = np.column_stack((df['AoAr'], df['vel'], df['AoAr']*df['vel'],df['AoAr']**2,
                         df['vel']**2))
    X = sm.add_constant(X)
    y = df[variable]
  
    model = sm.OLS(y,X)
    results = model.fit()
    print(results.summary())
    print("RMSE: ", np.sqrt(results.mse_model), np.sqrt(results.mse_resid))
    
    #print('R2: ', results.rsquared)
    #print(results.params)
    
    return results.params, pd.DataFrame([results.scale**0.5])



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


#%%
def CL_vs_AoA(df, v0_x, v3_x):
    
    p_color = '#D5D5D7'
    f9v0_color = '#4771b2'
    f9v3_color = '#c6793a'
    
    AoAs = np.arange(-5,13)
    v0_lin = np.zeros((len(AoAs),3))
    v3_lin = np.zeros((len(AoAs),3))
    base_lin = np.zeros((len(AoAs),3))
    for ind, a_deg in enumerate(AoAs):
        a = a_deg*np.pi/180
        vel10_x = np.array([1,a,11.525,11.525*a, a**2, 11.525**2])
        vel12_x = np.array([1,a,14.275,14.275*a, a**2, 14.275**2])
        vel15_x = np.array([1,a,17.025,17.025*a, a**2, 17.025**2])
        
        v0_lin[ind,0] = np.dot(v0_x,vel10_x)
        v3_lin[ind,0] = np.dot(v3_x,vel10_x)

        v0_lin[ind,1] = np.dot(v0_x,vel12_x)
        v3_lin[ind,1] = np.dot(v3_x,vel12_x)

        v0_lin[ind,2] = np.dot(v0_x,vel15_x)
        v3_lin[ind,2] = np.dot(v3_x,vel15_x)

    
    #PLot the Cl vs Alpha curves and analyse slope and such
    df = df.loc[df['AoA']<13]
    CL_200 = pd.pivot_table(df.loc[df['rpm'] == 200],
                   index='AoA', columns = 'config', values='CL')
    CL_250 = pd.pivot_table(df.loc[df['rpm'] == 250],
                   index='AoA', columns = 'config', values='CL')
    CL_300 = pd.pivot_table(df.loc[df['rpm'] == 300],
                   index='AoA', columns = 'config', values='CL')
    

    CLvar_200 = pd.pivot_table(df.loc[df['rpm'] == 200],
                   index='AoA', columns = 'config', values='CLvar')
    CLvar_250 = pd.pivot_table(df.loc[df['rpm'] == 250],
                   index='AoA', columns = 'config', values='CLvar')
    CLvar_300 = pd.pivot_table(df.loc[df['rpm'] == 300],
                   index='AoA', columns = 'config', values='CLvar')
    
    fig, ax = plt.subplots(3,1, figsize=(2,5), dpi=300)
    
    
    ax[0].errorbar(CL_200.index, CL_200['f9_v0'], yerr=CLvar_200['f9_v0']**0.5,
                   linewidth=1, elinewidth=0.5, capsize=1, capthick=0.5,
                   color=f9v0_color)
    ax[0].plot(AoAs, v0_lin[:,0], ls='--', color=f9v0_color)
    ax[0].errorbar(CL_200.index, CL_200['f9_v3'], yerr=CLvar_200['f9_v3']**0.5,
                   linewidth=1, elinewidth=0.5, capsize=1, capthick=0.5,
                   color=f9v3_color)
    ax[0].plot(AoAs, v3_lin[:,0], ls='--', color=f9v3_color)


    ax[1].errorbar(CL_250.index, CL_250['f9_v0'], yerr=CLvar_250['f9_v0']**0.5,
                   linewidth=1, elinewidth=0.5, capsize=1, capthick=0.5,
                   color=f9v0_color)
    ax[1].plot(AoAs, v0_lin[:,1], ls='--', color=f9v0_color)
    ax[1].errorbar(CL_250.index, CL_250['f9_v3'], yerr=CLvar_250['f9_v3']**0.5,
                   linewidth=1, elinewidth=0.5, capsize=1, capthick=0.5,
                   color=f9v3_color)
    ax[1].plot(AoAs, v3_lin[:,1], ls='--', color=f9v3_color)

    
 
    ax[2].errorbar(CL_300.index, CL_300['f9_v0'], yerr=CLvar_300['f9_v0']**0.5,
                   linewidth=1, elinewidth=0.5, capsize=1, capthick=0.5,
                   color=f9v0_color)
    ax[2].plot(AoAs, v0_lin[:,2], ls='--', color=f9v0_color)
    ax[2].errorbar(CL_300.index, CL_300['f9_v3'], yerr=CLvar_300['f9_v3']**0.5,
                   linewidth=1, elinewidth=0.5, capsize=1, capthick=0.5,
                   color=f9v3_color)
    ax[2].plot(AoAs, v3_lin[:,2], ls='--', color=f9v3_color)

    xlims = [-6,12]
    
    
    ax[0].spines[['right','top','bottom']].set_visible(False)
    ax[0].set_ylabel('$C_{L}$')
    ax[0].set_xlim(xlims)
    ax[0].get_xaxis().set_ticks([])
    ax[0].get_yaxis().set_ticks([-0.5,0,0.5,1])
    ax[0].set_title('11.5 m/s')
    ax[1].spines[['right','top','bottom']].set_visible(False)
    ax[1].set_ylabel('$C_{L}$')
    ax[1].set_xlim(xlims)
    ax[1].get_xaxis().set_ticks([])
    ax[1].get_yaxis().set_ticks([-0.5,0,0.5,1])
    ax[1].set_title('14.3 m/s')
    ax[2].spines[['right','top']].set_visible(False)
    ax[2].set_ylabel('$C_{L}$')
    ax[2].get_yaxis().set_ticks([-0.5,0,0.5,1])
    ax[2].set_xlim(xlims)
    ax[2].set_title('17 m/s')
    ax[2].set_xlabel('Angle of Attack (deg)')
    
    custom_lines1 = [Line2D([0], [0], color= f9v0_color, lw=1),
                Line2D([0], [0], color= f9v3_color, lw=1)]
    
    legend1 = fig.legend(custom_lines1,['f9v0','f9v3'],
                         ncol= 2, loc=8, bbox_to_anchor=(0.55,-0.03), frameon=False)
    
    fig.tight_layout()
    
def CD_vs_AoA(df, v0_x, v3_x):
    b_color = '#221F20'
    p_color = '#D5D5D7'
    f9v0_color = '#4771b2'
    f9v3_color = '#c6793a'
    df = df.loc[df['AoA']<13]
    AoAs = np.arange(-5,13)
    v0_lin = np.zeros((len(AoAs),3))
    v3_lin = np.zeros((len(AoAs),3))
    for ind, a_deg in enumerate(AoAs):
        a = a_deg*np.pi/180
        vel10_x = np.array([1,a,11.525,11.525*a, a**2, 11.525**2])
        vel12_x = np.array([1,a,14.275,14.275*a, a**2, 14.275**2])
        vel15_x = np.array([1,a,17.025,17.025*a, a**2, 17.025**2])
        
        v0_lin[ind,0] = np.dot(v0_x,vel10_x)
        v3_lin[ind,0] = np.dot(v3_x,vel10_x)
        v0_lin[ind,1] = np.dot(v0_x,vel12_x)
        v3_lin[ind,1] = np.dot(v3_x,vel12_x)
        v0_lin[ind,2] = np.dot(v0_x,vel15_x)
        v3_lin[ind,2] = np.dot(v3_x,vel15_x)
    
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
    
    
    ax[0].errorbar(CD_200.index, CD_200['f9_v0'], yerr=CDvar_200['f9_v0']**0.5,
                   linewidth=1, elinewidth=0.5, capsize=1, capthick=0.5,
                   color=f9v0_color)
    ax[0].plot(AoAs, v0_lin[:,0], ls='--', color=f9v0_color)
    ax[0].errorbar(CD_200.index, CD_200['f9_v3'], yerr=CDvar_200['f9_v3']**0.5,
                   linewidth=1, elinewidth=0.5, capsize=1, capthick=0.5,
                   color=f9v3_color)
    ax[0].plot(AoAs, v3_lin[:,0], ls='--', color=f9v3_color)
    

    ax[1].errorbar(CD_250.index, CD_250['f9_v0'], yerr=CDvar_250['f9_v0']**0.5,
                   linewidth=1, elinewidth=0.5, capsize=1, capthick=0.5,
                   color=f9v0_color)
    ax[1].plot(AoAs, v0_lin[:,1], ls='--', color=f9v0_color)
    ax[1].errorbar(CD_250.index, CD_250['f9_v3'], yerr=CDvar_250['f9_v3']**0.5,
                   linewidth=1, elinewidth=0.5, capsize=1, capthick=0.5,
                   color=f9v3_color)
    ax[1].plot(AoAs, v3_lin[:,1], ls='--', color=f9v3_color)
    
 
    ax[2].errorbar(CD_300.index, CD_300['f9_v0'], yerr=CDvar_300['f9_v0']**0.5,
                   linewidth=1, elinewidth=0.5, capsize=1, capthick=0.5,
                   color=f9v0_color)
    ax[2].plot(AoAs, v0_lin[:,2], ls='--', color=f9v0_color)
    ax[2].errorbar(CD_300.index, CD_300['f9_v3'], yerr=CDvar_300['f9_v3']**0.5,
                   linewidth=1, elinewidth=0.5, capsize=1, capthick=0.5,
                   color=f9v3_color)
    ax[2].plot(AoAs, v3_lin[:,2], ls='--', color=f9v3_color)
    
    xlims = [-6,12]
    ylims = [0,0.3]
    
    
    ax[0].spines[['right','top','bottom']].set_visible(False)
    ax[0].set_ylabel('$C_{D}$')
    ax[0].set_xlim(xlims)
    ax[0].set_ylim(ylims)
    ax[0].get_xaxis().set_ticks([])
    ax[0].set_title('11.5 m/s')
    ax[1].spines[['right','top','bottom']].set_visible(False)
    ax[1].set_ylabel('$C_{D}$')
    ax[1].set_xlim(xlims)
    ax[1].set_ylim(ylims)
    ax[1].get_xaxis().set_ticks([])
    ax[1].set_title('14.3 m/s')
    ax[2].spines[['right','top']].set_visible(False)
    ax[2].set_ylabel('$C_{D}$')
    ax[2].set_xlim(xlims)
    ax[2].set_ylim(ylims)
    ax[2].set_title('17 m/s')
    ax[2].set_xlabel('Angle of Attack (deg)')
    
    custom_lines1 = [Line2D([0], [0], color= b_color, lw=1),
                Line2D([0], [0], color= p_color, lw=1),
                Line2D([0], [0], color= f9v0_color, lw=1),
                Line2D([0], [0], color= f9v3_color, lw=1)]
    
    legend1 = fig.legend(custom_lines1,['Baseline','Plate','f9v0','f9v3'],
                         ncol= 4, loc=8, bbox_to_anchor=(0.55,-0.03), frameon=False)
    
    fig.tight_layout()


#%%


if __name__=="__main__":
    run(load_lins=False)
    print('Done')