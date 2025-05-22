# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np

class flyer_state_resp:
    def __init__(self, uc, a, q, th, mass, dt):
        
        self.uc = uc
        
        self.a = a
        self.u = self.uc*np.cos(self.a)
        self.w = self.uc*np.sin(self.a)
        
        self.q = q 
        self.th = th 
        self.gamma = th - a
        self.m = mass
        self.dt = dt
        
        # nonsense
        self.x = 0
        self.z = 0
        
        #nonsense
        self.wc = 0
        self.xc = 0
        self.zc = 0
        
        self.ue = self.uc*np.cos(self.gamma)
        self.we = self.uc*np.sin(self.gamma)
        self.xe = 0
        self.ze = 0
        self.alt = -self.ze
        
        self.a_dot = 0
        self.g_dot = 0
        
        self.body = np.array([self.x, self.z, self.u, self.w, self.a, self.a_dot,
                              self.q, self.th, self.gamma, self.g_dot])
        
        self.wind = np.array([self.xc, self.zc, self.uc, self.wc, self.a, self.a_dot,
                              self.q, self.th, self.gamma, self.g_dot])
        
        self.earth = np.array([self.xe, self.alt, self.ue, self.we, self.a,self.a_dot,
                               self.q, self.th, self.gamma, self.g_dot])
        
        self.due = 0
        self.dwe = 0
        
        self.ug = 0
        self.wg = 0
        
        self.uc_g = 0
        self.gamma_g = 0
        
    def step_pitch(self, L, D, T, dq):
        
        self.T = T
        
        Fxc_T =  - D - 9.81*self.m*np.sin(self.gamma)
        T = -Fxc_T/np.cos(self.a)
        Fxc = T + Fxc_T
        Fzc = 9.81*self.m*np.cos(self.gamma) - L - T*np.sin(self.a)
        
        
        uc_dot = Fxc/self.m
        self.g_dot = -Fzc/self.m/self.uc
        
        self.uc += uc_dot*self.dt
        self.gamma += self.g_dot*self.dt
        
        
        xe_dot = self.uc*np.cos(self.gamma)
        ze_dot = -self.uc*np.sin(self.gamma)
        
        self.xe +=  xe_dot*self.dt
        self.ze += ze_dot*self.dt
        
        self.ue = xe_dot
        self.we = ze_dot
        
        q_ = self.q + dq*self.dt
        
        self.th +=  (q_ + self.q)*self.dt/2
        self.q = q_
        
        a_ = self.th - self.gamma
        self.a_dot = (a_ - self.a)/self.dt
        self.a = a_
        
        self.alt = -self.ze
        
        self.body = np.array([self.x, self.z, self.u, self.w, self.a,self.a_dot,
                              self.q, self.th, self.gamma, self.g_dot])
        
        self.wind = np.array([self.xc, self.zc, self.uc, self.wc, self.a,self.a_dot,
                              self.q, self.th, self.gamma, self.g_dot])
        
        self.earth = np.array([self.xe, self.alt, self.ue, self.we, self.a,self.a_dot,
                               self.q, self.th, self.gamma, self.g_dot])
        
        
    def step_pitch_pert(self, dL, dD, dT, dq):
        
        
        
        duc_no_T = (- dD - 9.81*self.m*np.sin(self.gamma)*self.q )*self.dt/self.m
        #dT = -duc_no_T
        duc = duc_no_T + dT
        dwc = (9.81*self.m*np.sin(self.gamma)*self.q-dL-dT*np.sin(self.a))*self.dt/self.m
        
        self.q += dq*self.dt
        self.th += self.q*self.dt
        self.gamma = np.arctan(-dwc/(self.uc+duc))
        self.a = self.th - self.gamma
        
        self.uc = np.sqrt((self.uc+duc)**2+dwc**2)
        
        self.u += self.uc*np.cos(self.a)
        self.w += self.uc*np.sin(self.a)
        
        ue = self.uc*np.cos(self.gamma)
        we = self.uc*np.sin(-self.gamma)
        
        self.xe += (ue + self.ue)*self.dt/2 #averaged
        self.ze += (we + self.we)*self.dt/2 #averaged
        self.alt = -self.ze
        
        self.ue = ue
        self.we = we
        
       
        # Note x, z, xc, zc, wc are nonsense since the axes move and rotate
        
        self.body = np.array([self.x, self.z, self.u, self.w, self.a,
                              self.q, self.th, self.gamma])
        
        self.wind = np.array([self.xc, self.zc, self.uc, self.wc, self.a,
                              self.q, self.th, self.gamma])
        
        self.earth = np.array([self.xe, self.alt, self.ue, self.we, self.a,
                               self.q, self.th, self.gamma])
        
    def step_pitch_earth(self, L, D, T, dq):
         
         self.T = T
        
         #This is in earth frame
         Fxe = T*np.cos(self.th) - D*np.cos(self.gamma) + L*np.sin(self.gamma) 
         Fze = 9.81*self.m - L*np.cos(self.gamma) - D*np.sin(self.gamma)
         
         ue_ = self.ue + Fxe*self.dt/self.m
         we_ = self.we + Fze*self.dt/self.m
         
         self.xe += (ue_ + self.ue)*self.dt/2
         self.ze += (we_ + self.we)*self.dt/2
         self.alt = -self.ze
         
         self.ue = ue_
         self.we = we_ 
         
         U_mag = np.sqrt(ue_**2 + we_**2)
         
         self.gamma = np.arctan(-we_/ue_)
         
         q_ = self.q + dq*self.dt
         
         self.th += (q_ + self.q)*self.dt/2
         
         self.a = self.th - self.gamma
         
         
         self.body = np.array([self.x, self.z, self.u, self.w, self.a,
                               self.q, self.th, self.gamma])
         
         self.wind = np.array([self.xc, self.zc, self.uc, self.wc, self.a,
                               self.q, self.th, self.gamma])
         
         self.earth = np.array([self.xe, self.alt, self.ue, self.we, self.a,
                                self.q, self.th, self.gamma])
        
    def step_gust_pert(self, dL, dD, dT, dq, ug, Ag):
            
        dug_e = ug*np.cos(Ag)
        dwg_e = ug*np.sin(Ag)
        
        dug_c = ug*np.cos(Ag+self.gamma)
        dwg_c = ug*np.sin(Ag+self.gamma)
        
        

      
        duc = (dT*np.cos(self.a)- dD - 9.81*self.m*np.sin(self.gamma)*self.q )*self.dt/self.m
        dwc = (9.81*self.m*np.sin(self.gamma)*self.q-dL-dT*np.sin(self.a))*self.dt/self.m
        
        self.q += dq*self.dt
        self.th += self.q*self.dt
        self.gamma = np.arctan(-dwc/(self.uc+duc))
        self.uc = np.sqrt((self.uc+duc+dug_c)**2+(dwc+dwg_c)**2)
        no_gust_AoA = self.th - self.gamma
        da_g = np.arctan((dwg_c)/(self.uc+dug_c))
        self.a = no_gust_AoA + da_g
        
        self.u += self.uc*np.cos(self.a)
        self.w += self.uc*np.sin(self.a)
        
        ue = self.uc*np.cos(self.gamma)
        we = self.uc*np.sin(-self.gamma)
        
        self.xe += (ue + self.ue)*self.dt/2 #averaged
        self.ze += (we + self.we)*self.dt/2 #averaged
        self.alt = -self.ze
        
        self.ue = ue
        self.we = we
        
       
        # Note x, z, xc, zc, wc are nonsense since the axes move and rotate
        
        self.body = np.array([self.x, self.z, self.u, self.w, self.a,
                              self.q, self.th, self.gamma])
        
        self.wind = np.array([self.xc, self.zc, self.uc, self.wc, self.a,
                              self.q, self.th, self.gamma])
        
        self.earth = np.array([self.xe, self.alt, self.ue, self.we, self.a,
                               self.q, self.th, self.gamma])
            
     
    def step_gust(self, L, D, T, dq, Ug, Ag):
         
         self.T = T
         u_gust = Ug*np.cos(Ag)
         w_gust = Ug*np.sin(Ag)
         
         
         #This is in earth frame
         Fxe = T*np.cos(self.th) - D*np.cos(self.gamma) + L*np.sin(self.gamma) 
         Fze = 9.81*self.m - L*np.cos(self.gamma) - D*np.sin(self.gamma)
         
         ue_ = self.ue + Fxe*self.dt/self.m
         we_ = self.we + Fze*self.dt/self.m
         
         self.xe += (ue_ + self.ue)*self.dt/2
         self.ze += (we_ + self.we)*self.dt/2
         self.alt = -self.ze
         
         self.ue = ue_
         self.we = we_ 
         
         self.g_dot = 0
         self.a_dot = 0
         
         self.gamma = np.arctan(-we_/ue_)
         
         q_ = self.q + dq*self.dt
         
         self.th += (q_ + self.q)*self.dt/2
         
         a_no_gust = self.th - self.gamma
         
         u_w_gust = ue_ + u_gust
         w_w_gust = we_ + w_gust
         
         self.uc = np.sqrt(u_w_gust**2 + w_w_gust**2)
         self.a = self.th + np.arctan(w_w_gust/u_w_gust) 
         
         
         self.body = np.array([self.x, self.z, self.u, self.w, self.a,self.a_dot,
                               self.q, self.th, self.gamma, self.g_dot])
         
         self.wind = np.array([self.xc, self.zc, self.uc, self.wc, self.a,self.a_dot,
                               self.q, self.th, self.gamma, self.g_dot])
         
         self.earth = np.array([self.xe, self.alt, self.ue, self.we, self.a,self.a_dot,
                                self.q, self.th, self.gamma, self.g_dot])
    def step_gust2(self, L, D, T, dq, Ug, Ag):
        
       self.T = T
       u_gust = Ug*np.cos(Ag)
       w_gust = Ug*np.sin(Ag)
       
       uc_gust = Ug*np.cos(Ag+self.gamma)
       wc_gust = Ug*np.sin(Ag+self.gamma)
       
       Fxc_T =  - D - 9.81*self.m*np.sin(self.gamma)
       T = -Fxc_T/np.cos(self.a)
       Fxc = T + Fxc_T
       Fzc = 9.81*self.m*np.cos(self.gamma) - L - T*np.sin(self.a)
       
       
       uc_dot = Fxc/self.m
       self.g_dot = (-Fzc/self.m)/self.uc
       
       self.uc += uc_dot*self.dt 
       self.uc_g = self.uc + uc_gust
       self.gamma += self.g_dot*self.dt 
       self.gamma_g = self.gamma - wc_gust/self.uc
       
       
       xe_dot = self.uc*np.cos(self.gamma)
       ze_dot = -self.uc*np.sin(self.gamma)
       
       self.xe +=  xe_dot*self.dt
       self.ze += ze_dot*self.dt
       
       self.ue = xe_dot
       self.we = ze_dot
       
       q_ = self.q + dq*self.dt
       
       self.th +=  (q_ + self.q)*self.dt/2
       self.q = q_
       
       a_ = self.th - self.gamma_g
       self.a_dot = (a_ - self.a)/self.dt
       self.a = a_
       
       self.alt = -self.ze
       
       self.body = np.array([self.x, self.z, self.u, self.w, self.a,self.a_dot,
                             self.q, self.th, self.gamma, self.g_dot])
       
       self.wind = np.array([self.xc, self.uc, self.uc_g, self.wc, self.a,self.a_dot,
                             self.q, self.th, self.gamma_g, self.g_dot])
       
       self.earth = np.array([self.xe, self.alt, self.ue, self.we, self.a,self.a_dot,
                              self.q, self.th, self.gamma, self.g_dot])
        
         
#%%
