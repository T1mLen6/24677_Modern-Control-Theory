# Fill in the respective functions to implement the controller

# Import libraries
import numpy as np
from base_controller import BaseController
from scipy import signal, linalg
from util import *

# CustomController class (inherits from BaseController)
class CustomController(BaseController):

    def __init__(self, trajectory):

        super().__init__(trajectory)

        # Define constants
        # These can be ignored in P1
        self.lr = 1.39
        self.lf = 1.55
        self.Ca = 20000
        self.Iz = 25854
        self.m = 1888.6
        self.g = 9.81
        
        
        self.Kp_lat = 5
        self.Ki_lat = 0.5
        self.Kd_lat = 1
        
        self.Kp_lon = 160
        self.Ki_lon = 0.1
        self.Kd_lon = 8
        
        
        
        self.index_step_lat = 170
        self.index_step_lon = 2000
        
        
        
        self.index_nxt_lat = 0
        self.index_nxt_lon = 0
        
        self.intPsiErr = 0
        self.intXdotErr = 0
        
        self.pervPsiErr = 0
        self.pervXdotErr = 0
        
        self.cum_lat = 0
        self.cum_lon = 0

        self.i = 0
        # Add additional member variables according to your need here.

    def update(self, timestep):

        trajectory = self.trajectory

        lr = self.lr
        lf = self.lf
        Ca = self.Ca
        Iz = self.Iz
        m = self.m
        g = self.g

        # Fetch the states from the BaseController method
        delT, X, Y, xdot, ydot, psi, psidot = super().getStates(timestep)

        # Design your controllers in the spaces below. 
        # Remember, your controllers will need to use the states
        # to calculate control inputs (F, delta). 

        # ---------------|Lateral Controller|-------------------------
        """
        Please design your lateral controller below.
        .
        """
        index_step_lat = self.index_step_lat
        index_step_lon = self.index_step_lon
        
        sqindex, index = closestNode(X, Y, trajectory)
        #self.index_nxt_lat = index_nxt_lat
        if index + self.index_step_lat < len(trajectory):
            index_nxt_lat = index + self.index_step_lat
        else:
            index_nxt_lat = len(trajectory)-1
        
        arr1 = trajectory[index_nxt_lat, 1] - Y
        arr2 = trajectory[index_nxt_lat, 0] - X
        
        psi_nxt = np.arctan2(arr1, arr2)
        psi_err = wrapToPi(psi_nxt - psi)
        #print(psi_err)
        
        self.cum_lat += psi_err*delT
        
        delta = ((self.Kp_lat * psi_err) + 
                self.Ki_lat * self.cum_lat + 
                self.Kd_lat * (psi_err - self.pervPsiErr)/delT)
        
        self.pervPsiErr = psi_err

        # ---------------|Longitudinal Controller|-------------------------
        """
        Please design your longitudinal controller below.
 
        """
        
        if index + self.index_step_lon < len(trajectory):
            index_nxt_lon = index + self.index_step_lon
        else:
            index_nxt_lon = len(trajectory)-1
        
        arr1_lon = trajectory[index_nxt_lon, 1] - Y
        arr2_lon = trajectory[index_nxt_lon, 0] - X
        
        psi_nxt_lon = np.arctan2(arr1_lon, arr2_lon)
        psi_err_lon= wrapToPi(psi_nxt_lon - psi)
        
        ideal_velocity = 90
        
        
        
        dynamic_velocity = ideal_velocity / (abs(psi_err_lon)*6 + 1)
        
        self.i +=1
        xdot_err = (dynamic_velocity - xdot)
        self.cum_lon += xdot_err*delT
        
        F = ((self.Kp_lon * (abs(psi_err_lon)*3 + 1) * xdot_err) + 
              self.Ki_lon * self.cum_lon + 
              self.Kd_lon * (abs(xdot_err - self.pervXdotErr))/delT)
        
        # if self.i % 10 == 0:
        #     print("------", psi_err_lon)
        #     print("coeff:", (abs(psi_err_lon)*2 + 1))
        #     print("dynamic;", dynamic_velocity)
        #     print("speed:", F)
        
        self.pervXdotErr = xdot_err
        

        # Return all states and calculated control inputs (F, delta)
        return X, Y, xdot, ydot, psi, psidot, F, delta
