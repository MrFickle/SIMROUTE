     #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 23:10:59 2020
Version 31-01-2921  # Added testVrtx function
Version 29-01-2021
@author: manel grifoll (UPC-BarcelonaTech)
"""
""" Revision because np.complex is deprecated and doesn't work with numpy version
The aliases were originally deprecated in NumPy 1.20; for more details and guidance see the original release note at:
    https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations.
    
    Replaced np.complex with complex
"""

# Import simulation parameters from one of the available parameter files
# Currently using TUNIS_NICE parameters


from params import inc, LonMin, LonMax, LatMin, LatMax, name_Simu, t_ini, time_res, date_Ini, date_End, WEN_form, Lbp, DWT  # Import parameters for simulation between Tunis and Nice

# Import required libraries
import numpy as np
import math as math
import re
import os
import datetime
import time

################### First we create the mesh ###################


# x -> longtitude
# y -> latitude

# Convert grid step from minutes to degrees since longitude and latitude are measured in degrees
inc = inc/60.0    # Convert increment from minutes to degrees
# Calculate number of nodes in longitude (Nx) and latitude (Ny) directions
Nx = int(np.floor((LonMax-LonMin)/inc)+2)  # Number of nodes in longitude direction
Ny = int(np.floor((LatMax-LatMin)/inc)+2)  # Number of nodes in latitude direction

# Create arrays of longitude and latitude values for the grid
tira_lon = np.linspace(LonMin, LonMin + (Nx-1)*inc, Nx)  # Longitude values
tira_lat = np.linspace(LatMin, LatMin + (Ny-1)*inc, Ny)  # Latitude values

# Create node coordinates array with all grid points (lon, lat pairs)
# Fill nodes array with coordinate pairs
lon_nodes = np.tile(tira_lon, Ny)    # Repeats longitudes (fastest-varying)
lat_nodes = np.repeat(tira_lat, Nx)  # Repeats latitudes (slowest-varying)
# Create all nodes
nodes = np.column_stack((lon_nodes, lat_nodes))

inc = inc*60  # Convert increment back to minutes
LatMaxEfec = tira_lat[-1]  # Effective maximum latitude
LonMaxEfec = tira_lon[-1]  # Effective maximum longitude

# Load wave data if it exists
arx = 'in/' + name_Simu + '_wInt.npz'
if os.path.exists('in/' + name_Simu + '_wInt.npz'):
    if t_ini == 0:
        # Load all wave data if starting from t=0
        dat = np.load(arx)
        hs = dat['arr_0']  # Significant wave height
        dir = dat['arr_1']  # Wave direction
    else:
        # Load wave data and skip to t_ini
        dat = np.load(arx)
        hs1 = dat['arr_0'] # significant wave height
        dir1 = dat['arr_1'] # wave direction

        # Load wave data and remove initial time steps if t_ini > 0
        if time_res == 1:
            tt = range(0, t_ini)
        else:
            tt = range(0, int(t_ini/3))

        # Remove initial time steps from wave data
        hs = np.delete(hs1, tt, axis=1) # significant wave height
        del hs1
        dir = np.delete(dir1, tt, axis=1) # wave direction
        del dir1

def arxW():
    """Generate output filename for wave data based on simulation name and date range"""
    d1 = datetime.date(date_Ini[0],date_Ini[1],date_Ini[2]) # Date start
    d2 = datetime.date(date_End[0],date_End[1],date_End[2]) # Date end
    nomw1=d1.strftime('%Y-%m-%d') # Date start time
    #  dt_min= nomw1+'T00:00:00'
    nomw2=d2.strftime('%Y-%m-%d') # Date end time
    # dt_max= nomw2+'T23:59:59'
    nomarx='Waves_'+name_Simu+'_'+nomw1+'%'+nomw2+'.nc'  # Wave data filename

    return nomarx


# def ferNoms(data1,data2,p):
#     '''funcio nomArx(datein,date end,p)  normalment  llegira del params
#     donara una llista amb els noms dels arxius de sortida p el nom fix, producte oo nom de la simu
#     '''
#     Larx=[]
#     d1 = datetime.date(data1[0],data1[1],data1[2])
#     d2 = datetime.date(data2[0],data2[1],data2[2])
#     incDies=datetime.timedelta(days=1)
#     for i in range((d2-d1).days+1):
#         d=d1+incDies*i
#         dt_min= d.strftime('%Y-%m-%d')+'T00:00:00'
#         dt_max= d.strftime('%Y-%m-%d')+'T23:59:59'
#         Larx.append('Waves-'+p+'_'+d.strftime('%Y%m%d')+'.nc')
#     return Larx
# def segonaRepe(cadena, elem):
#     #funcio dona la posicio de la posicio de la repeticio d0'un carcater
#     count = 0
#     for index,char in enumerate(cadena):
#         if char == elem:
#             count +=1
#             if count == 2:
#                return index
#     return -1 # si no hi res


def cart2compass(deg):
    """Convert Cartesian (0°=East, CCW) to Compass (0°=North, CW)."""
    return (90 - deg) % 360

def compass2cart(degN):
    """Convert Compass (0°=North, CW) to Cartesian (0°=East, CCW)."""
    return (450 - degN) % 360


def arrayRect2Comp(Ax, Ay):
   """Convert rectangular (x,y) to compass angles (0°=North, CW)."""
   angles_rad = np.arctan2(Ay, Ax)  # Cartesian angles in radians
   angles_deg = np.degrees(angles_rad)  # Convert to degrees
   return (90 - angles_deg) % 360  # Compass convention

def get_time_tic():
   tic = time.time()

   return tic

def print_elapsed_time(tic, toc):
  print ('Elapsed time is  {}  seconds.'.format(toc - tic))



def dir2dir(dir1,dir2,n):
   """Interpolate between two compass directions with n steps"""
   # passar de dir 1 a dir2 amb npassos compass a commpass
   #N son els nds interiors, o sigui:  rightn-leftn -1
   # el leftn i rihtn no surtiran al resultat
   #Tambe s'utilitza per fer el sud nort
   # Convert directions to cartesian coordinates
   x1=np.cos(np.deg2rad(compass2cart(dir1)))
   y1=np.sin(np.deg2rad(compass2cart(dir1)))
   x2=np.cos(np.deg2rad(compass2cart(dir2)))
   y2=np.sin(np.deg2rad(compass2cart(dir2)))

   # Calculate step sizes
   dx=(x2-x1)/(n+1)
   dy=(y2-y1)/(n+1)

   # Calculate intermediate directions
   out=np.zeros(n)
   for i in range(1,n+1):
      zeta=complex(x1+dx*i,y1+dy*i)
      out[i-1]=cart2compass(np.angle(zeta,deg=True))
   return


def dist_nods(N1,N2):
   """Calculate distance between two nodes in nautical miles"""
  # print(N1,N2)
   if N1==N2:
       d=0
       return d
   lon1=nodes[N1,0]
   lat1=nodes[N1,1]
   lon2=nodes[N2,0]
   lat2=nodes[N2,1]
  # print(N1,N2)

   # Calculate great-circle distance using haversine formula
   a=(np.sin(np.deg2rad(lat1))*np.sin(np.deg2rad(lat2))+np.cos(np.deg2rad(lat1))*np.cos(np.deg2rad(lat2))*np.cos(np.deg2rad(lon1-lon2)))
   #
   # if a>1:
   #     a=1
   # Convert to nautical miles (60 nautical miles per degree)
   d=60*np.rad2deg((np.arccos(a)))
   #d=60*np.rad2deg((np.arccos(np.sin(np.deg2rad(lat1))*np.sin(np.deg2rad(lat2))+np.cos(np.deg2rad(lat1))*np.cos(np.deg2rad(lat2))*np.cos(np.deg2rad(lon1-lon2)))));
   if  a<-1 or a>1 :
       print(a,N1,N2)
       if a>1:
           a=1
       print(a)
       print('Fatal Error: this is a cosine')
   return d

def veloc(v0,nod_i,nod_f,cost_i):
    """
    Calculate ship speed considering wave effects

    Parameters:
        v0 - cruising speed in calm conditions
        nod_i - starting node of the edge
        nod_f - ending node of the edge
        cost_i - time from start to determine which wave data to use

    Returns:
        Effective ship speed considering wave effects
    """

    # Get ship course angle
    ang_ship=ang_edge(nod_f,nod_i)

    # Determine wave data index based on time and time resolution
    if time_res==1:
       iv=math.floor(cost_i)+1
       # Average wave height between start and end nodes
       hm=0.5*(hs[nod_i,iv]+hs[nod_f,iv])
       # Calculate encounter angle between ship and waves
       angEnc=ang_encounter(ang_ship,dir[nod_f,iv])
    elif time_res==3:
    #       print(nod_i,nod_f,cost_i)
       iv,a=np.divmod(cost_i,time_res)
       iv=int(iv)
    #     print(nod_i)
       hm=0.5*(hs[nod_i,iv]+hs[nod_f,iv])
       angEnc=ang_encounter(ang_ship,dir[nod_f,iv])
    #       print("veloc  ", hm   )
     #  hm=hs[nod_i,iv]+(hs[nod_i,iv+1]-hs[nod_i,iv])*a/time_res
       # sembla que no cal fer la mitjana entre nodes
    #      hm2=hs[nod_f,iv]+(hs[nod_f,iv+1]-hs[nod_f,iv])*a/time_res
    #      hm=0.5*(hm1+hm2)


    # Apply different wave effect formulations based on WEN_form parameter
    if WEN_form==1:
        # Bowditch formulation
        vel=v0-reduc_v_bow(angEnc)*hm*hm*3.2808*3.2808 #transform meters to feets
    elif WEN_form==2:
        # Aertssen formulation
        vel=v0-reduc_v_arte(angEnc,hm,Lbp,v0)
    elif WEN_form==3:
        # Khokhlov formulation
        vel=v0-reduc_v_khok(angEnc,hm)*(1 - 1.35e-6*DWT*v0)
    #
    else:
        # No wave effect
        vel=v0

    # Check for negative speed
    if vel<0:
      print('Negative Ship speed. Use other Wave Effect on Navigation formulation.')
      print(vel,angEnc,ang_ship,nod_i,nod_f,hm,cost_i )
      raise SystemExit

    return vel

def ang_edge(n_desti,n_ori):
    """Calculate compass angle from origin node to destination node"""
    #   x=nodes[n_desti,0]-nodes[n_ori,0];
#   y=nodes[n_desti,1]-nodes[n_ori,1];
#   at=math.atan2(y,x);
#   alfa =cart2compass(np.rad2deg(at));
#   return alfa
    loni , lati = nodes[n_ori,0],nodes[n_ori,1]
    lone , late = nodes[n_desti,0],nodes[n_desti,1]
    # Handle special case when nodes are at same latitude
    if lati==late:
        if loni >lone:
            return 270
        else:
            return 90
#    lati=lati+0.00001
    # Calculate great-circle distance between points
    k=dist_arc(loni,lati,lone,late)
    # Calculate initial bearing using spherical trigonometry
    cosI=(np.cos(np.deg2rad(90-late))- np.cos(k)*
          np.cos(np.deg2rad(90-lati))) /((np.sin(k)) *
          np.sin(np.deg2rad(90-lati)) )
    # Handle edge cases for cosI
    if cosI>1:   # millor if cosI>1 and cosI<1.001:
#        print('cosi 1 ',cosI)
        I=0
    elif cosI<-1:   # cosI<-1 ans cosI>-1.0001
#        print('cosi -1 ', cosI)
        I=np.pi
    else:
        I=np.arccos(cosI)

    # Convert to degrees and adjust for quadrant
    I=I*180/np.pi
    if loni>lone:
        return  360-I # westbound
    else:
        return I  # eastbound


def dist_arc(loni,lati,lone,late):
    """Calculate arc distance between two points in radians (not currently used)"""
    #resultat en radiants !!Funcio no utilitzada
    cosp=(np.cos(np.deg2rad(90-lati))*np.cos(np.deg2rad(90-late)) +
        np.sin(np.deg2rad(90-lati))*np.sin(np.deg2rad(90-late)) *
        np.cos(np.deg2rad(lone-loni)))
    return np.arccos(cosp)   #np.arccos(cosp)



def ang_encounter(ang_ship,ang_wave):
    """Calculate encounter angle between ship course and wave direction"""
    if(ang_wave>ang_ship):
        theta=ang_wave-ang_ship
        return theta
    else:
        theta=360-(ang_ship-ang_wave)
#        print('kkkk',theta)
        return theta

def reduc_v_bow(theta):
   """Bowditch speed penalty based on encounter angle theta"""
   #Bowditch speed penalty (theta angle of encounter)
   if ((theta>=45 and theta<=135) or (theta>=225 and theta<=315)):
      f_theta=0.0165 # BEAM SEA
      return f_theta
   if (theta>135 and theta<225):
      f_theta=0.0083#FOLLOWING SEA
      return f_theta

   if ((theta>=0 and theta<45) or(theta>315 and theta<=360)):
      f_theta=0.0248 #HEAD SEA
      return f_theta

   f_theta=0
   return f_theta

def reduc_v_arte(theta,h,Lb,v):
    """Aertssen speed penalty based on encounter angle, wave height, ship length and speed"""
    #Aertssen speed penalty (theta angle of encounter)
    # Ensure theta is in [0,180] degrees
    if (theta > 180 and theta <= 360):
        theta = 360 - theta
    m=0
    n=0

    # Determine coefficients based on wave height and encounter angle
    if (0 <= h and h < 2.5):
        m = 0
        n = 0

    if (2.5 <= h and h < 4.0):         #referencia aertssen
        if (0 <= theta and theta <= 30): #  head sea
            m = 900
            n = 2
        if (30 < theta and theta <= 60): #bow sea  (mar de proa)
            m = 700
            n = 2
        if (60 < theta and theta <= 150):  #beam sea  (mar de traves)
            m = 350
            n = 1
        if (150 < theta and theta <= 180): #following sea  (mar de popa)
            m = 100
            n = 0

    if (4.0 <= h and h < 5.5):
        if (0 <= theta and theta <= 30): #  head sea
            m = 1300
            n = 6
        if (30 < theta and theta <= 60): #bow sea  (mar de proa)
            m = 1000
            n = 5
        if (60 < theta and theta <= 150):  #beam sea  (mar de traves)
            m = 500
            n = 3
        if (150 < theta and theta <= 180): #following sea  (mar de popa)
            m = 200
            n = 1

    if (5.5 <= h and h < 7.5):
        if (0 <= theta and theta <= 30): #  head sea
            m = 2100
            n = 11
        if (30 < theta and theta <= 60): #bow sea  (mar de proa)
            m = 1400
            n = 8
        if (60 < theta and theta <= 150):  #beam sea  (mar de traves)
            m = 700
            n = 5
        if (150 < theta and theta <= 180): #following sea  (mar de popa)
            m = 400
            n = 2

    if (7.5 <= h):
        if (0 <= theta and theta <= 30): #  head sea
            m = 3600
            n = 18
        if (30 < theta and theta <= 60): #bow sea  (mar de proa)
            m = 2300
            n = 12
        if (60 < theta and theta <= 150):  #beam sea  (mar de traves)
            m = 1000
            n = 7
        if (150 < theta and theta <= 180): #following sea  (mar de popa)
            m = 700
            n = 3

    # Calculate speed reduction
    delta_v = v * (m / Lb + n) / 100

    return delta_v

def reduc_v_khok(theta,h):
    """Khokhlov speed penalty based on encounter angle and wave height"""
    #Khoklov speed penalty (theta angle of encounter)
    # Ensure theta is in [0,180] degrees
    if (theta > 180 and theta <= 360):
        theta = 360 - theta
    # Convert angle to radians
    theta_rad = np.pi/180*theta
    #print('theta?',theta_rad)
    # Calculate speed reduction factor
    khokhlov_factor = (0.745 - 0.245*theta_rad)*h

    return khokhlov_factor

def time_edge(v0,n_o,n_e,cost):
    """
        Calculate time to traverse an edge considering wave effects

        Parameters:
            v0 - calm water speed
            n_o - origin node
            n_e - destination node
            cost - time from start

        Returns:
            Total time from start after traversing the edge
    """
    #With a edge length given and knowing the cost (time from start point)
    # and figuring that height and direction are known, acumulated time is
    # found. It finishes travelling the edge, knowing that
    # THERE IS AN HOURLY SWELL RESOLUTION!!!!!!!!Nonhourly, time_res
    L=dist_nods(n_o,n_e)
    q=1
    costi=cost
    tau=time_res-np.divmod(costi,time_res)[1]  # remaining time in current time step
    while q==1:
      if veloc(v0,n_o,n_e,costi)*tau>=L:   # with tau, remaining time for travelling the edge, it makes it and leaves (q==0)
         # Can traverse entire edge in remaining time
         costi=costi+L/veloc(v0,n_o,n_e,costi)
    #        print( "Ldirect, veloc = ",L,veloc(v0,n_o,n_e,costi))
    #        print("veloc L ", veloc(v0,n_o,n_e,costi) )
         q=0
      else:
    #       print("tau icost L=",tau,costi,L)
         # Can only traverse part of edge in remaining time
         L=L-veloc(v0,n_o,n_e,costi)*tau  # remaining distance
         costi=costi+tau   # advance time
    #         print("veloc Lindi ", veloc(v0,n_o,n_e,costi) )
         tau=time_res-np.divmod(costi,time_res)[1] # reset remaining time for new time
    #         print("tau,icost,L,v=",tau,costi,L,veloc(v0,n_o,n_e,costi))
    return  costi


def veins(N):
    """
    Return list of neighboring nodes for node N in a grid pattern

    Parameters:
        N - node index (0 to Nx*Ny-1)

    Returns:
        List of neighboring node indices based on N's position in the grid
        Handles special cases for nodes near edges and corners
    """

    # Check if node index is valid
    if N >Nx*Ny-1:
        print("Error: value must be smaller than", N)
        return False

    # Calculate node's grid coordinates
    y=math.floor(N/Nx) # row index
    x=N%Nx  # column index
    #print(x,y)

    # Central region nodes (not near any edge)
    if((x>3 and x<Nx-4) and (y>3 and y<Ny-4)):
        # Return 44 neighbors in a star-like pattern
        A =[N+1+Nx, N+1+2*Nx, N+1+3*Nx, N+1+4*Nx, N+2+3*Nx, N+2+Nx, N+3+Nx, N+3+2*Nx, N+3+4*Nx, N+4+Nx, N+4+3*Nx,
            N+1-Nx, N+1-2*Nx, N+1-3*Nx, N+1-4*Nx, N+2-3*Nx, N+2-Nx, N+3-Nx, N+3-2*Nx, N+3-4*Nx, N+4-Nx, N+4-3*Nx,
            N-1+Nx, N-1+2*Nx, N-1+3*Nx, N-1+4*Nx, N-2+3*Nx, N-2+Nx, N-3+Nx, N-3+2*Nx, N-3+4*Nx, N-4+Nx, N-4+3*Nx,
            N-1-Nx, N-1-2*Nx, N-1-3*Nx, N-1-4*Nx, N-2-3*Nx, N-2-Nx, N-3-Nx, N-3-2*Nx, N-3-4*Nx, N-4-Nx, N-4-3*Nx,
            N+1, N-1, N+Nx, N-Nx]

        return A

    # Left edge nodes (not near corners)
    if (x==0) and (y>3) and (y<Ny-4) :    # marc esquerra pur
        A =[N+1+Nx, N+1+2*Nx,N+1+3*Nx,N+1+4*Nx, N+2+3*Nx, N+2+Nx, N+3+Nx, N+3+2*Nx, N+3+4*Nx, N+4+Nx, N+4+3*Nx,
            N+1-Nx, N+1-2*Nx,N+1-3*Nx,N+1-4*Nx, N+2-3*Nx, N+2-Nx, N+3-Nx, N+3-2*Nx, N+3-4*Nx, N+4-Nx, N+4-3*Nx,
            N+1,  N+Nx, N-Nx]
        return A

    # Left edge +1 column nodes (not near corners)
    if (x==1) and (y>3) and (y<Ny-4) :
        A =[N+1+Nx, N+1+2*Nx,N+1+3*Nx,N+1+4*Nx, N+2+3*Nx, N+2+Nx, N+3+Nx, N+3+2*Nx, N+3+4*Nx, N+4+Nx, N+4+3*Nx,
            N+1-Nx, N+1-2*Nx,N+1-3*Nx,N+1-4*Nx, N+2-3*Nx, N+2-Nx, N+3-Nx, N+3-2*Nx, N+3-4*Nx, N+4-Nx, N+4-3*Nx,
            N-1+Nx, N-1+2*Nx,N-1+3*Nx,N-1+4*Nx,
            N-1-Nx, N-1-2*Nx,N-1-3*Nx,N-1-4*Nx,
            N+1, N-1, N+Nx, N-Nx]
        return A

    # Left edge +2 columns nodes (not near corners)
    if (x==2) and (y>3) and (y<Ny-4) :
        A =[N+1+Nx, N+1+2*Nx, N+1+3*Nx, N+1+4*Nx, N+2+3*Nx, N+2+Nx, N+3+Nx, N+3+2*Nx, N+3+4*Nx, N+4+Nx, N+4+3*Nx,
            N+1-Nx, N+1-2*Nx, N+1-3*Nx, N+1-4*Nx, N+2-3*Nx, N+2-Nx, N+3-Nx, N+3-2*Nx, N+3-4*Nx, N+4-Nx, N+4-3*Nx,
            N-1+Nx, N-1+2*Nx, N-1+3*Nx, N-1+4*Nx, N-2+3*Nx, N-2+Nx,
            N-1-Nx, N-1-2*Nx, N-1-3*Nx, N-1-4*Nx, N-2-3*Nx, N-2-Nx,
            N+1, N-1, N+Nx, N-Nx]
        return A

    # Left edge +3 columns nodes (not near corners)
    if (x==3) and (y>3) and (y<Ny-4) :
         A =[N+1+Nx, N+1+2*Nx, N+1+3*Nx, N+1+4*Nx, N+2+3*Nx, N+2+Nx, N+3+Nx, N+3+2*Nx, N+3+4*Nx, N+4+Nx, N+4+3*Nx,
            N+1-Nx, N+1-2*Nx, N+1-3*Nx, N+1-4*Nx, N+2-3*Nx, N+2-Nx, N+3-Nx, N+3-2*Nx, N+3-4*Nx, N+4-Nx, N+4-3*Nx,
            N-1+Nx, N-1+2*Nx, N-1+3*Nx, N-1+4*Nx, N-2+3*Nx, N-2+Nx, N-3+Nx, N-3+2*Nx, N-3+4*Nx,
            N-1-Nx, N-1-2*Nx, N-1-3*Nx, N-1-4*Nx, N-2-3*Nx, N-2-Nx, N-3-Nx, N-3-2*Nx, N-3-4*Nx,
            N+1, N-1, N+Nx, N-Nx]
         return A

    # Right edge nodes (not near corners)
    if (x==Nx-1) and  (y>3) and (y<Ny-4) :
        A =[N-1+Nx, N-1+2*Nx, N-1+3*Nx, N-1+4*Nx, N-2+3*Nx, N-2+Nx, N-3+Nx, N-3+2*Nx, N-3+4*Nx, N-4+Nx, N-4+3*Nx,
            N-1-Nx, N-1-2*Nx, N-1-3*Nx, N-1-4*Nx, N-2-3*Nx, N-2-Nx, N-3-Nx, N-3-2*Nx, N-3-4*Nx, N-4-Nx, N-4-3*Nx,
            N-1, N+Nx, N-Nx]
        return A

    # Right edge -1 column nodes (not near corners)
    if (x==Nx-2) and  (y>3) and (y<Ny-4) :
         A =[N+1+Nx, N+1+2*Nx, N+1+3*Nx, N+1+4*Nx,
            N+1-Nx, N+1-2*Nx, N+1-3*Nx, N+1-4*Nx,
            N-1+Nx, N-1+2*Nx, N-1+3*Nx, N-1+4*Nx, N-2+3*Nx, N-2+Nx, N-3+Nx, N-3+2*Nx, N-3+4*Nx, N-4+Nx, N-4+3*Nx,
            N-1-Nx, N-1-2*Nx, N-1-3*Nx, N-1-4*Nx, N-2-3*Nx, N-2-Nx, N-3-Nx, N-3-2*Nx, N-3-4*Nx, N-4-Nx, N-4-3*Nx,
            N+1, N-1, N+Nx, N-Nx]

         return A

    # Right edge -2 columns nodes (not near corners)
    if (x==Nx-3) and  (y>3) and (y<Ny-4) :
         A =[N+1+Nx, N+1+2*Nx, N+1+3*Nx, N+1+4*Nx, N+2+3*Nx, N+2+Nx,
            N+1-Nx, N+1-2*Nx, N+1-3*Nx, N+1-4*Nx, N+2-3*Nx, N+2-Nx,
            N-1+Nx, N-1+2*Nx, N-1+3*Nx, N-1+4*Nx, N-2+3*Nx, N-2+Nx, N-3+Nx, N-3+2*Nx, N-3+4*Nx, N-4+Nx, N-4+3*Nx,
            N-1-Nx, N-1-2*Nx, N-1-3*Nx, N-1-4*Nx, N-2-3*Nx, N-2-Nx, N-3-Nx, N-3-2*Nx, N-3-4*Nx, N-4-Nx, N-4-3*Nx,
            N+1, N-1, N+Nx, N-Nx]

         return A

    # Right edge -3 columns nodes (not near corners)
    if (x==Nx-4) and  (y>3) and (y<Ny-4) :
         A =[N+1+Nx, N+1+2*Nx, N+1+3*Nx, N+1+4*Nx, N+2+3*Nx, N+2+Nx, N+3+Nx, N+3+2*Nx, N+3+4*Nx,
            N+1-Nx, N+1-2*Nx, N+1-3*Nx, N+1-4*Nx, N+2-3*Nx, N+2-Nx, N+3-Nx, N+3-2*Nx, N+3-4*Nx,
            N-1+Nx, N-1+2*Nx, N-1+3*Nx, N-1+4*Nx, N-2+3*Nx, N-2+Nx, N-3+Nx, N-3+2*Nx, N-3+4*Nx, N-4+Nx, N-4+3*Nx,
            N-1-Nx, N-1-2*Nx, N-1-3*Nx, N-1-4*Nx, N-2-3*Nx, N-2-Nx, N-3-Nx, N-3-2*Nx, N-3-4*Nx, N-4-Nx, N-4-3*Nx,
            N+1, N-1, N+Nx, N-Nx]

         return A

    # Bottom edge nodes (not near corners)
    if (y==0) and (x>3) and (x<Nx-4):   # marc inferior
        A =[N+1+Nx, N+1+2*Nx, N+1+3*Nx, N+1+4*Nx, N+2+3*Nx, N+2+Nx, N+3+Nx, N+3+2*Nx, N+3+4*Nx, N+4+Nx, N+4+3*Nx,
            N-1+Nx, N-1+2*Nx, N-1+3*Nx, N-1+4*Nx, N-2+3*Nx, N-2+Nx, N-3+Nx, N-3+2*Nx, N-3+4*Nx, N-4+Nx, N-4+3*Nx,
            N+1, N-1, N+Nx]
        return A

    # Bottom edge +1 row nodes (not near corners)
    if (y==1) and (x>3) and (x<Nx-4):
        A =[N+1+Nx, N+1+2*Nx, N+1+3*Nx, N+1+4*Nx, N+2+3*Nx, N+2+Nx, N+3+Nx, N+3+2*Nx, N+3+4*Nx, N+4+Nx, N+4+3*Nx,
            N+1-Nx, N+2-Nx, N+3-Nx,  N+4-Nx,
            N-1+Nx, N-1+2*Nx, N-1+3*Nx, N-1+4*Nx, N-2+3*Nx, N-2+Nx, N-3+Nx, N-3+2*Nx, N-3+4*Nx, N-4+Nx, N-4+3*Nx,
            N-1-Nx, N-2-Nx, N-3-Nx, N-4-Nx,
            N+1, N-1, N+Nx, N-Nx]
        return A

    # Bottom edge +2 rows nodes (not near corners)
    if (y==2) and (x>3) and (x<Nx-4):
        A =[N+1+Nx, N+1+2*Nx, N+1+3*Nx, N+1+4*Nx, N+2+3*Nx, N+2+Nx, N+3+Nx, N+3+2*Nx, N+3+4*Nx, N+4+Nx, N+4+3*Nx,
            N+1-Nx, N+1-2*Nx, N+2-Nx, N+3-Nx, N+3-2*Nx, N+4-Nx,
            N-1+Nx, N-1+2*Nx, N-1+3*Nx, N-1+4*Nx, N-2+3*Nx, N-2+Nx, N-3+Nx, N-3+2*Nx, N-3+4*Nx, N-4+Nx, N-4+3*Nx,
            N-1-Nx, N-1-2*Nx, N-2-Nx, N-3-Nx, N-3-2*Nx, N-4-Nx,
            N+1, N-1, N+Nx, N-Nx]
        return A

    # Bottom edge +3 rows nodes (not near corners)
    if (y==3) and (x>3) and (x<Nx-4):
        A =[N+1+Nx, N+1+2*Nx, N+1+3*Nx, N+1+4*Nx, N+2+3*Nx, N+2+Nx, N+3+Nx, N+3+2*Nx, N+3+4*Nx, N+4+Nx, N+4+3*Nx,
            N+1-Nx, N+1-2*Nx, N+1-3*Nx, N+2-3*Nx, N+2-Nx, N+3-Nx, N+3-2*Nx, N+4-Nx, N+4-3*Nx,
            N-1+Nx, N-1+2*Nx, N-1+3*Nx, N-1+4*Nx, N-2+3*Nx, N-2+Nx, N-3+Nx, N-3+2*Nx, N-3+4*Nx, N-4+Nx, N-4+3*Nx,
            N-1-Nx, N-1-2*Nx, N-1-3*Nx, N-2-3*Nx, N-2-Nx, N-3-Nx, N-3-2*Nx, N-4-Nx, N-4-3*Nx,
            N+1, N-1, N+Nx, N-Nx]
        return A

    if (y==Ny-1) and (x>3) and (x<Nx-4):
        A =[N+1-Nx, N+1-2*Nx, N+1-3*Nx, N+1-4*Nx, N+2-3*Nx, N+2-Nx, N+3-Nx, N+3-2*Nx, N+3-4*Nx, N+4-Nx, N+4-3*Nx,
            N-1-Nx, N-1-2*Nx, N-1-3*Nx, N-1-4*Nx, N-2-3*Nx, N-2-Nx, N-3-Nx, N-3-2*Nx, N-3-4*Nx, N-4-Nx, N-4-3*Nx,
            N+1, N-1, N-Nx]

        return A
    if (y==Ny-2) and (x>3) and (x<Nx-4):
          A =[N+1+Nx, N+2+Nx, N+3+Nx,  N+4+Nx,
            N+1-Nx, N+1-2*Nx, N+1-3*Nx, N+1-4*Nx, N+2-3*Nx, N+2-Nx, N+3-Nx, N+3-2*Nx, N+3-4*Nx, N+4-Nx, N+4-3*Nx,
            N-1+Nx,  N-2+Nx, N-3+Nx,  N-4+Nx,
            N-1-Nx, N-1-2*Nx, N-1-3*Nx, N-1-4*Nx, N-2-3*Nx, N-2-Nx, N-3-Nx, N-3-2*Nx, N-3-4*Nx, N-4-Nx, N-4-3*Nx,
            N+1, N-1, N+Nx, N-Nx]
          return A

    if (y==Ny-3) and (x>3) and (x<Nx-4):
    #        A =[N+1+Nx, N+1+2*Nx, N+2+Nx, N+3+Nx, N+3+2*Nx,  N+4+Nx,
    #            N+1-Nx, N+1-2*Nx, N+1-3*Nx, N+1-4*Nx, N+2-3*Nx, N+2-Nx, N+3-Nx, N+3-2*Nx, N+3-4*Nx, N+4-Nx, N+4-3*Nx,
    #            N-1+Nx, N-1+2*Nx, N-1+3*Nx, N-1+4*Nx, N-2+3*Nx, N-2+Nx, N-3+Nx, N-3+2*Nx, N-3+4*Nx, N-4+Nx, N-4+3*Nx,
    #            N-1-Nx, N-1-2*Nx, N-2-Nx, N-3-Nx, N-3-2*Nx, N-4-Nx,
    #            N+1, N-1, N+Nx, N-Nx]
        A =[N+1+Nx, N+1+2*Nx,    N+2+Nx, N+3+Nx, N+3+2*Nx,  N+4+Nx,
            N+1-Nx, N+1-2*Nx, N+1-3*Nx, N+1-4*Nx, N+2-3*Nx, N+2-Nx, N+3-Nx, N+3-2*Nx, N+3-4*Nx, N+4-Nx, N+4-3*Nx,
            N-1+Nx, N-1+2*Nx,  N-2+Nx, N-3+Nx, N-3+2*Nx,  N-4+Nx,
            N-1-Nx, N-1-2*Nx, N-1-3*Nx, N-1-4*Nx, N-2-3*Nx, N-2-Nx, N-3-Nx, N-3-2*Nx, N-3-4*Nx, N-4-Nx, N-4-3*Nx,
            N+1, N-1, N+Nx, N-Nx]

        return A

    if (y==Ny-4) and (x>3) and (x<Nx-4):
        A =[N+1+Nx, N+1+2*Nx, N+1+3*Nx, N+2+3*Nx, N+2+Nx, N+3+Nx, N+3+2*Nx, N+4+Nx, N+4+3*Nx,
            N+1-Nx, N+1-2*Nx, N+1-3*Nx, N+1-4*Nx, N+2-3*Nx, N+2-Nx, N+3-Nx, N+3-2*Nx, N+3-4*Nx, N+4-Nx, N+4-3*Nx,
            N-1+Nx, N-1+2*Nx, N-1+3*Nx, N-1+4*Nx, N-2+3*Nx, N-2+Nx, N-3+Nx, N-3+2*Nx, N-3+4*Nx, N-4+Nx, N-4+3*Nx,
            N-1-Nx, N-1-2*Nx, N-1-3*Nx, N-2-3*Nx, N-2-Nx, N-3-Nx, N-3-2*Nx, N-3-4*Nx, N-4-3*Nx,
            N+1, N-1, N+Nx, N-Nx]

        return A

    # Si arriba aqui es que el node es un dels 64 dels quatre vertex, el rebotem al seu pare i que dara tencat.
    return [N]
       
def testVrtx(N):
    y=math.floor(N/Nx) #  Miren si un dode esta en un vertex amb un unic vei,no valis per nodEnd o nodIni
    x=N%Nx
    if x<4 and y<4:
        return False
    if x>Nx-5  and y<4:
        return False
    if x<4 and y>Ny-5 :
        return False
    if x>Nx-5 and y>Ny-5:
        return False
    return True     
         
def nod2cart(N):  # dona llista [y,x]    
    y=math.floor(N/Nx)      
    x=N%Nx
    return [y,x]
       