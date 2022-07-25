#import math
import numpy as np
import sys
import time
#from mpmath import *
import mpmath as mp
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import LogFormatterSciNotation
from matplotlib import colors
#from decimal import *
mp.dps=32
np.set_printoptions(precision=3)

print('Choose parameters of the rock matrix to plot on heatmaps (porosity, rigidity, viscosity or permeability) in format (param1 param2).')
params_in=list(input().split(" "))

#print('Choose range of poroelastic/Biot coefficient in format (start stop steps). Suggested - 0.91 0.99 5 or 0.95:')
#a_in=list(input().split(" "))
a_in=[0.91, 0.99, 9]
if len(a_in)==3:
    alpharange=np.linspace(float(a_in[0]),float(a_in[1]),int(a_in[2]))
    
    if len(params_in)>0:
        print(alpharange)
        print('What value should alpha be held at for heatmaps over other 2D parameter spaces? Make sure this value is part of the range specified above.')
        alphaconstant_in=float(input())
        alphaconstant=np.where(alpharange==alphaconstant_in)
        print(alphaconstant[0])
        
elif len(a_in)==1:
    alpharange=np.array([float(a_in[0])])
else:
    print("Please enter one of these formats: [constant] or [min,max,steps].")

#print('Choose logarithmic range of rigidity of the rock matrix in format (start stop steps), where mu=10^(input). Suggested - 7 10 4 or 9:')
#m_in=list(input().split(" "))
m_in=[7, 10, 4]
if len(m_in)==3:
    murange=np.logspace(float(m_in[0]),float(m_in[1]),int(m_in[2]))
    str_murange=[str('$10^7$'),str('$10^8$'),str('$10^9$'),str('$10^{10}$')]
    
    if len(params_in)>0:
        print(murange)
        print('What value should rigidity be held at for heatmaps over other 2D parameter spaces? Make sure this value is part of the range specified above.')
        muconstant_in=float(input())
        muconstant=np.where(murange==muconstant_in)
        print(muconstant[0])
        
elif len(m_in)==1:
    murange=np.array([10**float(m_in[0])])
else:
    print("Please enter one of these formats: [constant] or [min,max,steps].")

#print('Choose logarithmic range of rock viscosity in format (start stop steps), where eta_m=10^(input). Suggested - 13 19 7 or 16:')
#eta_in=list(input().split(" "))
eta_in=[13, 19, 7]
if len(eta_in)==3:
    eta_m_range=np.logspace(float(eta_in[0]),float(eta_in[1]),int(eta_in[2]))
    str_eta_m_range=[str('$10^{13}$'),str('$10^{14}$'),str('$10^{15}$'),str('$10^{16}$'),str('$10^{17}$'),str('$10^{18}$'),str('$10^{19}$')]
    
    if len(params_in)>0:
        print(eta_m_range)
        print('What value should viscosity be held at for heatmaps over other 2D parameter spaces? Make sure this value is part of the range specified above.')
        eta_m_constant_in=float(input())
        eta_m_constant=np.where(eta_m_range==eta_m_constant_in)
        
elif len(eta_in)==1:
    eta_m_range=np.array([10**float(eta_in[0])])
else:
    print("Please enter one of these formats: [constant] or [min,max,steps].")

#print('Choose logarithmic range of rock permeability in format (start stop steps), where kappa=10^(input). Suggested - -11 -7 5 or -10:')
#kp_in=list(input().split(" "))
kp_in=[-11, -7, 5]
if len(kp_in)==3:
    kapparange=np.logspace(float(kp_in[0]),float(kp_in[1]),int(kp_in[2]))
    str_kapparange=[str('$10^{-11}$'),str('$10^{-10}$'),str('$10^{-9}$'),str('$10^{-8}$'),str('$10^{-7}$')]
    if len(params_in)>0:
        print(kapparange)
        print('What value should permeability be held at for heatmaps over other 2D parameter spaces? Make sure this value is part of the range specified above.')
        kappaconstant_in=float(input())
        kappaconstant=np.where(kapparange==kappaconstant_in)
        
elif len(kp_in)==1:
    kapparange=np.array([10**float(kp_in[0])])
else:
    print("Please enter one of these formats: [constant] or [min,max,steps].")
    
alphaconstant=int(alphaconstant[0])
muconstant=int(muconstant[0])
eta_m_constant=int(eta_m_constant[0])
kappaconstant=int(kappaconstant[0])

h_tide_heatplt=np.zeros((len(alpharange),len(murange),len(eta_m_range),len(kapparange)))
h_vis_heatplt=np.zeros((len(alpharange),len(murange),len(eta_m_range),len(kapparange)))
h_tot_heatplt=np.zeros((len(alpharange),len(murange),len(eta_m_range),len(kapparange)))

h_tide_2d=np.loadtxt('h_tide heating 4d datafile.txt', delimiter=' , ')
h_vis_2d=np.loadtxt('h_vis heating 4d datafile.txt', delimiter=' , ')
h_tot_2d=np.loadtxt('full tidal heating 4d datafile.txt', delimiter=' , ')

counter=-1

for al in range(len(alpharange)):
    for m in range(len(murange)):
        for et in range(len(eta_m_range)):
            for kp in range(len(kapparange)):
                counter+=1
                h_tide_heatplt[al,m,et,kp]=h_tide_2d[counter,4]
                h_vis_heatplt[al,m,et,kp]=h_vis_2d[counter,4]
                h_tot_heatplt[al,m,et,kp]=h_tot_2d[counter,4]
print(h_tide_heatplt)
print(h_vis_heatplt)
print(h_tot_heatplt)

if 'porosity' in params_in and 'rigidity'in params_in:
    ptcol=np.array(np.meshgrid(murange,alpharange)).T.reshape(len(alpharange)*len(murange),2)
    h_tot_flat=np.ndarray.flatten(h_tot_heatplt[:,:,eta_m_constant,kappaconstant],'F')
    np.savetxt('alpha v mu.txt',np.column_stack((ptcol,h_tot_flat)),fmt='%E',delimiter=' , ')
    
    f,ax = plt.subplots(figsize=(10, 10)); # 
    
    plt.imshow(h_tot_heatplt[:,:,eta_m_constant,kappaconstant], origin='lower',vmin=5E6,vmax=1E11,cmap='hot',norm=colors.LogNorm(), aspect='auto', interpolation='spline16', extent=[murange[0],murange[len(murange)-1],alpharange[0],alpharange[len(alpharange)-1]])
    plt.title("Total heat generated due to tidal dissapation in the core of Enceladus over a range of poroelastic coefficients and rigidities", wrap=True);
    plt.ylabel("Poroelastic coefficient of rock matrix");
    plt.yscale('linear')
    plt.yticks(np.linspace(alpharange[0],alpharange[len(alpharange)-1],len(alpharange)),alpharange); 
    plt.xlabel("Rigidity of rock matrix ($Pa$)");
    plt.xscale('linear')
    plt.xticks(np.linspace(murange[0],murange[len(murange)-1],len(murange)),str_murange);
    #ax.xaxis.set_major_formatter(ScalarFormatter())
    #ax.xaxis.set_minor_formatter(ScalarFormatter())
    #plt.ticklabel_format(style='sci', axis='x')
    plt.colorbar()
    plt.savefig('tidal heat generation - alpha v mu.png')

    plt.show()
    
if 'porosity' in params_in and 'viscosity' in params_in:
    ptcol=np.array(np.meshgrid(eta_m_range,alpharange)).T.reshape(len(alpharange)*len(eta_m_range),2)
    h_tot_flat=np.ndarray.flatten(h_tot_heatplt[:,muconstant,:,kappaconstant],'F')
    np.savetxt('alpha v eta.txt',np.column_stack((ptcol,h_tot_flat)),fmt='%E',delimiter=' , ')
    
    f = plt.figure(figsize=(10, 10)); #  

    plt.imshow(h_tot_heatplt[:,muconstant,:,kappaconstant], origin='lower',vmin=5E6,vmax=1E11,cmap='hot',norm=colors.LogNorm(), aspect='auto', interpolation='spline16', extent=[eta_m_range[0],eta_m_range[len(eta_m_range)-1],alpharange[0],alpharange[len(alpharange)-1]])
    plt.title("Total heat generated due to tidal dissapation in the core of Enceladus over a range of poroelastic coefficients and viscosities", wrap=True);
    plt.ylabel("Poroelastic coefficient of rock matrix");
    plt.yscale('linear')
    plt.yticks(np.linspace(alpharange[0],alpharange[len(alpharange)-1],len(alpharange)),alpharange); 
    plt.xlabel("Viscosity of rock matrix ($Pa \cdot s$)");
    plt.xscale('linear')
    plt.xticks(np.linspace(eta_m_range[0],eta_m_range[len(eta_m_range)-1],len(eta_m_range)),str_eta_m_range); 
    plt.colorbar()
    plt.savefig('tidal heat generation - alpha v eta.png')

    plt.show()
    
if 'porosity' in params_in and 'permeability'in params_in:
    ptcol=np.array(np.meshgrid(kapparange,alpharange)).T.reshape(len(alpharange)*len(kapparange),2)
    h_tot_flat=np.ndarray.flatten(h_tot_heatplt[:,muconstant,eta_m_constant,:],'F')
    np.savetxt('alpha v kappa.txt',np.column_stack((ptcol,h_tot_flat)),fmt='%E',delimiter=' , ')
    
    f = plt.figure(figsize=(10, 10)); #  

    plt.imshow(h_tot_heatplt[:,muconstant,eta_m_constant,:],origin='lower',vmin=5E6,vmax=1E11,cmap='hot',norm=colors.LogNorm(), aspect='auto', interpolation='spline16', extent=[kapparange[0],kapparange[len(kapparange)-1],alpharange[0],alpharange[len(alpharange)-1]])
    plt.title("Total heat generated due to tidal dissapation in the core of Enceladus over a range of poroelastic coefficients and permeabilities", wrap=True);
    plt.ylabel("Poroelastic coefficient of rock matrix");
    plt.yscale('linear')
    plt.yticks(np.linspace(alpharange[0],alpharange[len(alpharange)-1],len(alpharange)),alpharange); 
    plt.xlabel("Permability of rock matrix ($m^2$)");
    plt.xscale('linear')
    plt.xticks(np.linspace(kapparange[0],kapparange[len(kapparange)-1],len(kapparange)),str_kapparange); 
    plt.colorbar()
    plt.savefig('tidal heat generation - alpha v kappa.png')

    plt.show()
    
if 'rigidity' in params_in and 'viscosity' in params_in:
    ptcol=np.array(np.meshgrid(eta_m_range,murange)).T.reshape(len(murange)*len(eta_m_range),2)
    h_tot_flat=np.ndarray.flatten(h_tot_heatplt[alphaconstant,:,:,kappaconstant],'F')
    np.savetxt('mu v eta.txt',np.column_stack((ptcol,h_tot_flat)),fmt='%E',delimiter=' , ')
    
    f = plt.figure(figsize=(10, 10)); #  

    plt.imshow(h_tot_heatplt[alphaconstant,:,:,kappaconstant], origin='lower',vmin=5E6,vmax=1E11,cmap='hot',norm=colors.LogNorm(), aspect='auto', interpolation='spline16', extent=[eta_m_range[0],eta_m_range[len(eta_m_range)-1],murange[0],murange[len(murange)-1]])
    plt.title("Total heat generated due to tidal dissapation in the core of Enceladus over a range of rigidities and viscosities", wrap=True);
    plt.ylabel("Rigidity of rock matrix ($Pa$)");
    plt.yscale('linear')
    plt.yticks(np.linspace(murange[0],murange[len(murange)-1],len(murange)),str_murange); 
    plt.xlabel("Viscosity of rock matrix ($Pa \cdot s$)");
    plt.xscale('linear')
    plt.xticks(np.linspace(eta_m_range[0],eta_m_range[len(eta_m_range)-1],len(eta_m_range)),str_eta_m_range); 
    plt.colorbar()
    plt.savefig('tidal heat generation - mu v eta.png')

    plt.show()
    
if 'rigidity' in params_in and 'permeability'in params_in:
    ptcol=np.array(np.meshgrid(kapparange,murange)).T.reshape(len(murange)*len(kapparange),2)
    h_tot_flat=np.ndarray.flatten(h_tot_heatplt[alphaconstant,:,eta_m_constant,:],'F')
    np.savetxt('mu v kappa.txt',np.column_stack((ptcol,h_tot_flat)),fmt='%E',delimiter=' , ')
    
    f = plt.figure(figsize=(10, 10)); #  

    plt.imshow(h_tot_heatplt[alphaconstant,:,eta_m_constant,:], origin='lower',vmin=5E6,vmax=1E11,cmap='hot',norm=colors.LogNorm(), aspect='auto', interpolation='spline16', extent=[kapparange[0],kapparange[len(kapparange)-1],murange[0],murange[len(murange)-1]])
    plt.title("Total heat generated due to tidal dissapation in the core of Enceladus over a range of rigidities and permeabilities", wrap=True);
    plt.ylabel("Rigidity of rock matrix ($Pa$)");
    plt.yscale('linear')
    plt.yticks(np.linspace(murange[0],murange[len(murange)-1],len(murange)),str_murange); 
    plt.xlabel("Permability of rock matrix ($m^2$)");
    plt.xscale('linear')
    plt.xticks(np.linspace(kapparange[0],kapparange[len(kapparange)-1],len(kapparange)),str_kapparange); 
    plt.colorbar()
    plt.savefig('tidal heat generation - mu v kappa.png')

    plt.show()
    
if 'viscosity' in params_in and 'permeability' in params_in:
    ptcol=np.array(np.meshgrid(kapparange,eta_m_range)).T.reshape(len(eta_m_range)*len(kapparange),2)
    h_tot_flat=np.ndarray.flatten(h_tot_heatplt[alphaconstant,muconstant,:,:],'F')
    np.savetxt('eta v kappa.txt',np.column_stack((ptcol,h_tot_flat)),fmt='%E',delimiter=' , ')
    
    f = plt.figure(figsize=(10, 10)); #  

    plt.imshow(h_tot_heatplt[alphaconstant,muconstant,:,:], origin='lower',vmin=5E6,vmax=1E11,cmap='hot',norm=colors.LogNorm(), aspect='auto', interpolation='spline16', extent=[kapparange[0],kapparange[len(kapparange)-1],eta_m_range[0],eta_m_range[len(eta_m_range)-1]])
    plt.title("Total heat generated due to tidal dissapation in the core of Enceladus over a range of viscosities and permeabilities", wrap=True);
    plt.ylabel("Viscosity of rock matrix ($Pa \cdot s$)");
    plt.yscale('linear')
    plt.yticks(np.linspace(eta_m_range[0],eta_m_range[len(eta_m_range)-1],len(eta_m_range)),str_eta_m_range); 
    plt.xlabel("Permability of rock matrix ($m^2$)");
    plt.xscale('linear')
    plt.xticks(np.linspace(kapparange[0],kapparange[len(kapparange)-1],len(kapparange)),str_kapparange); 
    plt.colorbar()
    plt.savefig('tidal heat generation - eta v kappa.png')

    plt.show()

"""
plot template:
    
    ptcol=np.array(np.meshgrid(kapparange,eta_m_range)).T.reshape(len(eta_m_range)*len(kapparange),2)           #create 2 columns from meshgrid where lowest dimension is x and higher dimension is y 
    h_tot_flat=np.ndarray.flatten(h_tot_heatplt[alphaconstant,muconstant,:,:],'F')                              #flatten h_tot_plt into 1 column with F order
    np.savetxt('eta v kappa.txt',np.column_stack((ptcol,h_tot_flat)),fmt='%E',delimiter=' , ')                  #save output to 3 column txtfile with x,y,h_tot
    
    f = plt.figure(figsize=(10, 10)); #  

    plt.imshow(h_tot_heatplt[alphaconstant,muconstant,:,:], origin='lower',vmin=5E6,vmax=1E11,cmap='hot',norm=colors.LogNorm(), aspect='auto', interpolation='spline16', extent=[kapparange[0],kapparange[len(kapparange)-1],eta_m_range[0],eta_m_range[len(eta_m_range)-1]])
    plt.title("Total heat generated due to tidal dissapation in the core of Enceladus over parameter spaces of viscosity and permeability", wrap=True);
    plt.ylabel("Viscosity of rock matrix (Pa s)");
    plt.yscale('linear')
    plt.yticks(np.linspace(eta_m_range[0],eta_m_range[len(eta_m_range)-1],len(eta_m_range)),eta_m_range); 
    plt.xlabel("Permability of rock matrix (m^2)");
    plt.xscale('linear')
    plt.xticks(np.linspace(kapparange[0],kapparange[len(kapparange)-1],len(kapparange)),kapparange); 
    plt.colorbar()
    plt.savefig('tidal heat generation - eta v kappa.png')

    plt.show()
"""