#import math
import numpy as np
import sys
import time
#from mpmath import *
import mpmath as mp
import matplotlib.pyplot as plt
from matplotlib import colors
#from decimal import *
mp.dps=32
np.set_printoptions(precision=3)

"""
README

"""



#FUNCTIONS

def comp_norm(y):
    I=np.imag(y)
    R=np.real(y)
    a=np.square(I)+np.square(R)
    y_norm=np.sqrt(a)
    return y_norm
    
def comp_const(epsilon_0,alpha,k,K_m,mu_star,R_0,b_0,jC,ddjrC):

    F=(1/2)*b_0*ddjrC+(1/2)*b_0*(k**2)*jC-(jC/(4*mu_star))
    #C1=epsilon_0/(4*np.linalg.norm(2*F-b_0*ddjrC))
    C1=epsilon_0/(4*mp.norm(2*F-b_0*ddjrC,p=2))
    C2=C1*F
    return(F,C1,C2)

def bessel2int(k,rp):
        
    e_x=np.multiply(k,rp)
    e_j=(3*np.sin(e_x)/(e_x**3))-(3*np.cos(e_x)/(e_x**2))-(np.sin(e_x)/e_x)
    e_djr=k*((4*(e_x**2)-9)*np.sin(e_x)+(9*e_x-e_x**3)*np.cos(e_x))/(e_x**4)
    e_ddjr=(k**2)*(((e_x**4)-17*(e_x**2)+36)*np.sin(e_x)+(5*(e_x**3)-36*e_x)*np.cos(e_x))/(e_x**5)

    if (np.isnan(e_j)) or (np.isnan(e_djr)) or (np.isnan(e_ddjr)):
        """
        e_j=0*rp
        e_djr=0*rp
        e_ddjr=0*rp
        """
        e_x=mp.mpc(k*rp)
        e_sinx=mp.mpc(mp.sin(e_x))
        e_cosx=mp.mpc(mp.cos(e_x))
        e_j=mp.mpc(3*e_sinx/mp.power(e_x,3))-(3*e_cosx/mp.power(e_x,2))-(e_sinx/(e_x))
        e_djr=mp.mpc(k*((4*(e_x**2)-9)*e_sinx+(9*e_x-e_x**3)*e_cosx)/(e_x**4))
        e_ddjr=mp.mpc((k**2)*(((e_x**4)-17*(e_x**2)+36)*e_sinx+(5*(e_x**3)-36*e_x)*e_cosx)/(e_x**5))
        
    elif (k==0):
        e_j=0*rp
        e_djr=0*rp
        e_ddjr=0*rp
    elif (rp==0):
        e_j=0
        e_djr=0
        e_ddjr=0
    
    return(e_x,e_j,e_djr,e_ddjr)
    
def bessel2mesh(k,rmesh):
    x=mp.matrix(thetapts,rpts)
    j=mp.matrix(thetapts,rpts)
    djr=mp.matrix(thetapts,rpts)
    ddjr=mp.matrix(thetapts,rpts)
    for i in range (1,thetapts):
        for u in range (1,rpts):
            r=rmesh[i,u]
            x[i,u],j[i,u],djr[i,u],ddjr[i,u]=bessel2int(k,r)
            
    #print(x,j,djr,ddjr)
    
    return(j,djr,ddjr)
    
def comp_hat(thetamesh,rmesh,j,djr,ddjr,C1,C2,alpha,K_m,K_u,mu_star,kappa,eta_f):
    phi_N=mp.matrix(thetapts,rpts)
    P_hat=mp.matrix(thetapts,rpts)
    m_hat=mp.matrix(thetapts,rpts)
    qr_hat=mp.matrix(thetapts,rpts)
    qt_hat=mp.matrix(thetapts,rpts)

    for i in range (1,thetapts):
        for u in range (1,rpts):
            phi_N[i,u]=alpha*C1*(3*np.cos(2*thetamesh[i,u])+1)*(j[i,u]/(K_m+4*mu_star/3))
            P_hat[i,u]=C1*(3*np.cos(2*thetamesh[i,u])+1)*j[i,u]
            m_hat[i,u]=alpha*phi_N[i,u]+(alpha**2)*P_hat[i,u]/(K_u-K_m)
            qr_hat[i,u]=-(kappa/eta_f)*C1*(3*np.cos(2*thetamesh[i,u])+1)*djr[i,u]
            qt_hat[i,u]=6*(kappa/eta_f)*C1*np.sin(2*thetamesh[i,u])*(j[i,u]/rmesh[i,u])
    return(phi_N,P_hat,m_hat,qr_hat,qt_hat)    
    
def strain_comps(thetamesh,C2,b_0,C1,ddjr,j,djr,rmesh,phi_N):
    epsilon_rr=mp.matrix(thetapts,rpts)
    epsilon_tt=mp.matrix(thetapts,rpts)
    epsilon_pp=mp.matrix(thetapts,rpts)
    epsilon_rt=mp.matrix(thetapts,rpts)
    for i in range (1,thetapts):
        for u in range (1,rpts):
            epsilon_rr[i,u]=(3*np.cos(2*thetamesh[i,u])+1)*(2*C2-b_0*C1*ddjr[i,u])
            epsilon_tt[i,u]=(-12*np.cos(2*thetamesh[i,u]))*(-b_0*C1*j[i,u]/(rmesh[i,u]**2)+C2)+(3*np.cos(2*thetamesh[i,u])+1)*(-b_0*C1*djr[i,u]/rmesh[i,u]+2*C2)
            epsilon_pp[i,u]=phi_N[i,u]-epsilon_rr[i,u]-epsilon_tt[i,u]
            epsilon_rt[i,u]=(-6*b_0*C1*np.sin(2*thetamesh[i,u])*(j[i,u]-rmesh[i,u]*djr[i,u])/(rmesh[i,u]**2))-6*np.sin(2*thetamesh[i,u])*C2
    return(epsilon_rr,epsilon_tt,epsilon_pp,epsilon_rt)
    
def stress_comps(epsilon_rr,epsilon_tt,epsilon_pp,epsilon_rt,K_m,mu_star,phi_N,alpha,P_hat):
    sigma_rr=mp.matrix(thetapts,rpts)
    sigma_tt=mp.matrix(thetapts,rpts)
    sigma_pp=mp.matrix(thetapts,rpts)
    sigma_rt=mp.matrix(thetapts,rpts)
    for i in range (1,thetapts):
        for u in range (1,rpts):
            sigma_rr[i,u]=(K_m-2*mu_star/3)*phi_N[i,u]+2*mu_star*epsilon_rr[i,u]-alpha*P_hat[i,u]
            sigma_tt[i,u]=(K_m-2*mu_star/3)*phi_N[i,u]+2*mu_star*epsilon_tt[i,u]-alpha*P_hat[i,u]
            sigma_pp[i,u]=(K_m-2*mu_star/3)*phi_N[i,u]+2*mu_star*epsilon_pp[i,u]-alpha*P_hat[i,u]
            sigma_rt[i,u]=2*mu_star*epsilon_rt[i,u]
    return(sigma_rr,sigma_tt,sigma_pp,sigma_rt)
    
def comp_heat(epsilon_rr,epsilon_tt,epsilon_pp,epsilon_rt,sigma_rr,sigma_tt,sigma_pp,sigma_rt,omega,eta_f,kappa,qr_hat,qt_hat,P_hat,m_hat):
    h_tide_rr=mp.matrix(thetapts,rpts)
    h_tide_tt=mp.matrix(thetapts,rpts)
    h_tide_pp=mp.matrix(thetapts,rpts)
    h_tide_rt=mp.matrix(thetapts,rpts)
    h_tide_Pm=mp.matrix(thetapts,rpts)
    h_tide=mp.matrix(thetapts,rpts)
    h_vis=mp.matrix(thetapts,rpts)
    for i in range (1,thetapts):
        for u in range (1,rpts):
            h_tide_rr[i,u]=-(omega/2)*(np.real(sigma_rr[i,u])*np.imag(epsilon_rr[i,u])-np.imag(sigma_rr[i,u])*np.real(epsilon_rr[i,u]))
            h_tide_tt[i,u]=-(omega/2)*(np.real(sigma_tt[i,u])*np.imag(epsilon_tt[i,u])-np.imag(sigma_tt[i,u])*np.real(epsilon_tt[i,u]))
            h_tide_pp[i,u]=-(omega/2)*(np.real(sigma_pp[i,u])*np.imag(epsilon_pp[i,u])-np.imag(sigma_pp[i,u])*np.real(epsilon_pp[i,u]))
            h_tide_rt[i,u]=-(omega/2)*(np.real(sigma_rt[i,u])*np.imag(epsilon_rt[i,u])-np.imag(sigma_rt[i,u])*np.real(epsilon_rt[i,u]))
            h_tide_Pm[i,u]=-(omega/2)*(np.real(P_hat[i,u])*np.imag(m_hat[i,u])-np.real(m_hat[i,u])*np.imag(P_hat[i,u]))
            
            h_tide[i,u]=(h_tide_rr[i,u]+h_tide_tt[i,u]+h_tide_pp[i,u]+(2*h_tide_rt[i,u])+h_tide_Pm[i,u])
            h_vis[i,u]=(1/2)*(eta_f/kappa)*((comp_norm(qr_hat[i,u])**2)+(comp_norm(qt_hat[i,u])**2))
    return(h_tide,h_vis)
    
def total_heat(thetamesh,dtheta,rmesh,dr,h_tide,h_vis):
    H_tide=mp.matrix(thetapts,rpts)
    H_vis=mp.matrix(thetapts,rpts)
    H_tide_tot=0
    H_vis_tot=0
    for i in range (1,thetapts):
        for u in range (1,rpts):        
            H_tide[i,u]=2*np.pi*h_tide[i,u]*(rmesh[i,u]**2)*np.sin(thetamesh[i,u])*dr*dtheta
            H_vis[i,u]=2*np.pi*h_vis[i,u]*(rmesh[i,u]**2)*np.sin(thetamesh[i,u])*dr*dtheta
            H_tide_tot=H_tide_tot+H_tide[i,u]
            H_vis_tot=H_vis_tot+H_vis[i,u]
    #print(H_tide,H_vis)

    return(H_tide_tot,H_vis_tot)

#INPUTS

#rpts=465                                               #Number of intervals of depth to integrate over
#thetapts=45                                            #Number of latitude intervals to integrate over

T_omega=33*3600                                         #Orbital period
omega=2*np.pi/T_omega                                   #Angular frequency (of orbit but also of fluid-loading) 
R_0=186000                                              #Radius of core-sea boundary
e=0.0045                                                #Eccentricity of Enceladus' orbit

rho=2400                                                #
G=6.67E-11                                              #
g=0.125

#mu=10E8                                                 #Rigidity of rock matrix
#eta_m=10E15                                             #Viscosity of rock matrix

K_s=10E9                                                #Bulk modulus of rock
#kappa=10E-11                                            #Permeability of rock
                                            #Relaxation time for rock

phi_0=0.3                                               #Porosity of rock
eta_f=0.0019                                            #Pore water viscosity
K_f=2.2E9                                               #Bulk modulus of water
#alpha=0.95                                              #Poroelastic coefficient

#print('Choose parameters of the rock matrix to plot on heatmaps (porosity, rigidity, viscosity or permeability) in format (param1 param2).')
params_in=['porosity', 'rigidity', 'viscosity', 'permeability']
"""
if 'porosity' in params_in and 'rigidity'in params_in:
    print('porosity and rigidity')
if 'porosity' in params_in and 'viscosity' in params_in:
    print('porosity and viscosity')
if 'porosity' in params_in and 'permeability'in params_in:
    print('porosity and permeability')
if 'rigidity' in params_in and 'viscosity' in params_in:
    print('rigidity and viscosity')
if 'rigidity' in params_in and 'permeability'in params_in:
    print('rigidity and permeability')
if 'viscosity' in params_in and 'permeability' in params_in:
    print('viscosity and permeability')
"""

print('Choose number of volume elements to split core into in format (rpts,thetapts), where core is assumed axisymmetric for phi. Suggested - 465 45, 930 90, 1860 180 or 3720 360 (constant parameters only, memory intensive).')
pts_in=list(input().split(" "))
rpts=int(pts_in[0])
thetapts=int(pts_in[1])

print('Choose range of poroelastic/Biot coefficient in format (start stop steps). Suggested - 0.91 0.99 5 or 0.95:')
a_in=list(input().split(" "))
#a1,a2,a3=input().split(" ")
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

print('Choose logarithmic range of rigidity of the rock matrix in format (start stop steps), where mu=10^(input). Suggested - 7 10 4 or 9:')
m_in=list(input().split(" "))
if len(m_in)==3:
    murange=np.logspace(float(m_in[0]),float(m_in[1]),int(m_in[2]))
    
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

print('Choose logarithmic range of rock viscosity in format (start stop steps), where eta_m=10^(input). Suggested - 13 19 7 or 16:')
eta_in=list(input().split(" "))
if len(eta_in)==3:
    eta_m_range=np.logspace(float(eta_in[0]),float(eta_in[1]),int(eta_in[2]))
    
    if len(params_in)>0:
        print(eta_m_range)
        print('What value should viscosity be held at for heatmaps over other 2D parameter spaces? Make sure this value is part of the range specified above.')
        eta_m_constant_in=float(input())
        eta_m_constant=np.where(eta_m_range==eta_m_constant_in)
        
elif len(eta_in)==1:
    eta_m_range=np.array([10**float(eta_in[0])])
else:
    print("Please enter one of these formats: [constant] or [min,max,steps].")

print('Choose logarithmic range of rock permeability in format (start stop steps), where kappa=10^(input). Suggested - -11 -7 5 or -10:')
kp_in=list(input().split(" "))
if len(kp_in)==3:
    kapparange=np.logspace(float(kp_in[0]),float(kp_in[1]),int(kp_in[2]))
    
    if len(params_in)>0:
        print(kapparange)
        print('What value should permeability be held at for heatmaps over other 2D parameter spaces? Make sure this value is part of the range specified above.')
        kappaconstant_in=float(input())
        kappaconstant=np.where(kapparange==kappaconstant_in)
        
elif len(kp_in)==1:
    kapparange=np.array([10**float(kp_in[0])])
else:
    print("Please enter one of these formats: [constant] or [min,max,steps].")


#alpharange=np.linspace(0.91,0.99,3)                         #poroelasticity
#murange=np.logspace(8,10,3)                           #rigidity
#eta_m_range=np.logspace(12,18,3)                       #viscosity
#kapparange=np.logspace(-11,-7,3)                      #permeability

#alpharange=np.array([0.95])
#murange=np.array([10E8])
#eta_m_range=np.array([10E15])
#kapparange=np.array([10E-11])


r=np.linspace(0,R_0,rpts)
dr=r[2]-r[1]

theta=np.linspace(0,np.pi,thetapts)
dtheta=theta[2]-theta[1]

rmesh,thetamesh=np.meshgrid(r,theta)

#MAIN
start=time.time()
h_tide_heatplt=np.zeros((len(alpharange),len(murange),len(eta_m_range),len(kapparange)))
h_vis_heatplt=np.zeros((len(alpharange),len(murange),len(eta_m_range),len(kapparange)))

for al in range(len(alpharange)):
    alpha=alpharange[al]
    for m in range(len(murange)):
        mu=murange[m]
        for et in range(len(eta_m_range)):
            eta_m=eta_m_range[et]
            for kp in range(len(kapparange)):
                kappa=kapparange[kp]
                print('porosity = ' +str(alpha) +' , rigidity = ' +str(mu)+' , viscosity = ' +str(eta_m) +' permeability = ' +str(kappa) +'.')
                
                tau=eta_m/mu
                K_m=(1-alpha)*K_s
                K_u=K_m+alpha**2/(phi_0/K_f+(alpha-phi_0)/K_s)

                mu_star=1/((1/mu)+1/(1j*omega*eta_m))

                c=(kappa/eta_f)*K_f*(1-alpha+4*mu_star/(3*K_s))/(((alpha**2)*K_f/K_s)+(1-alpha+4*mu_star/(3*K_s))*(phi_0+(alpha-phi_0)*K_f/K_s))
                k=(1-1j)*(np.sqrt(omega/(2*c)))
                print('k=' + str(k))

                e_x=np.multiply(k,rmesh[0,0])
                print('zero_x='+str(e_x))
                
                e_x=np.multiply(k,rmesh[1,1])
                print('low_x='+str(e_x))
                e_x=np.multiply(k,rmesh[thetapts-1,rpts-1])
                print('high_x='+str(e_x))
                j,djr,ddjr=bessel2mesh(k,rmesh)
                _,jC,_,ddjrC=bessel2int(k,R_0) #The bessel function and its derivatives evaluated at R_0
                print('bessel functions acquired')

                epsilon_0=(9/(4*np.pi))*e*(omega**2/(G*rho))*(5/3)*(3/2)/comp_norm(1+19*mu_star/(2*rho*g*R_0))#np.sqrt((np.real(1+19*mu_star/(2*rho*g*R_0))**2)+(np.imag(1+19*mu_star/(2*rho*g*R_0))**2))
                b_0=alpha/((k**2)*(K_m+4*mu_star/3))
                
                F,C1,C2=comp_const(epsilon_0,alpha,k,K_m,mu_star,R_0,b_0,jC,ddjrC)

                phi_N,P_hat,m_hat,qr_hat,qt_hat=comp_hat(thetamesh,rmesh,j,djr,ddjr,C1,C2,alpha,K_m,K_u,mu_star,kappa,eta_f)
                print('complex components acquired')
                
                epsilon_rr,epsilon_tt,epsilon_pp,epsilon_rt=strain_comps(thetamesh,C2,b_0,C1,ddjr,j,djr,rmesh,phi_N)
                print('strain components acquired')
                
                sigma_rr,sigma_tt,sigma_pp,sigma_rt=stress_comps(epsilon_rr,epsilon_tt,epsilon_pp,epsilon_rt,K_m,mu_star,phi_N,alpha,P_hat)
                print('stress components acquired')
                
                h_tide,h_vis=comp_heat(epsilon_rr,epsilon_tt,epsilon_pp,epsilon_rt,sigma_rr,sigma_tt,sigma_pp,sigma_rt,omega,eta_f,kappa,qr_hat,qt_hat,P_hat,m_hat)
                print('complex heat components acquired')
                
                H_tide_tot,H_vis_tot=total_heat(thetamesh,dtheta,rmesh,dr,h_tide,h_vis)
                
                print('total heat calculated')
                print(H_tide_tot)
                print(H_vis_tot)
                h_tide_heatplt[al,m,et,kp]=H_tide_tot
                h_vis_heatplt[al,m,et,kp]=H_vis_tot

h_tot_heatplt=h_tide_heatplt+h_vis_heatplt
if len(alpharange)==len(murange)==len(eta_m_range)==len(kapparange)==1:
    np.savetxt('long sim constant parameters.txt',('h_tide = '+str(h_tide_heatplt)+ '\n' + 'h_vis =' +str(h_vis_heatplt)+'\n'+'h_tot_heatplt = '+str(h_tot_heatplt)))

elif len(murange)==len(eta_m_range)==len(kapparange)==1:
    np.savetxt('long sim variable alpha.txt',('h_tide = \n'+(h_tide_heatplt)+ '\n' + 'h_vis = \n' +(h_vis_heatplt)+'\n'+'h_tot_heatplt = \n'+(h_tot_heatplt)))
else:
    ptcol=np.array(np.meshgrid(eta_m_range,kapparange,murange,alpharange)).T.reshape(len(alpharange)*len(murange)*len(eta_m_range)*len(kapparange),4)

    h_tide_flat=np.ndarray.flatten(h_tide_heatplt)
    np.savetxt('h_tide heating 4d datafile.txt',np.column_stack((ptcol,h_tide_flat)),fmt='%E',delimiter=' , ')

    h_vis_flat=np.ndarray.flatten(h_vis_heatplt)
    np.savetxt('h_vis heating 4d datafile.txt',np.column_stack((ptcol,h_vis_flat)),fmt='%E',delimiter=' , ')

    h_tot_flat=np.ndarray.flatten(h_tot_heatplt)
    np.savetxt('full tidal heating 4d datafile.txt',np.column_stack((ptcol,h_tot_flat)),fmt='%E',delimiter=' , ')


""" 
alphaconstant=int(alphaconstant[0])
muconstant=int(muconstant[0])
eta_m_constant=int(eta_m_constant[0])
kappaconstant=int(kappaconstant[0])          
"""
"""
plt.imshow(h_tot_heatplt[0,0,:,:], aspect='auto', interpolation='bilinear', extent=[eta_m_range[0],eta_m_range[len(eta_m_range)-1],kapparange[0],kapparange[len(kapparange)-1]])
plt.title("Total heat generated due to tidal dissapation in the core of Enceladus over parameter space");
plt.xlabel("Viscosity of rock matrix (Pa s)");
plt.xscale('linear')
plt.xticks(np.linspace(eta_m_range[0],eta_m_range[len(eta_m_range)-1],len(eta_m_range)),eta_m_range); 
plt.ylabel("Permability of rock matrix (m^2)");
plt.yscale('linear')
plt.yticks(np.linspace(kapparange[0],kapparange[len(kapparange)-1],len(kapparange)),kapparange); 
plt.colorbar()
plt.savefig('heat generation.png')

plt.show()
"""
"""
if 'porosity' in params_in and 'rigidity'in params_in:
    print('porosity and rigidity')
    ptcol=np.array(np.meshgrid(alpharange,murange)).T.reshape(len(alpharange)*len(murange),2)
    h_tot_flat=np.ndarray.flatten(h_tot_heatplt[:,:,eta_m_constant,kappaconstant],'F')
    np.savetxt('alpha v mu.txt',np.column_stack((ptcol,h_tot_flat)),fmt='%E',delimiter=' , ')
    
    f = plt.figure(figsize=(20, 20)); # 
    
    plt.imshow(h_tot_heatplt[:,:,eta_m_constant,kappaconstant], origin='upper',vmin=5E6,vmax=1E11,cmap='hot',norm=colors.LogNorm(), aspect='auto', interpolation='spline16', extent=[alpharange[0],alpharange[len(alpharange)-1],murange[0],murange[len(murange)-1]])
    plt.title("Total heat generated due to tidal dissapation in the core of Enceladus over parameter spaces of viscosity and permeability");
    plt.xlabel("Porosity of rock matrix");
    plt.xscale('linear')
    plt.xticks(np.linspace(alpharange[0],alpharange[len(alpharange)-1],len(alpharange)),alpharange); 
    plt.ylabel("Rigidity of rock matrix (Pa.s)");
    plt.yscale('linear')
    plt.yticks(np.linspace(murange[0],murange[len(murange)-1],len(murange)),murange); 
    plt.colorbar()
    plt.savefig('tidal heat generation - alpha v mu.png')

    plt.show()
    
if 'porosity' in params_in and 'viscosity' in params_in:
    print('porosity and viscosity')
    ptcol=np.array(np.meshgrid(alpharange,eta_m_range)).T.reshape(len(alpharange)*len(eta_m_range),2)
    h_tot_flat=np.ndarray.flatten(h_tot_heatplt[:,muconstant,:,kappaconstant],'F')
    np.savetxt('alpha v eta.txt',np.column_stack((ptcol,h_tot_flat)),fmt='%E',delimiter=' , ')
    
    f = plt.figure(figsize=(20, 20)); #  

    plt.imshow(h_tot_heatplt[:,muconstant,:,kappaconstant], origin='upper',vmin=5E6,vmax=1E11,cmap='hot',norm=colors.LogNorm(), aspect='auto', interpolation='spline16', extent=[alpharange[0],alpharange[len(alpharange)-1],eta_m_range[0],eta_m_range[len(eta_m_range)-1]])
    plt.title("Total heat generated due to tidal dissapation in the core of Enceladus over parameter spaces of viscosity and permeability");
    plt.xlabel("Porosity of rock matrix");
    plt.xscale('linear')
    plt.xticks(np.linspace(alpharange[0],alpharange[len(alpharange)-1],len(alpharange)),alpharange); 
    plt.ylabel("Viscosity of rock matrix (Pa.s)");
    plt.yscale('linear')
    plt.yticks(np.linspace(eta_m_range[len(eta_m_range)-1],eta_m_range[0],len(eta_m_range)),eta_m_range); 
    plt.colorbar()
    plt.savefig('tidal heat generation - alpha v eta.png')

    plt.show()
    
if 'porosity' in params_in and 'permeability'in params_in:
    print('porosity and permeability')
    ptcol=np.array(np.meshgrid(alpharange,kapparange)).T.reshape(len(alpharange)*len(kapparange),2)
    h_tot_flat=np.ndarray.flatten(h_tot_heatplt[:,muconstant,eta_m_constant,:],'F')
    np.savetxt('alpha v kappa.txt',np.column_stack((ptcol,h_tot_flat)),fmt='%E',delimiter=' , ')
    
    f = plt.figure(figsize=(20, 20)); #  

    plt.imshow(h_tot_heatplt[:,muconstant,eta_m_constant,:],origin='upper',vmin=5E6,vmax=1E11,cmap='hot',norm=colors.LogNorm(), aspect='auto', interpolation='spline16', extent=[alpharange[0],alpharange[len(alpharange)-1],kapparange[0],kapparange[len(kapparange)-1]])
    plt.title("Total heat generated due to tidal dissapation in the core of Enceladus over parameter spaces of viscosity and permeability");
    plt.xlabel("Porosity of rock matrix");
    plt.xscale('linear')
    plt.xticks(np.linspace(alpharange[0],alpharange[len(alpharange)-1],len(alpharange)),alpharange); 
    plt.ylabel("Permability of rock matrix (m^2)");
    plt.yscale('linear')
    plt.yticks(np.linspace(kapparange[len(kapparange)-1],kapparange[0],len(kapparange)),kapparange); 
    plt.colorbar()
    plt.savefig('tidal heat generation - alpha v kappa.png')

    plt.show()
    
if 'rigidity' in params_in and 'viscosity' in params_in:
    print('rigidity and viscosity')
    ptcol=np.array(np.meshgrid(murange,eta_m_range)).T.reshape(len(murange)*len(eta_m_range),2)
    h_tot_flat=np.ndarray.flatten(h_tot_heatplt[alphaconstant,:,:,kappaconstant],'F')
    np.savetxt('mu v eta.txt',np.column_stack((ptcol,h_tot_flat)),fmt='%E',delimiter=' , ')
    
    f = plt.figure(figsize=(20, 20)); #  

    plt.imshow(h_tot_heatplt[alphaconstant,:,:,kappaconstant], origin='upper',vmin=5E6,vmax=1E11,cmap='hot',norm=colors.LogNorm(), aspect='auto', interpolation='spline16', extent=[murange[0],murange[len(murange)-1],eta_m_range[0],eta_m_range[len(eta_m_range)-1]])
    plt.title("Total heat generated due to tidal dissapation in the core of Enceladus over parameter spaces of viscosity and permeability");
    plt.xlabel("Rigidity of rock matrix (Pa.s)");
    plt.xscale('linear')
    plt.xticks(np.linspace(murange[0],murange[len(murange)-1],len(murange)),murange); 
    plt.ylabel("Viscosity of rock matrix (Pa.s)");
    plt.yscale('linear')
    plt.yticks(np.linspace(eta_m_range[len(eta_m_range)-1],eta_m_range[0],len(eta_m_range)),eta_m_range); 
    plt.colorbar()
    plt.savefig('tidal heat generation - mu v eta.png')

    plt.show()
    
if 'rigidity' in params_in and 'permeability'in params_in:
    print('rigidity and permeability')
    ptcol=np.array(np.meshgrid(murange,kapparange)).T.reshape(len(murange)*len(kapparange),2)
    h_tot_flat=np.ndarray.flatten(h_tot_heatplt[alphaconstant,:,eta_m_constant,:],'F')
    np.savetxt('mu v kappa.txt',np.column_stack((ptcol,h_tot_flat)),fmt='%E',delimiter=' , ')
    
    f = plt.figure(figsize=(20, 20)); #  

    plt.imshow(h_tot_heatplt[alphaconstant,:,eta_m_constant,:], origin='upper',vmin=5E6,vmax=1E11,cmap='hot',norm=colors.LogNorm(), aspect='auto', interpolation='spline16', extent=[murange[0],murange[len(murange)-1],kapparange[0],kapparange[len(kapparange)-1]])
    plt.title("Total heat generated due to tidal dissapation in the core of Enceladus over parameter spaces of viscosity and permeability");
    plt.xlabel("Rigidity of rock matrix (Pa.s)");
    plt.xscale('linear')
    plt.xticks(np.linspace(murange[0],murange[len(murange)-1],len(murange)),murange); 
    plt.ylabel("Permability of rock matrix (m^2)");
    plt.yscale('linear')
    plt.yticks(np.linspace(kapparange[len(kapparange)-1],kapparange[0],len(kapparange)),kapparange); 
    plt.colorbar()
    plt.savefig('tidal heat generation - mu v kappa.png')

    plt.show()
    
if 'viscosity' in params_in and 'permeability' in params_in:
    print('viscosity and permeability')
    ptcol=np.array(np.meshgrid(eta_m_range,kapparange)).T.reshape(len(eta_m_range)*len(kapparange),2)
    h_tot_flat=np.ndarray.flatten(h_tot_heatplt[alphaconstant,muconstant,:,:],'F')
    np.savetxt('eta v kappa.txt',np.column_stack((ptcol,h_tot_flat)),fmt='%E',delimiter=' , ')
    
    f = plt.figure(figsize=(20, 20)); #  

    plt.imshow(h_tot_heatplt[alphaconstant,muconstant,:,:], origin='upper',vmin=5E6,vmax=1E11,cmap='hot',norm=colors.LogNorm(), aspect='auto', interpolation='spline16', extent=[eta_m_range[0],eta_m_range[len(eta_m_range)-1],kapparange[0],kapparange[len(kapparange)-1]])
    plt.title("Total heat generated due to tidal dissapation in the core of Enceladus over parameter spaces of viscosity and permeability");
    plt.xlabel("Viscosity of rock matrix (Pa s)");
    plt.xscale('linear')
    plt.xticks(np.linspace(eta_m_range[0],eta_m_range[len(eta_m_range)-1],len(eta_m_range)),eta_m_range); 
    plt.ylabel("Permability of rock matrix (m^2)");
    plt.yscale('linear')
    plt.yticks(np.linspace(kapparange[len(kapparange)-1],kapparange[0],len(kapparange)),kapparange); 
    plt.colorbar()
    plt.savefig('tidal heat generation - eta v kappa.png')

    plt.show()
"""
end=time.time()
print('Total runtime of '+str(end-start)+' seconds.')
sys.exit("Finished")