"""
Various functions for modelling and fitting dipoles and quadrupoles

Author: Paula Boubel
"""
import astropy.coordinates as coord
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np
import math
import os
import healpy as hp

def parabola_lin(x, a0, a1, a2, bp=0):
    linear_TFR = a1*x + a0
    
    curvature = a2 * (x-bp)**2 
    curvature_mask = np.logical_not(x<=bp)
        
    return linear_TFR + curvature*curvature_mask
pi = np.pi
def IndexToDeclRa(index,NSIDE):
    theta,phi=hp.pixelfunc.pix2ang(NSIDE,index)
    return -np.degrees(theta-pi/2.),np.degrees(pi*2.-phi)

def DeclRaToIndex(decl,RA,NSIDE):
    return hp.pixelfunc.ang2pix(NSIDE,np.radians(-decl+90.),np.radians(360.-RA))

def nu_approx(vp, z):
    #dC = cosmo.comoving_distance(z).value
    nu = vp * (1 + z) / np.log(10) / c / z
    #log10_dC = np.log10(dC) - nu 
    
    return nu

def sph_dipole(ra, dec, monopole, a11r, a11i, a01r, a01i):
    theta,phi = np.radians(-dec+90.),np.radians(360.-ra)
    a00 = monopole/(1/(2*np.sqrt(np.pi)))
    am11i = -a11i
    am11r = a11r
    
    Y11 = -0.5*np.sqrt(3/2/np.pi)*np.exp(phi*1j)*np.sin(theta)
    Y01 = 0.5*np.sqrt(3/np.pi)*np.cos(theta)
    Ym11 = 0.5*np.sqrt(3/2/np.pi)*np.exp(phi*-1j)*np.sin(theta)
    Y00 = 1/(2*np.sqrt(np.pi))
    
    return (a11r + a11i*1j)*Y11 + (a01r + a01i*1j)*Y01 + (am11r + am11i*1j)*Ym11 + a00*Y00


def radec_to_lb(ra, dec):
    l = SkyCoord(ra*u.degree, dec*u.degree, frame='icrs').transform_to('galactic').l.value
    b = SkyCoord(ra*u.degree, dec*u.degree, frame='icrs').transform_to('galactic').b.value
    l = coord.Angle(l*u.degree)
    l= l.wrap_at(180*u.degree)
    b = coord.Angle(b*u.degree)
    return l, b

def sph_dipole_moment(l, b, px, py, pz):
    ## px, py, pz, is u, v, w in galactic coordinates
    theta,phi = np.radians(-b.value+90.),np.radians(360.+l.value)

    a=np.sin(theta)*np.cos(phi)*px
    b=np.sin(theta)*np.sin(phi)*py
    c=np.cos(theta)*pz
    
    return (a + b + c)

def sph_quadrupole_moment(l, b, a20, a21, b21, a22, b22):
    theta,phi = np.radians(-b.value+90.),np.radians(360.+l.value)

    a = a20 * (3*np.cos(theta)**2 - 1)
    b = a21 * np.sin(theta) * np.cos(theta) * np.cos(phi)
    c = b21 * np.sin(theta) * np.cos(theta) * np.sin(phi)
    d = a22 * np.sin(theta)**2 * np.cos(2*phi)
    e = b22 * np.sin(theta)**2 * np.sin(2*phi)
    
    return (a + b + c + d + e)

def log_likelihood_dipole(theta, z, l, b, w, m, xerr, yerr, params, form):

    a1 = params[1]
    a2 = params[2]
    e0 = params[3]
    e1 = params[4]

    if form == 'quadrupole':
        a0, a20, a21, b21, a22, b22 = theta
        quadrupole = sph_quadrupole_moment(l, b, a20, a21, b21, a22, b22)
    elif form == 'quaddip':
        a0, px, py, pz, a20, a21, b21, a22, b22 = theta
        quadrupole = sph_quadrupole_moment(l, b, a20, a21, b21, a22, b22)
        dipole = sph_dipole_moment(l, b, px, py, pz)
    else:
        a0, px, py, pz = theta
        dipole = sph_dipole_moment(l, b, px, py, pz)
            
    TF_params = [a0, a1, a2]

    sigma2_linear = (e1*w + e0) ** 2  + yerr ** 2 + (xerr*a1)**2
    sigma2_curved = (e1*w + e0) ** 2  + yerr ** 2 + (xerr**2)*(a1 + 2*a2*w)**2
    sigma2 = sigma2_linear*(w<=0) + sigma2_curved*(w>0)

    model = parabola_lin(w,*TF_params,bp=0)
    if form == 'velocity':
        model += 5*nu_approx(dipole, z)
    if form == 'zeropoint':
        model += dipole
    if form == 'quadrupole':
        model += quadrupole
    if form == 'quaddip':
        model += dipole + quadrupole

    diff = (m-model)**2

    lh = (diff / sigma2 + np.log(sigma2))
    
    lp = -0.5 * np.nansum(lh)
    
    if np.isfinite(lp):
        return lp
    else:
        return -np.inf
    
def log_likelihood_vp_dipole(theta, z, l, b, w, m, xerr, yerr, params):

    a1 = params[1]
    a2 = params[2]
    e0 = params[3]
    e1 = params[4]

    a0, px, py, pz = theta
        
    vp_dipole = sph_dipole_moment(l, b, px, py, pz)
    
    TF_params = [a0, a1, a2]

    sigma2_linear = (e1*w + e0) ** 2  + yerr ** 2 + (xerr*a1)**2
    sigma2_curved = (e1*w + e0) ** 2  + yerr ** 2 + (xerr**2)*(a1 + 2*a2*w)**2
    sigma2 = sigma2_linear*(w<=0) + sigma2_curved*(w>0)

    model = parabola_lin(w,*TF_params,bp=0)
    model += 5*nu_approx(vp_dipole, z)

    diff = (m-model)**2

    lh = (diff / sigma2 + np.log(sigma2))
    
    lp = -0.5 * np.nansum(lh)
    
    if np.isfinite(lp):
        return lp
    else:
        return -np.inf

def log_likelihood_ZP_variable(theta, z, l, b, w, m, xerr, yerr, params, fixed_monopole, fit_quadrupole, fit_velocity_dipole):

    a1 = params[1]
    a2 = params[2]
    e0 = params[3]
    e1 = params[4]
    
    if fit_velocity_dipole:
        if fixed_monopole:
            monopole = params[0]
            if fit_quadrupole:
                px, py, pz, a20, a21, b21, a22, b22, vx, vy, vz = theta
            else:
                px, py, pz, vx, vy, vz = theta
        else:
            if fit_quadrupole:
                monopole, px, py, pz, a20, a21, b21, a22, b22, vx, vy, vz = theta
            else:
                monopole, px, py, pz, vx, vy, vz = theta
        
        vp_dipole = sph_dipole_moment(l, b, vx, vy, vz)
    
    else:
        if fixed_monopole:
            monopole = params[0]
            if fit_quadrupole:
                px, py, pz, a20, a21, b21, a22, b22 = theta
            else:
                px, py, pz = theta
        else:
            if fit_quadrupole:
                monopole, px, py, pz, a20, a21, b21, a22, b22 = theta
            else:
                monopole, px, py, pz = theta

    zp = monopole + sph_dipole_moment(l, b, px, py, pz)
    if fit_quadrupole:
        quadrupole = sph_quadrupole_moment(l, b, a20, a21, b21, a22, b22)
        zp += quadrupole
    
    TF_params = [zp, a1, a2]

    sigma2_linear = (e1*w + e0) ** 2  + yerr ** 2 + (xerr*a1)**2
    sigma2_curved = (e1*w + e0) ** 2  + yerr ** 2 + (xerr**2)*(a1 + 2*a2*w)**2
    sigma2 = sigma2_linear*(w<=0) + sigma2_curved*(w>0)

    model = parabola_lin(w,*TF_params,bp=0)
    if fit_velocity_dipole:
        model += 5*nu_approx(vp_dipole, z)

    diff = (m-model)**2

    lh = (diff / sigma2 + np.log(sigma2))
    
    lp = -0.5 * np.nansum(lh)
    
    if np.isfinite(lp):
        return lp
    else:
        return -np.inf

def log_likelihood_ZP_variable_multiphot(theta, z, l, b, w, m1, m2, xerr, yerr, params1, params2, fixed_monopole, fit_quadrupole, fit_velocity_dipole):
    ## Fit a zp dipole and quadrupole to two photometry datasets
    
    a11 = params1[1]
    a21 = params1[2]
    e01 = params1[3]
    e11 = params1[4]
    
    a12 = params2[1]
    a22 = params2[2]
    e02 = params2[3]
    e12 = params2[4]
    
    if fit_velocity_dipole:
        if fixed_monopole:
            monopole1 = params1[0]
            monopole2 = params2[0]
            if fit_quadrupole:
                px, py, pz, a20, a21, b21, a22, b22, vx, vy, vz = theta
            else:
                px, py, pz, vx, vy, vz = theta
        else:
            if fit_quadrupole:
                monopole1, monopole2, px, py, pz, a20, a21, b21, a22, b22, vx, vy, vz = theta
            else:
                monopole1, monopole2, px, py, pz, vx, vy, vz = theta
        
        vp_dipole = sph_dipole_moment(l, b, vx, vy, vz)
    
    else:
        if fixed_monopole:
            monopole1 = params1[0]
            monopole2 = params2[0]
            if fit_quadrupole:
                px, py, pz, a20, a21, b21, a22, b22 = theta
            else:
                px, py, pz = theta
        else:
            if fit_quadrupole:
                monopole1, monopole2, px, py, pz, a20, a21, b21, a22, b22 = theta
            else:
                monopole1, monopole2, px, py, pz = theta

    dipole = sph_dipole_moment(l, b, px, py, pz)
    zp1 = monopole1 + dipole
    zp2 = monopole2 + dipole

    if fit_quadrupole:
        quadrupole = sph_quadrupole_moment(l, b, a20, a21, b21, a22, b22)
        zp1 += quadrupole
        zp2 += quadrupole

    TF_params1 = [zp1, a11, a21]
    TF_params2 = [zp2, a12, a22]

    sigma2_linear = (e11*w + e01) ** 2  + yerr ** 2 + (xerr*a11)**2
    sigma2_curved = (e11*w + e01) ** 2  + yerr ** 2 + (xerr**2)*(a11 + 2*a21*w)**2
    sigma21 = sigma2_linear*(w<=0) + sigma2_curved*(w>0)

    sigma2_linear = (e12*w + e02) ** 2  + yerr ** 2 + (xerr*a12)**2
    sigma2_curved = (e12*w + e02) ** 2  + yerr ** 2 + (xerr**2)*(a12 + 2*a22*w)**2
    sigma22 = sigma2_linear*(w<=0) + sigma2_curved*(w>0)

    model1 = parabola_lin(w,*TF_params1,bp=0)
    model2 = parabola_lin(w,*TF_params2,bp=0)
    if fit_velocity_dipole:
        model1 += 5*nu_approx(vp_dipole, z)
        model2 += 5*nu_approx(vp_dipole, z)


    inds1 = np.where(m1!=0)[0]
    inds2 = np.where(m2!=0)[0]

    diff1 = (m1-model1)**2
    diff2 = (m2-model2)**2

    lh1 = (diff1 / sigma21 + np.log(sigma21))[inds1]
    lh2 = (diff2 / sigma22 + np.log(sigma22))[inds2]
    return -0.5 * np.nansum(lh1) - 0.5 * np.nansum(lh2)

def log_likelihood_ZP(theta, w, m, xerr, yerr, params):

    a1 = params[1]
    a2 = params[2]
    e0 = params[3]
    e1 = params[4]

    a0 = theta
    TF_params = [a0, a1, a2]

    sigma2_linear = (e1*w + e0) ** 2  + yerr ** 2 + (xerr*a1)**2
    sigma2_curved = (e1*w + e0) ** 2  + yerr ** 2 + (xerr**2)*(a1 + 2*a2*w)**2
    sigma2 = sigma2_linear*(w<=0) + sigma2_curved*(w>0)

    model = parabola_lin(w,*TF_params,bp=0)

    diff = (m-model)**2

    lh = (diff / sigma2 + np.log(sigma2))
    lp = -0.5 * np.nansum(lh)

    if np.isfinite(lp):
        return lp
    else:
        return -np.inf
