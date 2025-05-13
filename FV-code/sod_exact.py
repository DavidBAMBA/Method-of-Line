
import os, glob, re
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton

# ════════════════════════════════════════════════════════════════════════
#  Solución exacta (idéntica a la que compartiste, solo cambio gamma = 1.4)
# ════════════════════════════════════════════════════════════════════════
def exact_riemann_solution(rhoL, uL, pL, rhoR, uR, pR, gamma=5.0/3.0, x0=0.5, t=0.2):
    """
    Solución exacta del problema de Riemann para condiciones iniciales arbitrarias.
    Devuelve un diccionario con todos los parámetros relevantes de la solución.
    """
    # Constantes del gas
    gm = gamma
    gm1 = gm - 1.0
    gm_p1 = gm + 1.0

    # Velocidades de sonido iniciales
    cL = np.sqrt(gm * pL / rhoL)
    cR = np.sqrt(gm * pR / rhoR)

    # Función para ecuación de p_star
    def pressure_star_eq(p_star):
        # Onda izquierda
        if p_star > pL:
            AL = 2.0 / (gm_p1 * rhoL)
            BL = gm1/gm_p1 * pL
            fL = (p_star - pL) * np.sqrt(AL/(p_star + BL))
        else:
            fL = 2*cL/gm1 * ((p_star/pL)**((gm1)/(2*gm)) - 1)
        
        # Onda derecha
        if p_star > pR:
            AR = 2.0 / (gm_p1 * rhoR)
            BR = gm1/gm_p1 * pR
            fR = (p_star - pR) * np.sqrt(AR/(p_star + BR))
        else:
            fR = 2*cR/gm1 * ((p_star/pR)**((gm1)/(2*gm)) - 1)
        
        return fL + fR + (uR - uL)

    # Resolver para p_star
    p_pvrs = 0.5 * (pL + pR) - 0.125 * (uR - uL) * (rhoL + rhoR) * (cL + cR)
    p_guess = max(1e-6, p_pvrs)  # Estimación robusta para casos extremos
    p_star = newton(pressure_star_eq, p_guess, maxiter=500, tol=1e-10)
    """ p_guess = 0.5*(pL + pR)
    p_star = newton(pressure_star_eq, p_guess, maxiter=500)
 """
    # Calcular u_star
    if p_star > pL:
        AL = 2.0/(gm_p1 * rhoL)
        BL = gm1/gm_p1 * pL
        u_star = uL + (p_star - pL)*np.sqrt(AL/(p_star + BL))
    else:
        u_star = uL + 2*cL/gm1*(1 - (p_star/pL)**((gm1)/(2*gm)))

    # Calcular densidades estrella
    if p_star > pL:
        rho_starL = rhoL*((p_star/pL) + (gm1)/gm_p1)/((gm1)/gm_p1*(p_star/pL) + 1)
    else:
        rho_starL = rhoL*(p_star/pL)**(1/gm)
    
    if p_star > pR:
        rho_starR = rhoR*((p_star/pR) + (gm1)/gm_p1)/((gm1)/gm_p1*(p_star/pR) + 1)
    else:
        rho_starR = rhoR*(p_star/pR)**(1/gm)

    # Calcular velocidades de las ondas
    wave_speeds = {}
    if p_star <= pL:
        c_starL = cL*(p_star/pL)**((gm1)/(2*gm))
        wave_speeds['x_head'] = x0 - cL*t
        wave_speeds['x_tail'] = x0 + (u_star - c_starL)*t
    else:
        shock_speed_L = uL + cL*np.sqrt((gm_p1/(2*gm))*(p_star/pL) + gm1/(2*gm))
        wave_speeds['x_shock_L'] = x0 + shock_speed_L*t

    wave_speeds['x_contact'] = x0 + u_star*t

    if p_star <= pR:
        c_starR = cR*(p_star/pR)**((gm1)/(2*gm))
        wave_speeds['x_tail_R'] = x0 + (u_star + c_starR)*t
    else:
        shock_speed_R = uR + cR*np.sqrt((gm_p1/(2*gm))*(p_star/pR) + gm1/(2*gm))
        wave_speeds['x_shock_R'] = x0 + shock_speed_R*t

    return {
        'p_star': p_star,
        'u_star': u_star,
        'rho_starL': rho_starL,
        'rho_starR': rho_starR,
        'cL': cL,
        'cR': cR,
        **wave_speeds
    }

def general_shock_tube_solution(x, t, rhoL, uL, pL, rhoR, uR, pR, gamma=5.0/3.0, x0=0.5):
    """
    Solución analítica general para cualquier condición inicial.
    """
    sol = exact_riemann_solution(rhoL, uL, pL, rhoR, uR, pR, gamma, x0, t)
    
    # Extraer parámetros
    p_star = sol['p_star']
    u_star = sol['u_star']
    rho_starL = sol['rho_starL']
    rho_starR = sol['rho_starR']
    cL = sol['cL']
    cR = sol['cR']
    
    # Determinar posiciones de las ondas
    if 'x_shock_L' in sol:
        x_head = sol['x_shock_L']
        x_tail = sol['x_shock_L']
    else:
        x_head = sol['x_head']
        x_tail = sol['x_tail']
    
    x_contact = sol['x_contact']
    
    if 'x_shock_R' in sol:
        x_shock_R = sol['x_shock_R']
    else:
        x_shock_R = sol['x_tail_R']

    # Inicializar arrays
    rho = np.zeros_like(x)
    u = np.zeros_like(x)
    p = np.zeros_like(x)

    # Asignar estados
    for i, xi in enumerate(x):
        if xi < x_head:
            rho[i] = rhoL
            u[i] = uL
            p[i] = pL
        elif xi < x_tail:
            if p_star <= pL:
                c = (xi - x0)/t
                u[i] = 2/(gamma+1)*(c + (gamma-1)/2*uL + cL)
                c_local = cL - 0.5*(gamma-1)*u[i]
                rho[i] = rhoL*(c_local/cL)**(2/(gamma-1))
                p[i] = pL*(rho[i]/rhoL)**gamma
        elif xi < x_contact:
            rho[i] = rho_starL
            u[i] = u_star
            p[i] = p_star
        elif xi < x_shock_R:
            rho[i] = rho_starR
            u[i] = u_star
            p[i] = p_star
        else:
            rho[i] = rhoR
            u[i] = uR
            p[i] = pR

    e_int = p/((gamma-1.0)*rho)
    return rho, u, p, e_int



