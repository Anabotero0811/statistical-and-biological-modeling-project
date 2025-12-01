import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares


# ============================
# 1. MODELO DE GOODWIN
# ============================
def goodwin_rhs(t, y, params):
    X, Y, Z = y
    a, b, c, d, e, f, n = params

    dXdt = a / (1 + Z ** n) - b * X
    dYdt = c * X - d * Y
    dZdt = e * Y - f * Z
    return [dXdt, dYdt, dZdt]


def simulate_goodwin(t_eval, y0, params):
    sol = solve_ivp(
        fun=lambda t, y: goodwin_rhs(t, y, params),
        t_span=(t_eval[0], t_eval[-1]),
        y0=y0,
        t_eval=t_eval,
        method="RK45"
    )
    return sol.y


# ============================
# 2. GENERAR DATOS SINTÉTICOS
# ============================
np.random.seed(10)

true_params = [2.0, 0.7, 3.0, 0.9, 1.1, 0.8, 10]  # a,b,c,d,e,f,n
t = np.linspace(0, 48, 120)
y0_true = [0.5, 0.5, 0.5]

X_true, Y_true, Z_true = simulate_goodwin(t, y0_true, true_params)
sigma_noise = 0.3
X_obs = X_true + np.random.normal(0, sigma_noise, len(X_true))

# ============================
# 3. AJUSTE DE PARÁMETROS
# ============================

# Ajuste para a, b, c, n (los otros se fijan)
params_fixed = [None, None, None, 0.9, 1.1, 0.8, None]


def residuals_to_fit(p_vec):
    a, b, c, n = p_vec
    params = [a, b, c, 0.9, 1.1, 0.8, n]
    X_sim = simulate_goodwin(t, y0_true, params)[0]
    return X_sim - X_obs


p0 = [1.0, 0.5, 2.0, 8.0]  # a, b, c, n (iniciales)
bounds = ([0, 0, 0, 1], [10, 5, 10, 20])

result = least_squares(residuals_to_fit, p0, bounds=bounds)
a_fit, b_fit, c_fit, n_fit = result.x

params_fit = [a_fit, b_fit, c_fit, 0.9, 1.1, 0.8, n_fit]
X_fit, Y_fit, Z_fit = simulate_goodwin(t, y0_true, params_fit)

# ============================
# 4. GRÁFICAS
# ============================

# ---- Gráfica 1: Observado vs Ajustado vs Verdadero ----
plt.figure(figsize=(10, 5))
plt.plot(t, X_obs, 'o', markersize=4, label="Datos observados (ruido)")
plt.plot(t, X_fit, '-', linewidth=2, label="Modelo ajustado")
plt.plot(t, X_true, '--', linewidth=2, label="Modelo verdadero (sin ruido)")
plt.xlabel("Tiempo (h)")
plt.ylabel("Concentración X")
plt.title("Ajuste del modelo de Goodwin")
plt.legend()
plt.grid()
#plt.savefig("Ajuste del modelo de Goodwin", dpi=600)
plt.show()

# ---- Gráfica 2: Trayectorias ajustadas (X, Y, Z) ----
plt.figure(figsize=(10, 5))
plt.plot(t, X_fit, label="X ajustado")
plt.plot(t, Y_fit, label="Y ajustado")
plt.plot(t, Z_fit, label="Z ajustado")
plt.xlabel("Tiempo (h)")
plt.ylabel("Concentración")
plt.title("Solución del modelo ajustado (Goodwin)")
plt.legend()
plt.grid()
#plt.savefig("Solución del modelo ajustado (Goodwin)", dpi=600)
plt.show()

# ---- Gráfica 3: Residuos ----
residuos = X_fit - X_obs
plt.figure(figsize=(10, 4))
plt.plot(t, residuos, marker='o', linestyle='None',
         color="steelblue",
         markersize=3, label="Residuos")
plt.axhline(0, color="blue", linewidth=1, linestyle='--')
plt.xlabel("Tiempo (h)")
plt.ylabel("Residuo (X_ajustado − X_obs)")
plt.title("Residuos del ajuste")
plt.grid()
plt.legend()
#plt.savefig("Residuos del ajuste", dpi=600)
plt.show()

