# ============================================================
#  Análisis del comportamiento dinámico del modelo de Goodwin
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from scipy.signal import find_peaks
from mpl_toolkits.mplot3d import Axes3D  # necesario para 3D

# ------------------------------------------------------------
# 1. Definición del modelo de Goodwin (3 variables)
# ------------------------------------------------------------
def goodwin_rhs(t, y, params):
    X, Y, Z = y
    a, b, c, d, e, f, n = params

    dXdt = a / (1 + Z**n) - b * X
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
    if not sol.success:
        raise RuntimeError("Falló la integración: " + sol.message)
    return sol.y  # devuelve array de forma (3, len(t_eval))


# ------------------------------------------------------------
# 2. Generación de datos sintéticos oscilatorios
# ------------------------------------------------------------
np.random.seed(1)

# Parámetros "verdaderos" (escogidos para que haya oscilaciones claras)
true_params = [1.0, 0.3, 1.0, 0.3, 1.0, 0.3, 12.0]  # a,b,c,d,e,f,n

# Condiciones iniciales
y0_true = [0.1, 0.1, 0.1]
t = np.linspace(0, 100, 500)
X_true, Y_true, Z_true = simulate_goodwin(t, y0_true, true_params)

# Añadimos ruido gaussiano (simula mediciones experimentales)
sigma_X = 0.1
sigma_Y = 0.1
sigma_Z = 0.1

rng = np.random.default_rng(1)
X_obs = X_true + rng.normal(0, sigma_X, size=X_true.shape)
Y_obs = Y_true + rng.normal(0, sigma_Y, size=Y_true.shape)
Z_obs = Z_true + rng.normal(0, sigma_Z, size=Z_true.shape)


# ------------------------------------------------------------
# 3. Ajuste de parámetros (a,b,c,d,e,f,n) al conjunto X,Y,Z
# ------------------------------------------------------------
def residuals_all(p_vec):
    params = p_vec
    X_sim, Y_sim, Z_sim = simulate_goodwin(t, y0_true, params)

    rX = (X_sim - X_obs) / sigma_X
    rY = (Y_sim - Y_obs) / sigma_Y
    rZ = (Z_sim - Z_obs) / sigma_Z

    return np.concatenate([rX, rY, rZ])


# Valores iniciales razonables para el ajuste
p0 = [0.8, 0.5, 0.8, 0.5, 0.8, 0.5, 8.0]

# Cotas para los parámetros: (a..f) entre 0.01 y 5, n entre 1 y 20
lower_bounds = [0.01]*6 + [1.0]
upper_bounds = [5.0]*6 + [20.0]

result = least_squares(
    residuals_all,
    p0,
    bounds=(lower_bounds, upper_bounds),
    max_nfev=100,
    verbose=1
)

params_fit = result.x
print("Parámetros verdaderos :", true_params)
print("Parámetros ajustados  :", params_fit)

X_fit, Y_fit, Z_fit = simulate_goodwin(t, y0_true, params_fit)


# ------------------------------------------------------------
# 4. Estimación del período de oscilación (a partir de X)
# ------------------------------------------------------------
peaks, _ = find_peaks(X_fit, distance=20)  # distance controla separación mínima
t_peaks = t[peaks]
periods = np.diff(t_peaks)

if len(periods) > 0:
    period_mean = periods.mean()
    print("Tiempos de picos en X:", t_peaks)
    print("Períodos individuales:", periods)
    print("Período promedio     :", period_mean)
else:
    print("No se detectaron picos suficientemente separados.")


# ------------------------------------------------------------
# 5. Gráficas: datos vs modelo ajustado vs modelo verdadero
# ------------------------------------------------------------
plt.figure(figsize=(12, 8))

# X
plt.subplot(3, 1, 1)
plt.plot(t, X_obs, 'o', markersize=3, label="X observado (ruido)")
plt.plot(t, X_true, '--', linewidth=2, label="X verdadero")
plt.plot(t, X_fit, '-', linewidth=2, label="X ajustado")
plt.ylabel("X")
plt.title("Ajuste del modelo de Goodwin (3 componentes)")
plt.legend()
plt.grid()

# Y
plt.subplot(3, 1, 2)
plt.plot(t, Y_obs, 'o', markersize=3, label="Y observado (ruido)")
plt.plot(t, Y_true, '--', linewidth=2, label="Y verdadero")
plt.plot(t, Y_fit, '-', linewidth=2, label="Y ajustado")
plt.ylabel("Y")
plt.legend()
plt.grid()

# Z
plt.subplot(3, 1, 3)
plt.plot(t, Z_obs, 'o', markersize=3, label="Z observado (ruido)")
plt.plot(t, Z_true, '--', linewidth=2, label="Z verdadero")
plt.plot(t, Z_fit, '-', linewidth=2, label="Z ajustado")
plt.xlabel("Tiempo")
plt.ylabel("Z")
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig("Ajuste del modelo de Goodwin (X,,Y,Z)", dpi=600)
plt.show()


# ------------------------------------------------------------
# 6. Análisis de residuos (para X, Y, Z)
# ------------------------------------------------------------
resX = X_fit - X_obs
resY = Y_fit - Y_obs
resZ = Z_fit - Z_obs

plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(t, resX, 'o', markersize=3)
plt.axhline(0, color='k', linestyle='--', linewidth=1)
plt.ylabel("Residuos X")
plt.title("Residuos del ajuste")
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(t, resY, 'o', markersize=3)
plt.axhline(0, color='k', linestyle='--', linewidth=1)
plt.ylabel("Residuos Y")
plt.grid()

plt.subplot(3, 1, 3)
plt.plot(t, resZ, 'o', markersize=3)
plt.axhline(0, color='k', linestyle='--', linewidth=1)
plt.xlabel("Tiempo")
plt.ylabel("Residuos Z")
plt.grid()

plt.tight_layout()
plt.savefig("Análisis de residuos (para X, Y, Z)", dpi=600)
plt.show()


# ------------------------------------------------------------
# 7. Espacio de fase 3D (X–Y–Z) con parámetros ajustados
# ------------------------------------------------------------
fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111, projection='3d')

ax.plot(X_fit, Y_fit, Z_fit, lw=2, label="Trayectoria ajustada")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Espacio de fase (modelo de Goodwin ajustado)")
ax.legend()

plt.tight_layout()
plt.savefig("Espacio de fase (modelo de Goodwin ajustado)", dpi=600)
plt.show()

# ============================
# 8 Autocorrelación de X_fit
# ============================
X_centered = X_fit - np.mean(X_fit)
autocorr = np.correlate(X_centered, X_centered, mode='full')
autocorr = autocorr[autocorr.size // 2:]
lags = t - t[0]

plt.figure(figsize=(7,4))
plt.plot(lags, autocorr / autocorr[0])   # normalizada
plt.xlabel("Retardo (tiempo)")
plt.ylabel("Autocorrelación normalizada")
plt.title("Autocorrelación de X(t)")
plt.grid()
plt.savefig("Autocorrelación de X(t)", dpi=600)
plt.show()

# ============================
# 9 Espectro de potencia de X_fit
# ============================
dt = t[1] - t[0]
N = len(t)

freqs = np.fft.rfftfreq(N, d=dt)
X_fft = np.fft.rfft(X_fit - np.mean(X_fit))
power = np.abs(X_fft)**2

plt.figure(figsize=(7,4))
plt.plot(freqs, power)
plt.xlabel("Frecuencia (1/tiempo)")
plt.ylabel("Potencia")
plt.title("Espectro de potencia de X(t)")
plt.grid()
plt.savefig("Espectro de potencia de X(t)", dpi=600)
plt.show()

# ============================
# Diagrama de bifurcación en función de n
# ============================
n_values = np.linspace(2, 16, 30)
amp_max = []
amp_min = []

for n_val in n_values:
    params_bif = [true_params[0], true_params[1], true_params[2],
                  true_params[3], true_params[4], true_params[5], n_val]
    X_bif, Y_bif, Z_bif = simulate_goodwin(t, y0_true, params_bif)
    # descartamos el transitorio (por ej. primera mitad del tiempo)
    X_ss = X_bif[len(t)//2:]
    amp_max.append(np.max(X_ss))
    amp_min.append(np.min(X_ss))

plt.figure(figsize=(7,5))
plt.plot(n_values, amp_max, 'o-', label="Máximo de X (estado oscilatorio)")
plt.plot(n_values, amp_min, 'o-', label="Mínimo de X")
plt.xlabel("n (coeficiente de Hill)")
plt.ylabel("X")
plt.title("Diagrama de bifurcación en función de n")
plt.legend()
plt.grid()
plt.savefig("Diagrama de bifurcación en función de n", dpi=600)
plt.show()


