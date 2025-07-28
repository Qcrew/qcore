""" """

from inspect import isfunction
import math

import lmfit
from lmfit import Model
from lmfit.models import LinearModel
import numpy as np

def create_params(**kwargs):
    """patch method because lmfit does not like working with np datatypes"""
    params = {}
    for name, value in kwargs.items():
        if isinstance(value, dict):
            value = {k: v.item() for k, v in value.items() if isinstance(v, np.number)}
        elif isinstance(value, np.number):
            value = value.item()
        params[name] = value
    return lmfit.create_params(**params)


def atan(y, x):
    """ """

    def fn(x, fr, Ql, theta, sign=1):
        """
        Arctan fit for resonator phase response around complex plane origin
        x: array of probe frequencies (independent variables)
        fr: resonant frequency
        Ql: loaded (total) quality factor
        theta: arbitrary phase y-offset
        sign: +1 if S shape, -1 if reverse S shape phase response, not varied
        note that np.arctan return real values in the interval [-pi/2, pi/2]
        """
        return theta + 2 * np.arctan(2 * Ql * sign * (x / fr - 1))

    def params(y, x):
        """ """
        sign = 1 if y[0] < y[-1] else -1
        fr = x[np.argmax(np.gradient(y))]
        Ql = (fr / (max(x) - min(x))) * np.sqrt(len(x))
        pts = len(y) // 8
        theta = np.average((y[:pts] + y[-pts:]) / 2)
        params = create_params(fr=fr, Ql=Ql, theta=theta, sign=sign)
        params["sign"].set(vary=False)
        return params

    result = Model(fn).fit(y, params(y, x), x=x)
    return result.best_fit, result.best_values


def cohstate_decay(y, x):
    """ """

    def fn(x, amp, alpha0, tau, ofs, n):
        """
        Poissonian distribution given photon projection:
        alpha = alpha0 * exp(-xs / tau)
        """
        alphas = alpha0 * np.exp(-x / 2.0 / tau)
        nbars = alphas**2
        return ofs + amp * nbars**n / 1 * np.exp(-nbars)

    def params(y, x):
        """ """
        mul = 1 if (x[-1] > x[0]) else -1
        amp = mul * (y[-1] - y[0])
        if amp < 0:
            ofs = np.max(y)
        else:
            ofs = np.min(y)
        tau = x[-1] / 5
        return create_params(n=0, amp=amp, ofs=ofs, alpha0=1.0, tau=tau)

    result = Model(fn).fit(y, params(y, x), x=x)
    return result.best_fit, result.best_values


def displacement_cal(y, x):
    """ """

    def fn(x, dispscale, ofs, amp, n):
        """ """
        alphas = x * dispscale
        nbars = alphas**2
        return ofs + amp * nbars**0 / math.factorial(0) * np.exp(-nbars)

    def params(y, x):
        """ """
        mul = -1 if (x[-1] > x[0]) else 1
        amp = mul * (y[-1] - y[0])
        ofs = np.max(y) if (amp < 0) else np.min(y)
        return create_params(dispscale=1.0, ofs=ofs, amp=amp, n=0)

    result = Model(fn).fit(y, params(y, x), x=x)
    return result.best_fit, result.best_values


def double_gaussian_2dhist(z, y, x):
    """ """

    def fn(y, x, y0, x0, y1, x1, a0, a1, ofs, sigma=2):
        """ """
        r0 = (x - x0) ** 2 + (y - y0) ** 2
        r1 = (x - x1) ** 2 + (y - y1) ** 2
        a0_exp, a1_exp = np.exp(-0.5 * r0 / sigma**2), np.exp(-0.5 * r1 / sigma**2)
        return ofs + a0 * a0_exp + a1 * a1_exp

    def params(z, y, x):
        """ """
        zofs = np.mean([z[0, :], z[-1, :], z[:, 0], z[:, -1]])
        z = z - zofs

        # Locate first max
        maxidx0 = np.argmax(np.abs(z))
        x0 = x.flatten()[maxidx0]
        y0 = y.flatten()[maxidx0]
        a0 = z.flatten()[maxidx0]

        # Other estimates
        dmin = (np.max(x) - np.min(x)) / 8
        mask = ((x - x0) ** 2 + (y - y0) ** 2) > dmin**2
        sigma = np.abs(dmin)

        # Locate second max
        maxidx1 = np.argmax(np.abs(z[mask]))
        x1 = x[mask].flatten()[maxidx1]
        y1 = y[mask].flatten()[maxidx1]
        a1 = z[mask].flatten()[maxidx1]

        return create_params(y0=y0, x0=x0, y1=y1, x1=x1, a0=a0, a1=a1, sigma=sigma)

    result = Model(fn, independent_vars=["x", "y"]).fit(z, params(z, y, x), y=y, x=x)
    return result.best_fit, result.best_values


def exp_decay(y, x):
    """ """

    def fn(x, A, tau, ofs):
        """ """
        return A * np.exp(-x / tau) + ofs

    def params(y, x):
        """ """
        ofs = y[-1]
        y = y - ofs
        tau = (x[-1] - x[0]) / 5
        tau_dict = {"value": tau, "min": 0, "max": 100 * tau}
        return create_params(A=y[0], tau=tau_dict, ofs=ofs)

    result = Model(fn).fit(y, params(y, x), x=x)
    return result.best_fit, result.best_values


def exp_decay_sine(y, x):
    """ """

    def fn(x, amp=1, f0=0.05, phi=np.pi / 4, ofs=0, tau=0.5):
        return amp * np.sin(2 * np.pi * x * f0 + phi) * np.exp(-x / tau) + ofs

    def params(y, x):
        """ """
        params = sine(y, x, return_params=True)
        params.add("tau", value=np.average(x), min=0, max=10 * x[-1])
        return params

    result = Model(fn).fit(y, params(y, x), x=x)
    return result.best_fit, result.best_values


def char_func_coh_state_2(y, x):
    def fn(x, amp, alpha, ofs):
        scale = 2
        return (
            amp * np.exp(-np.abs(x * scale) ** 2 / 2) * np.cos(2 * alpha * x * scale)
            + ofs
        )

    def params(y, x):
        ofs = (y[0] + y[-1]) / 2
        peak_idx = np.argmax(abs(y - ofs))
        yrange = np.max(y) - np.min(y)
        ofs_min, ofs_max = np.min(y) - 0.3 * yrange, np.max(y) + 0.3 * yrange
        return create_params(
            ofs={"value": ofs, "min": ofs_min, "max": ofs_max},
            amp={
                "value": y[peak_idx] - ofs,
                "min": -3 * yrange - ofs,
                "max": 3 * yrange - ofs,
            },
            alpha=1.5,
        )

    result = Model(fn).fit(y, params(y, x), x=x)
    return result.best_fit, result.best_values

def char_func_coh_state_3(y, x):
    # set offset manually
    def fn(x, alpha, amp, ofs):
        #amp = 0.4
        scale = 3 
        #ofs = 0.48 # remove this if want to let the plotter decide.
        return (
            amp * np.exp(-np.abs(x * scale) ** 2 / 2) * np.cos(2 * alpha * x * scale)
            + ofs
        )

    def params(y, x, alpha_guess=3):
        ofs = (y[0] + y[-1]) / 2
        peak_idx = np.argmax(abs(y - ofs))
        yrange = np.max(y) - np.min(y)
        ofs_min, ofs_max = np.min(y) - 0.3 * yrange, np.max(y) + 0.3 * yrange
        return create_params(
            ofs={"value": ofs, "min": ofs_min, "max": ofs_max},
            amp={
                "value": y[peak_idx] - ofs,
                "min": -3 * yrange - ofs,
                "max": 3 * yrange - ofs,
            },
            alpha=alpha_guess,
        )

    # Try just two reasonable guesses - one high and one low
    # This is fast enough for live plotting but gives flexibility
    alpha_guesses = [1.0, 3.0, 5.0, 7.0, 10.0, 12.0]#, 10.0]#, 8.0, 11.0, 15.0]  # Low and high alpha values
    
    best_result = None
    best_chisqr = float('inf')
    
    for alpha_guess in alpha_guesses:
        # Create model with current alpha guess
        model = Model(fn)
        result = model.fit(y, params(y, x, alpha_guess), x=x)
        
        # Check if this is the best fit
        if result.chisqr < best_chisqr:
            best_chisqr = result.chisqr
            best_result = result
    
    return best_result.best_fit, best_result.best_values


def cat_and_back_sigma_x(y, x):
    def fn(
        x,
        amp,
        ofs,
        eta_0,
        eta_2,
        eta_4,
        xi,
    ):
        scale = 2
        return (
            amp
            * (
                1
                - eta_0
                - eta_2 * (np.abs(x * scale)) ** 2
                - eta_4 * (np.abs(x * scale)) ** 4
            )
            * np.cos(2 * xi * (np.abs(x * scale)) ** 2)
            + ofs
        )

    def params(y, x):
        ofs = (y[0] + y[-1]) / 2
        amp_0 = np.max(y) - ofs
        yrange = np.max(y) - np.min(y)
        index_y_min = np.argmin(abs(y - ofs))
        ofs_min, ofs_max = np.min(y) - 0.3 * yrange, np.max(y) + 0.3 * yrange
        xi_val = x[index_y_min] / (2 * (np.abs(np.average(x))) ** 2)
        return create_params(
            ofs={"value": ofs, "min": ofs_min, "max": ofs_max},
            amp={
                "value": y[peak_idx] - ofs,
                "min": -3 * yrange - ofs,
                "max": 3 * yrange - ofs,
            },
            eta_0={"value": 0.01, "min": 0, "max": 1 / 3},
            eta_2={"value": 0.01, "min": 0, "max": 1 / 3},
            eta_4={"value": 0.01, "min": 0, "max": 1 / 3},
            xi={"value": xi_val, "min": 0, "max": 1},
        )

    result = Model(fn).fit(y, params(y, x), x=x)
    return result.best_fit, result.best_values


def cat_and_back(y, x):
    scale = 6

    def fn(x, amp, ofs, xi, eta_0, eta_2, eta_4):
        argument = 2 * xi * (np.abs(x * scale)) ** 2
        return (
            amp
            * (
                1
                - eta_0
                - eta_2 * (np.abs(x * scale)) ** 2
                - eta_4 * (np.abs(x * scale)) ** 4
            )
            * np.cos(argument)
            + ofs
        )

    def params(y, x):
        ofs = (np.max(y) + np.min(y)) / 2
        amp_0 = np.max(y) - ofs
        yrange = np.max(y) - np.min(y)
        ofs_min, ofs_max = np.min(y) - 0.3 * yrange, np.max(y) + 0.3 * yrange
        xi_val = np.pi * 1 / (2 * (scale * x[np.argmin(y)]) ** 2)

        return create_params(
            ofs={"value": ofs, "min": ofs_min, "max": ofs_max},
            amp={
                "value": amp_0,
                "min": -3 * yrange - ofs,
                "max": 3 * yrange - ofs,
            },
            eta_0={
                "value": np.float64(0.01),
                "min": np.float64(0),
                "max": np.float64(1 / 4),
            },
            eta_2={
                "value": np.float64(0.01),
                "min": np.float64(0),
                "max": np.float64(1 / 4),
            },
            eta_4={
                "value": np.float64(0.01),
                "min": np.float64(0),
                "max": np.float64(1 / 4),
            },
            xi={"value": xi_val},
        )

    result = Model(fn).fit(y, params(y, x), x=x)
    return result.best_fit, result.best_values


def T1_cav_char_point_chi(y, x):
    scale = 2
    beta = 1

    def fn(x, amp, ofs, tau, alpha, phi):
        Delta = Charlie["chi"] / 2

        alpha = alpha * np.exp(-1j * (Delta * x + phi) - x / (tau * 2))
        # char_point = ofs + amp * np.exp(-1 / 2 * np.abs(beta * scale) ** 2) * np.exp(
        #     -1j * np.imag(alpha * np.conjugate(beta * scale))
        # )
        return np.real(
            amp
            * np.exp(-np.abs(beta * scale) ** 2 / 2)
            * np.cos(2 * beta * alpha * scale * np.exp(-0.5 * (1j * Delta) * x))
            + ofs
        )
        # return np.real(char_point)

    def params(y, x):
        ofs = (np.max(y) + np.min(y)) / 2
        amp_0 = np.max(y) - ofs
        yrange = np.max(y) - np.min(y)
        ofs_min, ofs_max = np.min(y) - 0.7 * yrange, np.max(y) + 0.7 * yrange
        tau_guess = np.float64(1e5)
        alpha_guess = np.float64(4)
        phi_guess = 0
        return create_params(
            ofs={"value": ofs, "min": ofs_min, "max": ofs_max},
            amp={
                "value": amp_0 - ofs,
                "min": -1.5 * yrange - ofs,
                "max": 1.5 * yrange - ofs,
            },
            tau={"value": tau_guess, "min": 0, "max": 5 * x[-1]},
            alpha={
                "value": alpha_guess,
                "min": np.float64(3.5),
                "max": np.float64(4.5),
            },  # edit this
            phi={
                "value": phi_guess,
                "min": np.float64(0),
                "max": np.float64(np.pi),
            },
        )

    result = Model(fn).fit(y, params(y, x), x=x)
    return result.best_fit, result.best_values


def T1_cav_char_point(y, x):
    def fn(x, amp, ofs, tau, alpha, phi):
        scale = 2
        beta = 1
        chi = 0  # Alice["chi"]
        kappa = 1 / tau
        return np.real(
            amp
            * np.exp(-np.abs(beta * scale) ** 2 / 2)
            * np.cos(
                2
                * beta
                * alpha
                * scale
                * np.exp(-0.5 * (1j * chi + kappa) * x + 1j * phi)
            )
            + ofs
        )

    def params(y, x):
        ofs = (np.max(y) + np.min(y)) / 2
        amp_0 = np.max(y) - ofs
        yrange = np.max(y) - np.min(y)
        ofs_min, ofs_max = np.min(y) - 0.7 * yrange, np.max(y) + 0.7 * yrange
        tau_guess = np.float64(1e5)
        alpha_guess = np.float64(8)
        phi_guess = np.float64(0.4)
        return create_params(
            ofs={"value": ofs, "min": ofs_min, "max": ofs_max},
            amp={
                "value": amp_0 - ofs,
                "min": -1.5 * yrange - ofs,
                "max": 1.5 * yrange - ofs,
            },
            tau={"value": tau_guess, "min": 0, "max": 5 * x[-1]},
            alpha={
                "value": alpha_guess,
                "min": np.float64(2),
                "max": np.float64(5),
            },  # edit this
            phi={
                "value": phi_guess,
                "min": np.float64(0),
                "max": np.float64(2 * np.pi),
            },
        )

    result = Model(fn).fit(y, params(y, x), x=x)
    return result.best_fit, result.best_values


def gaussian(y, x):
    """ """

    def fn(x, x0, sig, ofs, amp):
        """ """
        return ofs + amp * np.exp(-((x - x0) ** 2) / (2 * sig**2))

    def params(y, x):
        """ """
        ofs = (y[0] + y[-1]) / 2
        peak_idx = np.argmax(abs(y - ofs))
        sig = abs(x[-1] - x[0]) / 10
        yrange = np.max(y) - np.min(y)
        ofs_min, ofs_max = np.min(y) - 0.3 * yrange, np.max(y) + 0.3 * yrange
        return create_params(
            x0={"value": x[peak_idx], "min": np.min(x), "max": np.max(x)},
            sig={"value": sig, "min": abs(x[1] - x[0]), "max": abs(x[-1] - x[0])},
            ofs={"value": ofs, "min": ofs_min, "max": ofs_max},
            amp={"value": y[peak_idx] - ofs, "min": -3 * yrange, "max": 3 * yrange},
        )

    result = Model(fn).fit(y, params(y, x), x=x)
    return result.best_fit, result.best_values


def gaussian2d_symmetric(z, y, x):
    """ """

    def fn(y, x, y0=0, x0=0, sigma=1, area=1, ofs=0):
        """ """
        r2 = (x - x0) ** 2 / (2 * sigma**2) + (y - y0) ** 2 / (2 * sigma**2)
        return ofs + area / (2 * sigma**2) / np.sqrt(np.pi / 2) * np.exp(-r2)

    def params(z, y, x):
        """ """
        zofs = np.mean([z[0, :], z[-1, :], z[:, 0], z[:, -1]])
        z = z - zofs
        maxidxy = np.argmax(np.abs(z).sum(axis=1))
        maxidxx = np.argmax(np.abs(z).sum(axis=0))

        xspan = x[-1, 0] - x[0, 0]
        yspan = y[0, -1] - y[0, 0]

        sigma = (xspan + yspan) / 2 / 5

        maxidx0 = np.argmax(np.abs(z))
        dmin = (np.max(x) - np.min(x)) / 8
        mask = (
            (x - x.flatten()[maxidxx]) ** 2 + (y - y.flatten()[maxidx0]) ** 2
        ) > dmin**2
        area = np.sum(z[mask])

        return create_params(
            x0=x[maxidxy, maxidxx],
            y0=y[maxidxy, maxidxx],
            sigma=sigma,
            area=area,
            ofs=zofs,
        )

    result = Model(fn, independent_vars=["x", "y"]).fit(z, params(z, y, x), y=y, x=x)
    return result.best_fit, result.best_values


def linear(y, x):
    """ """
    result = LinearModel.fit(y, x=x)
    return result.best_fit, result.best_values


def lorentzian(y, x, return_params=False):
    """ """

    def fn(x, fr, ofs, height, fwhm):
        """ """
        return np.abs(ofs + height / (1 + 2j * ((x - fr) / fwhm)))

    def params(y, x):
        """ """
        pts = len(y) // 8
        ofs = np.average((y[:pts] + y[-pts:]) / 2)
        height = np.abs(np.max(y) - np.min(y))
        fr_idx = (y - np.abs(ofs + height)).argmin()
        fr = x[fr_idx]
        is_inverted = np.abs(y[0] - y.max()) < np.abs(y[-1] - y.min())
        height = -height if is_inverted else height
        amp, left, right = height / 2 + ofs, y[:fr_idx], y[fr_idx:]
        fwhm = x[fr_idx + np.abs(right - amp).argmin()] - x[np.abs(left - amp).argmin()]
        return create_params(fr=fr, ofs=ofs, height=height, fwhm=fwhm)

    fit_params = params(y, x)
    if return_params:
        return fit_params
    result = Model(fn).fit(y, fit_params, x=x)
    return result.best_fit, result.best_values


def lorentzian_asymmetric(y, x):
    """ """

    def fn(x, fr, ofs, height, fwhm, phi):
        """ """
        return np.abs(ofs + height * np.exp(1j * phi) / (1 + 2j * ((x - fr) / fwhm)))

    def params(y, x):
        """ """
        params = lorentzian(y, x, return_params=True)
        ofs, height, fr = params["ofs"].value, params["height"].value, params["fr"]
        phi = 4 * np.arcsin((np.max(y) - ofs) / height)
        params.add("phi", value=phi)
        fr_idx = (y - np.abs(ofs + height * np.exp(1j * phi))).argmin()
        fr.set(value=x[fr_idx])
        return params

    result = Model(fn).fit(y, params(y, x), x=x)
    return result.best_fit, result.best_values


def sine(y, x, return_params=False):
    """ """

    def fn(x, f0, ofs, amp, phi):
        """ """
        return ofs + amp * np.sin(2 * np.pi * f0 * x + phi)

    def params(y, x):
        """ """
        fs = np.fft.rfftfreq(len(x), x[1] - x[0])
        ofs = np.mean(y)
        fft = np.fft.rfft(y - ofs)
        idx = np.argmax(abs(fft))
        return create_params(
            f0={"value": fs[idx], "min": fs[0], "max": fs[-1]},
            ofs={"value": ofs, "min": np.min(y), "max": np.max(y)},
            amp={"value": np.std(y - ofs), "min": 0, "max": np.max(y) - np.min(y)},
            phi={"value": np.angle(fft[idx]), "min": -2 * np.pi, "max": 2 * np.pi},
        )

    fit_params = params(y, x)
    if return_params:
        return fit_params
    result = Model(fn).fit(y, fit_params, x=x)
    return result.best_fit, result.best_values


def sine_gf(y, x, return_params=False):

    def fn(x, f0, ofs, amp, phi):
        return ofs + amp * np.sin(2 * np.pi * f0 * x**2 + phi)

    def params(y, x):
        """ """
        fs = np.fft.rfftfreq(len(x), x[1] - x[0])
        ofs = np.mean(y)
        fft = np.fft.rfft(y - ofs)
        idx = np.argmax(abs(fft))

        return create_params(
            f0={"value": fs[idx], "min": fs[0], "max": fs[-1]},
            ofs={"value": ofs, "min": np.min(y), "max": np.max(y)},
            amp={"value": np.std(y - ofs), "min": 0, "max": np.max(y) - np.min(y)},
            phi={"value": np.angle(fft[idx]), "min": -2 * np.pi, "max": 2 * np.pi},
        )

    fit_params = params(y, x)
    if return_params:
        return fit_params
    result = Model(fn).fit(y, fit_params, x=x)
    return result.best_fit, result.best_values


FITFN_MAP = {
    k: v for k, v in locals().items() if not k == "isfunction" and isfunction(v)
}
