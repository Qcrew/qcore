""" """

import matplotlib.pyplot as plt
import numpy as np
from qm import qua as qm_qua
from scipy import signal, special
from scipy.optimize import curve_fit

from qcore import qua
from qcore.helpers.logger import logger
from qcore.modes.mode import Mode
from qcore.libs.fit_fns import gaussian2d_symmetric

ADC_TO_VOLTS = 2**-12
TS = 1e-9  # Sampling time of the OPX in seconds
T_OFS = 35.96


class ReadoutTrainer:
    """ """

    def __init__(
        self,
        rr: Mode,
        qubit: Mode,
        qm,
        reps,
        wait_time,
        readout_pulse,
        qubit_pi_pulse,
        ddrop_params=None,
        weights_file_path=None,
    ):
        """ """
        self._rr: Mode = rr
        self._qubit: Mode = qubit
        self._qm = qm
        self.modes = [rr, qubit]
        self.mode_names = [mode.name for mode in self.modes]
        self.reps = reps
        self.wait_time = wait_time
        self.readout_pulse = readout_pulse
        self.qubit_pi_pulse = qubit_pi_pulse
        self.ddrop_params = ddrop_params
        self.weights_file_path = weights_file_path

        logger.info(f"Initialized ReadoutTrainer with {self._rr} and {self._qubit}")

    def train_weights(self) -> None:
        """
        Obtain integration weights of rr given the excited and ground states of qubit and update rr mode.
        """

        # Start with constant integration weights. Not really necessary
        # self._reset_weights()

        # Get traces and average envelope when qubit in ground state
        trace_g_list, timestamps_g = self._acquire_traces(self._qm, excite_qubit=False)
        env_g = self._calc_average_envelope(trace_g_list, timestamps_g, T_OFS)

        # Get traces and average envelope when qubit in excited state
        trace_e_list, timestamps_e = self._acquire_traces(self._qm, excite_qubit=True)
        env_e = self._calc_average_envelope(trace_e_list, timestamps_e, T_OFS)

        # Get difference between average envelopes
        envelope_diff = env_g - env_e

        # Normalize and squeeze by 1/4th
        norm = np.max(np.abs(envelope_diff))
        norm_envelope_diff = envelope_diff / norm
        squeezed_diff = self._squeeze_array(norm_envelope_diff)  # convert shape

        # Update readout with optimal weights
        weights = self._update_weights(squeezed_diff)

        # Plot envelopes
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(7, 10))
        axes[0].plot(1000 * np.real(env_g), label="Re")
        axes[0].plot(1000 * np.imag(env_g), label="Imag")
        axes[0].set_title("|g> envelope")
        axes[0].set_ylabel("Amplitude (mV)")
        axes[0].legend()
        axes[1].plot(1000 * np.real(env_e))
        axes[1].plot(1000 * np.imag(env_e))
        axes[1].set_title("|e> envelope")
        axes[1].set_ylabel("Amplitude (mV)")
        axes[1].set_xlabel("Time (ns)")
        plt.show()

        return env_g, env_e

    def _reset_weights(self):
        """
        Start the pulse with constant integration weights
        """
        (readout_pulse,) = self._rr.get_operations(self.readout_pulse)
        readout_pulse.weights = (1.0, 0.0, 0.0, 1.0)

    def _acquire_traces(self, qm, excite_qubit: bool = False) -> tuple[list]:
        """
        Run QUA program to obtain traces of the readout pulse.
        """

        # Execute script
        qua_program = self._get_QUA_trace_acquisition(excite_qubit)
        job = self._qm.execute(qua_program)

        handle = job.result_handles
        handle.wait_for_all_values()
        timestamps = handle.get("timestamps").fetch_all()["value"]
        adc = handle.get("adc").fetch_all()["value"]

        return ADC_TO_VOLTS * adc, timestamps

    def _get_QUA_trace_acquisition(self, excite_qubit: bool = False):
        """ """
        reps = self.reps
        wait_time = self.wait_time
        (readout_pulse,) = self._rr.get_operations(self.readout_pulse)
        (qubit_pi_pulse,) = self._qubit.get_operations(self.qubit_pi_pulse)

        with qm_qua.program() as acquire_traces:
            adc = qm_qua.declare_stream(adc_trace=True)
            n = qm_qua.declare(int)

            with qm_qua.for_(n, 0, n < reps, n + 1):
                # if self.ddrop_params:
                #     self._macro_DDROP_reset()

                # qua.measure(readout_pulse, self._rr.name, adc)
                # qua.wait(wait_time, self._rr.name)
                # qua.reset_phase(self._rr.name)

                # qua.play("predist_square_plusminus_pulse" * qua.amp(-0.28), "FLUX")
                # qua.wait(int(2500 // 4), self._qubit.name, self._rr.name)
                if self.ddrop_params:
                    self._macro_DDROP_reset()

                if excite_qubit:
                    # qua.align("FLUX", self._qubit.name)
                    self._qubit.play(qubit_pi_pulse)
                    qua.align(self._rr, self._qubit)

                self._rr.measure(readout_pulse, stream=adc)
                qua.wait(wait_time, self._rr)

            with qm_qua.stream_processing():
                # streams for envelope calculation
                adc.input1().timestamps().save_all("timestamps")
                adc.input1().save_all("adc")

                # streams for plotting/bug fixing
                adc.input1().average().save("adc_avg")
                adc.input1().average().fft().save("adc_fft")

        return acquire_traces

    def _calc_average_envelope(self, trace_list, timestamps, t_ofs):
        int_freq = np.abs(self._rr.int_freq)

        # demodulate
        s = trace_list * np.exp(1j * 2 * np.pi * int_freq * TS * (timestamps - t_ofs))

        # filter 2*omega_IF using hann filter
        hann = signal.hann(int(2 / TS / int_freq), sym=True)
        hann = hann / np.sum(hann)
        s_filtered = np.array([np.convolve(s_single, hann, "same") for s_single in s])

        # adjust envelope
        env = 2 * s_filtered.conj()

        # get average envelope
        avg_env = np.average(env, axis=0)

        return avg_env

    def _squeeze_array(self, s):
        """
        Split the array in bins of 4 values and average them. QM requires the weights to have 1/4th of the length of the readout pulse.
        """
        return np.average(np.reshape(s, (-1, 4)), axis=1)

    def _update_weights(self, squeezed_diff):
        weights = {}
        weights["I"] = np.array(
            [np.real(squeezed_diff).tolist(), (np.imag(squeezed_diff)).tolist()]
        )
        weights["Q"] = np.array(
            [np.imag(-squeezed_diff).tolist(), np.real(squeezed_diff).tolist()]
        )

        path = self.weights_file_path

        # Save weights to npz file
        np.savez(path, **weights)

        # Update the readout pulse with the npz file path
        (readout_pulse,) = self._rr.get_operations(self.readout_pulse)
        readout_pulse.weights = str(path)

        return weights

    def _fit_hist_double_gaussian(self, guess, data_g):
        """
        The E population is estimated from the amplitudes of the two gaussians
        fitted from the G state blob.
        """

        p0 = [
            guess["x0"],
            guess["x1"],
            guess["a0"],
            guess["a1"],
            guess["ofs"],
            guess["sigma"],
        ]

        popt, _ = curve_fit(double_gaussian, data_g["xs"], data_g["ys"], p0=p0)
        return popt

    def calculate_threshold(self):
        # Get IQ for qubit in ground state
        IQ_acquisition_program = self._get_QUA_IQ_acquisition()
        job = self._qm.execute(IQ_acquisition_program)
        handle = job.result_handles
        handle.wait_for_all_values()
        Ig_list = handle.get("I").fetch_all()["value"]
        Qg_list = handle.get("Q").fetch_all()["value"]

        # Get IQ for qubit in excited state
        IQ_acquisition_program = self._get_QUA_IQ_acquisition(excite_qubit=True)
        job = self._qm.execute(IQ_acquisition_program)
        handle = job.result_handles
        handle.wait_for_all_values()
        Ie_list = handle.get("I").fetch_all()["value"]
        Qe_list = handle.get("Q").fetch_all()["value"]

        # Fit each blob to a 2D gaussian and retrieve the center
        params_g, data_g = self._fit_IQ_blob(Ig_list, Qg_list)
        params_e, data_e = self._fit_IQ_blob(Ie_list, Qe_list)

        IQ_center_g = (params_g["x0"], params_g["y0"])  # G blob center
        IQ_center_e = (params_e["x0"], params_e["y0"])  # E blob center

        # Calculate threshold
        threshold = (IQ_center_g[0] + IQ_center_e[0]) / 2

        # Update readout with optimal threshold
        self._update_threshold(threshold)

        # Calculates the confusion matrix of the readout
        conf_matrix = self._calculate_confusion_matrix(Ig_list, Ie_list, threshold)

        # Plot scatter and contour of each blob
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_aspect("equal")
        ax.scatter(Ig_list, Qg_list, label="|g>", s=5)
        ax.scatter(Ie_list, Qe_list, label="|e>", s=5)
        ax.contour(
            data_g["I_grid"],
            data_g["Q_grid"],
            data_g["counts_fit"],
            levels=5,
            cmap="winter",
        )
        ax.contour(
            data_e["I_grid"],
            data_e["Q_grid"],
            data_e["counts_fit"],
            levels=5,
            cmap="autumn",
        )
        ax.plot(
            [threshold, threshold],
            [np.min(data_g["Q_grid"]), np.max(data_g["Q_grid"])],
            label="threshold",
            c="k",
            linestyle="--",
        )

        ax.set_title("IQ blobs for each qubit state")
        ax.set_ylabel("Q")
        ax.set_xlabel("I")
        ax.legend()
        plt.show()

        # Plot I histogram
        fig, ax = plt.subplots(figsize=(7, 4))
        n_g, bins_g, _ = ax.hist(Ig_list, bins=50, alpha=1)
        n_e, bins_e, _ = ax.hist(Ie_list, bins=50, alpha=0.8)

        # Estimate excited state population from G blob double gaussian fit

        pge = conf_matrix["pge"]  # first estimate of excited state population
        guess = {
            "x0": params_g["x0"],
            "x1": params_e["x0"],
            "a0": max(n_g),
            "a1": max(n_g) * pge / (1 - pge),
            "ofs": 0.0,
            "sigma": params_g["sigma"],
        }
        data_hist_g = {"xs": (bins_g[1:] + bins_g[:-1]) / 2, "ys": n_g}

        popt = self._fit_hist_double_gaussian(guess, data_hist_g)
        print(popt)
        a0 = popt[2]
        a1 = popt[3]
        e_population = a1 / (a1 + a0)
        print("Excited state population: ", e_population)

        ax.plot(bins_g, [double_gaussian(x, *popt) for x in bins_g])

        ax.set_title("Projection of the IQ blobs onto the I axis")
        ax.set_ylabel("counts")
        ax.set_xlabel("I")
        ax.legend()
        plt.show()

        # Organize the raw I and Q data for each G and E measurement
        data = {
            "Ig": Ig_list,
            "Qg": Qg_list,
            "Ie": Ie_list,
            "Qe": Qe_list,
        }

        # Plot manual threshold selectiveness on |g>
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.set_title("Is qubit really in ground state if state = 0?")
        ax.set_ylabel("Certainty")
        ax.set_xlabel("Threshold")
        popt
        p_pass_if_g = lambda t: 0.5 * special.erfc((t - popt[0]) / 2**0.5 / popt[-1])
        p_pass_if_e = lambda t: 0.5 * special.erfc((t - popt[1]) / 2**0.5 / popt[-1])
        p_g = popt[2] / (popt[2] + popt[3])
        p_e = popt[3] / (popt[2] + popt[3])
        certainty = (
            lambda t: p_pass_if_g(t)
            * p_g
            / (p_pass_if_e(t) * p_e + p_pass_if_g(t) * p_g)
        )
        t_rng = np.linspace(bins_g[0], bins_g[-1], 1000)
        ax.plot(t_rng, [certainty(t) for t in t_rng])
        ax.plot(
            [threshold, threshold],
            [0.4, 1.1],
            linestyle="--",
            label="calculated threshold",
        )
        ax.legend()
        plt.show()

        return threshold, data

    def _fit_IQ_blob(self, I_list, Q_list):
        fit_fn = "gaussian2d_symmetric"

        # Make ground IQ blob in a 2D histogram
        zs, xs, ys = np.histogram2d(I_list, Q_list, bins=50)

        # Replace "bin edge" by "bin center"
        dx = xs[1] - xs[0]
        xs = (xs - dx / 2)[1:]
        dy = ys[1] - ys[0]
        ys = (ys - dy / 2)[1:]

        # Get fit to 2D gaussian
        xs_grid, ys_grid = np.meshgrid(xs, ys)
        best_fit, fit_params = gaussian2d_symmetric(zs, ys_grid.T, xs_grid.T)
        best_fit = best_fit.T

        data = {
            "I_grid": xs_grid,
            "Q_grid": ys_grid,
            "counts": zs,
            "counts_fit": best_fit,
        }

        return fit_params, data

    def _get_QUA_IQ_acquisition(self, excite_qubit: bool = False):
        """ """
        reps = self.reps
        wait_time = self.wait_time

        (readout_pulse,) = self._rr.get_operations(self.readout_pulse)
        (qubit_pi_pulse,) = self._qubit.get_operations(self.qubit_pi_pulse)

        with qm_qua.program() as acquire_IQ:
            I = qm_qua.declare(qm_qua.fixed)
            Q = qm_qua.declare(qm_qua.fixed)
            n = qm_qua.declare(int)

            with qm_qua.for_(n, 0, n < reps, n + 1):
                # qua.play("predist_square_plusminus_pulse" * qua.amp(-0.3), "FLUX")
                # qua.wait(int(2500 // 4), self._qubit.name, self._rr.name)

                if self.ddrop_params:
                    self._macro_DDROP_reset()

                if excite_qubit:
                    # qua.align(self._rr.name, self._qubit.name)
                    self._qubit.play(qubit_pi_pulse)
                    qua.align(self._rr, self._qubit)

                self._rr.measure(readout_pulse, (I, Q))
                qm_qua.save(I, "I")
                qm_qua.save(Q, "Q")
                qua.wait(wait_time, self._rr)

        return acquire_IQ

    def _update_threshold(self, threshold):
        (readout_pulse,) = self._rr.get_operations(self.readout_pulse)
        readout_pulse.threshold = threshold

    def _calculate_confusion_matrix(self, Ig_list, Ie_list, threshold):
        pgg = 100 * round((np.sum(Ig_list > threshold) / len(Ig_list)), 3)
        pge = 100 * round((np.sum(Ig_list < threshold) / len(Ig_list)), 3)
        pee = 100 * round((np.sum(Ie_list < threshold) / len(Ie_list)), 3)
        peg = 100 * round((np.sum(Ie_list > threshold) / len(Ie_list)), 3)
        print("\nState prepared in |g>")
        print(f"   Measured in |g>: {pgg}%")
        print(f"   Measured in |e>: {pge}%")
        print("State prepared in |e>")
        print(f"   Measured in |e>: {pee}%")
        print(f"   Measured in |g>: {peg}%")
        return {"pgg": pgg, "pge": pge, "pee": pee, "peg": peg}

    def _macro_DDROP_reset(self):
        rr_ddrop_freq = self.ddrop_params["rr_ddrop_freq"]
        rr_ddrop = self.ddrop_params["rr_ddrop"]
        qubit_ddrop = self.ddrop_params["qubit_ddrop"]
        steady_state_wait = self.ddrop_params["steady_state_wait"]
        qubit_ef = self.ddrop_params["qubit_ef_mode"]

        qua.align(
            self._qubit, self._rr, qubit_ef
        )  # wait qubit pulse to end
        qua.update_frequency(self._rr, rr_ddrop_freq)
        qm_qua.play(self._rr.name, rr_ddrop)  # play rr ddrop excitation
        qua.wait(
            steady_state_wait, self._qubit, qubit_ef
        )  # wait resonator in steady state
        qm_qua.play(self._qubit.name, qubit_ddrop)  # play qubit ddrop excitation
        qm_qua.play(qubit_ef.name, "ddrop_pulse")  # play qubit ddrop excitation
        qua.wait(
            steady_state_wait, self._qubit, qubit_ef
        )  # wait resonator in steady state
        qua.align(
            self._qubit, self._rr, qubit_ef
        )  # wait qubit pulse to end
        qua.update_frequency(self._rr, self._rr.int_freq)


def double_gaussian(xs, x0, x1, a0, a1, ofs, sigma):
    """
    Gaussian defined by it's area <area>, sigma <s>, position <x0> and
    y-offset <ofs>.
    """
    r0 = (xs - x0) ** 2
    r1 = (xs - x1) ** 2
    ys = ofs + a0 * np.exp(-0.5 * r0 / sigma**2) + a1 * np.exp(-0.5 * r1 / sigma**2)
    return ys
