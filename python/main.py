import time 
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import fft
import pywt
import fcwt

__t = 0

def main():
    f_s = int(1e5)
    omega_min = 0.001
    omega_max = 0.5
    n_scales = 128
    N = int(1e5)
    nds = 100

    times, x = generate_chirp_exp(N, f_s)
    times, y = generate_chirp_lin(N, f_s)
    x += y
    x += 1 * np.random.normal(size=N)

    # plt.plot(times, instant_frequency)
    # plt.xlabel("Time [s]")
    # plt.ylabel("Freq [Hz]")
    # plt.title("Instantaneous Frequency")
    # plt.grid()
    # plt.savefig('instant_frequency.png')
    # plt.clf()

    plt.plot(times, x)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title("Generated Signal")
    plt.grid()
    plt.savefig('plot.png')
    plt.clf()

    X = fft.fft(x)

    plt.plot(f_s * np.arange(N) / N, np.abs(X))
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude")
    plt.title("FFT of Generated Signal")
    plt.grid()
    plt.savefig('fft.png')
    plt.clf()

    # ----------------------------------------------

    # print(pywt.wavelist(kind="discrete"))
    # print(pywt.wavelist(family="cmor"))

    tick()
    wavelet = pywt.ContinuousWavelet('cmor3-1')
    tock("generate")

    scales = pywt.frequency2scale(wavelet, np.geomspace(omega_min, omega_max, n_scales))
    tock("scale convert")

    cwtmatr, freqs1 = pywt.cwt(x, scales, wavelet, sampling_period=1/f_s, method="conv")
    tock("CWT")
    plt.pcolormesh(times[..., ::nds], freqs1, np.abs(cwtmatr[..., ::nds]), shading='gouraud')
    tock("plot")

    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    plt.title('CWT using Morlet Wavelet')
    plt.gca().invert_yaxis()
    plt.colorbar(label='Magnitude')
    plt.yscale("log")
    plt.gca().invert_yaxis()
    plt.savefig('cwt.png')
    tock("save")
    plt.clf()

    # ----------------------------------------------
    tick()
    wavelet = fcwt.Morlet(3)
    scales = fcwt.Scales(wavelet, fcwt.FCWT_LOGSCALES, f_s, f_s*omega_min, f_s*omega_max, n_scales)
    freqs2 = np.zeros(n_scales, dtype=np.float32)
    scales.getFrequencies(freqs2)
    fcwt_obj = fcwt.FCWT(wavelet, 1, False, True)
    output = np.zeros((n_scales, x.size), dtype=np.complex64)

    tock("Generate")
    fcwt_obj.cwt(x, scales, output)

    tock("CWT")
    plt.pcolormesh(times[..., ::nds], freqs2, np.abs(output[..., ::nds]), shading='gouraud')
    tock("plot")

    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    plt.title('CWT using Morlet Wavelet')
    plt.gca().invert_yaxis()
    plt.colorbar(label='Magnitude')
    plt.yscale("log")
    plt.gca().invert_yaxis()
    plt.savefig('fcwt.png')
    tock("save")
    plt.clf()

    plt.stem(freqs1)
    plt.stem(freqs2)
    plt.savefig('freqs.png')


def generate_chirp_lin(N, f_s):
    samples = np.arange(N)
    times = samples / f_s

    f_0 = 3000
    c = 10000

    signal = np.sin(2 * np.pi * (c*times/2 + f_0) * times)
    return times, np.array(signal, dtype=np.float32)

def generate_chirp_exp(N, f_s):
    samples = np.arange(N)
    times = samples / f_s
    f_0 = 5e3
    f_1 = 3e4
    T_end = N / f_s
    k = f_1 / f_0

    signal = np.sin(2 * np.pi * f_0 * T_end * (k**(times / T_end) - 1) / np.log(k))
    return times, np.array(signal, dtype=np.float32)

def tick():
    global __t
    __t = time.time()

def tock(s):
    global __t
    t_new = time.time()
    print(f"Lap \"{s}\": {(t_new - __t) * 1000} ms")
    __t = t_new

if __name__ == "__main__":
    main()