import numpy as np
import fcwt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt
import scipy.signal as sig
from scipy.signal.windows import gaussian

N = 1000
f_s = 1000
f_0 = 6
f_1 = 20
n_scales = 200
sigma = 1
wavelet = fcwt.Morlet(sigma)
scales = fcwt.Scales(wavelet, fcwt.FCWT_LOGSCALES, f_s, f_0, f_1, n_scales)
freqs = np.zeros(n_scales, dtype=np.float32)
scales.getFrequencies(freqs)
scales_arr = np.zeros(n_scales, dtype=np.float32)
scales.getScales(scales_arr)
largest_scale = round(scales_arr.max())
largest_support = wavelet.getSupport(largest_scale)
freqs_arr = np.zeros(n_scales, dtype=np.float32)
scales.getFrequencies(freqs_arr)
largets_freq = freqs_arr.max()
print("Largest support:", largest_support)
wavelet_arr = np.zeros(2*largest_support, dtype=np.complex64)
wavelet.getWavelet(largest_scale, wavelet_arr)
wavelet_arr = np.pad(wavelet_arr, [N, N], "constant", constant_values=0)

sine_wave = np.tile(np.sin(2*np.pi*np.arange(N)/f_s), 3)

plt.clf() 
plt.plot(np.real(wavelet_arr), label="real")
plt.plot(np.imag(wavelet_arr), label="imag")
plt.plot(sine_wave, label="sin")
plt.legend()
plt.savefig("example_wavelet.png")

tfm_len = 2**(3 + int(np.log2(largest_support)))
xcrop = 4

plt.clf()
plt.plot(np.arange(tfm_len // xcrop) / tfm_len, np.abs(np.fft.fft(wavelet_arr, tfm_len))[:tfm_len // xcrop], label="wavelet")
plt.plot(np.arange(tfm_len // xcrop) / tfm_len, np.abs(np.fft.fft(sine_wave, tfm_len))[:tfm_len // xcrop], label="sin")
plt.legend()
plt.savefig("example_wavelet_tfm.png")

win = gaussian(600, 70)

plt.clf()
plt.plot(win)
plt.savefig("foo.png")

stft = sig.ShortTimeFFT(win, 100, f_s, fft_mode="centered")


wavelet_stft = stft.stft(wavelet_arr)
sine_stft = stft.stft(sine_wave)

# Calculate time and frequency axes
window_length = win.shape[0]
hop_size = 100
num_frames = wavelet_stft.shape[1]
num_freqs = wavelet_stft.shape[0]
times = np.arange(num_frames) * hop_size / f_s
freqs = np.fft.fftfreq(window_length, d=1/f_s)[:num_freqs]

# Plot STFT of wavelet
plt.clf()
plt.pcolormesh(times, freqs, np.abs(wavelet_stft), shading='auto', cmap='turbo')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('STFT of Wavelet')
plt.colorbar(label='Magnitude')
plt.savefig("stft_wavelet.png")

# Plot STFT of sine
plt.clf()
plt.pcolormesh(times, freqs, np.abs(sine_stft), shading='auto', cmap='turbo')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('STFT of Sine Wave')
plt.colorbar(label='Magnitude')
plt.savefig("stft_sine.png")
