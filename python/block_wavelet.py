import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import fcwt
import numpy as np
from timing import tick, tock
from tqdm import tqdm


def main():
    data = np.genfromtxt("python/synthetic_dataset.csv", delimiter=",", skip_header=1, dtype=np.float32)
    times = data[:, 0]
    x = data[:, 1]
    x_noisy = data[:, 2]
    N = x.size
    print("===================================")
    print(f"Using dataset of size {N} (log2: {int(np.log2(N)) + 1})")
    print("===================================")

    signal = x
    # signal = np.pad(x_noisy, (N, N), mode='constant', constant_values=0)
    # signal = np.pad(x, (N, N), mode='reflect')

    test_block_transform(2, times, signal, 1024)

    # for sigma in [1.5, 2, 3, 4, 5, 6, 7]:
    #     output = plot_fcwt_blocked(sigma, times, signal, 256)
    #     # output = plot_fcwt(sigma, times, signal, N)

def test_block_transform(sigma, times, x, block_size):
    omega_min = 1 / 500 / 4
    omega_max = 0.5
    n_scales = 480

    sigma = float(sigma)

    T = round(np.diff(times).mean(), 5)
    f_s = int(round(1 / T, 0))

    print("Signal length:", len(x))
    print("Sampling time:", T)
    print("Sampling freq:", f_s)

    tick()
    wavelet = fcwt.Morlet(sigma)
    scales = fcwt.Scales(wavelet, fcwt.FCWT_LOGSCALES, f_s, f_s*omega_min, f_s*omega_max, n_scales)

    scales_arr = np.zeros(n_scales, dtype=np.float32)
    scales.getScales(scales_arr)
    print(scales_arr)
    
    freqs = np.zeros(n_scales, dtype=np.float32)
    scales.getFrequencies(freqs)

    fcwt_obj = fcwt.FCWT(wavelet, 1, False, True)
    output1 = np.zeros((n_scales, x.size), dtype=np.complex64)
    output2 = np.zeros((n_scales, x.size), dtype=np.complex64)

    x1 = np.copy(x)
    x1[:block_size] = 0
    x1[2*block_size:] = 0

    x2 = np.copy(x)
    x2[:2*block_size] = 0
    x2[3*block_size:] = 0

    fcwt_obj.cwt(x1, scales, output1)
    output_power1 = (np.abs(output1))

    fcwt_obj.cwt(x2, scales, output2)
    output_power2 = (np.abs(output2))

    plot_cwt(f'fcwt{str(sigma).replace(".", "_")}_blocktest1.png', times, freqs, output_power1)
    plot_cwt(f'fcwt{str(sigma).replace(".", "_")}_blocktest2.png', times, freqs, output_power2)
    plot_cwt(f'fcwt{str(sigma).replace(".", "_")}_blocktest12.png', times, freqs, np.abs(output1 + output2))

def plot_fcwt(sigma, times, x, N_pad):
    omega_min = 1 / 500 / 4
    omega_max = 0.5
    n_scales = 480
    nds = 4

    sigma = float(sigma)

    T = round(np.diff(times).mean(), 5)
    f_s = int(round(1 / T, 0))

    print("Signal length:", len(x))
    print("Sampling time:", T)
    print("Sampling freq:", f_s)

    tick()
    wavelet = fcwt.Morlet(sigma)
    scales = fcwt.Scales(wavelet, fcwt.FCWT_LOGSCALES, f_s, f_s*omega_min, f_s*omega_max, n_scales)

    scales_arr = np.zeros(n_scales, dtype=np.float32)
    scales.getScales(scales_arr)
    print(scales_arr)
    
    freqs = np.zeros(n_scales, dtype=np.float32)
    scales.getFrequencies(freqs)

    fcwt_obj = fcwt.FCWT(wavelet, 1, False, True)
    output = np.zeros((n_scales, x.size), dtype=np.complex64)

    tock("Generate")
    fcwt_obj.cwt(x, scales, output)
    # output_power = (np.abs(output[:, ::nds]))
    output_power = (np.abs(output[:, N_pad:-N_pad:nds]))
    # times = np.pad(times, (N_pad, N_pad), 'linear_ramp', end_values=(times[0]-N_pad*T,times[-1]+N_pad*T))

    tock("CWT")
    plt.pcolormesh(times[..., ::nds], freqs, output_power, shading='auto', cmap='turbo')
    tock("plot")

    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    plt.title('CWT using Morlet Wavelet')
    plt.gca().invert_yaxis()
    plt.colorbar(label='Magnitude')
    plt.yscale("log")
    plt.gca().invert_yaxis()
    # plt.gca().set_aspect(aspect=6)
    plt.savefig(f'fcwt{str(sigma).replace(".", "_")}.png', )
    tock("save")
    plt.clf()

def plot_fcwt_blocked(sigma:float | int, times:np.ndarray, x: np.ndarray, block_size: int):
    omega_min = 0.1
    omega_max = 0.5
    n_scales = 480
    nds = 4
    N = x.size

    sigma = float(sigma)

    T = round(np.diff(times).mean(), 5)
    f_s = int(round(1 / T, 0))

    print("Signal length:", len(x))
    print("Sampling time:", T)
    print("Sampling freq:", f_s)


    block_limits = np.arange(N)[::block_size]

    tick()
    wavelet = fcwt.Morlet(sigma)
    scales = fcwt.Scales(wavelet, fcwt.FCWT_LOGSCALES, f_s, f_s*omega_min, f_s*omega_max, n_scales)
    freqs = np.zeros(n_scales, dtype=np.float32)
    scales.getFrequencies(freqs)
    scales_arr = np.zeros(n_scales, dtype=np.float32)
    scales.getScales(scales_arr)
    largest_scale = round(scales_arr.max())
    largest_support = wavelet.getSupport(largest_scale)
    print("Largest scale:", largest_scale)
    fcwt_obj = fcwt.FCWT(wavelet, 1, False, True)
    tock("Initialize")

    out = np.zeros((n_scales, N), dtype=np.complex64)

    for i in tqdm(range(block_limits.size - 1)):
        extras = int(largest_support) - 1
        l = block_limits[i]
        u_in = block_limits[i+1]
        u_out = u_in + extras
        x_block = np.pad(x[l:u_in], (0, extras), mode='constant', constant_values=0)
        out_block = np.zeros((n_scales, u_out - l), dtype=np.complex64)

        fcwt_obj.cwt(x_block, scales, out_block)
        out[:, l:u_out] = out_block
        output_power = (np.abs(out[..., ::nds]))
        plt.pcolormesh(times[..., ::nds], freqs, output_power, shading='auto', cmap='turbo')
        plt.xlabel('Time [s]')
        plt.ylabel('Frequency [Hz]')
        plt.title('CWT using Morlet Wavelet')
        plt.gca().invert_yaxis()
        plt.colorbar(label='Magnitude')
        plt.yscale("log")
        plt.gca().invert_yaxis()
        plt.savefig(f'fcwt{str(sigma).replace(".", "_")}_b.png', )
        plt.clf()


    return out

def plot_cwt(filename, times, freqs, power):
    plt.pcolormesh(times, freqs, power, shading='auto', cmap='turbo')
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    plt.title('CWT using Morlet Wavelet')
    plt.gca().invert_yaxis()
    plt.colorbar(label='Magnitude')
    plt.yscale("log")
    plt.gca().invert_yaxis()
    plt.savefig(filename)
    plt.clf()

if __name__ == "__main__":
    main()