import fcwt
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from timing import tick, tock


def main():
    block_size = 2**13
    overlap_size = block_size // 2
    n_threads = 6

    data = np.genfromtxt("synthetic_dataset.csv", delimiter=",", skip_header=1, dtype=np.float32)
    t_data = data[:, 0]
    x_data = data[:, 1]
    x_data_noisy = data[:, 2]

    T = float(round(np.diff(t_data).mean(), 5))
    f_s = int(round(1 / T, 0))
    print("Sampling time:", T)
    print("Sampling freq:", f_s)

    signal = np.concatenate(5*[x_data])
    N_in = signal.size

    # Actually need padding depending on scales, but this will do for now
    N_pad_l = 0
    N_pad_r = signal.size
    
    signal = np.pad(signal, (N_pad_l, N_pad_r), mode='constant', constant_values=0)

    times = np.arange(N_in + N_pad_l + N_pad_r, dtype=np.float32) * T

    print("===================================")
    print(f"Using signal of size {N_in}, padded by ({N_pad_l}, {N_pad_r}) to ({N_in + N_pad_l + N_pad_r})")
    print("===================================")
    # test_block_transform(2, times, signal, 1024)

    n_scales = 480
    nds = 16
    omega_min = 0.0008
    omega_max = 0.5

    for sigma in [1.5,]:
        wavelet = fcwt.Morlet(sigma)
        fcwt_obj = fcwt.FCWT(wavelet, n_threads, True, False)

        optimize(fcwt_obj, n_threads, 2*block_size)

        output_block = plot_fcwt_blocked(times, wavelet, fcwt_obj, signal, f_s, block_size, overlap_size, omega_min, omega_max, n_scales, nds)
        output = plot_fcwt(sigma, times, signal, omega_min, omega_max, n_scales, nds, n_threads)

    output_block /= output_block.max()
    output /= output.max()

    
    scales = fcwt.Scales(wavelet, fcwt.FCWT_LOGSCALES, f_s, f_s*omega_min, f_s*omega_max, n_scales)
    freqs = np.zeros(n_scales, dtype=np.float32)
    scales.getFrequencies(freqs)

    plt.pcolormesh(times[..., ::nds], freqs, np.abs(output[..., ::nds] - output_block[..., ::nds]), shading='auto', cmap='turbo')
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    plt.title('CWT using Morlet Wavelet')
    plt.gca().invert_yaxis()
    plt.colorbar(label='Magnitude')
    plt.yscale("log")
    plt.gca().invert_yaxis()
    plt.savefig(f'fcwt_comparison.png')
    plt.clf()

def optimize(obj, n_threads, size):
    optimization_file = f"n{size}_t{n_threads}.wis"
    if not os.path.exists(optimization_file):
        print(f"Could not find optimization plan \"{optimization_file}\"")
        obj.create_FFT_optimization_plan(size, "FFTW_PATIENT")
    else:
        print(f"Found plan \"{optimization_file}\"!")

def test_block_transform(sigma, times, x, block_size):
    omega_min = 0.001
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

def plot_fcwt(sigma, times, x, omega_min, omega_max, n_scales, nds, n_threads):
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
    
    freqs = np.zeros(n_scales, dtype=np.float32)
    scales.getFrequencies(freqs)

    fcwt_obj = fcwt.FCWT(wavelet, n_threads, True, False)
    output = np.zeros((n_scales, x.size), dtype=np.complex64)

    tock("Generate")
    fcwt_obj.cwt(x, scales, output)
    t = tock("CWT")
    output_power = (np.abs(output[:, ::nds]))
    # output_power = (np.abs(output[:, N_pad:-N_pad:nds]))
    # times = np.pad(times, (N_pad, N_pad), 'linear_ramp', end_values=(times[0]-N_pad*T,times[-1]+N_pad*T))

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

    print(f"Transforms took {t} ms")

    return output

def plot_fcwt_blocked(times:np.ndarray, wavelet: fcwt.Morlet, fcwt_obj: fcwt.FCWT, x_in: np.ndarray, f_s: float, block_size: int, overlap_size: int, omega_min:float, omega_max:float, n_scales:int, nds:int):
    N_blocks = (x_in.size // block_size) + 1
    N_pad_l = overlap_size
    N_pad_r = overlap_size + (block_size - (x_in.size % block_size))
    x = np.pad(x_in, (N_pad_l, N_pad_r), mode='constant', constant_values=0)

    scales = fcwt.Scales(wavelet, fcwt.FCWT_LOGSCALES, f_s, f_s*omega_min, f_s*omega_max, n_scales)
    freqs = np.zeros(n_scales, dtype=np.float32)
    scales.getFrequencies(freqs)
    scales_arr = np.zeros(n_scales, dtype=np.float32)
    scales.getScales(scales_arr)
    largest_scale = round(scales_arr.max())
    largest_support = wavelet.getSupport(largest_scale)
    print("Largest scale:", largest_scale)
    print("Largest support:", largest_support)

    tick()

    out = np.zeros((n_scales, x.size), dtype=np.complex64)
    print("Output shape:", out.shape)

    t_tot = 0.0

    for i in tqdm(range(N_blocks)):
        tick()
        l_block = overlap_size + i * block_size
        u_block = overlap_size + (i + 1) * block_size
        l = l_block - overlap_size
        u = u_block + overlap_size

        # print(f"{l_block}:{u_block} -> {l}:{u}")

        x_block = np.pad(x[l_block:u_block], (overlap_size, overlap_size), mode='constant', constant_values=0)
        t_tot += tock("Padding", supress=False)

        out_block = np.zeros((n_scales, block_size+2*overlap_size), dtype=np.complex64)
        t_tot += tock("Zero initialization", supress=False)

        fcwt_obj.cwt(x_block, scales, out_block)
        t_tot += tock("Block CWT", supress=True)

        out[:, l:u] += out_block

        output_power = np.abs(out[..., N_pad_l:-N_pad_r:nds])

        # l_next = (l_block + block_size) // nds
        # u_next = (u_block + block_size) // nds
        # output_power[:, (l_next-10):l_next] = 1e4
        # output_power[:, (u_next-10):u_next] = 1e4

        # plt.pcolormesh(times[..., ::nds], freqs, output_power, shading='auto', cmap='turbo')
        # plt.xlabel('Time [s]')
        # plt.ylabel('Frequency [Hz]')
        # plt.title('CWT using Morlet Wavelet')
        # plt.gca().invert_yaxis()
        # plt.colorbar(label='Magnitude')
        # plt.yscale("log")
        # plt.gca().invert_yaxis()
        # # plt.savefig(f'fcwt_b.png')
        # plt.clf()

        # input("ENTER to continue...")

    print(f"Transforms took {t_tot} ms")

    plt.pcolormesh(times[..., ::nds], freqs, np.abs(out[..., N_pad_l:-N_pad_r:nds]), shading='auto', cmap='turbo')
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    plt.title('CWT using Morlet Wavelet')
    plt.gca().invert_yaxis()
    plt.colorbar(label='Magnitude')
    plt.yscale("log")
    plt.gca().invert_yaxis()
    plt.savefig(f'fcwt_b.png')
    plt.clf()

    return out[..., N_pad_l:-N_pad_r]

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