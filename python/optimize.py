import fcwt


def optimize_fcwt(fcwt_obj: fcwt.FCWT, size: int):
    fcwt_obj.create_FFT_optimization_plan(size, "FFTW_PATIENT")


if __name__ == "__main__":
    threads = 4
    N = 2**14
    wavelet = fcwt.Morlet(2)
    obj = fcwt.FCWT(wavelet, threads, True, False)
    optimize_fcwt(obj, N)