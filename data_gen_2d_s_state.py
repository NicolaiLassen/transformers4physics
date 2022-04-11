import numpy as np
import matplotlib.pyplot as plt
import h5py

from magtense import magtense, micromag_problem


def generate_s_state(
    res=[36, 36, 1],
    NIST_field=1,
    use_CUDA=False,
) -> np.ndarray:
    mu0 = 4*np.pi*1e-7

    ### Magnetization to s-state
    # Setup problem
    problem_ini = micromag_problem.DefaultMicroMagProblem(res)
    problem_ini.dem_appr = micromag_problem.get_micromag_demag_approx(None)
    problem_ini.set_use_CUDA(use_CUDA)

    problem_ini.grid_L = [500e-9, 125e-9, 3e-9]

    problem_ini.alpha = 4.42e3
    problem_ini.gamma = 0

    # Initial magnetization
    problem_ini.m0[:] = (1 / np.sqrt(3))

    # Time-dependent applied field
    t_end = 100e-9
    Hyst_dir = 1 / mu0 *  np.array([1, 1, 1])
    HextFct = lambda t: np.expand_dims(1e-9 - t, 0).T * Hyst_dir * np.expand_dims(t < 1e-9, 0).T
    problem_ini.set_Hext(HextFct, np.linspace(0, t_end, 2000))

    timesteps = 200
    M_ini_out = np.zeros(shape=(timesteps, problem_ini.m0.shape[0], 1, 3))
    t_ini_out = np.zeros(shape=(timesteps))

    for n_t in range(timesteps//50):
        dt = t_end / ( timesteps // 50 )
        problem_ini.set_time(np.linspace(n_t * dt, (n_t + 1) * dt , 50))    

        t, M, pts, _, _, _, _ = magtense.run_micromag_simulation(problem_ini)
        problem_ini.m0 = M[-1,:,0,:]
        M_ini_out[n_t*50:(n_t+1)*50] = M.copy()
        t_ini_out[n_t*50:(n_t+1)*50] = t.copy()

    return M_ini_out

if __name__ == '__main__':
    res = [64,16,1]
    hf = h5py.File('./s_state.h5', 'w')
    seq = generate_s_state(
        res=res,
    )
    seq = seq.reshape(200, res[0], res[1], 3).swapaxes(1,3).swapaxes(2,3)
    hf.create_dataset('m',data=seq)
    hf.close()
