import numpy as np
import matplotlib.pyplot as plt
import h5py

from magtense import magtense, micromag_problem


def generate_sequence(
    rng,
    res=[36, 36, 1],
    grid_L=[500e-9, 500e-9, 3e-9],
    timesteps=500,
    use_CUDA=False,
    show=True,
    field_micro_t=[0, 0, 0],
    time_per_step = 4e-12,
) -> np.ndarray:
    assert timesteps % 50 == 0, 'Timesteps must be multiple of 50'
    mu0 = 4*np.pi*1e-7

    problem = micromag_problem.DefaultMicroMagProblem(res)
    problem.dem_appr = micromag_problem.get_micromag_demag_approx(None)
    problem.set_use_CUDA(use_CUDA)

    problem.grid_L = grid_L
    # Note, alpha = 0.02 gives a lambda val equal to 4.42e3
    problem.alpha = 4.42e3
    problem.gamma = 2.21e5
    problem.setTimeDis = 10

    # Material properties
    # problem.A0 = 1.3e-11
    # problem.Ms = 8.0e5

    timesteps = timesteps
    t_end = time_per_step*timesteps
    Hyst_dir = 1 / mu0 * np.array(field_micro_t) / 1000
    HextFct = lambda t: np.expand_dims(t > -1, 0).T * Hyst_dir
    problem.set_Hext(HextFct, np.linspace(0, t_end, 2000))

    M_out = np.zeros(shape=(timesteps, problem.m0.shape[0], 1, 3))
    t_out = np.zeros(shape=(timesteps))

    # Starting state
    for i in range(problem.m0.shape[0]):
        v = 2 * rng.random((3)) - 1
        problem.m0[i] = 1 / np.linalg.norm(v) * v

    for n_t in range(timesteps//50):
        dt = t_end / (timesteps // 50)
        problem.set_time(np.linspace(n_t * dt, (n_t + 1) * dt, 50))

        t, M, _, _, _, _, _ = magtense.run_micromag_simulation(problem)
        problem.m0 = M[-1, :, 0, :]
        M_out[n_t*50:(n_t+1)*50] = M
        t_out[n_t*50:(n_t+1)*50] = t

    if show:
        plt.plot(t_out, np.mean(M_out[:, :, 0, 0], axis=1), 'rx')
        plt.plot(t_out, np.mean(M_out[:, :, 0, 1], axis=1), 'gx')
        plt.plot(t_out, np.mean(M_out[:, :, 0, 2], axis=1), 'bx')
        plt.show()

        # TODO Plot start and end state
    return M_out


def generate_data_set(
    save_to_file,
    res=[36, 36, 1],
    num_sequences=4,
    timesteps=500,
    use_CUDA=False,
    seed=None,
    field_interval=[-50, 50],
    grid_L=[500e-9, 500e-9, 3e-9],
    time_per_step = 4e-12,
):
    rng = np.random.default_rng(seed)
    hf = h5py.File(save_to_file, 'w')
    for i in range(num_sequences):
        print('Generating sequence: {}/{}'.format(i+1, num_sequences))
        field = np.zeros(3)
        field[0:2] = (field_interval[1]-field_interval[0]) * \
            rng.random((2)) + field_interval[0]
        seq = generate_sequence(
            rng,
            res=res,
            timesteps=timesteps,
            use_CUDA=use_CUDA,
            show=False,
            field_micro_t=field,
            grid_L=grid_L,
            time_per_step=time_per_step,
        )
        seq = seq.reshape(timesteps, res[0], res[1], 3).swapaxes(1,3).swapaxes(2,3)
        g = hf.create_group(str(i))
        g.create_dataset('sequence', data=seq)
        g.create_dataset('field', data=field)
    hf.close()


if __name__ == '__main__':
    generate_data_set(
        './mag_data_with_field_train.h5',
        res=[64, 16, 1],
        grid_L=[500e-9, 125e-9, 3e-9],
        num_sequences=50,
        timesteps=400,
        seed=42,
        field_interval=[-25,25],
    )
