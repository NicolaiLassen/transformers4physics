import numpy as np
import matplotlib.pyplot as plt
import h5py

from magtense import magtense, micromag_problem


def generate_sequence(
    res=[36, 36, 1],
    grid_L=[500e-9, 500e-9, 3e-9],
    timesteps=500,
    use_CUDA=False,
    show=True,
    field_micro_t=[0, 0, 0],
    starting_state=np.zeros([3, 36, 36, 1]),
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
    problem.A0 = 1.3e-11
    problem.Ms = 8.0e5

    timesteps = timesteps
    t_end = time_per_step*timesteps
    Hyst_dir = 1 / mu0 * np.array(field_micro_t) / 1000
    HextFct = lambda t: np.expand_dims(t > -1, 0).T * Hyst_dir
    problem.set_Hext(HextFct, np.linspace(0, t_end, 2000))

    M_out = np.zeros(shape=(timesteps, problem.m0.shape[0], 1, 3))
    t_out = np.zeros(shape=(timesteps))

    # Starting state
    for i in range(problem.m0.shape[0]):
        v = starting_state.reshape(3,res[0]*res[1])[:,i]
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
    read_starting_states,
    res=[36, 36, 1],
    timesteps=500,
    use_CUDA=False,
    grid_L=[500e-9, 500e-9, 3e-9],
    time_per_step = 4e-12,
):
    hf = h5py.File(save_to_file, 'w')
    read = h5py.File(read_starting_states, 'r')
    starting_states = read['states']
    fields = read['fields']
    for i in range(len(starting_states)):
        print('Generating sequence: {}/{}'.format(i+1, len(starting_states)))
        field = fields[i]
        starting_state = starting_states[i]
        seq = generate_sequence(
            res=res,
            timesteps=timesteps,
            use_CUDA=use_CUDA,
            show=False,
            starting_state=starting_state,
            field_micro_t=field,
            grid_L=grid_L,
            time_per_step=time_per_step,
        )
        seq = seq.reshape(timesteps, res[0], res[1], 3).swapaxes(1,3).swapaxes(2,3)
        g = hf.create_group(str(i))
        g.create_dataset('sequence', data=seq)
        g.create_dataset('field', data=field)
    hf.close()
    read.close()


if __name__ == '__main__':
    generate_data_set(
        './mag_data_field_9x9_test.h5',
        './starting_states_9x9_test.h5',
        res=[9, 9, 1],
        grid_L=[125e-9, 125e-9, 3e-9],
        timesteps=400,
    )
    generate_data_set(
        './mag_data_field_18x18_test.h5',
        './starting_states_18x18_test.h5',
        res=[18, 18, 1],
        grid_L=[250e-9, 250e-9, 3e-9],
        timesteps=400,
    )
    generate_data_set(
        './mag_data_field_36x36_test.h5',
        './starting_states_36x36_test.h5',
        res=[36, 36, 1],
        grid_L=[500e-9, 500e-9, 3e-9],
        timesteps=400,
    )
    generate_data_set(
        './mag_data_field_8x8_test.h5',
        './starting_states_8x8_test.h5',
        res=[8, 8, 1],
        grid_L=[125e-9, 125e-9, 3e-9],
        timesteps=400,
    )
    generate_data_set(
        './mag_data_field_16x16_test.h5',
        './starting_states_16x16_test.h5',
        res=[16, 16, 1],
        grid_L=[250e-9, 250e-9, 3e-9],
        timesteps=400,
    )
    generate_data_set(
        './mag_data_field_32x32_test.h5',
        './starting_states_32x32_test.h5',
        res=[32, 32, 1],
        grid_L=[500e-9, 500e-9, 3e-9],
        timesteps=400,
    )

    
    generate_data_set(
        './mag_data_field_9x9_train.h5',
        './starting_states_9x9_train.h5',
        res=[9, 9, 1],
        grid_L=[125e-9, 125e-9, 3e-9],
        timesteps=400,
    )
    generate_data_set(
        './mag_data_field_18x18_train.h5',
        './starting_states_18x18_train.h5',
        res=[18, 18, 1],
        grid_L=[250e-9, 250e-9, 3e-9],
        timesteps=400,
    )
    generate_data_set(
        './mag_data_field_36x36_train.h5',
        './starting_states_36x36_train.h5',
        res=[36, 36, 1],
        grid_L=[500e-9, 500e-9, 3e-9],
        timesteps=400,
    )
    generate_data_set(
        './mag_data_field_8x8_train.h5',
        './starting_states_8x8_train.h5',
        res=[8, 8, 1],
        grid_L=[125e-9, 125e-9, 3e-9],
        timesteps=400,
    )
    generate_data_set(
        './mag_data_field_16x16_train.h5',
        './starting_states_16x16_train.h5',
        res=[16, 16, 1],
        grid_L=[250e-9, 250e-9, 3e-9],
        timesteps=400,
    )
    generate_data_set(
        './mag_data_field_32x32_train.h5',
        './starting_states_32x32_train.h5',
        res=[32, 32, 1],
        grid_L=[500e-9, 500e-9, 3e-9],
        timesteps=400,
    )