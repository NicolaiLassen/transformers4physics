import os
import numpy as np
import ray

from asyncio import Event
from typing import Tuple
from ray.actor import ActorHandle
from tqdm import tqdm
from datetime import datetime
from multiprocessing import cpu_count

from magtense import magtense

def create_halbach(
    idx,
    res=256,
    datapath='',
    t_start=None,
    shared=False,
    check=False,
    cube=False,
    extended=False,
    test=False,
    pba=None
):
    """ Create the database for a 3-D magnetic field surrounded by a randomized halbach magnet setup.
    Parameters:
        idx: Internal iterator. A new random generator is initiated.
        res: Resolution of magnetic field.
        datapath: Indicates where to store the database.
        t_start: Starting time for progress bar.
        shared: True if multiprocessing is in use.
        check: Boolean for possible visual output of created field and correspondingt magnetic setup.
        cube: Boolean for 3-D magnetic field being enclosed by magnets and output is field in 3 dims.
        test: Creating data with different seeds for random number generator.
        pba: Ray actor for progress bar.
    """
    # Omit already created files for restart
    if os.path.isfile(f'{datapath}/{idx}.npy'): 
        if shared: pba.update.remote(1)
        return

    seed = idx + 100000 if test else idx
    rng = np.random.default_rng(seed)
    sample_dict = {}
        
    # 10x10 grid with randomly varying size of empty center
    hole_dict = {0:[4,5], 1:[4,6], 2:[3,6], 3:[3,7]}
    empty_pos = hole_dict[rng.integers(4)]
    empty_z = [empty_pos[0] / 2, empty_pos[1] / 2] if cube else [2,2]

    # Setup tiles - Quadratic prisms for now with same z-component for offset
    z_places = 5
    z_area = 0.5
    
    A = rng.integers(2, size=(10,10,z_places))
    filled_pos = [[i, j, k] for i in range(10) for j in range(10) for k in range(z_places) 
                    if A[i][j][k] == 1 and (i < empty_pos[0] or i > empty_pos[1] or j < empty_pos[0]
                    or j > empty_pos[1] or k < np.floor(empty_z[0]) or k > empty_z[1])]
    
    # Randomly oriented magnetization vector in 3D
    rand_arr = rng.random(size=(len(filled_pos),2))
    mag_angles = [[np.pi * rand_arr[i,0], 2 * np.pi * rand_arr[i,1]] for i in range(len(filled_pos))]

    (tiles, _, _) = magtense.setup(
        places=[10, 10, z_places],
        area=[1, 1, z_area],
        mag_angles=mag_angles,
        filled_positions=filled_pos
    )
    
    # Area to evaluate field in
    x_eval = np.linspace(empty_pos[0] + 0.5, empty_pos[1] + 0.5, res + 1) / 10
    y_eval = np.linspace(empty_pos[0] + 0.5, empty_pos[1] + 0.5, res + 1) / 10
    if cube:
        res_z = int(res * 0.5)
        z_eval = np.linspace(empty_z[0] + 0.25, empty_z[1] + 0.25, res_z + 1) / 10
        xv, yv, zv = np.meshgrid(x_eval[:res], y_eval[:res], z_eval[:res_z])
    elif extended:
        res_z = 5
        z_eval = np.linspace(-2*(empty_pos[1] - empty_pos[0]) / res, 2*(empty_pos[1] - empty_pos[0]) / res, res_z) / 10 + z_area / 2
        xv, yv, zv = np.meshgrid(x_eval[:res], y_eval[:res], z_eval)
    else:
        xv, yv = np.meshgrid(x_eval[:res], y_eval[:res])
        zv = np.zeros(res * res) + z_area / 2
    pts_eval = np.hstack([xv.reshape(-1,1), yv.reshape(-1,1), zv.reshape(-1,1)])

    # Running simulation
    iterated_tiles = magtense.iterate_magnetization(tiles)
    N = magtense.get_N_tensor(iterated_tiles, pts_eval)
    H = magtense.get_H_field(iterated_tiles, pts_eval, N)

    if cube or extended:
        # Tensor image with shape CxHxWxD
        field = np.zeros(shape=(3, res, res, res_z), dtype=np.float32)
        field = H.reshape((res,res,res_z,3)).transpose((3,0,1,2))
    else:
        # Tensor image with shape CxHxW
        field = np.zeros(shape=(3, res, res), dtype=np.float32)
        field = H.reshape((res,res,3)).transpose((2,0,1))

    # Saving field in [T]
    mu0 = 4 * np.pi * 1e-7
    field = field * mu0

    # Plot first ten samples
    if check and idx < 20:
        print("sample_check")
        # sample_check(field, iterated_tiles, pts_eval, v_max=0.2, filename=f'{t_start.strftime("%y%m%d_%H%M")}_{idx}', cube=False, structure=True)
    
    sample_dict[idx] = field
    if sample_dict[idx] is not None: np.save(f'{datapath}/{idx}.npy', sample_dict[idx])
    
    # Progress bar
    if shared: pba.update.remote(1)    


@ray.remote
class ProgressBarActor:
    counter: int
    delta: int
    event: Event

    def __init__(self) -> None:
        self.counter = 0
        self.delta = 0
        self.event = Event()

    def update(self, num_items_completed: int) -> None:
        """Updates the ProgressBar with the incremental
        number of items that were just completed.
        """
        self.counter += num_items_completed
        self.delta += num_items_completed
        self.event.set()

    async def wait_for_update(self) -> Tuple[int, int]:
        """Blocking call.

        Waits until somebody calls `update`, then returns a tuple of
        the number of updates since the last call to
        `wait_for_update`, and the total number of completed items.
        """
        await self.event.wait()
        self.event.clear()
        saved_delta = self.delta
        self.delta = 0
        return saved_delta, self.counter

    def get_counter(self) -> int:
        """
        Returns the total number of complete items.
        """
        return self.counter


class ProgressBar:
    progress_actor: ActorHandle
    total: int
    description: str
    pbar: tqdm

    def __init__(self, total: int, description: str = ""):
        # Ray actors don't seem to play nice with mypy, generating
        # a spurious warning for the following line,
        # which we need to suppress. The code is fine.
        self.progress_actor = ProgressBarActor.remote()  # type: ignore
        self.total = total
        self.description = description

    @property
    def actor(self) -> ActorHandle:
        """Returns a reference to the remote `ProgressBarActor`.

        When you complete tasks, call `update` on the actor.
        """
        return self.progress_actor

    def print_until_done(self) -> None:
        """Blocking call.

        Do this after starting a series of remote Ray tasks, to which you've
        passed the actor handle. Each of them calls `update` on the actor.
        When the progress meter reaches 100%, this method returns.
        """
        pbar = tqdm(desc=self.description, total=self.total)
        while True:
            delta, counter = ray.get(self.actor.wait_for_update.remote())
            pbar.update(delta)
            if counter >= self.total:
                pbar.close()
                return


def create_db(size, name='', res=400, num_proc=None, check=False, cube=False, ext=False, test=False, start_idx=0):
    datapath = os.path.dirname(os.path.abspath(__file__)) + f'/../data/{name}_{res}'
    if not os.path.exists(datapath): os.makedirs(datapath)
    worker = cpu_count() if num_proc is None else num_proc
    if worker > 1: ray.init(num_cpus=worker, include_dashboard=False, local_mode=False)

    t_start = datetime.utcnow()
    print(f'[INFO {t_start.strftime("%d/%m %H:%M:%S")}] #Data: {size} | #Worker: {worker} | #Path: {datapath}')

    if num_proc == 1:
        for idx in range(start_idx, size):
            create_halbach(idx=idx, res=res, datapath=datapath, t_start=t_start, check=check, cube=cube, extended=ext, test=test)
    else:
        pb = ProgressBar(size - start_idx)
        actor = pb.actor
        create_halbach_ray = ray.remote(create_halbach)
        res = [create_halbach_ray.remote(idx, res, datapath, t_start, True, check, cube, ext, test, actor) for idx in range(start_idx, size)]
        pb.print_until_done()
        _ = [ray.get(r) for r in res]
        ray.shutdown()


if __name__ == '__main__':
    create_db(
        size=5, # Number of overall samples
        # start_idx=10000,
        name='IntermagMMM_viz', # Name of experiment
        res=256, # Resolution
        num_proc=1, # Number of processors, if num_proc is None then os.cpu_count() is used
        check=True, # Boolean if samples are plotted after creation
        cube=False,
        ext=False,
        # test=False
    )
