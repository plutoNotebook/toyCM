
from tqdm import tqdm

from cm.ct import ConsistencyModel
from cm.toy_tasks.data_generator import DataGenerator
from cm.visualization.vis_utils import plot_main_figure, plot_fid_results, plot_trajectory_comparison
from copy import deepcopy


"""
Discrete training of the consistency model on a toy task.
For better performance, one can pre-training the model with the karras diffusion objective
and then use the weights as initialization for the consistency model.
"""

if __name__ == "__main__":

    device = 'cpu'
    use_pretraining = True
    save_path = 'ct2'
    train_method = 'ct2'
    fids = []
    n_sampling_steps = 10
    cm = ConsistencyModel(
        lr=1e-4,
        sampler_type='onestep',
        sigma_data=0.5,
        sigma_min=0.05,
        sigma_max=1,
        conditioned=False,
        device='mps',
        rho=7,
        t_steps_min=100,
        t_steps=100,
        ema_rate=0.999,
        n_sampling_steps=n_sampling_steps,
        use_karras_noise_conditioning=True,   
        score_corrector=True
    )
    train_epochs = 1500
    # chose one of the following toy tasks: 'three_gmm_1D' 'uneven_two_gmm_1D' 'two_gmm_1D' 'single_gaussian_1D'
    data_manager = DataGenerator('two_gmm_1D')
    samples, cond = data_manager.generate_samples(10000)
    samples = samples.reshape(-1, 1).to(device)
    pbar = tqdm(range(train_epochs))

    
    # Pretraining if desired
    if use_pretraining:
        
        for i in range(train_epochs):
            cond = cond.reshape(-1, 1).to(device)        
            loss = cm.diffusion_train_step(samples, cond, i, train_epochs)
            pbar.set_description(f"Step {i}, Loss: {loss:.8f}")
            pbar.update(1)
        
        dm = deepcopy(cm)

        # plot the results of the pretraining diffusion model to compare with the consistency model
        plot_main_figure(
            data_manager.compute_log_prob, 
            cm, 
            100, 
            train_epochs, 
            sampling_method='euler', 
            n_sampling_steps=n_sampling_steps,
            x_range=[-4, 4], 
            save_path=save_path
        )
        
        cm.update_target_network()
        pbar = tqdm(range(train_epochs))
        
    for i in range(train_epochs):
        cond = cond.reshape(-1, 1).to(device)        
        loss = cm.train_step(samples, cond, i, train_epochs)
        pbar.set_description(f"Step {i}, Loss: {loss:.8f}")
        pbar.update(1)
        fids.append(cm.compute_fid(samples))
    
    # Plotting the results of the training
    # We do this for the one-step and the multi-step sampler to compare the results
    plot_main_figure(
        data_manager.compute_log_prob, 
        cm, 
        100, 
        train_epochs, 
        sampling_method='onestep', 
        x_range=[-4, 4], 
        save_path=save_path
    )

    plot_main_figure(
        data_manager.compute_log_prob, 
        cm, 
        100, 
        train_epochs, 
        sampling_method='multistep', 
        n_sampling_steps=n_sampling_steps,
        x_range=[-4, 4], 
        save_path=save_path
    )

    plot_main_figure(
        data_manager.compute_log_prob, 
        cm, 
        100, 
        train_epochs, 
        sampling_method='euler', 
        n_sampling_steps=n_sampling_steps,
        x_range=[-4, 4], 
        save_path=save_path
    )

    plot_trajectory_comparison(
        diffusion_model=dm,
        consistency_model=cm,
        data_manager=data_manager,
        n_samples=50,
        x_range=[-4, 4],
        save_path=save_path
    )

    plot_fid_results(
        cm.deltas, 
        train_epochs,
        train_method,
        save_path=save_path
    )

    print('done')
