
import torch 
import numpy as np
from torch.distributions import Normal
from cm.ct import ConsistencyModel
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.interpolate import interpn
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d, interp2d
from matplotlib.colors import LogNorm, PowerNorm
from matplotlib.collections import LineCollection
from scipy.stats import gaussian_kde, norm
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection

import torch
from torch.distributions.normal import Normal
from cm.ct import ConsistencyModel
import os
import csv


def get_test_samples(model, n_samples, sampling_method, n_sampling_steps):
    """
    Obtain the test samples from the given model, based on the specified sampling method and diffusion sampling type.

    Args:
        model (object): Model to be used for sampling (ConsistencyModel or Beso).
        n_samples (int): Number of samples to be taken.
        sampling_method (str, optional): Method to be used for sampling ('multistep', 'onestep', or 'euler').
        n_sampling_steps (int, optional): Number of sampling steps. Defaults to 10.

    Returns:
        test_samples (list): List of test samples obtained from the given model.
    """
    if sampling_method == 'multistep':
        return model.sample_multistep(torch.zeros((n_samples, 1)), None, return_seq=True, n_sampling_steps=n_sampling_steps)
    elif sampling_method == 'onestep':
        return model.sample_singlestep(torch.zeros((n_samples, 1)), None, return_seq=True)
    elif sampling_method == 'euler':
        return model.sample_diffusion_euler(torch.zeros((n_samples, 1)), None, return_seq=True, n_sampling_steps=n_sampling_steps)
    else:
        raise ValueError('sampling_method must be either multistep, onestep or euler')
    

def plot_main_figure(
    fn, 
    model, 
    n_samples, 
    train_epochs, 
    sampling_method='euler',
    x_range=[-4, 4], 
    n_sampling_steps = 10,
    save_path='./plots',
    name = 'dm'
):  
    """
    Plot the main figure for the given model and sampling method.
    Args:
    fn (callable): Target function to be plotted.
    model (object): Model to be used for sampling (ConsistencyModel or Beso).
    n_samples (int): Number of samples to be taken.
    train_epochs (int): Number of training epochs.
    sampling_method (str, optional): Method to be used for sampling ('multistep', 'onestep', or 'euler'). Defaults to False.
    x_range (list, optional): Range of x values to be plotted. Defaults to [-5, 5].
    n_sampling_steps (int, optional): Number of sampling steps. Defaults to 10.
    save_path (str, optional): Directory to save the plot. Defaults to '/home/moritz/code/cm_1D_Toy_Task/plots'.

    Raises ValueError: If the sampling_method is not one of the specified options ('multistep', 'onestep', or 'euler').
    """
    test_samples = get_test_samples(model, n_samples, sampling_method, n_sampling_steps) 
    test_samples = [x.detach().cpu().numpy() for x in test_samples] 
    test_samples = np.stack(test_samples, axis=1)

    x_test = np.linspace(x_range[0], x_range[1], n_samples)
    target_fn = fn(torch.tensor(x_test), exp=True)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    ax1.set_xlim(*x_range)
    ax2.set_xlim(*x_range)
    ax3.set_xlim(*x_range)

    # Plot target distribution
    ax1.plot(x_test, target_fn, color='black', label='Target Distribution')

    # Plot predicted distribution
    kde = gaussian_kde(test_samples[:, -1, 0], bw_method=0.1)
    predicted_distribution = kde(x_test)
    ax1.plot(x_test, predicted_distribution, label='Predicted Distribution')

    # Create a LineCollection to show colors on the predicted distribution line
    points = np.array([x_test, predicted_distribution]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap='viridis', norm=plt.Normalize(predicted_distribution.min(), predicted_distribution.max()))
    lc.set_array(predicted_distribution)
    lc.set_linewidth(2)

    ax1.add_collection(lc)
    stepsize = np.linspace(0, 1, model.n_sampling_steps)
    # stepsize = cm.get_noise_schedule(model.n_sampling_steps, noise_schedule_type='exponential').flip(0)
    # ax2.set_ylim(-0.1, 1.1)
    if sampling_method == 'onestep':
        n_sampling_steps = 1
        stepsize = np.linspace(0, 1, 2)
        ax2.quiver(test_samples[:, 0].reshape(-1),
                    stepsize[0] * np.ones(n_samples),
                    test_samples[:, 1].reshape(-1) - test_samples[:, 0].reshape(-1),
                    stepsize[1] * np.ones(n_samples) - stepsize[0] * np.ones(n_samples),
                    angles='xy', scale_units='xy', scale=1,
                    width=0.001
                    )
    else:
        n_sampling_steps = model.n_sampling_steps
        for i in range(1, n_sampling_steps):
            ax2.quiver(test_samples[:, i - 1].reshape(-1),
                    stepsize[i - 1] * np.ones(n_samples),
                    test_samples[:, i].reshape(-1) - test_samples[:, i-1].reshape(-1),
                    stepsize[i] * np.ones(n_samples) - stepsize[i - 1] * np.ones(n_samples),
                    angles='xy', scale_units='xy', scale=1,
                    width=0.001
                    )
    ax2.set_yticks([stepsize.min(), stepsize.max()])
    ax2.set_ylim(stepsize.min(), stepsize.max())
    
    mu = 0  # mean
    sigma = model.sigma_max  # standard deviation

    # Compute the PDF values for x_test
    prob_samples = norm.pdf(x_test, loc=mu, scale=sigma)
    # Create a LineCollection to show colors on the normal distribution line
    points = np.array([x_test, prob_samples]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap='viridis', norm=plt.Normalize(prob_samples.min(), prob_samples.max()))
    lc.set_array(prob_samples)
    lc.set_linewidth(2)

    ax3.add_collection(lc)
    ax3.set_ylim(0, 0.5)

    # ... (previous code remains unchanged)
    ax2.set_xticks([])
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax3.set_yticks([])
    ax2.set_yticklabels(['T', '0'])
    ax2.tick_params(axis='y', labelsize=16)
    # ax2.set_yticks('log')
    plt.subplots_adjust(hspace=0)
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(save_path + f'/{name}_' + sampling_method + f'_epochs_{train_epochs}.png', bbox_inches='tight', pad_inches=0.1)    
    
    print('Plot saved!')



def plot_results(
    stats, 
    train_epochs, 
    name,
    save_path='./plots',
):

    os.makedirs(save_path, exist_ok=True)
    
    stats = np.array([stat.cpu().numpy() if stat.is_cuda or stat.device.type == 'mps' else stat.numpy() for stat in stats])
    epochs = np.arange(1, len(stats) + 1)  # x-axis values (e.g., epoch numbers)
    
    # Plot the scores
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, stats, marker='o', color='blue', label=f'difference of two sample {name}')
    plt.title(f'difference of two sample', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('abs of difference', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    
    # Save the plot
    plot_filename = f"results_{name}_epochs_{train_epochs}.png"
    plot_path = os.path.join(save_path, plot_filename)
    plt.savefig(plot_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()  # Close the plot to avoid display during batch processing
    print(f"Plot saved at {plot_path}")
    
    csv_path = os.path.join(save_path, f"results_{name}_epochs_{train_epochs}.csv")
    with open(csv_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Epoch', 'values'])
        for epoch, stat in zip(epochs, stats):
            writer.writerow([epoch, stat])
    print(f"difference of {name} results saved to {csv_path}")


def plot_trajectory_comparison(
    diffusion_model,
    consistency_model,
    data_manager,
    n_samples,
    x_range=[-4, 4],
    save_path='./plots'
):
    """
    Plot and compare the trajectories of three methods: pretrained model, 
    trained consistency model, and network-free score estimator.

    Args:
    diffusion_model (object): The pretrained diffusion model.
    consistency_model (object): The trained consistency model.
    data_manager (DataGenerator): Data manager to compute log probabilities.
    n_samples (int): Number of samples to visualize.
    x_range (list, optional): Range of x values to plot. Defaults to [-4, 4].
    save_path (str, optional): Directory to save the plot. Defaults to './plots'.
    """
    os.makedirs(save_path, exist_ok=True)
    device = consistency_model.device

    # Generate initial samples and noise
    x_0, cond = data_manager.generate_samples(n_samples)  # Target distribution에서 샘플링
    x_test = x_0 + consistency_model.sigma_min * torch.randn_like(x_0)
    cond = cond.reshape(-1, 1).to(device)

    x_test = x_test.reshape(-1, 1).to(device)  # Reshape and move to device
    noise = torch.randn_like(x_test).to(device)  # 랜덤 노이즈 생성
    sigmas = diffusion_model.sample_seq_timesteps(N=consistency_model.n_sampling_steps+1, type='karras')

########################################################################################################################################################################
    # diffusino reverse trajectory, CD , CT , CM 

    x_r_score_estimate = [] # x_r along various noise level when Consistency Training
    x_r_pretrained = [] # x_r along various noise level when Consistency Distillation
    x_r_consistency = [] # x_r along various noise level when ours method

    # 진행 경로 설정 (역방향 또는 정방향)
    for i in range(consistency_model.n_sampling_steps):
        noise = torch.randn_like(x_test).to(device)  # 랜덤 노이즈 생성

        # 1. Score Estimator
        x_1 = x_test + sigmas[i+1] * noise  # 노이즈와 함께 이동
        x_r_score_estimate.append(x_1)

        # 2. Pretrained Model (denoise toward target)
        x_2 = x_test + sigmas[i] * noise  # 노이즈와 함께 이동
        denoised = diffusion_model.diffusion_wrapper(diffusion_model.model, x_2, cond, sigmas[i])
        x_1 = diffusion_model.euler_update_step_wx0(x_2, sigmas[i], sigmas[i+1], denoised)
        x_r_pretrained.append(x_1)

        # 3. Consistency Model (denoise with consistency model)
        x_2 = x_test + sigmas[i] * noise  # 노이즈와 함께 이동
        denoised = consistency_model.consistency_wrapper(consistency_model.model, x_2, cond, torch.tensor([sigmas[i]]))
        x_1 = consistency_model.euler_update_step_wx0(x_2, sigmas[i], sigmas[i+1], denoised)
        x_r_consistency.append(x_1)


    x_r_score_estimate.append(x_test)
    x_r_pretrained.append(x_test)
    x_r_consistency.append(x_test)


    reverse_samples = [x_test]
    for i in range(diffusion_model.n_sampling_steps):
        denoised = diffusion_model.diffusion_wrapper(diffusion_model.model, reverse_samples[-1], cond, sigmas[-1-i])
        #denoised = x_0.reshape(-1, 1).to(device)
        dxdt = (reverse_samples[-1] - denoised) / sigmas[-1-i]
        x_next = reverse_samples[-1] + (sigmas[-2-i] - sigmas[-1-i]) * dxdt
        reverse_samples.append(x_next)

    reverse_samples = reverse_samples[::-1]


###################################################################################################################################################


# # Plot trajectories
    plt.figure(figsize=(8, 10))
    plt.ylim(0, consistency_model.n_sampling_steps)


    for i in range(n_samples):
        plt.plot(
            [x_r_score_estimate[j][i].detach().cpu().numpy() for j in range(len(x_r_score_estimate))],
            np.arange(len(x_r_score_estimate)),
            color='#e41a1c', alpha=0.7, linestyle='-', linewidth=1,
            label='Diffusion Model' if i == 0 else ""  # Label only once
        )


    for i in range(n_samples):
        plt.plot(
            [x_r_pretrained[j][i].detach().cpu().numpy() for j in range(len(x_r_pretrained))],
            np.arange(len(x_r_pretrained)),
            color='#377eb8', alpha=0.7, linestyle='-', linewidth=1,
            label='Consistency Model' if i == 0 else ""  # Label only once
        )


    for i in range(n_samples):
        plt.plot(
            [x_r_consistency[j][i].detach().cpu().numpy() for j in range(len(x_r_consistency))],
            np.arange(len(x_r_consistency)),
            color='#4daf4a', alpha=0.7, linestyle='-', linewidth=1,
            label='True Trajectory' if i == 0 else ""  # Label only once
        )


    for i in range(n_samples):
        plt.plot(
            [reverse_samples[j][i].detach().cpu().numpy() for j in range(len(reverse_samples))],
            np.arange(len(reverse_samples)),
            color='#ff7f00', alpha=0.7, linestyle='-', linewidth=1,
            label='CM sampling' if i == 0 else ""  # Label only once
        )


# Final plot adjustments
    plt.title("Trajectory Comparison: Consistency Model vs Diffusion Model")
    plt.xticks([])  # Remove x-axis ticks
    plt.yticks([])  # Remove y-axis ticks
    plt.xlabel("")  # Remove x-axis label
    plt.ylabel("")  # Remove y-axis label

    # Add y-axis top (0) and bottom (T) labels manually
    plt.text(-0.05, 0, 'T', ha='center', va='center', fontsize=12, transform=plt.gca().transAxes)
    plt.text(-0.05, 1, '0', ha='center', va='center', fontsize=12, transform=plt.gca().transAxes)

# Fixing the legend labels for 4 methods
    plt.legend(loc="upper right")  # Automatically picks labels with `label` defined above

# Add grid for better readability
    plt.grid(False)

# Save the plot
    plt.savefig(f"{save_path}/trajectory_comparison_4methods.png", bbox_inches='tight')
    plt.close()
    print('Trajectory comparison plot saved!')
