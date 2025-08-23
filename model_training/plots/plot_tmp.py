for mode in ['Control', 'Alzheimer', 'AMD', 'GA', 'Glaucoma', 'Neavus', 'Ischaemia']:
    f_result = display_rgb(filter_epoch_results(train_results, mode), filter_epoch_results(val_results, mode), epoch,
                           cfg.hparams.nb_epochs)
    mlflow.log_figure(f_result, artifact_file=f"epoch_{epoch:04d}_{mode}.png")
    plt.close(f_result)

    f_result = display_latent(filter_epoch_results(train_results, mode), filter_epoch_results(val_results, mode), epoch,
                              cfg.hparams.nb_epochs)
    mlflow.log_figure(f_result, artifact_file=f"epoch_{epoch:04d}_{mode}_latent.png")
    plt.close(f_result)

latent_outputs.append(reconstructed_output.cpu().detach().numpy())
rgb_images.append(rgb.cpu().detach().numpy())
mse_spectra.append(mse(reconstructed_output, hs_cube).mean(dim=(2, 3)).cpu().detach().numpy())
input_spectra.append(hs_cube.mean(dim=(2, 3)).cpu().detach().numpy())
output_spectra.append(reconstructed_output.mean(dim=(2, 3)).cpu().detach().numpy())
# unet_outputs.append(unet_output.cpu().detach().numpy())
# targets.append(target_unet.cpu().detach().numpy())
# patient_ids.append(patient_id)
labels.append(label)

mse = nn.MSELoss(reduction='none')
