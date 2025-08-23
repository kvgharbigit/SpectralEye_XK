import torch
from PIL import Image
from path import Path

from src.segmentation.utils.register_image import register_image
from src.utils.normalise_parameters import unnormalize_parameters
from src.utils.single_to_three_channels import single_to_three_channel


def evalutate_model_dataloader(model, transformation, dataloader, output_folder):
    """ Evaluate the model using the given dataloader and save the results to the output folder. """
    model.eval()
    with torch.no_grad():
        for batch_idx, (im_fixed, im_moving, target_params, shape_fixed, shape_moving, _, _, _, fixed_image_files,
            moving_image_files) in enumerate(dataloader, 1):
            if next(model.parameters()).is_cuda:
                im_fixed, im_moving, target_params = im_fixed.cuda(), im_moving.cuda(), target_params.cuda()


            # prediction = model(torch.cat([im_fixed, im_moving], dim=1))
            # prediction = model(single_to_three_channel(im_fixed), single_to_three_channel(im_moving), transformation)
            prediction = model(single_to_three_channel(im_fixed), single_to_three_channel(im_moving))
            # prediction = model(im_fixed, im_moving, transformation)
            prediction = unnormalize_parameters(prediction, transformation)

            for i in range(len(prediction)):
                # im_moved_pred = register_image(prediction[i], im_moving[i], shape_fixed[i], shape_moving[i],
                #                                transformation_type)
                curent_folder = Path(output_folder) / f'{fixed_image_files[i].stem}'
                curent_folder.makedirs_p()
                im_moved_pred = register_image(prediction[i].cpu().numpy(), im_moving[i].cpu().numpy(),
                                               shape_fixed[i],
                                               shape_moving[i],
                                               transformation)

                im_moved_pred = im_moved_pred.squeeze()
                im_moved_pred = (im_moved_pred * 255).astype('uint8')
                im_moved_pred = Image.fromarray(im_moved_pred)
                im_moved_pred.save(Path(curent_folder) / f'{moving_image_files[i].stem}_moved.tif')

                im_fix = im_fixed[i].squeeze().cpu().numpy()
                im_fix = (im_fix * 255).astype('uint8')
                im_fix = Image.fromarray(im_fix)
                im_fix.save(Path(curent_folder) / f'{fixed_image_files[i].stem}.tif')

                im_mov = im_moving[i].squeeze().cpu().numpy()
                im_mov = (im_mov * 255).astype('uint8')
                im_mov = Image.fromarray(im_mov)
                im_mov.save(Path(curent_folder) / f'{moving_image_files[i].stem}.tif')



# if __name__ == "__main__":
#     model_path = r'C:\Users\xhadoux\Data_projects\registration\src\model_training\working_env\multirun\2024-06-09\12-50-44\0\model.pth'
#     model = torch.load(model_path)
#     model = model.module
#     model = model.to('cuda')
#     model = model.eval()
#
#     transformation = 'similarity'
