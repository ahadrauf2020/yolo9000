from EnsembleTable import EnsembleTable



paths = {
    'resnet152': "./models/resnet152_best_model_state_dict.pth",
    'vgg19_bn': './models/vgg19_bn_best_model.pth',
    'dense169': './models/densenet169_best_model_state_dict_v2_65.pth',
    'resatt': './models/res_att_best_model_epoch_15.pth'
}
table = EnsembleTable(paths=paths, fgsm_dataloader=None, blurred_dataloader=None, fgsm_dataset_sizes=None, blurred_dataset_sizes=None)
table.print_table()
