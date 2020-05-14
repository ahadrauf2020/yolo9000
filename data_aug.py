import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
import os
import imageio

if __name__ == '__main__':
    path = 'data/tiny-imagenet-200/train'
    path_aug = 'augmented_data/train'
    ia.seed(2)

    seq = iaa.Sequential([
        iaa.Fliplr(0.5), # horizontal flips
        iaa.Crop(percent=(0, 0.1)), # random crops
        # Small gaussian blur with random sigma between 0 and 0.5.
        # But we only blur about 50% of all images.
        iaa.Sometimes(
            0.5,
            iaa.GaussianBlur(sigma=(0, 0.5))
        ),
        # Strengthen or weaken the contrast in each image.
        iaa.LinearContrast((0.75, 1.5)),
        # Add gaussian noise.
        # For 50% of all images, we sample the noise once per pixel.
        # For the other 50% of all images, we sample the noise per pixel AND
        # channel. This can change the color (not only brightness) of the
        # pixels.
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        iaa.Multiply((0.8, 1.2), per_channel=0.2),
        # Apply affine transformations to each image.
        # Scale/zoom them, translate/move them, rotate them and shear them.
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-25, 25),
            shear=(-8, 8)
        )
    ], random_order=True) # apply augmenters in random order

    for folder in os.listdir(path):
        i = 0
        if not folder.startswith('.'):
            for fname in os.listdir(path + '/' + folder):
                if not fname.endswith('.txt'):
                    
                    for item in os.listdir(path + '/' + folder + '/' + fname):
                        if not item.startswith('.'):
                            img = imageio.imread(path + '/' + folder + '/' + fname + '/' + item)
                            # print('Original:')
                            # ia.imshow(img)
                            img_aug = seq.augment_image(img)
                            # print('Augmented:')
                            # ia.imshow(img_aug)
                            # print(os.path.join(path_aug + '/' + folder + '/' + fname + "%06d.png" % (i,)))
                            print(os.path.join(path_aug + '/' + folder + '/' + fname))
                            if not (os.path.join(path_aug + '/' + folder)):
                                os.mkdir(os.path.join(path_aug + '/' + folder))
                            if not (os.path.join(path_aug + '/' + folder + '/' + fname)):
                                os.mkdir(os.path.join(path_aug + '/' + folder + '/' + fname))
                            imageio.imwrite(os.path.join(path_aug + '/' + folder + '/' + fname + "%06d.png" % (i,)), img_aug)
                            i += 1
                else:
                    continue

