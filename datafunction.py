import os
import SimpleITK as sitk
import torch
import random


class MedData_pretrain(torch.utils.data.Dataset):
    def __init__(self, data_path, crop_size=16):
        self.data_path = data_path
        self.data = os.listdir(data_path)
        self.crop_size = crop_size

    def __len__(self):
        return len(self.data)

    def rand_crop(self, image):
        t, _, _ = image.size()
        new_t = random.randint(0, t - self.crop_size)
        new_image = image[new_t:new_t + self.crop_size, :, :]
        return new_image

    def znorm(self, image):
        mn = image.mean()
        sd = image.std()
        return (image - mn) / sd

    def __getitem__(self, index):
        patient = self.data[index]
        patient_path = os.path.join(self.data_path, patient)
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(patient_path)
        reader.SetFileNames(dicom_names)
        img = reader.Execute()
        image_tensor = torch.FloatTensor(sitk.GetArrayFromImage(img))

        # the type of origin img is int 16,so we convert it to tensor float 32

        znorm_image = self.znorm(image_tensor)
        out_image = self.rand_crop(znorm_image)
        out_image = out_image.unsqueeze(0)
        return out_image


if __name__ == "__main__":
    path = 'postprocess'
    dataset_train = MedData_pretrain(path)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=1)
    for epoch in range(10):
        for ite, batch in enumerate(train_loader):
            img = batch
            print(epoch, ite)
            print(img.size())