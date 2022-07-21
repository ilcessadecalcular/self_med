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


class MedData_train(torch.utils.data.Dataset):
    def __init__(self, source_dir, label_dir,crop_size=16):
        self.source_dir = source_dir
        self.label_dir = label_dir
        self.patient_dir = os.listdir(source_dir)
        self.crop_size = crop_size

    def __len__(self):
        return len(self.patient_dir)

    def rand_crop(self, image_source,image_label):
        t, _, _ = image_source.size()
        new_t = random.randint(0, t - self.crop_size)
        new_image_source = image_source[new_t:new_t + self.crop_size, :, :]
        new_image_label = image_label[new_t:new_t + self.crop_size, :, :]
        return new_image_source,new_image_label

    def znorm(self, image):
        mn = image.mean()
        sd = image.std()
        return (image - mn) / sd

    def __getitem__(self, index):
        patient = self.patient_dir[index]
        source_path = os.path.join(self.source_dir, patient)
        label_path = os.path.join(self.label_dir,patient)
        source_reader = sitk.ImageSeriesReader()
        source_dicom_names = source_reader.GetGDCMSeriesFileNames(source_path)
        source_reader.SetFileNames(source_dicom_names)
        source_img = source_reader.Execute()
        source_image_tensor = torch.FloatTensor(sitk.GetArrayFromImage(source_img))
        # the type of origin img is int 16,so we convert it to tensor float 32
        label_reader = sitk.ImageSeriesReader()
        label_dicom_names = label_reader.GetGDCMSeriesFileNames(label_path)
        label_reader.SetFileNames(label_dicom_names)
        label_img = label_reader.Execute()
        label_image_tensor = torch.FloatTensor(sitk.GetArrayFromImage(label_img))
        znorm_image = self.znorm(source_image_tensor)
        out_image_source, out_image_label= self.rand_crop(znorm_image,label_image_tensor)
        out_image_source = out_image_source.unsqueeze(0)
        out_image_label = out_image_label.unsqueeze(0)
        return out_image_source,out_image_label



# if __name__ == "__main__":
#     source_dir = 'test/source'
#     label_dir = 'test/label'
#     dataset_train = MedData_train(source_dir,label_dir)
#     train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=1)
#     for epoch in range(10):
#         for ite, sample in enumerate(train_loader):
#             img = sample[0]
#             label = sample[1]
#             # label = target
#             print(epoch, ite)
#             print(img)
#             print(label)