import os
import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset
from pysofaconventions import *
import numpy as np
import matplotlib.pyplot as plt

class HUTUBSDataset(Dataset):
    def __init__(self, hrtf_directory, anthro_csv_path):
        self.hrtf_directory = hrtf_directory
        self.anthro_csv_path = anthro_csv_path
        self.load_data()

    def load_data(self):
        #excluded_subjects = [18, 79, 92]
        subject = []

        hrtf_points = []
        for n in range(1, 97):


            file_path = os.path.join(self.hrtf_directory, f'pp{n}_HRIRs_measured.sofa')

            sofa = SOFAFile(file_path, 'r')
            sourcePositions = sofa.getVariableValue('SourcePosition')
            subject.append(sofa)

        for n, sofa_file in enumerate(subject):
            # print("sub: ", n)
            hrtf_data = sofa_file.getDataIR()
            for point in range(440):
                p = sourcePositions[point, 1]

                hrtf_point = hrtf_data[point, :, :]
                hrtf_point = hrtf_point.data
                #hrtf_point = hrtf_point[:, 20:128]
                #print("hrtf_point:", hrtf_point.shape)
                if np.isnan(hrtf_point.any()):
                    print(f"nan detected at subject: {n} point: {point}")

                hrtf_points.append({'hrtf': hrtf_point, 'point': point, 'subj': n})

        self.all_hrtf_data = torch.from_numpy(np.array([item['hrtf'] for item in hrtf_points]))
        self.all_hrtf_points = torch.from_numpy(np.array([item['point'] for item in hrtf_points]))

        self.global_mean = torch.mean(self.all_hrtf_data)
        self.global_std = torch.std(self.all_hrtf_data)

        af_csv = pd.read_csv(self.anthro_csv_path, header=0)
        # af_csv_filtered = af_csv[~af_csv['SubjectID'].isin(excluded_subjects)]
        self.subject_ids = torch.from_numpy(af_csv.iloc[:, 0].values)
        # print("sub size: ", self.subject_ids)
        self.head_measurements = torch.from_numpy(af_csv.iloc[:, 1:14].values)
        # self.head_measurements = self.head_measurements[~af_csv['SubjectID'].isin(excluded_subjects)]
        # print("head: ", self.head_measurements)
        self.ear_measurements = torch.from_numpy(af_csv.iloc[:, 14:].values)
        # self.ear_measurements = self.ear_measurements[~af_csv['SubjectID'].isin(excluded_subjects)]
        # print("NaN in Head Measurements:", torch.isnan(self.head_measurements).any())
        self.head_measurements[torch.isnan(self.head_measurements)] = 0
        self.ear_measurements[torch.isnan(self.ear_measurements)] = 0
        # print("NaN in Head Measurements:", torch.isnan(self.head_measurements).any())
        self.global_head_mean = torch.mean(self.head_measurements)
        self.global_head_std = torch.std(self.head_measurements)
        self.global_ears_mean = torch.mean(self.ear_measurements)
        self.global_ears_std = torch.std(self.ear_measurements)

        self.normalized_dataset = [
            {
                'hrtf': (torch.from_numpy(item['hrtf']) - self.global_mean) / self.global_std,
                'subject_id': item['subj'] + 1,
                'measurement_point': item['point'],
                'head_measurements': (self.head_measurements[self.subject_ids[item[
                    'subj']] - 1] - self.global_head_mean) / self.global_head_std,
                'ear_measurements': (self.ear_measurements[self.subject_ids[item[
                    'subj']] - 1] - self.global_ears_mean) / self.global_ears_std,
                'global_std': self.global_std,
                'global_mean': self.global_mean,
            }

            for item in hrtf_points
        ]

    def __len__(self):
        return len(self.normalized_dataset)

    def __getitem__(self, idx):
        return self.normalized_dataset[idx]

hutubs_dataset = HUTUBSDataset(
    hrtf_directory='/nas/home/jalbarracin/datasets/HUTUBS/HRIRs',
    anthro_csv_path='/nas/home/jalbarracin/datasets/HUTUBS/AntrhopometricMeasures.csv'
)


indices_to_remove = []
for n in range(92):
    for p in range(440):
        hutubs_inspect = hutubs_dataset[n * 440 + p]
        if hutubs_inspect['subject_id'] in [18,79,92]:
            indices_to_remove.append(n * 440 + p)

for idx in sorted(indices_to_remove, reverse=True):
    hutubs_dataset.normalized_dataset.pop(idx)


subject_idx = 1
measurement_point = 9

data_point = hutubs_dataset[subject_idx * 440 + measurement_point]
#print("HRTF shape: ", data_point['hrtf'].shape)
#print("HRTF shape: ", data_point['hrtf'])

#plt.figure(figsize=(14, 5))
#plt.plot(data_point['hrtf'][0],label='L', linewidth=0.5, marker='o', markersize=1)
#plt.plot(data_point['hrtf'][1],label='R', linewidth=0.5, marker='o', markersize=1)
#plt.xlim([0,256])
#plt.title(f'Subject: {data_point["subject_id"]}, Position: {data_point["measurement_point"]}')
#plt.legend()
#plt.show()

def collate_fn(batch):

    audio_batch = [item['hrtf'] for item in batch]
    subject_id_batch = [item['subject_id'] for item in batch]
    measurement_point_batch = [item['measurement_point'] for item in batch]
    head_measurements_batch = [item['head_measurements']for item in batch]
    ear_measurements_batch = [item['ear_measurements'] for item in batch]

    return {'hrtf': torch.stack(audio_batch),
            'measurement_point': torch.tensor(measurement_point_batch),
            'subject_id': torch.LongTensor(subject_id_batch),
            'head_measurements': torch.stack(head_measurements_batch),
            'ear_measurements': torch.stack(ear_measurements_batch)
            }