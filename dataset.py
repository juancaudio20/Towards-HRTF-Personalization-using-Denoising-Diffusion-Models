import os
import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset
from pysofaconventions import *
import numpy as np
import matplotlib.pyplot as plt


class HUTUBSDataset(Dataset):
    def __init__(self, hrtf_directory, anthro_csv_path, val_sub_idx):
        self.hrtf_directory = hrtf_directory
        self.anthro_csv_path = anthro_csv_path
        self.val_sub_idx = val_sub_idx
        self.load_data()

    def load_data(self):
        #excluded_subjects = [18, 79, 92]
        subject = []
        angles = set()
        hrtf_points = []
        hrtf_lo = []
        for n in range(1, 97):
            file_path = os.path.join(self.hrtf_directory, f'pp{n}_HRIRs_measured.sofa')

            sofa = SOFAFile(file_path, 'r')
            sourcePositions = sofa.getVariableValue('SourcePosition')
            #print("Source Positions")
            #print(sourcePositions)
            subject.append(sofa)
        m = 0
        mn = 0
        for n, sofa_file in enumerate(subject):
            angle = 0
            if n not in {17, 78, 91, self.val_sub_idx}:
                m += 1
                hrtf_data = sofa_file.getDataIR()
                for point in range(440):
                    p = sourcePositions[point]
                    azimuth = p[0]
                    elevation = p[1]
                    if elevation ==0:
                        angle +=1
                        hrtf_point = hrtf_data[point, :, :]
                        hrtf_point = hrtf_point.data
                        if np.isnan(hrtf_point.any()):
                            print(f"nan detected at subject: {n} point: {point}")
                        hrtf_points.append({'hrtf': hrtf_point, 'point': angle, 'azimuth': azimuth, 'elevation': elevation, 'subj': n, 'subj_2': m})

            if n == self.val_sub_idx:
                 mn += 1
                 hrtf_data = sofa_file.getDataIR()
                 for point in range(440):
                    p = sourcePositions[point]
                    azimuth = p[0]
                    elevation = p[1]
                    if elevation == 0:
                        angle += 1
                        #print(angle)
                        #angles.add(azimuth)
                        #print(len(angles))
                        #print(azimuth)
                        #print(elevation)
                        hrtf_point = hrtf_data[point, :, :]
                        hrtf_point = hrtf_point.data
                        if np.isnan(hrtf_point.any()):
                            print(f"nan detected at subject: {n} point: {point}")
                        hrtf_lo.append({'hrtf': hrtf_point, 'point': angle, 'azimuth': azimuth, 'elevation': elevation, 'subj': n, 'subj_2': mn})


        self.all_hrtf_data = torch.from_numpy(np.array([item['hrtf'] for item in hrtf_points]))
        self.lo_hrtf_data = torch.from_numpy(np.array([item['hrtf'] for item in hrtf_lo]))
        self.combined_hrtf_data = torch.cat((self.all_hrtf_data, self.lo_hrtf_data), dim=0)
        self.global_mean = torch.mean(self.combined_hrtf_data)
        self.global_std = torch.std(self.combined_hrtf_data)

        af_csv = pd.read_csv(self.anthro_csv_path, header=0)
        self.sub_lo = torch.tensor(af_csv.iloc[self.val_sub_idx, 0])
        self.head_lo = torch.from_numpy(af_csv.iloc[self.val_sub_idx, 1:14].values)
        self.ears_lo = torch.from_numpy(af_csv.iloc[self.val_sub_idx, 26:].values)
        subs_to_delete = torch.tensor([17, 78, 91])
        self.subject_ids = torch.from_numpy(np.delete(af_csv.iloc[:, 0].values, subs_to_delete, axis=0))
        #print(len(self.subject_ids))
        #print(self.subject_ids)
        self.head_measurements = torch.from_numpy(np.delete(af_csv.iloc[:, 1:14].values, subs_to_delete, axis=0))
        self.ear_measurements = torch.from_numpy(np.delete(af_csv.iloc[:, 26:].values, subs_to_delete, axis=0))
        self.anthro_measurements = torch.from_numpy(np.delete(af_csv.iloc[:, 1:].values, subs_to_delete, axis=0))
        self.anthro_mean = torch.mean(self.anthro_measurements)
        self.anthro_std = torch.std(self.anthro_measurements)
        #self.global_head_mean = torch.mean(self.head_measurements)
        #self.global_head_std = torch.std(self.head_measurements)
        #self.global_ears_mean = torch.mean(self.ear_measurements)
        #self.global_ears_std = torch.std(self.ear_measurements)

        self.normalized_dataset = [
            {
                'hrtf': (torch.from_numpy(item['hrtf']) - self.global_mean) / self.global_std,
                'point': item['point'],
                'subject_id': item['subj'] + 1,
                'azimuth': item['azimuth'],
                'elevation': item['elevation'],
                'head_measurements': torch.reciprocal(1 + torch.exp((self.head_measurements[item['subj_2'] - 1] - self.anthro_mean)/ self.anthro_std)),
                'ear_measurements': torch.reciprocal(1 + torch.exp((self.ear_measurements[item['subj_2'] - 1] - self.anthro_mean)/ self.anthro_std)),
                'global_std': self.global_std,
                'global_mean': self.global_mean
            }

            for item in hrtf_points
        ]

        self.leave_out_subject = [

            {
            'hrtf': (torch.from_numpy(item['hrtf']) - self.global_mean) / self.global_std,
            'point': item['point'],
            'subject_id': item['subj'] + 1,
            'azimuth': item['azimuth'],
            'elevation': item['elevation'],
            'head_measurements': torch.reciprocal(
                1 + torch.exp((self.head_lo - self.anthro_mean) / self.anthro_std)),
            'ear_measurements': torch.reciprocal(
                1 + torch.exp((self.ears_lo - self.anthro_mean) / self.anthro_std)),
            'global_std': self.global_std,
            'global_mean': self.global_mean
            }
            for item in hrtf_lo
        ]

    def __len__(self):
        return len(self.normalized_dataset), len(self.leave_out_subject)

    def __getitem__(self, idx):
        if idx == 0:
            return self.normalized_dataset

        elif idx == 1:
            return self.leave_out_subject


def collate_fn(batch):

    audio_batch = [item['hrtf'] for item in batch]
    subject_id_batch = [item['subject_id'] for item in batch]
    point_batch = [item['point'] for item in batch]
    azimuth_batch = [item['azimuth'] for item in batch]
    elevation_batch = [item['elevation'] for item in batch]
    head_measurements_batch = [item['head_measurements']for item in batch]
    ear_measurements_batch = [item['ear_measurements'] for item in batch]

    return {'hrtf': torch.stack(audio_batch),
            'point': torch.tensor(point_batch),
            'azimuth': torch.tensor(azimuth_batch),
            'elevation': torch.tensor(elevation_batch),
            'subject_id': torch.LongTensor(subject_id_batch),
            'head_measurements': torch.stack(head_measurements_batch),
            'ear_measurements': torch.stack(ear_measurements_batch),
            }
