import os
from collections import OrderedDict
import numpy as np
import pandas as pd
from glob import glob
import SimpleITK as sitk
base_dir = os.path.join('../..', 'data/datasets/crawford', 'qureai-headct')
reads_dir = os.path.join('../..', 'data/datasets/crawford/qureai-headct/versions', '2')

all_dicom_paths = glob(os.path.join(base_dir, '*', '*', '*', '*', '*', '*', '*'))
print(len(all_dicom_paths), 'dicom files')
dicom_df = pd.DataFrame(dict(path = all_dicom_paths))
dicom_df['SliceNumber'] = dicom_df['path'].map(lambda x: int(os.path.splitext(x.split('/')[-1])[0][2:]))
dicom_df['SeriesName'] = dicom_df['path'].map(lambda x: x.split('/')[-2])
dicom_df['StudyID'] = dicom_df['path'].map(lambda x: x.split('/')[-3])
dicom_df['PatientID'] = dicom_df['path'].map(lambda x: x.split('/')[-4].split(' ')[0])
dicom_df['PatSeries'] = dicom_df.apply(lambda x: '{PatientID}-{SeriesName}'.format(**x), 1)

read_overview_df = pd.read_csv(os.path.join(reads_dir, 'reads.csv'))
read_overview_df['PatientID'] = read_overview_df['name'].map(lambda x: x.replace('-', '')) 

new_reads = []
for _, c_row in read_overview_df.iterrows():
    base_dict = OrderedDict(PatientID = c_row['PatientID'], Category = c_row['Category'])
    for reader in ['R1', 'R2', 'R3']:
        c_dict = base_dict.copy()
        c_dict['Reader'] = reader
        for k,v in c_row.items():
            if (reader+':') in k:
                c_dict[k.split(':')[-1]] = v
        new_reads += [c_dict]
new_reads_df = pd.DataFrame(new_reads)
new_reads_df.to_csv('formatted_reads.csv')

maj_reads_df = (
    new_reads_df
    .drop(columns=['Reader'])
    .groupby(['PatientID', 'Category'])
    .agg(lambda x: x.value_counts().idxmax())
    .reset_index()
)

read_dicom_df = pd.merge(maj_reads_df, dicom_df, on = 'PatientID')
print(read_dicom_df.shape[0], 'total weakly-labeled slices')

meta_df = read_dicom_df[["PatientID", "ICH", "Fracture", "MassEffect", "MidlineShift", "path", "SliceNumber", "SeriesName"]]
meta_df[['ICH', 'Fracture', "MassEffect", "MidlineShift"]] = meta_df[['ICH', 'Fracture', "MassEffect", "MidlineShift"]].astype(bool)

def reduce_patients(meta_df, class_name, n_keep, random_state=42):

    pathology_cols = ['ICH', 'Fracture', 'MassEffect', 'MidlineShift']

    patient_pathology = meta_df.groupby('PatientID')[pathology_cols].any()

    if class_name == 'Normal':
        class_patients = patient_pathology[
            ~patient_pathology.any(axis=1)
        ].index

    elif class_name in pathology_cols:
        other_cols = [c for c in pathology_cols if c != class_name]

        class_patients = patient_pathology[
            (patient_pathology[class_name] == True) &
            (~patient_pathology[other_cols].any(axis=1))
        ].index

    else:
        raise ValueError(
            "class_name must be one of: Normal, ICH, Fracture, MassEffect, MidlineShift"
        )

    original_count = len(class_patients)

    keep_patients = (
        pd.Series(class_patients)
        .sample(min(n_keep, original_count), random_state=random_state)
    )

    new_meta_df = meta_df[
        meta_df['PatientID'].isin(keep_patients) |
        ~meta_df['PatientID'].isin(class_patients)
    ]

    print(f"{class_name}-only patients: {original_count} → {len(keep_patients)} kept")

    return new_meta_df


prev_counts = meta_df.value_counts(["ICH", "Fracture", "MassEffect", "MidlineShift"]).reset_index(name='count')

n_keep = 70
meta_df = reduce_patients(meta_df, "Normal", n_keep)
meta_df = reduce_patients(meta_df, "ICH", n_keep)

meta_df.to_csv("./meta_df.csv", index=False)
print("\nmeta_df saved to a csv file!\n")