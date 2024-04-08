import torch
import torch.nn as nn
import numpy as np
import librosa


# Definig an ANN Model
class Model(nn.Module):
    def __init__(self, n_input_feature):
        super(Model, self).__init__()
        self.linear = nn.Linear(n_input_feature, 1)

    def forward(self, x):
        out = torch.sigmoid(self.linear(x))
        return out


# Now we will extract the MFCC(Mel-frequency cepstral coefficients) of the audio filename
def feature_extracter(file_path):
    sample, sample_rate = librosa.load(file_path)
    mfcc_feature = librosa.feature.mfcc(y=sample, sr=sample_rate, n_mfcc=40)
    mfcc_mean = np.mean(mfcc_feature, axis=1)
    return torch.tensor(mfcc_mean, dtype=torch.float32)
