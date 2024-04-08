from utils import Model, feature_extracter
import torch

n_features = 40
model = Model(n_input_feature=n_features)
model.load_state_dict(torch.load("Audio_Classifier.pth"))
model.eval()

file_path = "cats_dogs/test/dogs/dog_barking_64.wav"
# file_path = "cats_dogs/test/cats/cat_129.wav"
inputs = feature_extracter(file_path=file_path)
with torch.no_grad():
    outputs = model(inputs)
    predicted = torch.round(outputs.squeeze())
    if predicted == torch.tensor(1.0, dtype=torch.float32):
        print("Dog")
    else:
        print("Cat")
