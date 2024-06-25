import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
import torch.nn as nn
from torchvision import transforms, models

class CombinedModel(nn.Module):
    def __init__(self, num_classes):
        super(CombinedModel, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        num_ftrs_resnet = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        
        self.inception_resnet = models.inception_v3(pretrained=True, aux_logits=True)
        num_ftrs_inception_resnet = self.inception_resnet.fc.in_features
        self.inception_resnet.fc = nn.Identity()
        
        self.fc1 = nn.Linear(num_ftrs_resnet + num_ftrs_inception_resnet, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        resnet_features = self.resnet(x)
        inception_resnet_features = self.inception_resnet(x)
        
        if isinstance(inception_resnet_features, tuple):
            inception_resnet_features = inception_resnet_features[0]
        
        combined_features = torch.cat((resnet_features, inception_resnet_features), dim=1)
        x = torch.relu(self.fc1(combined_features))
        x = self.fc2(x)
        
        return x

# Load the model
num_classes = 7  # Replace with the actual number of classes
model = CombinedModel(num_classes)

# Map the model to the correct device
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model.load_state_dict(torch.load('C:/Users/Asus/Desktop/Thesis/Project Development/Tester/Model/combined_model.pth', map_location=torch.device(device)))
model = model.to(device)
model.eval()

# Define transformations
data_transforms = transforms.Compose([
    transforms.Resize(320),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Define class names
class_names = ['brightpixel', 'narrowband', 'narrowbanddrd', 'noise', 'squarepulsednarrowband', 'squiggle', 'squigglesquarepulsednarrowband']  # Replace with actual class names

def predict(image, model, transform, class_names, threshold=0.5):
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
        confidence = torch.nn.functional.softmax(outputs, dim=1)[0][preds[0]].item()

    if confidence < threshold:
        pred_class = "Not identified"
    else:
        pred_class = class_names[preds[0]]

    return pred_class

# Function to handle image selection and prediction
def select_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        image = Image.open(file_path).convert('RGB')
        image = image.resize((200, 200))  # Resize image for display (optional)
        photo = ImageTk.PhotoImage(image)
        image_label.config(image=photo)
        image_label.image = photo
        prediction = predict(image, model, data_transforms, class_names)
        prediction_label.config(text=f'Predicted Class: {prediction}')

# Create main application window
root = tk.Tk()
root.title("Image Classifier")
root.geometry("400x400")  # Set initial window size

# Create UI components
upload_button = tk.Button(root, text="Upload Image", command=select_image)
upload_button.pack(pady=20)

image_label = tk.Label(root)
image_label.pack(pady=20)

prediction_label = tk.Label(root, text="Predicted class: ")
prediction_label.pack(pady=20)

root.mainloop()
