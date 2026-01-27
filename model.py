import torch
import torch.nn as nn
import collections

class PlantClassifier(nn.Module):
    def __init__(self, num_classes=39, in_features=3, base_filters=16, # conv stride stays 1, kernel size stays 3
                 n_conv_layers=6, num_fc_units=256):
        super(PlantClassifier, self).__init__()
        # want to modularize the conv layers
        def make_conv_block(in_channels, out_channels, layer_idx=""):
            return nn.Sequential(
                collections.OrderedDict([
                    ('conv'+layer_idx, nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding="valid")),
                    ('bn'+layer_idx, nn.BatchNorm2d(out_channels)),
                    ('relu'+layer_idx, nn.ReLU(inplace=True)),
                    ('pool'+layer_idx, nn.MaxPool2d(kernel_size=2, stride=2))
                ])
            )
        layers = []
        current_in_channels = in_features
        for i in range(n_conv_layers):
            layers.append(make_conv_block(current_in_channels, base_filters * (2 ** i), layer_idx=str(i)))
            current_in_channels = base_filters * (2 ** i)
        self.features = nn.Sequential(*layers)

        with torch.no_grad():
            dummy_input = torch.randn(1, in_features, 256, 256)
            dummy_output = self.features(dummy_input)
            flattened_size = dummy_output.view(1, -1).shape[1]
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, num_fc_units),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(num_fc_units, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
class QuantizedPlantClassifier(nn.Module): # this is the same PlantClassifier but with quantization stubs
    def __init__(self, num_classes=39, in_features=3, base_filters=16, # conv stride stays 1, kernel size stays 3
                 n_conv_layers=6, num_fc_units=256):
        super(QuantizedPlantClassifier, self).__init__()
        # want to modularize the conv layers
        def make_conv_block(in_channels, out_channels, layer_idx=""):
            return nn.Sequential(
                collections.OrderedDict([
                    ('conv'+layer_idx, nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0)),
                    ('bn'+layer_idx, nn.BatchNorm2d(out_channels)),
                    ('relu'+layer_idx, nn.ReLU(inplace=True)),
                    ('pool'+layer_idx, nn.MaxPool2d(kernel_size=2, stride=2))
                ])
            )
        self.quant = torch.quantization.QuantStub()
        layers = []
        current_in_channels = in_features
        for i in range(n_conv_layers):
            layers.append(make_conv_block(current_in_channels, base_filters * (2 ** i), layer_idx=str(i)))
            current_in_channels = base_filters * (2 ** i)
        self.features = nn.Sequential(*layers)

        with torch.no_grad():
            dummy_input = torch.randn(1, in_features, 256, 256)
            dummy_output = self.features(dummy_input)
            flattened_size = dummy_output.view(1, -1).shape[1]
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, num_fc_units),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(num_fc_units, num_classes)
        )
        self.dequant = torch.quantization.DeQuantStub()
        
    def forward(self, x):
        x = self.quant(x)
        x = self.features(x)
        x = self.classifier(x)
        x = self.dequant(x)
        return x

