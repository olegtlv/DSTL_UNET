import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionGate(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(out_channels, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.psi(F.relu(g1 + x1, inplace=True))
        return x * psi, psi


class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True))
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=1):
        super().__init__()
        if upsample:
            self.upconv = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        else:
            self.upconv = nn.Identity()
        self.attention = AttentionGate(in_channels, in_channels)
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    #     def forward(self, x, skip_connection):
    #         upsampled = self.upconv(x)
    #         concatenated = torch.cat([skip_connection, upsampled], dim=1)
    #         output = self.layers(concatenated)
    #         return output

    def forward(self, x, skip_connection):
        upsampled = self.upconv(x)
        gated_skip, psi = self.attention(upsampled, skip_connection)
        concatenated = torch.cat([gated_skip, upsampled], dim=1)
        output = self.layers(concatenated)
        return output, psi


class DoubleConv(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs):
        return self.layers(inputs)


class FinalLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        return self.layers(inputs)


class UNet_orig(nn.Module):
    def __init__(self, num_classes, input_features=3,
                 layer1_features=16, layer2_features=18, layer3_features=20,
                 layer4_features=22, layer5_features=24):
        super(UNet_orig, self).__init__()

        self.num_classes = num_classes

        # Layer feature sizes
        self.input_features = input_features
        self.layer1_features = layer1_features
        self.layer2_features = layer2_features
        self.layer3_features = layer3_features
        self.layer4_features = layer4_features
        self.layer5_features = layer5_features

        # Encoder layers
        self.encoder1 = EncoderBlock(self.input_features, self.layer1_features)
        self.encoder2 = EncoderBlock(self.layer1_features, self.layer2_features)
        self.encoder3 = EncoderBlock(self.layer2_features, self.layer3_features)
        self.encoder4 = EncoderBlock(self.layer3_features, self.layer4_features)
        self.encoder5 = EncoderBlock(self.layer4_features, self.layer5_features)

        # Bottleneck Layer
        self.bottleneck = DoubleConv(self.layer5_features, self.layer5_features, self.layer5_features, )

        # Decoder layers
        self.decoder1 = DecoderBlock(self.layer5_features, self.layer4_features)
        self.decoder2 = DecoderBlock(self.layer4_features, self.layer3_features)
        self.decoder3 = DecoderBlock(self.layer3_features, self.layer2_features)
        self.decoder4 = DecoderBlock(self.layer2_features, self.layer1_features, upsample=0)
        self.decoder5 = DecoderBlock(self.layer1_features, self.layer1_features)

        # Final convolution
        self.final_conv = FinalLayer(self.layer1_features, self.num_classes)

    def forward(self, x):
        # Encoder (contracting path)
        output1, p1 = self.encoder1(x)
        output2, _ = self.encoder2(p1)
        output3, p3 = self.encoder3(output2)
        output4, p4 = self.encoder4(p3)
        output5, p5 = self.encoder5(p4)

        # Bottleneck Layer
        bn = self.bottleneck(p5)

        up1, psi1 = self.decoder1(bn, output5)
        up2, psi2 = self.decoder2(up1, output4)
        up3, psi3 = self.decoder3(up2, output3)
        up4, psi4 = self.decoder4(up3, output2)
        up5, psi5 = self.decoder5(up4, output1)

        #         up1 = self.decoder1(bn,   output5)
        #         up2 = self.decoder2(up1 , output4)
        #         up3 = self.decoder3(up2 , output3)
        #         up4 = self.decoder4(up3 , output2)
        #         up5 = self.decoder5(up4 , output1)
        # Final convolution to produce segmentation mask
        res = self.final_conv(up5)

        return res, psi1, psi5


