import torch
from torch import nn


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1)
        self.bn1 = nn.InstanceNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.InstanceNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1)
        self.bn3 = nn.InstanceNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.InstanceNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(inplanes, planes, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.InstanceNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class StageModule(nn.Module):
    def __init__(self, stage, output_branches, c):
        super(StageModule, self).__init__()
        self.stage = stage
        self.output_branches = output_branches

        self.branches = nn.ModuleList()
        for i in range(self.stage):
            w = c * (2 ** i)
            branch = nn.Sequential(
                BasicBlock(w, w),
                BasicBlock(w, w),
                BasicBlock(w, w),
                BasicBlock(w, w),
            )
            self.branches.append(branch)

        self.fuse_layers = nn.ModuleList()
        # for each output_branches (i.e. each branch in all cases but the very last one)
        for i in range(self.output_branches):
            self.fuse_layers.append(nn.ModuleList())
            for j in range(self.stage):  # for each branch
                if i == j:
                    self.fuse_layers[-1].append(nn.Sequential())  # Used in place of "None" because it is callable
                elif i < j:
                    self.fuse_layers[-1].append(nn.Sequential(
                        nn.Conv3d(c * (2 ** j), c * (2 ** i), kernel_size=1, stride=1),
                        nn.InstanceNorm3d(c * (2 ** i), eps=1e-05),
                        nn.Upsample(scale_factor=(2.0 ** (j - i)), mode='trilinear', align_corners=True),
                    ))
                elif i > j:
                    ops = []
                    for k in range(i - j - 1):
                        ops.append(nn.Sequential(
                            nn.Conv3d(c * (2 ** j), c * (2 ** j), kernel_size=3, stride=2, padding=1),
                            nn.InstanceNorm3d(c * (2 ** j)),
                            nn.ReLU(inplace=True),
                        ))
                    ops.append(nn.Sequential(
                        nn.Conv3d(c * (2 ** j), c * (2 ** i), kernel_size=3, stride=2, padding=1),
                        nn.InstanceNorm3d(c * (2 ** i)),
                    ))
                    self.fuse_layers[-1].append(nn.Sequential(*ops))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        assert len(self.branches) == len(x)

        x = [branch(b) for branch, b in zip(self.branches, x)]

        x_fused = []
        for i in range(len(self.fuse_layers)):
            for j in range(0, len(self.branches)):
                if j == 0:
                    x_fused.append(self.fuse_layers[i][0](x[0]))
                else:
                    x_fused[i] = x_fused[i] + self.fuse_layers[i][j](x[j])

        for i in range(len(x_fused)):
            x_fused[i] = self.relu(x_fused[i])

        return x_fused


class FeatureRecalibration(nn.Module):
    def __init__(self, channels):
        super(FeatureRecalibration, self).__init__()
        self.global_avg_pooling = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.conv1 = nn.Conv3d(channels, int(channels/2), 1, 1, 0)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(int(channels / 2), channels, 1, 1, 0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.global_avg_pooling(x)
        x1 = self.conv1(x1)
        x1 = self.relu1(x1)
        x1 = self.conv2(x1)
        x1 = self.sigmoid(x1)
        return x * x1


class HRNet3D(nn.Module):
    def __init__(self, c=48, nof_joints=1):
        super(HRNet3D, self).__init__()
        # high resolution complementary
        self.hconv1 = nn.Sequential(
            nn.Conv3d(1, 3, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(3),
            nn.ReLU(inplace=True),
        )
        self.hconv2 = nn.Sequential(
            nn.Conv3d(3, 6, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(6),
            nn.ReLU(inplace=True),
        )

        self.hconv3 = nn.Sequential(
            nn.Conv3d(6, 12, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(12),
            nn.ReLU(inplace=True),
        )
        self.hconv4 = nn.Sequential(
            nn.Conv3d(12, 24, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(24),
            nn.ReLU(inplace=True),
        )

        self.hconv5 = nn.Sequential(
            nn.Conv3d(24, 48, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(48),
            nn.ReLU(inplace=True),
        )

        # Input (stem net)
        self.conv1 = nn.Conv3d(1, 64, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.InstanceNorm3d(64)
        self.conv2 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.InstanceNorm3d(64)
        self.relu = nn.ReLU(inplace=True)

        # Stage 1 (layer1)      - First group of bottleneck (resnet) modules
        downsample = nn.Sequential(
            nn.Conv3d(64, 256, kernel_size=1, stride=1),
            nn.InstanceNorm3d(256),
        )
        self.layer1 = nn.Sequential(
            Bottleneck(64, 64, downsample=downsample),
            Bottleneck(256, 64),
            Bottleneck(256, 64),
            Bottleneck(256, 64),
        )

        # Fusion layer 1 (transition1)      - Creation of the first two branches (one full and one half resolution)
        self.transition1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(256, c, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(c),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(nn.Sequential(  # Double Sequential to fit with official pretrained weights
                nn.Conv3d(256, c * (2 ** 1), kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm3d(c * (2 ** 1)),
                nn.ReLU(inplace=True),
            )),
        ])

        # Stage 2 (stage2)      - Second module with 1 group of bottleneck (resnet) modules. This has 2 branches
        self.stage2 = nn.Sequential(
            StageModule(stage=2, output_branches=2, c=c),
        )

        # Fusion layer 2 (transition2)      - Creation of the third branch (1/4 resolution)
        self.transition2 = nn.ModuleList([
            nn.Sequential(),  # None,   - Used in place of "None" because it is callable
            nn.Sequential(),  # None,   - Used in place of "None" because it is callable
            nn.Sequential(nn.Sequential(  # Double Sequential to fit with official pretrained weights
                nn.Conv3d(c * (2 ** 1), c * (2 ** 2), kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm3d(c * (2 ** 2)),
                nn.ReLU(inplace=True),
            )),  # ToDo Why the new branch derives from the "upper" branch only?
        ])

        # Stage 3 (stage3)      - Third module with 4 groups of bottleneck (resnet) modules. This has 3 branches
        self.stage3 = nn.Sequential(
            StageModule(stage=3, output_branches=3, c=c),
            StageModule(stage=3, output_branches=3, c=c),
            StageModule(stage=3, output_branches=3, c=c),
            StageModule(stage=3, output_branches=3, c=c),
        )

        # Fusion layer 3 (transition3)      - Creation of the fourth branch (1/8 resolution)
        self.transition3 = nn.ModuleList([
            nn.Sequential(),  # None,   - Used in place of "None" because it is callable
            nn.Sequential(),  # None,   - Used in place of "None" because it is callable
            nn.Sequential(),  # None,   - Used in place of "None" because it is callable
            nn.Sequential(nn.Sequential(  # Double Sequential to fit with official pretrained weights
                nn.Conv3d(c * (2 ** 2), c * (2 ** 3), kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm3d(c * (2 ** 3)),
                nn.ReLU(inplace=True),
            )),  # ToDo Why the new branch derives from the "upper" branch only?
        ])

        # Stage 4 (stage4)      - Fourth module with 3 groups of bottleneck (resnet) modules. This has 4 branches
        self.stage4 = nn.Sequential(
            StageModule(stage=4, output_branches=4, c=c),
            StageModule(stage=4, output_branches=4, c=c),
            StageModule(stage=4, output_branches=4, c=c),
        )

        # Final layer (final_layer)
        self.upsample  = nn.ConvTranspose3d(c * (2 ** 0), c * (2 ** 0), 2, 2)
        self.upsample3 = nn.ConvTranspose3d(c * (2 ** 3), c * (2 ** 2), 2, 2)
        self.upsample2 = nn.ConvTranspose3d(c * (2 ** 2), c * (2 ** 1), 2, 2)
        self.upsample1 = nn.ConvTranspose3d(c * (2 ** 1), c * (2 ** 0), 2, 2)

        self.f_conv3 = nn.Sequential(
            nn.Conv3d(c * (2 ** 2), c * (2 ** 2), 3, 1, 1),
            nn.InstanceNorm3d(3),
            nn.ReLU(inplace=True),
        )
        self.f_conv2 = nn.Sequential(
            nn.Conv3d(c * (2 ** 1), c * (2 ** 1), 3, 1, 1),
            nn.InstanceNorm3d(3),
            nn.ReLU(inplace=True),
        )
        self.f_conv1 = nn.Sequential(
            nn.Conv3d(c * (2 ** 1), c * (2 ** 0), 3, 1, 1),
            nn.InstanceNorm3d(3),
            nn.ReLU(inplace=True),
        )

        self.feature_r3 = FeatureRecalibration(c * (2 ** 3))
        self.feature_r2 = FeatureRecalibration(c * (2 ** 2))
        self.feature_r1 = FeatureRecalibration(c * (2 ** 1))
        self.final_layer = nn.Conv3d(c*2, nof_joints, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_full_res = self.hconv5(self.hconv4(self.hconv3(self.hconv2(self.hconv1(x)))))
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = [trans(x) for trans in self.transition1]  # Since now, x is a list (# == nof branches)

        x = self.stage2(x)
        # x = [trans(x[-1]) for trans in self.transition2]    # New branch derives from the "upper" branch only
        x = [
            self.transition2[0](x[0]),
            self.transition2[1](x[1]),
            self.transition2[2](x[-1])
        ]  # New branch derives from the "upper" branch only

        x = self.stage3(x)
        # x = [trans(x) for trans in self.transition3]    # New branch derives from the "upper" branch only
        x = [
            self.transition3[0](x[0]),
            self.transition3[1](x[1]),
            self.transition3[2](x[2]),
            self.transition3[3](x[-1])
        ]  # New branch derives from the "upper" branch only

        x = self.stage4(x)

        temp = self.f_conv3(self.upsample3(self.feature_r3(x[3])) + self.feature_r2(x[2]))
        temp = self.f_conv2(self.upsample2(temp) + self.feature_r1(x[1]))

        x = self.f_conv1(torch.cat((self.upsample1(temp), x[0]), dim=1))
        # x = self.sigmoid(self.final_layer(self.upsample(x) + x_full_res))
        x = self.sigmoid(self.final_layer(torch.cat((self.upsample(x), x_full_res), dim=1)))
        return x


net = HRNet3D()


if __name__ == '__main__':
    model = HRNet3D(48, 1)
    # model = HRNet3D(32, 17)

    # print(model)

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    # print(device)

    model = model.to(device)

    y = model(torch.rand((1, 1, 96, 160, 96)).to(device))
    print(y.shape)
    print(torch.min(y).item(), torch.mean(y).item(), torch.max(y).item())
