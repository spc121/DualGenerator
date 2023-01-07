
import torch
from torch import nn, einsum
import torch.nn.functional as F
from models.pointnet2_utils import furthest_point_sample, \
    gather_operation, ball_query, three_nn, three_interpolate, grouping_operation


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, S = x.size()
        width = 16
        height = S//width
        x =x.view(-1, 128, width, height)
        proj_query = self.query_conv(x).view(m_batchsize, -1, S).permute(0, 2, 1)  # B X N*C
        proj_key = self.key_conv(x).view(m_batchsize, -1, S)  # B X C x N(*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check  B*N*N
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, S)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        out = out.view(m_batchsize, C, S)
        return out

class Self_RelAttn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super(Self_RelAttn, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        S = width * height
        proj_query = self.query_conv(x).view(m_batchsize, -1, S).permute(0, 2, 1)  # B X N*C
        proj_key = self.key_conv(x).view(m_batchsize, -1, S)  # B X C x N(*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check  B*N*N
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, S)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out

class Conv1d(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 kernel_size=1,
                 stride=1,
                 if_bn=True,
                 activation_fn=torch.relu):
        super(Conv1d, self).__init__()
        self.conv = nn.Conv1d(in_channel,
                              out_channel,
                              kernel_size,
                              stride=stride)
        self.if_bn = if_bn
        self.bn = nn.BatchNorm1d(out_channel)
        self.activation_fn = activation_fn

    def forward(self, input):
        out = self.conv(input)
        if self.if_bn:
            out = self.bn(out)

        if self.activation_fn is not None:
            out = self.activation_fn(out)

        return out


class Conv2d(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 kernel_size=(1, 1),
                 stride=(1, 1),
                 if_bn=True,
                 activation_fn=torch.relu):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channel,
                              out_channel,
                              kernel_size,
                              stride=stride)
        self.if_bn = if_bn
        self.bn = nn.BatchNorm2d(out_channel)
        self.activation_fn = activation_fn

    def forward(self, input):
        out = self.conv(input)
        if self.if_bn:
            out = self.bn(out)

        if self.activation_fn is not None:
            out = self.activation_fn(out)

        return out


class MLP(nn.Module):
    def __init__(self, in_channel, layer_dims, bn=None):
        super(MLP, self).__init__()
        layers = []
        last_channel = in_channel
        for out_channel in layer_dims[:-1]:
            layers.append(nn.Linear(last_channel, out_channel))
            if bn:
                layers.append(nn.BatchNorm1d(out_channel))
            layers.append(nn.ReLU())
            last_channel = out_channel
        layers.append(nn.Linear(last_channel, layer_dims[-1]))
        self.mlp = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.mlp(inputs)


class MLP_CONV(nn.Module):
    def __init__(self, in_channel, layer_dims, bn=None):
        super(MLP_CONV, self).__init__()
        layers = []
        last_channel = in_channel
        for out_channel in layer_dims[:-1]:
            layers.append(nn.Conv1d(last_channel, out_channel, 1))
            if bn:
                layers.append(nn.BatchNorm1d(out_channel))
            layers.append(nn.ReLU())
            last_channel = out_channel
        layers.append(nn.Conv1d(last_channel, layer_dims[-1], 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.mlp(inputs)


class MLP_Res(nn.Module):
    def __init__(self, in_dim=128, hidden_dim=None, out_dim=128):
        super(MLP_Res, self).__init__()
        if hidden_dim is None:
            hidden_dim = in_dim
        self.conv_1 = nn.Conv1d(in_dim, hidden_dim, 1)
        self.conv_2 = nn.Conv1d(hidden_dim, out_dim, 1)
        self.conv_shortcut = nn.Conv1d(in_dim, out_dim, 1)

    def forward(self, x):
        """
        Args:
            x: (B, out_dim, n)
        """
        shortcut = self.conv_shortcut(x)
        out = self.conv_2(torch.relu(self.conv_1(x))) + shortcut
        return out


def sample_and_group(xyz, points, npoint, nsample, radius, use_xyz=True):
    """
    Args:
        xyz: Tensor, (B, 3, N)
        points: Tensor, (B, f, N)
        npoint: int
        nsample: int
        radius: float
        use_xyz: boolean

    Returns:
        new_xyz: Tensor, (B, 3, npoint)
        new_points: Tensor, (B, 3 | f+3 | f, npoint, nsample)
        idx_local: Tensor, (B, npoint, nsample)
        grouped_xyz: Tensor, (B, 3, npoint, nsample)

    """
    xyz_flipped = xyz.permute(0, 2, 1).contiguous()  # (B, N, 3)
    new_xyz = gather_operation(xyz,
                               furthest_point_sample(xyz_flipped,
                                                     npoint))  # (B, 3, npoint)

    idx = ball_query(radius, nsample, xyz_flipped,
                     new_xyz.permute(0, 2,
                                     1).contiguous())  # (B, npoint, nsample)
    grouped_xyz = grouping_operation(xyz, idx)  # (B, 3, npoint, nsample)
    grouped_xyz -= new_xyz.unsqueeze(3).repeat(1, 1, 1, nsample)

    if points is not None:
        grouped_points = grouping_operation(points,
                                            idx)  # (B, f, npoint, nsample)
        if use_xyz:
            new_points = torch.cat([grouped_xyz, grouped_points], 1)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz

    return new_xyz, new_points, idx, grouped_xyz


def sample_and_group_all(xyz, points, use_xyz=True):
    """
    Args:
        xyz: Tensor, (B, 3, nsample)
        points: Tensor, (B, f, nsample)
        use_xyz: boolean

    Returns:
        new_xyz: Tensor, (B, 3, 1)
        new_points: Tensor, (B, f|f+3|3, 1, nsample)
        idx: Tensor, (B, 1, nsample)
        grouped_xyz: Tensor, (B, 3, 1, nsample)
    """
    b, _, nsample = xyz.shape
    device = xyz.device
    new_xyz = torch.zeros((1, 3, 1), dtype=torch.float,
                          device=device).repeat(b, 1, 1)
    grouped_xyz = xyz.reshape((b, 3, 1, nsample))
    idx = torch.arange(nsample,
                       device=device).reshape(1, 1, nsample).repeat(b, 1, 1)
    if points is not None:
        if use_xyz:
            new_points = torch.cat([xyz, points], 1)
        else:
            new_points = points
        new_points = new_points.unsqueeze(2)
    else:
        new_points = grouped_xyz

    return new_xyz, new_points, idx, grouped_xyz


class PointNet_SA_Module(nn.Module):
    def __init__(self,
                 npoint,
                 nsample,
                 radius,
                 in_channel,
                 mlp,
                 if_bn=True,
                 group_all=False,
                 use_xyz=True):
        """
        Args:
            npoint: int, number of points to sample
            nsample: int, number of points in each local region
            radius: float
            in_channel: int, input channel of features(points)
            mlp: list of int,
        """
        super(PointNet_SA_Module, self).__init__()
        self.npoint = npoint
        self.nsample = nsample
        self.radius = radius
        self.mlp = mlp
        self.group_all = group_all
        self.use_xyz = use_xyz
        if use_xyz:
            in_channel += 3

        last_channel = in_channel
        self.mlp_conv = []
        for out_channel in mlp:
            self.mlp_conv.append(Conv2d(last_channel, out_channel,
                                        if_bn=if_bn))
            last_channel = out_channel

        self.mlp_conv = nn.Sequential(*self.mlp_conv)

    def forward(self, xyz, points):
        """
        Args:
            xyz: Tensor, (B, 3, N)
            points: Tensor, (B, f, N)

        Returns:
            new_xyz: Tensor, (B, 3, npoint)
            new_points: Tensor, (B, mlp[-1], npoint)
        """
        if self.group_all:
            new_xyz, new_points, idx, grouped_xyz = sample_and_group_all(
                xyz, points, self.use_xyz)
        else:
            new_xyz, new_points, idx, grouped_xyz = sample_and_group(
                xyz, points, self.npoint, self.nsample, self.radius,
                self.use_xyz)

        new_points = self.mlp_conv(new_points)
        new_points = torch.max(new_points, 3)[0]

        return new_xyz, new_points


class PointNet_FP_Module(nn.Module):
    def __init__(self,
                 in_channel,
                 mlp,
                 use_points1=False,
                 in_channel_points1=None,
                 if_bn=True):
        """
        Args:
            in_channel: int, input channel of points2
            mlp: list of int
            use_points1: boolean, if use points
            in_channel_points1: int, input channel of points1
        """
        super(PointNet_FP_Module, self).__init__()
        self.use_points1 = use_points1

        if use_points1:
            in_channel += in_channel_points1

        last_channel = in_channel
        self.mlp_conv = []
        for out_channel in mlp:
            self.mlp_conv.append(Conv1d(last_channel, out_channel,
                                        if_bn=if_bn))
            last_channel = out_channel

        self.mlp_conv = nn.Sequential(*self.mlp_conv)

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Args:
            xyz1: Tensor, (B, 3, N)
            xyz2: Tensor, (B, 3, M)
            points1: Tensor, (B, in_channel, N)
            points2: Tensor, (B, in_channel, M)

        Returns:MLP_CONV
            new_points: Tensor, (B, mlp[-1], N)
        """
        dist, idx = three_nn(
            xyz1.permute(0, 2, 1).contiguous(),
            xyz2.permute(0, 2, 1).contiguous())
        dist = torch.clamp_min(dist, 1e-10)  # (B, N, 3)
        recip_dist = 1.0 / dist
        norm = torch.sum(recip_dist, 2, keepdim=True).repeat((1, 1, 3))
        weight = recip_dist / norm
        interpolated_points = three_interpolate(points2, idx,
                                                weight)  # B, in_channel, N

        if self.use_points1:
            new_points = torch.cat([interpolated_points, points1], 1)
        else:
            new_points = interpolated_points

        new_points = self.mlp_conv(new_points)
        return new_points


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))  # B, N, M
    dist += torch.sum(src**2, -1).view(B, N, 1)
    dist += torch.sum(dst**2, -1).view(B, 1, M)
    return dist


def query_knn(nsample, xyz, new_xyz, include_self=True):
    """Find k-NN of new_xyz in xyz"""
    pad = 0 if include_self else 1
    sqrdists = square_distance(new_xyz, xyz)  # B, S, N
    idx = torch.argsort(sqrdists, dim=-1, descending=False)[:, :,
                                                            pad:int(nsample) + pad]
    return idx.int()


def sample_and_group_knn(xyz, points, npoint, k, use_xyz=True, idx=None):
    """
    Args:
        xyz: Tensor, (B, 3, N)
        points: Tensor, (B, f, N)
        npoint: int
        nsample: int
        radius: float
        use_xyz: boolean

    Returns:
        new_xyz: Tensor, (B, 3, npoint)
        new_points: Tensor, (B, 3 | f+3 | f, npoint, nsample)
        idx_local: Tensor, (B, npoint, nsample)
        grouped_xyz: Tensor, (B, 3, npoint, nsample)

    """
    xyz_flipped = xyz.permute(0, 2, 1).contiguous()  # (B, N, 3)
    new_xyz = gather_operation(xyz,
                               furthest_point_sample(xyz_flipped,
                                                     npoint))  # (B, 3, npoint)
    if idx is None:
        idx = query_knn(k, xyz_flipped, new_xyz.permute(0, 2, 1).contiguous())
    grouped_xyz = grouping_operation(xyz, idx)  # (B, 3, npoint, nsample)
    grouped_xyz -= new_xyz.unsqueeze(3).repeat(1, 1, 1, k)

    if points is not None:
        grouped_points = grouping_operation(points,
                                            idx)  # (B, f, npoint, nsample)
        if use_xyz:
            new_points = torch.cat([grouped_xyz, grouped_points], 1)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz

    return new_xyz, new_points, idx, grouped_xyz


class Convlayer(nn.Module):
    def __init__(self,point_scales):
        super(Convlayer,self).__init__()
        self.point_scales = point_scales
        self.conv1 = torch.nn.Conv2d(1, 64, (1, 3))
        self.conv2 = torch.nn.Conv2d(64, 64, 1)
        self.conv3 = torch.nn.Conv2d(64, 128, 1)
        self.conv4 = torch.nn.Conv2d(128, 256, 1)
        self.conv5 = torch.nn.Conv2d(256, 512, 1)
        self.conv6 = torch.nn.Conv2d(512, 1024, 1)
        self.maxpool = torch.nn.MaxPool2d((self.point_scales, 1), 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(512)
        self.bn6 = nn.BatchNorm2d(1024)
    def forward(self,x):
        x = torch.unsqueeze(x,1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x_128 = F.relu(self.bn3(self.conv3(x)))
        x_256 = F.relu(self.bn4(self.conv4(x_128)))
        x_512 = F.relu(self.bn5(self.conv5(x_256)))
        x_1024 = F.relu(self.bn6(self.conv6(x_512)))
        x_128 = torch.squeeze(self.maxpool(x_128),2)
        x_256 = torch.squeeze(self.maxpool(x_256),2)
        x_512 = torch.squeeze(self.maxpool(x_512),2)
        x_1024 = torch.squeeze(self.maxpool(x_1024),2)
        L = [x_1024,x_512,x_256,x_128]
        x = torch.cat(L,1)
        return x


class Latentfeature(nn.Module):
    def __init__(self,num_scales,each_scales_size,point_scales_list):
        super(Latentfeature,self).__init__()
        self.num_scales = num_scales
        self.each_scales_size = each_scales_size
        self.point_scales_list = point_scales_list
        self.Convlayers1 = nn.ModuleList([Convlayer(point_scales = self.point_scales_list[0]) for i in range(self.each_scales_size)])
        self.Convlayers2 = nn.ModuleList([Convlayer(point_scales = self.point_scales_list[1]) for i in range(self.each_scales_size)])
        self.Convlayers3 = nn.ModuleList([Convlayer(point_scales = self.point_scales_list[2]) for i in range(self.each_scales_size)])
        self.conv1 = torch.nn.Conv1d(3,1,1)
        self.bn1 = nn.BatchNorm1d(1)

    def forward(self,x):
        outs = []
        for i in range(self.each_scales_size):
            outs.append(self.Convlayers1[i](x[0]))
        for j in range(self.each_scales_size):
            outs.append(self.Convlayers2[j](x[1]))
        for k in range(self.each_scales_size):
            outs.append(self.Convlayers3[k](x[2]))
        latentfeature = torch.cat(outs,2)
        latentfeature = latentfeature.transpose(1,2)
        latentfeature = F.relu(self.bn1(self.conv1(latentfeature)))
        latentfeature = torch.squeeze(latentfeature,1)
        return latentfeature


class netG(nn.Module):
    def __init__(self, crop_point_num):
        super(netG, self).__init__()
        self.crop_point_num = crop_point_num
        # self.latentfeature = Latentfeature(num_scales, each_scales_size, point_scales_list)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)

        self.fc1_1 = nn.Linear(1024, 128 * 512)
        self.fc2_1 = nn.Linear(512, 64 * 128)  # nn.Linear(512,64*256) !
        self.fc3_1 = nn.Linear(256, 64 * 3)

        self.conv1_1 = torch.nn.Conv1d(512, 512, 1)  # torch.nn.Conv1d(256,256,1) !
        self.conv1_2 = torch.nn.Conv1d(512, 256, 1)
        self.conv1_3 = torch.nn.Conv1d(256, int((self.crop_point_num * 3) / 128), 1)
        self.conv2_1 = torch.nn.Conv1d(128, 6, 1)  # torch.nn.Conv1d(256,12,1) !

    def forward(self, x):
        # x = self.latentfeature(x)
        x_1 = F.relu(self.fc1(x))  # 1024
        x_2 = F.relu(self.fc2(x_1))  # 512
        x_3 = F.relu(self.fc3(x_2))  # 256

        pc1_feat = self.fc3_1(x_3)
        pc1_xyz = pc1_feat.reshape(-1, 64, 3)  # 64x3 center1

        pc2_feat = F.relu(self.fc2_1(x_2))
        pc2_feat = pc2_feat.reshape(-1, 128, 64)
        pc2_xyz = self.conv2_1(pc2_feat)  # 6x64 center2

        pc3_feat = F.relu(self.fc1_1(x_1))
        pc3_feat = pc3_feat.reshape(-1, 512, 128)
        pc3_feat = F.relu(self.conv1_1(pc3_feat))
        pc3_feat = F.relu(self.conv1_2(pc3_feat))
        pc3_xyz = self.conv1_3(pc3_feat)  # 12x128 fine

        pc1_xyz_expand = torch.unsqueeze(pc1_xyz, 2)
        pc2_xyz = pc2_xyz.transpose(1, 2)
        pc2_xyz = pc2_xyz.reshape(-1, 64, 2, 3)
        pc2_xyz = pc1_xyz_expand + pc2_xyz
        pc2_xyz = pc2_xyz.reshape(-1, 128, 3)

        pc2_xyz_expand = torch.unsqueeze(pc2_xyz, 2)
        pc3_xyz = pc3_xyz.transpose(1, 2)
        pc3_xyz = pc3_xyz.reshape(-1, 128, int(self.crop_point_num / 128), 3)
        pc3_xyz = pc2_xyz_expand + pc3_xyz
        pc3_xyz = pc3_xyz.reshape(-1, self.crop_point_num, 3)

        return pc1_xyz, pc2_xyz, pc3_xyz  # center1 ,center2 ,fine


class PointNet_SA_Module_KNN(nn.Module):
    def __init__(self,
                 npoint,
                 nsample,
                 in_channel,
                 mlp,
                 if_bn=True,
                 group_all=False,
                 use_xyz=True,
                 if_idx=False):
        """
        Args:
            npoint: int, number of points to sample
            nsample: int, number of points in each local region
            radius: float
            in_channel: int, input channel of features(points)
            mlp: list of int,
        """
        super(PointNet_SA_Module_KNN, self).__init__()
        self.npoint = npoint
        self.nsample = nsample
        self.mlp = mlp
        self.group_all = group_all
        self.use_xyz = use_xyz
        self.if_idx = if_idx
        if use_xyz:
            in_channel += 3

        last_channel = in_channel
        self.mlp_conv = []
        for out_channel in mlp[:-1]:
            self.mlp_conv.append(Conv2d(last_channel, out_channel,
                                        if_bn=if_bn))
            last_channel = out_channel
        self.mlp_conv.append(
            Conv2d(last_channel, mlp[-1], if_bn=False, activation_fn=None))
        self.mlp_conv = nn.Sequential(*self.mlp_conv)

    def forward(self, xyz, points, idx=None):
        """
        Args:
            xyz: Tensor, (B, 3, N)
            points: Tensor, (B, f, N)

        Returns:
            new_xyz: Tensor, (B, 3, npoint)
            new_points: Tensor, (B, mlp[-1], npoint)
        """
        if self.group_all:
            new_xyz, new_points, idx, grouped_xyz = sample_and_group_all(
                xyz, points, self.use_xyz)
        else:
            new_xyz, new_points, idx, grouped_xyz = sample_and_group_knn(
                xyz, points, self.npoint, self.nsample, self.use_xyz, idx=idx)

        new_points = self.mlp_conv(new_points)
        new_points = torch.max(new_points, 3)[0]

        if self.if_idx:
            return new_xyz, new_points, idx
        else:
            return new_xyz, new_points


def fps_subsample(pcd, n_points=2048):
    """
    Args
        pcd: (b, 16384, 3)

    returns
        new_pcd: (b, n_points, 3)
    """
    if pcd.shape[1] == n_points:
        return pcd
    elif pcd.shape[1] < n_points:
        raise ValueError(
            'FPS subsampling receives a larger n_points: {:d} > {:d}'.format(
                n_points, pcd.shape[1]))
    new_pcd = gather_operation(
        pcd.permute(0, 2, 1).contiguous(),
        furthest_point_sample(pcd, n_points))
    new_pcd = new_pcd.permute(0, 2, 1).contiguous()
    return new_pcd


def get_nearest_index(target, source, k=1, return_dis=False):
    """
    Args:
        target: (bs, 3, v1)
        source: (bs, 3, v2)
    Return:
        nearest_index: (bs, v1, 1)
    """
    inner = torch.bmm(target.transpose(1, 2), source)  # (bs, v1, v2)
    s_norm_2 = torch.sum(source**2, dim=1)  # (bs, v2)
    t_norm_2 = torch.sum(target**2, dim=1)  # (bs, v1)
    d_norm_2 = s_norm_2.unsqueeze(1) + t_norm_2.unsqueeze(
        2) - 2 * inner  # (bs, v1, v2)
    nearest_dis, nearest_index = torch.topk(d_norm_2,
                                            k=k,
                                            dim=-1,
                                            largest=False)
    if not return_dis:
        return nearest_index
    else:
        return nearest_index, nearest_dis


def indexing_neighbor(x, index):
    """
    Args:
        x: (bs, dim, num_points0)
        index: (bs, num_points, k)
    Return:
        feature: (bs, dim, num_points, k)
    """
    batch_size, num_points, k = index.size()

    id_0 = torch.arange(batch_size).view(-1, 1, 1)

    x = x.transpose(2, 1).contiguous()  # (bs, num_points, num_dims)
    feature = x[id_0, index]  # (bs, num_points, k, num_dims)
    feature = feature.permute(0, 3, 1,
                              2).contiguous()  # (bs, num_dims, num_points, k)

    return feature


class vTransformer(nn.Module):
    def __init__(self,
                 in_channel,
                 dim=256,
                 n_knn=16,
                 pos_hidden_dim=64,
                 attn_hidden_multiplier=4):
        super(vTransformer, self).__init__()
        self.n_knn = n_knn
        self.conv_key = nn.Conv1d(dim, dim, 1)
        self.conv_query = nn.Conv1d(dim, dim, 1)
        self.conv_value = nn.Conv1d(dim, dim, 1)

        self.pos_mlp = nn.Sequential(nn.Conv2d(3, pos_hidden_dim, 1),
                                     nn.BatchNorm2d(pos_hidden_dim), nn.ReLU(),
                                     nn.Conv2d(pos_hidden_dim, dim, 1))

        self.attn_mlp = nn.Sequential(
            nn.Conv2d(dim, dim * attn_hidden_multiplier, 1),
            nn.BatchNorm2d(dim * attn_hidden_multiplier), nn.ReLU(),
            nn.Conv2d(dim * attn_hidden_multiplier, dim, 1))

        # self.RelAttn = Self_RelAttn(in_dim=64)
        self.linear_start = nn.Conv1d(in_channel, dim, 1)
        self.linear_end = nn.Conv1d(dim, in_channel, 1)

    def forward(self, x, pos):
        """feed forward of transformer
        Args:
            x: Tensor of features, (B, in_channel, n)
            pos: Tensor of positions, (B, 3, n)

        Returns:
            y: Tensor of features with attention, (B, in_channel, n)
        """

        identity = x

        x = self.linear_start(x)
        b, dim, n = x.shape

        pos_flipped = pos.permute(0, 2, 1).contiguous()
        idx_knn = query_knn(self.n_knn, pos_flipped, pos_flipped)
        key = self.conv_key(x)  # (B, dim, N)
        value = self.conv_value(x)
        query = self.conv_query(x)

        key = grouping_operation(key, idx_knn)  # (B, dim, N, k)
        qk_rel = query.reshape((b, -1, n, 1)) - key

        pos_rel = pos.reshape(
            (b, -1, n, 1)) - grouping_operation(pos, idx_knn)  # (B, 3, N, k)
        pos_embedding = self.pos_mlp(pos_rel)  # (B, dim, N, k)

        attention = self.attn_mlp(qk_rel + pos_embedding)
        # attention = self.RelAttn(qk_rel + pos_embedding)
        attention = torch.softmax(attention, -1)

        # knn value is correct
        value = grouping_operation(value,
                                   idx_knn) + pos_embedding  # (B, dim, N, k)

        agg = einsum('b c i j, b c i j -> b c i', attention,
                     value)  # (B, dim, N)
        y = self.linear_end(agg)  # (B, in_dim, N)

        return y + identity
