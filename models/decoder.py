import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import tinycudann as tcnn
    TCNN_EXISTS = True
except ImportError:
    TCNN_EXISTS = False

class PEMAP(nn.Module):
    def __init__(self, feature_embedding_dim):
        super(PEMAP, self).__init__()

        self.learnable_pe_map = nn.Parameter(
                        0.05 * torch.randn(1, feature_embedding_dim // 2, 80, 120),
                        requires_grad=True,
                    )
        # a PE head to decode PE features
        self.pe_head = nn.Sequential(
                        torch.nn.Linear(feature_embedding_dim // 2, feature_embedding_dim),
                    )
    
    def forward(self, pixel):
        learnable_pe_map = (
                            F.grid_sample(
                                self.learnable_pe_map,
                                # assume pixel coords have been normalize to [-1, 1]
                                # observation["vu"].reshape(1, 1, -1, 2) * 2 - 1,
                                pixel,
                                align_corners=False,  # didn't test with True
                                mode="bilinear",  # didn't test with other modes
                            )
                            .squeeze(2)
                            .squeeze(0)
                            .permute(1, 0)
                        )
        dino_pe = self.pe_head(learnable_pe_map)
        return dino_pe

class Embedder(nn.Module):
    def __init__(self, input_dim, max_freq_log2, N_freqs,
                 log_sampling=True, include_input=True,
                 periodic_fns=(torch.sin, torch.cos)):
        '''
        :param input_dim: dimension of input to be embedded
        :param max_freq_log2: log2 of max freq; min freq is 1 by default
        :param N_freqs: number of frequency bands
        :param log_sampling: if True, frequency bands are linerly sampled in log-space
        :param include_input: if True, raw input is included in the embedding
        :param periodic_fns: periodic functions used to embed input
        '''
        super(Embedder, self).__init__()

        self.input_dim = input_dim
        self.include_input = include_input
        self.periodic_fns = periodic_fns

        self.out_dim = 0
        if self.include_input:
            self.out_dim += self.input_dim

        self.out_dim += self.input_dim * N_freqs * len(self.periodic_fns)

        if log_sampling:
            self.freq_bands = 2. ** torch.linspace(0., max_freq_log2, N_freqs)
        else:
            self.freq_bands = torch.linspace(
                2. ** 0., 2. ** max_freq_log2, N_freqs)

        self.freq_bands = self.freq_bands.numpy().tolist()

    def forward(self, input: torch.Tensor):
        '''
        :param input: tensor of shape [..., self.input_dim]
        :return: tensor of shape [..., self.out_dim]
        '''
        assert (input.shape[-1] == self.input_dim)

        out = []
        if self.include_input:
            out.append(input)

        for i in range(len(self.freq_bands)):
            freq = self.freq_bands[i]
            for p_fn in self.periodic_fns:
                out.append(p_fn(input * freq))
        out = torch.cat(out, dim=-1)

        assert (out.shape[-1] == self.out_dim)
        return out


def get_embedder(multires, input_dim=3):
    if multires < 0:
        return nn.Identity(), input_dim

    embed_kwargs = {
        "include_input": True,
        "input_dim": input_dim,
        "max_freq_log2": multires - 1,
        "N_freqs": multires,
        "log_sampling": True,
        "periodic_fns": [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    return embedder_obj, embedder_obj.out_dim


class DenseLayer(nn.Module):
    def __init__(self, in_dim, out_dim, activation=None):
        super(DenseLayer, self).__init__()

        self.linear_layer = nn.Linear(in_dim, out_dim)

        if activation is None:
            # self.activation = nn.ReLU()
            self.activation = nn.Softplus(beta=100.0)
        else:
            self.activation = activation

    def forward(self, x):
        out = self.linear_layer(x)
        out = self.activation(out)
        return out

class SemanticDecoder(nn.Module):
    def __init__(self, W=128, D=4, skips=[], input_feat_dim=16, n_freq=-1, sem_num=44, weight_norm=False, concat_qp=False):
        super(SemanticDecoder, self).__init__()

        self.embed_fn, input_ch = get_embedder(n_freq, input_dim=input_feat_dim + concat_qp * 3)
        self.W = W
        self.D = D
        self.skips = skips
        layers = []

        for l in range(D+1):
            if l == D:
                out_dim = sem_num
            elif l + 1 in self.skips:
                out_dim = W - input_ch
            else:
                out_dim = W

            if l == 0:
                in_dim = input_ch
            else:
                in_dim = W

            if l != D:
                # layer = DenseLayer(in_dim, out_dim, activation=nn.Sigmoid())
                layer = DenseLayer(in_dim, out_dim)
            else:
                layer = nn.Linear(in_dim, out_dim)

            if weight_norm:
                layer = nn.utils.weight_norm(layer)

            layers.append(layer)

        self.layers = nn.ModuleList(layers)

    def forward(self, feat, return_h=False):
        feat = self.embed_fn(feat)
        h = feat
        
        for i in range(self.D+1):
            if i in self.skips:
                h = torch.cat([h, feat], dim=-1)
            h = self.layers[i](h)

        if return_h:  # return feature
            return h[..., :1], h[..., 1:]
        else:
            return h

class PositionDecoder(nn.Module):
    def __init__(self, W=128, D=4, skips=[], input_feat_dim=16, n_freq=-1, output_ch=64, weight_norm=False, concat_qp=False):
        super(PositionDecoder, self).__init__()

        self.embed_fn, input_ch = get_embedder(n_freq, input_dim=input_feat_dim + concat_qp * 3)
        self.W = W
        self.D = D
        self.skips = skips
        layers = []

        for l in range(D+1):
            if l == D:
                out_dim = output_ch
            elif l + 1 in self.skips:
                out_dim = W - input_ch
            else:
                out_dim = W

            if l == 0:
                in_dim = input_ch
            else:
                in_dim = W

            if l != D:
                # layer = DenseLayer(in_dim, out_dim, activation=nn.Sigmoid())
                layer = DenseLayer(in_dim, out_dim)
            else:
                layer = nn.Linear(in_dim, out_dim)

            if weight_norm:
                layer = nn.utils.weight_norm(layer)

            layers.append(layer)

        self.layers = nn.ModuleList(layers)

    def forward(self, feat, return_last=False):
        feat = self.embed_fn(feat)
        h = feat
        
        for i in range(self.D+1):
            if i in self.skips:
                h = torch.cat([h, feat], dim=-1)
            h_last = h
            h = self.layers[i](h)
        # return h
        if return_last:  # return feature
            return h,h_last
        else:
            return h

class instanceDecoder(nn.Module):
    def __init__(self, W=128, D=4, skips=[], input_feat_dim=16, n_freq=-1, output_ch=64, weight_norm=False, concat_qp=False):
        super(instanceDecoder, self).__init__()

        self.embed_fn, input_ch = get_embedder(n_freq, input_dim=input_feat_dim + concat_qp * 3)
        self.W = W
        self.D = D
        self.skips = skips
        layers = []

        for l in range(D+1):
            if l == D:
                out_dim = output_ch
            elif l + 1 in self.skips:
                out_dim = W - input_ch
            else:
                out_dim = W

            if l == 0:
                in_dim = input_ch
            else:
                in_dim = W

            if l != D:
                # layer = DenseLayer(in_dim, out_dim, activation=nn.Sigmoid())
                layer = DenseLayer(in_dim, out_dim, activation=nn.ReLU())
            else:
                layer = nn.Linear(in_dim, out_dim)

            if weight_norm:
                layer = nn.utils.weight_norm(layer)

            layers.append(layer)

        self.layers = nn.ModuleList(layers)

    def forward(self, feat):
        h = feat
        
        for i in range(self.D+1):
            if i in self.skips:
                h = torch.cat([h, feat], dim=-1)
            h_last = h
            h = self.layers[i](h)

            return h

class GeometryDecoder(nn.Module):
    def __init__(self, W=128, D=1, output_ch=1, skips=[], input_feat_dim=4, n_freq=-1, weight_norm=False, concat_qp=False):
        super(GeometryDecoder, self).__init__()

        self.embed_fn, input_ch = nn.Identity(), input_feat_dim
        self.W = W
        self.D = D
        self.skips = skips
        layers = []

        for l in range(D+1):
            if l == D:
                out_dim = output_ch
            elif l + 1 in self.skips:
                out_dim = W - input_ch
            else:
                out_dim = W

            if l == 0:
                in_dim = input_ch
            else:
                in_dim = W

            if l != D:
                layer = DenseLayer(in_dim, out_dim)
            else:
                layer = nn.Linear(in_dim, out_dim)

            if weight_norm:
                layer = nn.utils.weight_norm(layer)

            layers.append(layer)

        self.layers = nn.ModuleList(layers)

    def forward(self, feat, return_h=False):
        feat = self.embed_fn(feat)
        h = feat
        
        for i in range(self.D+1):
            if i in self.skips:
                h = torch.cat([h, feat], dim=-1)
            h = self.layers[i](h)

        if return_h:  # return feature
            return h[..., :1], h[..., 1:]
        else:
            return h[..., :1]
    # def forward(self, feat, return_h=False):
    #     feat = self.embed_fn(feat)
    #     h = feat
        
    #     for i in range(self.D+1):
    #         if i in self.skips:
    #             h = torch.cat([h, feat], dim=-1)
    #         h = self.layers[i](h)

    #     if return_h:  # return feature
    #         return h[..., :1], h[..., 1:]
    #     else:
    #         return h[..., :1]


class RadianceDecoder(nn.Module):
    def __init__(self, W=64, D=4, skips=[], use_view_dirs=False, use_normals=False, output_ch=3,
                 input_feat_dim=64, n_freq=4, weight_norm=False, concat_qp=False, use_dot_prod=False, embedding_a_dim=0, encoding='frequency'):
        super(RadianceDecoder, self).__init__()
        if use_view_dirs or use_normals or use_dot_prod:
            if encoding == 'spherical_harmonics':
                from shencoder import SHEncoder
                self.embed_fn= SHEncoder(input_dim=3, degree=n_freq)
                # input_ch = self.embed_fn.output_dim + use_normals * 3 #raw
                input_ch = use_normals * 3 
                # self.embed_fn = tcnn.Encoding(
                #     n_input_dims=3, encoding_config={"otype": "SphericalHarmonics","degree": n_freq}, dtype=torch.float32)
                # input_ch = self.embed_fn.n_output_dims + use_normals * 3
            if encoding == 'frequency':
                # self.embed_fn, input_ch = get_embedder(n_freq, use_view_dirs * 3 + use_normals * 3 + concat_qp * 3 + use_dot_prod * 1)
                self.embed_fn, input_ch = get_embedder(n_freq, use_view_dirs * 3)
                input_ch += use_normals * 3
            input_ch += input_feat_dim
        else:
            self.embed_fn = None
            input_ch = input_feat_dim
        input_ch += embedding_a_dim
        self.use_view_dirs = use_view_dirs
        self.use_normals = use_normals
        self.use_embedding = (embedding_a_dim > 0)
        self.W = W
        self.D = D
        self.skips = skips
        layers = []

        for l in range(D+1):
            if l == D:
                out_dim = output_ch
            else:
                out_dim = W

            if l == 0:
                in_dim = input_ch
            elif l in self.skips:
                in_dim = input_ch + W
            elif l == 1:
                in_dim = W + self.embed_fn.output_dim
            else:
                in_dim = W

            if l != D:
                layer = DenseLayer(in_dim, out_dim)
            else:
                layer = nn.Linear(in_dim, out_dim)

            if weight_norm:
                layer = nn.utils.weight_norm(layer)

            layers.append(layer)

        self.layers = nn.ModuleList(layers)

    def forward(self, radiance_feats, view_dirs, appearance_embedding=None, grads=None, return_last=False, returnh=False):
        radiance_input = radiance_feats
        if self.use_view_dirs:
            # normals = F.normalize(grads, p=2, dim=-1)
            # refdirs = 2.0 * torch.sum(normals * -view_dirs, axis=-1, keepdims=True) * normals + view_dirs
            refdirs = view_dirs
            # radiance_input = torch.cat([radiance_input,self.embed_fn(refdirs)], dim=-1) #raw
            radiance_input = torch.cat([radiance_input], dim=-1)
        if grads is not None and self.use_normals:
            radiance_input = torch.cat([radiance_input,grads], dim=-1)
        if self.use_embedding and appearance_embedding is not None:
            radiance_input = torch.cat([radiance_input, appearance_embedding], dim=-1)
        h = radiance_input
        for i in range(self.D + 1):
            if i in self.skips:
                h = torch.cat([h, radiance_input], dim=-1)
            elif i == 1:
                h_sem = h
                h = torch.cat([h, self.embed_fn(refdirs)], dim=-1)
            h_last = h
            h = self.layers[i](h)

        if return_last:  # return feature
            return h,h_sem #h_last
        elif returnh:
            return h[:,:3],h[:,3:]
        else:
            return h


class NeRFDecoder(nn.Module):
    def __init__(self, semantic_kwargs, sem_feat_dim, sem_num):
        super(NeRFDecoder, self).__init__()
        self.semantic_net = SemanticDecoder(**semantic_kwargs, input_feat_dim=sem_feat_dim, sem_num=sem_num)
        # self.geometry_net = GeometryDecoder(**geometry_kwargs, input_feat_dim=sdf_feat_dim)
        # self.radiance_net = RadianceDecoder(**radiance_kwargs, input_feat_dim=rgb_feat_dim)

    def forward(self, feat, view_dirs=None):
        # if view_dirs is not None:
        #     geometry, h = self.geometry_net(feat, return_h=True)
        #     rgb = self.radiance_net(h, view_dirs=view_dirs)
        #     return torch.cat([geometry, rgb], dim=-1)
        # else:
        return self.semantic_net(feat, return_h=False)


class SDFDecoder(nn.Module):
    def __init__(self, sdf_kwargs, sdf_feat_dim):
        super(SDFDecoder, self).__init__()
        self.sdf_net = GeometryDecoder(**sdf_kwargs, input_feat_dim=rgb_feat_dim)
        # self.geometry_net = GeometryDecoder(**geometry_kwargs, input_feat_dim=sdf_feat_dim)
        # self.radiance_net = RadianceDecoder(**radiance_kwargs, input_feat_dim=rgb_feat_dim)

    def forward(self, feat, view_dirs=None):
        return self.sdf_net(feat, return_h=False)
    def forward_sdf(self, feat):
        return self.sdf_net(feat, return_h=False)

class RGBDecoder(nn.Module):
    def __init__(self, radiance_kwargs, rgb_feat_dim):
        super(RGBDecoder, self).__init__()
        self.radiance_net = RadianceDecoder(**radiance_kwargs, input_feat_dim=rgb_feat_dim)

    def forward(self, feat, view_dirs=None):
        return self.radiance_net(feat, view_dirs=view_dirs)

class SemanticSDFDecoder(nn.Module):
    def __init__(self, sdf_kwargs, semantic_kwargs, sdf_feat_dim):
        super(SemanticSDFDecoder, self).__init__()
        self.sdf_net = GeometryDecoder(**sdf_kwargs, input_feat_dim=sdf_feat_dim)
        self.semantic_net = SemanticDecoder(**semantic_kwargs, input_feat_dim=sdf_kwargs['output_ch']-1)

    def forward(self, feat):
        sdf, semantic_feats = self.sdf_net(feat, return_h=True)
        semantic = self.semantic_net(semantic_feats, return_h=False)
        return sdf, semantic

    def forward_sdf(self, feat):
        return self.sdf_net(feat, return_h=False)

class SemanticRGBDecoder(nn.Module):
    def __init__(self, rgb_kwargs, semantic_kwargs, rgb_feat_dim, sem_num):
        super(SemanticRGBDecoder, self).__init__()
        self.rgb_net = RadianceDecoder(**rgb_kwargs, input_feat_dim=rgb_feat_dim)
        self.semantic_net = SemanticDecoder(**semantic_kwargs, input_feat_dim=rgb_kwargs['output_ch']-3, sem_num=sem_num)

    def forward(self, feat, view_dirs, appearance_embedding, grads):
        rgb, semantic_feats = self.rgb_net(feat, view_dirs, appearance_embedding=appearance_embedding, grads=grads, returnh=True)
        semantic = self.semantic_net(semantic_feats, return_h=False)
        return rgb, semantic

    def forward_rgb(self, feat, view_dirs, appearance_embedding, grads):
        return self.rgb_net(feat, view_dirs, appearance_embedding=appearance_embedding, grads=grads)

# class NeRFSDFDecoder(nn.Module):
#     def __init__(self, sdf_kwargs, radiance_kwargs, sdf_feat_dim):
#         super(NeRFSDFDecoder, self).__init__()
#         self.sdf_net = GeometryDecoder(**sdf_kwargs, input_feat_dim=sdf_feat_dim)
#         # self.semantic_net = SemanticDecoder(**semantic_kwargs, input_feat_dim=sdf_kwargs['output_ch']-1)
#         self.radiance_net = RadianceDecoder(**radiance_kwargs, input_feat_dim=sdf_kwargs['output_ch']-1)

#     def forward(self, feat, view_dirs):
#         sdf, radiance_feats = self.sdf_net(feat, return_h=True)
#         rgb = self.radiance_net(radiance_feats, view_dirs=view_dirs)
#         return sdf, rgb

#     def forward_sdf(self, feat):
#         return self.sdf_net(feat, return_h=False)

    
# class NeRFSDFDecoder(nn.Module):
#     def __init__(self, sdf_kwargs, radiance_kwargs, sdf_feat_dim, rgb_feat_dim):
#         super(NeRFSDFDecoder, self).__init__()
#         self.sdf_net = GeometryDecoder(**sdf_kwargs, input_feat_dim=sdf_feat_dim)
#         # self.semantic_net = SemanticDecoder(**semantic_kwargs, input_feat_dim=sdf_kwargs['output_ch']-1)
#         self.radiance_net = RadianceDecoder(**radiance_kwargs, input_feat_dim=rgb_feat_dim)

#     def forward(self, sdf_feat, radiance_feats=None, view_dirs=None):
#         sdf = self.sdf_net(sdf_feat, return_h=False)
#         rgb = self.radiance_net(radiance_feats, view_dirs=view_dirs)
#         return sdf, rgb

#     def forward_sdf(self, feat):
#         return self.sdf_net(feat, return_h=False)
    
class NeRFSDFDecoder(nn.Module):
    def __init__(self, sdf_kwargs, feature_kwargs, sdf_feat_dim, rgb_feat_dim):
        super(NeRFSDFDecoder, self).__init__()
        self.sdf_net = GeometryDecoder(**sdf_kwargs, input_feat_dim=sdf_feat_dim)
        # self.semantic_net = SemanticDecoder(**semantic_kwargs, input_feat_dim=sdf_kwargs['output_ch']-1)
        self.feature_net = PositionDecoder(**feature_kwargs, input_feat_dim=rgb_feat_dim)

    def forward(self, sdf_feat, radiance_feats=None):
        sdf = self.sdf_net(sdf_feat, return_h=False)
        feats = self.feature_net(radiance_feats)
        return sdf, feats

    def forward_sdf(self, feat):
        return self.sdf_net(feat, return_h=False)