import torch
# import hydra
import torch.nn as nn
# import MinkowskiEngine.MinkowskiOps as me
# from MinkowskiEngine.MinkowskiPooling import MinkowskiAvgPooling
import numpy as np
from torch.nn import functional as F
# from instance.models.modules.common import conv
from instance.models.position_embedding import PositionEmbeddingCoordsSine
# from third_party.pointnet2.pointnet2_utils import furthest_point_sample
from instance.models.modules.helpers_3detr import GenericMLP
from torch_scatter import scatter_mean, scatter_max, scatter_min
from torch.cuda.amp import autocast
from instance.models.swin import WindowAttention
from timm.models.layers import to_3tuple

from instance.models.query_model import QueryModel

class Mask3D(nn.Module):
    def __init__(
        self,
        # config,
        hidden_dim,
        num_heads,
        dim_feedforward,
        # sample_sizes,
        shared_decoder,
        num_classes,
        num_decoders,
        dropout,
        pre_norm,
        positional_encoding_type,
        non_parametric_queries,
        train_on_segments,
        normalize_pos_enc,
        use_level_embed,
        # hlevels,
        use_np_features,
        # voxel_size,
        max_sample_size,
        random_queries,
        gauss_scale,
        random_query_both,
        random_normal,
        query_position,
        # radius,
        device,
        distance_weight,
        class_need_flag = True,
    ):
        super().__init__()
        self.random_normal = random_normal
        self.random_query_both = random_query_both
        self.random_queries = random_queries
        self.max_sample_size = max_sample_size
        self.gauss_scale = gauss_scale
        # self.voxel_size = voxel_size
        self.hlevels = [0] #[0,1,2] #hlevels
        self.use_level_embed = use_level_embed
        self.train_on_segments = train_on_segments
        self.normalize_pos_enc = normalize_pos_enc
        self.num_decoders = num_decoders
        self.num_classes = num_classes
        self.dropout = dropout
        self.pre_norm = pre_norm
        self.shared_decoder = shared_decoder
        # self.sample_sizes = sample_sizes
        self.non_parametric_queries = non_parametric_queries
        self.use_np_features = use_np_features
        self.mask_dim = hidden_dim
        self.num_heads = num_heads
        # self.query_position = query_position
        self.num_queries = query_position.shape[0]
        self.query_update_map = {}
        self.pos_enc_type = positional_encoding_type
        # self.window_size = 30
        self.softmax = nn.Softmax(dim=-1)
        self.device = device
        # self.radius = radius
        self.distance_weight = distance_weight

        # self.backbone = hydra.utils.instantiate(config.backbone)
        self.num_levels = len(self.hlevels)
        sizes = [0]#self.backbone.PLANES[-5:]
        self.class_need_flag = class_need_flag


        self.mask_features_head = nn.Linear(32, hidden_dim).to(self.device) #feature_dim

        self.scatter_fn = scatter_mean

        assert (
            not use_np_features
        ) or non_parametric_queries, "np features only with np queries"

    
        # PARAMETRIC QUERIES
        # learnable query features
        # self.query_feat = nn.Embedding(self.num_queries, hidden_dim)
        # learnable query p.e.
        # self.query_pos = nn.Embedding(self.num_queries, hidden_dim)
        # self.query_mask = torch.ones(self.num_queries).to(self.device).bool()

        #         # self.mask_embed_head = nn.Linear(hidden_dim, hidden_dim)
        self.mask_embed_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.class_embed_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_classes+1),
        )
        # self.class_embed_head = nn.Linear(hidden_dim, self.num_classes+1)

        #         # if self.pos_enc_type == "legacy":
        #     self.pos_enc = PositionalEncoding3D(channels=self.mask_dim)
        # elif self.pos_enc_type == "fourier":
        #     self.pos_enc = PositionEmbeddingCoordsSine(
        #         pos_type="fourier",
        #         d_pos=self.mask_dim,
        #         gauss_scale=self.gauss_scale,
        #         normalize=self.normalize_pos_enc,
        #     )
        # elif self.pos_enc_type == "sine":
        #     self.pos_enc = PositionEmbeddingCoordsSine(
        #         pos_type="sine",
        #         d_pos=self.mask_dim,
        #         normalize=self.normalize_pos_enc,
        #     )
        # else:
        #     assert False, "pos enc type not known"

        self.decoder_norm = nn.LayerNorm(hidden_dim)

    def add_query(self, query_position, scale):
        self.gaussians = QueryModel()
        self.gaussians.create_from_pcd(query_position, None, scale=scale)

    def get_pos_encs(self, coords):
        pos_encodings_pcd = []

        # for i in range(len(coords)):
            # pos_encodings_pcd.append([[]])
            # for coords_batch in coords[i].decomposed_features:
        scene_min = coords.min(dim=0)[0][None, ...]
        scene_max = coords.max(dim=0)[0][None, ...]

        with autocast(enabled=False):
            tmp = self.pos_enc(
                coords[None,...].float(),
                input_range=[scene_min, scene_max],
            )

        # pos_encodings_pcd[-1][0].append(tmp.squeeze(0).permute((1, 0)))

        return tmp.squeeze(0).permute((1, 0)) #pos_encodings_pcd

    def update_network(self, query_mask, query_update_map):
        self.num_queries = query_mask.sum()
        self.query_update_map = query_update_map
        self.query_mask = query_mask
        # self.query_feat.weight
        # self.query_pos.weight

    def forward(
        self, x, coords, scale, semantic_x=None, point2segment=None, raw_coordinates=None, is_eval=False
    ):
        # pcd_features, aux = self.backbone(x)
        batch_size = 0
        for i in range(len(x)):
            pcd_features = x[i] if i==0 else torch.vstack((pcd_features, x[i]))
            batch_size += x[i].shape[0] #len(x.decomposed_coordinates)
            # pos_encodings_pcd = self.get_pos_encs(coords[i]) if i==0 else torch.vstack((pos_encodings_pcd, self.get_pos_encs(coords[i])))
            coords_all = coords[i] if i==0 else torch.vstack((coords_all, coords[i]))

        mask_features = pcd_features

        if self.train_on_segments:
            mask_segments = []
            for i, mask_feature in enumerate(
                mask_features #.decomposed_features
            ):
                mask_segments.append(
                    self.scatter_fn(mask_feature, point2segment[i], dim=0)
                )

        sampled_coords = None

        # PARAMETRIC QUERIES
        # queries = self.query_feat.weight[self.query_mask,:].unsqueeze(0).repeat(
        #     batch_size, 1, 1
        # )
        queries = self.gaussians._features_token.unsqueeze(0).repeat(
            batch_size, 1, 1
        )
        queries = self.mask_features_head(queries)
        # queries_class = self.gaussians._label_features_token.unsqueeze(0).repeat(
        #     batch_size, 1, 1
        # )

        predictions_class = []
        predictions_mask = []

        outputs_mask, outputs_class = self.mask_module( #output_class
            queries,
            self.gaussians._label_features_token,
            mask_features,
            semantic_x,
            0,
            ret_attn_mask=False,
            point2segment=None,
            coords=coords[0] #,
            # mask=attn_mask,
            # dis=d.transpose(1,0)
        )
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)

        # query loss
        query = queries[0]
        distance = torch.cdist(query, query, p=2)
        mask = torch.eye(query.shape[0]).bool().to(query.device)
        distance = distance.masked_fill(mask,float('inf'))
        distance_loss = torch.mean(torch.clamp(10-distance, min=0))

        return predictions_mask[-1].squeeze(), predictions_class[-1], \
                self.gaussians.get_xyz.detach().cpu().numpy(), 10*distance_loss #0 #query_loss


    def mask_module(
        self,
        query_feat,
        queries_class,
        mask_features,
        class_features,
        num_pooling_steps,
        ret_attn_mask=True,
        point2segment=None,
        coords=None,
        # dis=None,
    ):
        query_feat = self.decoder_norm(query_feat)
        # query_feat = self.mask_embed_head(query_feat) 
        if self.class_need_flag:
            outputs_class = self.class_embed_head(queries_class) #.detach() query_feat[0]
        else:
            outputs_class = None
        # outputs_class = queries_class
        outputs_mask = mask_features[:,None,:] @ query_feat.transpose(1,2)  # point feature dot* instance feature
        if self.distance_weight:
            weight = self.gaussians.gaussian_3d_probability(coords)
            prob_max = self.gaussians.gaussian_3d_probability_mu()
            # weight = (1/dis).sigmoid()
            outputs_mask = outputs_mask.squeeze(1).sigmoid() * (weight/prob_max+1e-5) #torch.clip(torch.FloatTensor([1e-5]).to(weight.device),weight.max(-1).values[:,None])) #
        else:
            outputs_mask = outputs_mask.squeeze(1).sigmoid()

        return torch.clip(outputs_mask,0,1), outputs_class #.decomposed_features #outputs_class

    # def mask_module(
    #     self,
    #     query_feat,
    #     mask_features,
    #     class_features,
    #     num_pooling_steps,
    #     ret_attn_mask=True,
    #     point2segment=None,
    #     coords=None,
    #     mask=None,
    #     dis=None,
    # ):
    #     query_feat = self.decoder_norm(query_feat)
    #     mask_embed = self.mask_embed_head(query_feat)  # instance feature
    #     # mask_embed = query_feat
    #     outputs_class = self.class_embed_head(query_feat[0]) #.detach()
    #     outputs_mask = mask_features[:,None,:] @ mask_embed.transpose(1,2)  # point feature dot* instance feature
    #     if self.distance_weight:
    #         weight = (1/dis - 1/self.attention_radius).sigmoid()
    #         # weight = (1/dis).sigmoid()
    #         outputs_mask = outputs_mask.squeeze(1).sigmoid() * weight #
    #     else:
    #         outputs_mask = outputs_mask.squeeze(1).sigmoid()

    #     return outputs_mask, outputs_class #.decomposed_features #outputs_class

    # @torch.jit.unused
    # def _set_aux_loss(self, outputs_class, outputs_seg_masks):
    #     # this is a workaround to make torchscript happy, as torchscript
    #     # doesn't support dictionary with non-homogeneous values, such
    #     # as a dict having both a Tensor and a list.
    #     return [
    #         {"pred_logits": a, "pred_masks": b}
    #         for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
    #     ]


class PositionalEncoding3D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        self.orig_ch = channels
        super(PositionalEncoding3D, self).__init__()
        channels = int(np.ceil(channels / 6) * 2)
        if channels % 2:
            channels += 1
        self.channels = channels
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, channels, 2).float() / channels)
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, tensor, input_range=None):
        """
        :param tensor: A 5d tensor of size (batch_size, x, y, z, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, z, ch)
        """
        pos_x, pos_y, pos_z = tensor[:, :, 0], tensor[:, :, 1], tensor[:, :, 2]
        sin_inp_x = torch.einsum("bi,j->bij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("bi,j->bij", pos_y, self.inv_freq)
        sin_inp_z = torch.einsum("bi,j->bij", pos_z, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1)

        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1)
        emb_z = torch.cat((sin_inp_z.sin(), sin_inp_z.cos()), dim=-1)

        emb = torch.cat((emb_x, emb_y, emb_z), dim=-1)
        return emb[:, :, : self.orig_ch].permute((0, 2, 1))
