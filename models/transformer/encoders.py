import spacy
from torch.nn import functional as F
from models.transformer.grid_aug import BoxRelationalEmbedding, AllRelationalEmbedding
from models.transformer.utils import PositionWiseFeedForward
import torch
from torch import nn
from models.transformer.attention import MultiHeadBoxAttention as MultiHeadAttention, MultiHeadBoxAttention
from models.transformer.attention import MyMultiHeadBoxAttention
from models.transformer.attention import MultiHeadAttention as MultiHeadAttention_int
import numpy as np

class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, d_k=128, d_v=128, h=4, d_ff=2048, dropout=.1, identity_map_reordering=False,
                 attention_module=None, attention_module_kwargs=None):
        super(EncoderLayer, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.mhatt = MultiHeadAttention(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
                                                attention_module=attention_module,
                                                attention_module_kwargs=attention_module_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.lnorm1 = nn.LayerNorm(d_model)
        self.lnorm2 = nn.LayerNorm(d_model)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

    def forward(self, queries, keys, values, relative_geometry_weights, attention_mask=None, attention_weights=None, pos=None):

        
        
        att = self.mhatt(queries, keys, values, pos, relative_geometry_weights, attention_mask, attention_weights)
        att = self.lnorm2(queries + self.dropout(att))
        
        ff = self.pwff(att)

        return ff
class EncoderLayerTwo(nn.Module):
    def __init__(self, d_model=512, d_k=128, d_v=128, h=4, d_ff=2048, dropout=.1, identity_map_reordering=False,
                 attention_module=None, attention_module_kwargs=None):
        super(EncoderLayerTwo, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.mhatt = MultiHeadAttention_int(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
                                                attention_module=attention_module,
                                                attention_module_kwargs=attention_module_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.lnorm = nn.LayerNorm(d_model)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)


    def forward(self, queries, keys, values, relative_geometry_weights, attention_mask=None, attention_weights=None, pos=None):

        
        
        att = self.mhatt(queries, keys, values, attention_mask, attention_weights)
        att = self.lnorm(queries + self.dropout(att))
        ff = self.pwff(att)
        

        return ff

class MyInterestLayer(nn.Module):
    def __init__(self, d_model=512, d_k=128, d_v=128, h=4, d_ff=2048, dropout=.1, identity_map_reordering=False,
                 attention_module=None, attention_module_kwargs=None):
        super(MyInterestLayer,self).__init__()
        self.identity_map_reordering = identity_map_reordering
         
        self.mhatt = MyMultiHeadBoxAttention(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
                                                attention_module=attention_module,
                                                attention_module_kwargs=attention_module_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.lnorm1 = nn.LayerNorm(d_model)
        self.lnorm2 = nn.LayerNorm(d_model)
        self.pwff1 = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)
        self.pwff2 = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

    def forward(self,regions_out,grids_out,interests,attention_mask_regions, attention_mask_grids,attention_weights):
        att1,att2 = self.mhatt(regions_out,grids_out,interests,attention_mask_regions, attention_mask_grids,attention_weights)
        att1 = self.lnorm1(regions_out + self.dropout(att1))
        att2 = self.lnorm2(grids_out + self.dropout(att2))
        ff1 = self.pwff1(att1)
        ff2 = self.pwff2(att2)
        return ff1,ff2
    

class MultiLevelEncoder(nn.Module):
    def __init__(self, N, padding_idx, d_model=512, d_k=128, d_v=128, h=4, d_ff=2048, dropout=.1,
                 identity_map_reordering=False, attention_module=None, attention_module_kwargs=None):
        super(MultiLevelEncoder, self).__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.layers_regions = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout,
                                                  identity_map_reordering=identity_map_reordering,
                                                  attention_module=attention_module,
                                                  attention_module_kwargs=attention_module_kwargs)
                                     for _ in range(N)])
        self.layers_grids = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout,
                                                  identity_map_reordering=identity_map_reordering,
                                                  attention_module=attention_module,
                                                  attention_module_kwargs=attention_module_kwargs)
                                     for _ in range(N)])
        self.layers_interests = nn.ModuleList([EncoderLayerTwo(d_model, d_k, d_v, h, d_ff, dropout,
                                                  identity_map_reordering=identity_map_reordering,
                                                  attention_module=attention_module,
                                                  attention_module_kwargs=attention_module_kwargs)
                                     for _ in range(N)])
        
        self.layers_my_attention = nn.ModuleList([MyInterestLayer(d_model, d_k, d_v, h, d_ff, dropout,
                                                  identity_map_reordering=identity_map_reordering,
                                                  attention_module=attention_module,
                                                  attention_module_kwargs=attention_module_kwargs)
                                     for _ in range(N)])
        self.padding_idx = padding_idx

        self.WGs = nn.ModuleList([nn.Linear(64, 1, bias=True) for _ in range(h)])

    def forward(self, regions, grids, interests, boxes, aligns, attention_weights=None, region_embed=None, grid_embed=None):
        # input (b_s, seq_len, d_in)
        attention_mask_regions = (torch.sum(regions, -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)  # (b_s, 1, 1, seq_len)
        attention_mask_grids = (torch.sum(grids, -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)  # (b_s, 1, 1, seq_len)

        # region geometry embedding
        regions_relative_geometry_embeddings = AllRelationalEmbedding(boxes)
        regions_flatten_relative_geometry_embeddings = regions_relative_geometry_embeddings.view(-1, 64)
        regions_box_size_per_head = list(regions_relative_geometry_embeddings.shape[:3])
        regions_box_size_per_head.insert(1, 1)
        regions_relative_geometry_weights_per_head = [l(regions_flatten_relative_geometry_embeddings).view(regions_box_size_per_head) for l in
                                              self.WGs]
        regions_relative_geometry_weights = torch.cat((regions_relative_geometry_weights_per_head), 1)
        regions_relative_geometry_weights = F.relu(regions_relative_geometry_weights)

        # grid geometry embedding
        # follow implementation of https://github.com/yahoo/object_relation_transformer/blob/ec4a29904035e4b3030a9447d14c323b4f321191/models/RelationTransformerModel.py
        grids_relative_geometry_embeddings = BoxRelationalEmbedding(grids)

        grids_flatten_relative_geometry_embeddings = grids_relative_geometry_embeddings.view(-1, 64)
        grids_size_per_head = list(grids_relative_geometry_embeddings.shape[:3])
        grids_size_per_head.insert(1, 1)
        grids_relative_geometry_weights_per_head = \
            [layer(grids_flatten_relative_geometry_embeddings).view(grids_size_per_head) for layer in self.WGs]
        grids_relative_geometry_weights = torch.cat((grids_relative_geometry_weights_per_head), 1)
        grids_relative_geometry_weights = F.relu(grids_relative_geometry_weights)

        grids_out = grids
        regions_out = regions


        for l_region,l_grid,l_interest,l_my in zip(self.layers_regions,self.layers_grids,self.layers_interests,self.layers_my_attention):
            regions_out = l_region(regions_out, regions_out, regions_out, regions_relative_geometry_weights, attention_mask_regions, attention_weights, pos=region_embed)
            grids_out = l_grid(grids_out, grids_out, grids_out, grids_relative_geometry_weights, attention_mask_grids, attention_weights, pos=grid_embed)
            interests_out = l_interest(interests,interests,interests,None,None)
            regions_out,grids_out = l_my(regions_out,grids_out,interests_out,attention_mask_regions, attention_mask_grids,attention_weights) 


        attention_mask = torch.cat([attention_mask_regions, attention_mask_grids], dim=-1) 

        out = torch.cat([regions_out,grids_out],dim=1)  
        return out, attention_mask

class TransformerEncoder(MultiLevelEncoder): 
    def __init__(self, N, padding_idx, d_in=2048, **kwargs):
        super(TransformerEncoder, self).__init__(N, padding_idx, **kwargs)
        self.fc_region = nn.Linear(d_in, self.d_model)
        self.dropout_region = nn.Dropout(p=self.dropout)
        self.layer_norm_region = nn.LayerNorm(self.d_model)

        self.fc_grid = nn.Linear(d_in, self.d_model)
        self.dropout_grid = nn.Dropout(p=self.dropout)
        self.layer_norm_grid = nn.LayerNorm(self.d_model)

        self.fc_interest = nn.Linear(300, self.d_model)
        self.dropout_interest = nn.Dropout(p=self.dropout)
        self.layer_norm_interest = nn.LayerNorm(self.d_model)
        self.tokenizer = spacy.load("en_core_web_lg")

    def interest_emb(self,interests):  
        max_len = 49
        interes = []
        for cs in interests:
            cs = cs.split(' ')
            intes = []
            
            for c in cs: 
                intes.append(self.tokenizer.vocab[c].vector)
            _len = max_len - len(intes)
            if _len >= 0:
                for _ in range(_len):
                    intes.append(self.tokenizer.vocab["<pad>"].vector)
            else:
                intes = intes[:max_len]
            interes.append(intes)
        return torch.tensor(interes)



    def forward(self, regions, grids, boxes, aligns, attention_weights=None, region_embed=None, grid_embed=None, interests = None):
        device = torch.device('cuda')
        interests = self.interest_emb(interests).to(device)
        
        
        mask_regions = (torch.sum(regions, dim=-1) == 0).unsqueeze(-1)
        mask_grids = (torch.sum(grids, dim=-1) == 0).unsqueeze(-1)
        mask_interests = (torch.sum(interests,dim=-1) == 0).unsqueeze(-1)

        out_interest = F.relu(self.fc_interest(interests))
        out_interest = self.dropout_interest(out_interest)
        out_interest = self.layer_norm_interest(out_interest)
        out_interest = out_interest.masked_fill(mask_interests,0)

        out_region = F.relu(self.fc_region(regions))
        out_region = self.dropout_region(out_region)
        out_region = self.layer_norm_region(out_region)
        out_region = out_region.masked_fill(mask_regions, 0)

        out_grid = F.relu(self.fc_grid(grids))
        out_grid = self.dropout_grid(out_grid)
        out_grid = self.layer_norm_grid(out_grid)
        out_grid = out_grid.masked_fill(mask_grids, 0)
        return super(TransformerEncoder, self).forward(out_region, out_grid, out_interest, boxes, aligns,
                                                       attention_weights=attention_weights,
                                                       region_embed=region_embed, grid_embed=grid_embed)

