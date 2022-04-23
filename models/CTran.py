 
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pdb import set_trace as stop
from .transformer_layers import SelfAttnLayer
from .utils import custom_replace,weights_init
from .position_enc import PositionEmbeddingSine,positionalencoding2d

 
class CTranModel(nn.Module):
    def __init__(self,num_labels,use_lmt,device, backbone_model, C=96, pos_emb=False,layers=3,heads=4,dropout=0.1, int_loss=0, no_x_features=False, grad_cam=False):
        super(CTranModel, self).__init__()
        self.use_lmt = use_lmt
        
        self.no_x_features = no_x_features # (for no image features)

        self.backbone = None
        print(backbone_model)

        # ResNet backbone
        if 'resnet' in backbone_model or 'resnext' in backbone_model:
            self.backbone = ResNetBackbone(backbone_model)
        elif 'efficientnet' in backbone_model:
            self.backbone = EfficientNetBackbone(backbone_model)
        elif 'convnext' in backbone_model:
            self.backbone = ConvNextBackbone(backbone_model)
        elif 'test' in backbone_model:
            self.backbone = BasicConv2d(3, 1024, kernel_size=3)
        else:
            print('unknown ', backbone_model, ' model')
            exit(0)

        self.hidden = self.backbone.features
        print('hidden', self.hidden)

        self.downsample = True
        if self.downsample:
            #self.proj = torch.nn.Linear(, C)
            self.conv_downsample = torch.nn.Conv2d(self.hidden, self.hidden, (1,1))
        
        # Label Embeddings
        self.label_input = torch.Tensor(np.arange(num_labels)).view(1,-1).long()
        self.label_lt = torch.nn.Embedding(num_labels, C, padding_idx=None)

        # State Embeddings
        self.known_label_lt = torch.nn.Embedding(3, C, padding_idx=0)

        # Position Embeddings (for image features)
        self.use_pos_enc = pos_emb
        if self.use_pos_enc:
            # self.position_encoding = PositionEmbeddingSine(int(hidden/2), normalize=True)
            self.position_encoding = positionalencoding2d(self.hidden, 18, 18).unsqueeze(0)


        # Here we have to add the modified version of the swin transformer
        # This version should not calculate the patch embeddings, instead it should receive them as parameter

        # Transformer
        self.self_attn_layers = nn.ModuleList([SelfAttnLayer(self.hidden, heads, dropout) for _ in range(layers)])

        # Classifier
        # Output is of size num_labels because we want a separate classifier for each label
        self.output_linear = torch.nn.Linear(self.hidden, num_labels)

        # Other
        self.LayerNorm = nn.LayerNorm(C)
        self.dropout = nn.Dropout(dropout)

        # Init all except pretrained backbone
        self.label_lt.apply(weights_init)
        self.known_label_lt.apply(weights_init)
        self.LayerNorm.apply(weights_init)
        self.self_attn_layers.apply(weights_init)
        self.output_linear.apply(weights_init)


        #Swin Projection
        self.project = True
        self.C = C

    def forward(self,images,mask=None):
        const_label_input = self.label_input.repeat(images.size(0),1).to(self.device)
        init_label_embeddings = self.label_lt(const_label_input)
        print('label embeddings', init_label_embeddings.shape)

        features = self.backbone(images)
        print('features', features.shape)

        if self.use_pos_enc:
            pos_encoding = self.position_encoding(features,torch.zeros(features.size(0),18,18, dtype=torch.bool).cuda())
            features = features + pos_encoding

        features = features.view(features.size(0), features.size(1),-1).permute(0,2,1)
        print('features after weird stuff', features.shape)

        # Don't forget to check how to train this
        if self.project:
            #features = self.conv_downsample(features)
            proj = torch.nn.Linear(features.shape[2], self.C)
            #print(features)
            features = proj(features.squeeze())
            features = features[None, :]
            print('new features', features.shape)

        #features = features.permute(0, 2, 1)

        if self.use_lmt:
            # Convert mask values to positive integers for nn.Embedding
            label_feat_vec = custom_replace(mask,0,1,2).long()

            # Get state embeddings
            state_embeddings = self.known_label_lt(label_feat_vec)
            print("state embeddings", state_embeddings.shape)

            # Add state embeddings to label embeddings
            init_label_embeddings += state_embeddings
        
        if self.no_x_features:
            embeddings = init_label_embeddings 
        else:
            # Concat image and label embeddings
            print('features shape', features.shape)
            print('label emb', init_label_embeddings.shape)
            #embeddings = torch.cat((features, init_label_embeddings), 1)

        #print('embeddings', embeddings.shape)

        # Add padding to make it square for the siwn

        # Feed image and label embeddings through Transformer
        #embeddings = self.LayerNorm(embeddings)

        # Input for Swin Transformer
        #embeddings = embeddings.permute(0, 2, 1)
        return  features, init_label_embeddings, None

        # We have to change this part to use swin transformer
        attns = []
        for layer in self.self_attn_layers:
            embeddings,attn = layer(embeddings,mask=None)
            attns += attn.detach().unsqueeze(0).data


        # Readout each label embedding using a linear layer
        label_embeddings = embeddings[:,-init_label_embeddings.size(1):,:]
        output = self.output_linear(label_embeddings) 
        diag_mask = torch.eye(output.size(1)).unsqueeze(0).repeat(output.size(0),1,1).cuda()
        output = (output*diag_mask).sum(-1)

        return output,None,attns

