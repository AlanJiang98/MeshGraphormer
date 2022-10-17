# ----------------------------------------------------------------------------------------------
# FastMETRO Official Code
# Copyright (c) POSTECH Algorithmic Machine Intelligence Lab. (P-AMI Lab.) All Rights Reserved
# Licensed under the MIT license.
# ----------------------------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved [see https://github.com/facebookresearch/detr/blob/main/LICENSE for details]
# ----------------------------------------------------------------------------------------------

"""
Transformer encoder-decoder architecture in FastMETRO model.
"""
import copy
import torch
import torch.nn.functional as F
from typing import Optional
from torch import nn, Tensor


class Encoder(nn.Module):
    def __init__(self, model_dim=512, nhead=8, num_layers=3,
                feedforward_dim=2048, dropout=0.1, activation="relu"):
        """
        Parameters:
            - model_dim: The hidden dimension size in the transformer architecture
            - nhead: The number of attention heads in the attention modules
            - num_enc_layers: The number of encoder layers in the transformer encoder
            - num_dec_layers: The number of decoder layers in the transformer decoder
            - feedforward_dim: The hidden dimension size in MLP
            - dropout: The dropout rate in the transformer architecture
            - activation: The activation function used in MLP
        """
        super().__init__()
        self.model_dim = model_dim
        self.nhead = nhead

        # transformer encoder
        encoder_layer = TransformerEncoderLayer(model_dim, nhead, feedforward_dim, dropout, activation)
        encoder_norm = nn.LayerNorm(model_dim)
        self.encoder = TransformerEncoder(encoder_layer, num_layers, encoder_norm)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, img_features, pos_embed, mask=None, return_att=True):
        device = img_features.device
        hw, bs, _ = img_features.shape
        if mask is None:
            mask = torch.zeros((bs, hw), dtype=torch.bool, device=device)
        # Transformer Encoder
        outputs, atts = self.encoder(img_features, src_key_padding_mask=mask, pos=pos_embed, return_att=return_att)
        return outputs, atts


class Decoder(nn.Module):
    """Transformer encoder-decoder"""
    def __init__(self, model_dim=512, nhead=8, num_layers=3,
                feedforward_dim=2048, dropout=0.1, activation="relu"):
        """
        Parameters:
            - model_dim: The hidden dimension size in the transformer architecture
            - nhead: The number of attention heads in the attention modules
            - num_enc_layers: The number of encoder layers in the transformer encoder
            - num_dec_layers: The number of decoder layers in the transformer decoder
            - feedforward_dim: The hidden dimension size in MLP
            - dropout: The dropout rate in the transformer architecture
            - activation: The activation function used in MLP
        """
        super().__init__()
        self.model_dim = model_dim
        self.nhead = nhead
        # transformer decoder
        decoder_layer_ = TransformerDecoderLayer(model_dim, nhead, feedforward_dim, dropout, activation)
        decoder_norm_ = nn.LayerNorm(model_dim)
        self.decoder = TransformerDecoder(decoder_layer_, num_layers, decoder_norm_)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input, jv_tokens, pos_embed, attention_mask=None, return_att=True):
        device = input.device
        L, bs, _ = input.shape

        # Transformer Decoder
        mask = torch.zeros((bs, L), dtype=torch.bool, device=device)
        zero_tgt = torch.zeros_like(jv_tokens, device=device)
        outputs, d_atts = self.decoder(jv_tokens,
                                        input,
                                        tgt_mask=attention_mask,
                                        memory_key_padding_mask=mask,
                                        pos=pos_embed,
                                        query_pos=zero_tgt,
                                       return_att=return_att)

        return outputs, d_atts


class Transformer(nn.Module):
    """Transformer encoder-decoder"""
    def __init__(self, model_dim=512, nhead=8, num_enc_layers=3, num_dec_layers=3,
                feedforward_dim=2048, dropout=0.1, activation="relu", input_shapes=[[6, 6]], att=False, decoder_features=[False, True, False]):
        """
        Parameters:
            - model_dim: The hidden dimension size in the transformer architecture
            - nhead: The number of attention heads in the attention modules
            - num_enc_layers: The number of encoder layers in the transformer encoder
            - num_dec_layers: The number of decoder layers in the transformer decoder
            - feedforward_dim: The hidden dimension size in MLP
            - dropout: The dropout rate in the transformer architecture
            - activation: The activation function used in MLP
        """
        super().__init__()
        self.model_dim = model_dim
        self.nhead = nhead
        self.input_shapes = input_shapes
        self.decoder_features=decoder_features

        # transformer encoder
        encoder_layer = TransformerEncoderLayer(model_dim, nhead, feedforward_dim, dropout, activation)
        encoder_norm = nn.LayerNorm(model_dim)
        self.encoder = TransformerEncoder(encoder_layer, num_enc_layers, encoder_norm)

        # transformer decoder
        self.decoder = torch.nn.ModuleList()
        decoder_layer_ = TransformerDecoderLayer(model_dim, nhead, feedforward_dim, dropout, activation)
        decoder_norm_ = nn.LayerNorm(model_dim)
        self.decoder.append(TransformerDecoder(decoder_layer_, num_dec_layers, decoder_norm_))
        if len(self.input_shapes) == 3:
            decoder_layer_ = TransformerDecoderLayer(model_dim, nhead, feedforward_dim, dropout, activation)
            decoder_norm_ = nn.LayerNorm(model_dim)
            self.decoder.append(TransformerDecoder(decoder_layer_, num_dec_layers, decoder_norm_))
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, img_features, jv_tokens, pos_embed, attention_mask=None):
        device = img_features.device
        hw, bs, _ = img_features.shape
        mask = torch.zeros((bs, hw), dtype=torch.bool, device=device)

        # Transformer Encoder
        e_outputs, e_atts = self.encoder(img_features, src_key_padding_mask=mask, pos=pos_embed)

        lens = [int(hw[0]*hw[1]) for hw in self.input_shapes]
        if len(self.input_shapes) == 3:
            de_ind_l = [0, lens[0]+lens[1]]
            de_ind_r = [sum(lens[:2]), sum(lens)]
        else:
            de_ind_l = [0, sum(lens)]

        # Transformer Decoder
        mask_l = torch.zeros((bs, de_ind_l[1]-de_ind_l[0]), dtype=torch.bool, device=device)
        pos_embed_l = pos_embed[de_ind_l[0]: de_ind_l[1]]
        zero_tgt_l = torch.zeros_like(jv_tokens[0], device=device)
        jv_features_l, d_atts_l = self.decoder[0](jv_tokens[0],
                                        e_outputs[de_ind_l[0]: de_ind_l[1]],
                                        tgt_mask=attention_mask,
                                        memory_key_padding_mask=mask_l,
                                        pos=pos_embed_l,
                                        query_pos=zero_tgt_l)
        output = {
            'e_outputs': e_outputs,
            'e_atts': e_atts,
            'jv_features_l': jv_features_l,
            'd_atts_l': d_atts_l
        }
        if len(self.input_shapes) == 3:
            mask_r = torch.zeros((bs, de_ind_r[1] - de_ind_r[0]), dtype=torch.bool, device=device)
            pos_embed_r = pos_embed[de_ind_r[0]: de_ind_r[1]]
            zero_tgt_r = torch.zeros_like(jv_tokens[1], device=device)
            jv_features_r, d_atts_r = self.decoder[1](jv_tokens[1],
                                            e_outputs[de_ind_r[0]: de_ind_r[1]],
                                            tgt_mask=attention_mask,
                                            memory_key_padding_mask=mask_r,
                                            pos=pos_embed_r,
                                            query_pos=zero_tgt_r)
            output['jv_features_r'] = jv_features_r
            output['d_atts_r'] = d_atts_r

        return output


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.num_layers = num_layers
        self.norm = norm
        self.layers = _get_clones(encoder_layer, num_layers)

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                return_att=True):
        atts = ()
        output = src

        for layer in self.layers:
            output, att_ = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos, return_att=return_att)
            atts = atts + (att_, )
        if self.norm is not None:
            output = self.norm(output)

        return output, atts


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.num_layers = num_layers
        self.norm = norm
        self.layers = _get_clones(decoder_layer, num_layers)

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                return_att=True):
        output = tgt
        atts = ()
        for layer in self.layers:
            output, att_ = layer(output, memory, tgt_mask=tgt_mask,
                                   memory_mask=memory_mask,
                                   tgt_key_padding_mask=tgt_key_padding_mask,
                                   memory_key_padding_mask=memory_key_padding_mask,
                                   pos=pos, query_pos=query_pos,
                                    return_att=return_att)
            atts = atts + (att_, )
        if self.norm is not None:
            output = self.norm(output)

        return output, atts


class TransformerEncoderLayer(nn.Module):

    def __init__(self, model_dim, nhead, feedforward_dim=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(model_dim, nhead, dropout=dropout)

        # MLP
        self.linear1 = nn.Linear(model_dim, feedforward_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(feedforward_dim, model_dim)

        # Layer Normalization & Dropout
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        # tensor[0] is for a camera token (no positional encoding)
        return tensor if pos is None else tensor + pos

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                return_att=True):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2, att = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        if not return_att:
            att = None
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src, att


class TransformerDecoderLayer(nn.Module):

    def __init__(self, model_dim, nhead, feedforward_dim=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(model_dim, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(model_dim, nhead, dropout=dropout)

        # MLP
        self.linear1 = nn.Linear(model_dim, feedforward_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(feedforward_dim, model_dim)

        # Layer Normalization & Dropout
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.norm3 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                return_att=True):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2, att_self = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                            key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2, att_query = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        if not return_att:
            att_self, att_query = None, None
        return tgt, (att_self, att_query)


class SelfAttention(nn.Module):
    def __init__(self, model_dim, nhead, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(model_dim)
        self.self_attn = nn.MultiheadAttention(model_dim, nhead, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        pass

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, latent, query_pos, latent_mask, tgt_key_padding_mask, return_att=True):
        latent_ = self.norm(latent)
        q = k = self.with_pos_embed(latent_, query_pos)
        latent_, att_self = self.self_attn(q, k, value=latent_, attn_mask=latent_mask,
                                        key_padding_mask=tgt_key_padding_mask)
        latent = latent + self.dropout(latent_)
        if return_att:
            att_self = None
        return latent, att_self


class FeedForwardLayer(nn.Module):
    def __init__(self, model_dim, feedforward_dim, dropout, activation='relu'):
        super().__init__()
        self.norm = nn.LayerNorm(model_dim)
        self.linear1 = nn.Linear(model_dim, feedforward_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(feedforward_dim, model_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)

    def forward(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout1(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout2(tgt2)
        return tgt


class PerceiverLayer(nn.Module):
    def __init__(self, latent_dim, context_dim, nhead, n_self_att_layers, ff_dim=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.to_kv = nn.Linear(context_dim, latent_dim * 2, bias=False)
        # cross
        self.cross_attn = nn.MultiheadAttention(latent_dim, nhead, dropout=dropout)
        self.cross_ff = FeedForwardLayer(latent_dim, ff_dim, dropout)
        # self
        self.n_self_att_layers = n_self_att_layers
        self.self_att_layers = nn.ModuleList()
        self.self_ff_layers = nn.ModuleList()
        for _ in range(n_self_att_layers):
            self.self_att_layers.append(SelfAttention(latent_dim, nhead, dropout))
            self.self_ff_layers.append(FeedForwardLayer(latent_dim, ff_dim, dropout, activation))
        # MLP
        # Layer Normalization & Dropout
        self.dropout_cross = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(latent_dim)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, latent, context, iterations=2,
                latent_mask: Optional[Tensor] = None,
                cross_mask: Optional[Tensor] = None,
                latent_key_padding_mask: Optional[Tensor] = None,
                cross_key_padding_mask: Optional[Tensor] = None,
                context_pos: Optional[Tensor] = None,
                latent_pos: Optional[Tensor] = None,
                return_att=True):
        atts = ()
        for i in range(iterations):
            k, v = self.to_kv(context).chunk(2, dim=-1)
            latent1, att_query = self.cross_attn(query=self.with_pos_embed(latent, latent_pos),
                                       key=self.with_pos_embed(k, context_pos),
                                       value=v, attn_mask=cross_mask,
                                       key_padding_mask=cross_key_padding_mask)
            if not return_att:
                att_query = None
            atts += (att_query, )
            latent = latent + self.dropout_cross(latent1)
            for j in range(self.n_self_att_layers):
                latent, att_latent = self.self_att_layers[j](latent, latent_pos, latent_mask, latent_key_padding_mask, return_att=return_att)
                latent = self.self_ff_layers[j](latent)
                if not return_att:
                    att_latent = None
                atts += (att_latent,)
        return latent, atts


class PerceiverBlock(nn.Module):
    def __init__(self, num_layers, latent_dim, context_dim, nhead, n_self_att_layers, ff_dim=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.num_layers = num_layers
        layer = PerceiverLayer(latent_dim, context_dim, nhead, n_self_att_layers, ff_dim, dropout, activation)
        self.layers = _get_clones(layer, num_layers)

    def forward(self, latent, context, iterations=2,
                context_pos: Optional[Tensor] = None,
                return_att=True):
        atts = ()
        device = context.device
        C, bs, _ = context.shape
        L, bs, _ = latent.shape
        # Transformer Decoder
        mask = torch.zeros((L, C), dtype=torch.bool, device=device)
        zero_latent = torch.zeros_like(latent, device=device)
        for layer in self.layers:
            latent, att_ = layer(
                latent=latent,
                context=context,
                iterations=iterations,
                cross_mask=mask,
                context_pos=context_pos,
                latent_pos=zero_latent, return_att=return_att
            )
            atts += (att_)
        return latent, atts


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def build_transformer(transformer_config):
    return Transformer(model_dim=transformer_config['model_dim'],
                       dropout=transformer_config['dropout'],
                       nhead=transformer_config['nhead'],
                       feedforward_dim=transformer_config['feedforward_dim'],
                       num_enc_layers=transformer_config['num_enc_layers'],
                       num_dec_layers=transformer_config['num_dec_layers'],
                       input_shapes=transformer_config['input_shapes'],
                       att=transformer_config['att'],
                       decoder_features=transformer_config['decoder_features'])


def build_transformer_block(block, transformer_config):
    if block == 'encoder':
        return Encoder(model_dim=transformer_config['model_dim'],
                       dropout=transformer_config['dropout'],
                       nhead=transformer_config['nhead'],
                       feedforward_dim=transformer_config['feedforward_dim'],
                       num_layers=transformer_config['num_layers'],
                       )
    elif block == 'decoder':
        return Decoder(model_dim=transformer_config['model_dim'],
                       dropout=transformer_config['dropout'],
                       nhead=transformer_config['nhead'],
                       feedforward_dim=transformer_config['feedforward_dim'],
                       num_layers=transformer_config['num_layers'],
                       )
    elif block == 'perceiver':
        return PerceiverBlock(
            num_layers=transformer_config['perceiver_layers'],
            latent_dim=transformer_config['model_dim'],
            context_dim=transformer_config['model_dim'],
            nhead=transformer_config['nhead'],
            n_self_att_layers=transformer_config['n_self_att_layers'],
            ff_dim=transformer_config['feedforward_dim'],
            dropout=transformer_config['dropout']
        )