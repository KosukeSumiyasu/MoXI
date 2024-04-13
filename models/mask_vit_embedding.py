import numpy as np
import copy
import torch
from typing import Union, Optional, Tuple
from transformers import ViTForImageClassification, ViTConfig, ViTModel
from transformers.models.vit.modeling_vit import ViTSelfAttention, ViTAttention, ViTEncoder, ViTLayer, ViTModel, ViTEmbeddings, BaseModelOutputWithPooling, ImageClassifierOutput
from transformers.modeling_outputs import BaseModelOutput
from torch import nn
import math

def MyViTForward(
    self,
    pixel_values: Optional[torch.Tensor] = None,
    bool_masked_pos: Optional[torch.BoolTensor] = None,
    head_mask: Optional[torch.Tensor] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    interpolate_pos_encoding: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    target_mask_layer = None,
    mask_list = None,
) -> Union[Tuple, BaseModelOutputWithPooling]:
    r"""
    bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`, *optional*):
        Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
    """
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if pixel_values is None:
        raise ValueError("You have to specify pixel_values")

    # Prepare head mask if needed
    # 1.0 in head_mask indicate we keep the head
    # attention_probs has shape bsz x n_heads x N x N
    # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
    # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
    head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

    # TODO: maybe have a cleaner way to cast the input (from `ImageProcessor` side?)
    expected_dtype = self.embeddings.patch_embeddings.projection.weight.dtype
    if pixel_values.dtype != expected_dtype:
        pixel_values = pixel_values.to(expected_dtype)

    embedding_output = self.embeddings(
        pixel_values, bool_masked_pos=bool_masked_pos, interpolate_pos_encoding=interpolate_pos_encoding
    )

    if mask_list is not None and target_mask_layer is not None:
        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            mask_list = mask_list,
            target_mask_layer = target_mask_layer,
        )
    else:
        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
    sequence_output = encoder_outputs[0]
    sequence_output = self.layernorm(sequence_output)
    pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

    if not return_dict:
        head_outputs = (sequence_output, pooled_output) if pooled_output is not None else (sequence_output,)
        return head_outputs + encoder_outputs[1:]

    return BaseModelOutputWithPooling(
        last_hidden_state=sequence_output,
        pooler_output=pooled_output,
        hidden_states=encoder_outputs.hidden_states,
        attentions=encoder_outputs.attentions,
    )

def MyViTForImageClassificationForward(
    self,
    pixel_values: Optional[torch.Tensor] = None,
    bool_masked_pos: Optional[torch.BoolTensor] = None,
    head_mask: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    interpolate_pos_encoding: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    target_mask_layer = None,
    mask_list = None,
) -> Union[tuple, ImageClassifierOutput]:
    r"""
    labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
        Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
        config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
        `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
    """
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    outputs = self.vit(
        pixel_values,
        bool_masked_pos=bool_masked_pos,
        head_mask=head_mask,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        interpolate_pos_encoding=interpolate_pos_encoding,
        return_dict=return_dict,
        mask_list = mask_list,
        target_mask_layer = target_mask_layer,
    )

    sequence_output = outputs[0]

    logits = self.classifier(sequence_output[:, 0, :])

    loss = None
    if labels is not None:
        # move labels to correct device to enable model parallelism
        labels = labels.to(logits.device)
        if self.config.problem_type is None:
            if self.num_labels == 1:
                self.config.problem_type = "regression"
            elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                self.config.problem_type = "single_label_classification"
            else:
                self.config.problem_type = "multi_label_classification"

        if self.config.problem_type == "regression":
            loss_fct = MSELoss()
            if self.num_labels == 1:
                loss = loss_fct(logits.squeeze(), labels.squeeze())
            else:
                loss = loss_fct(logits, labels)
        elif self.config.problem_type == "single_label_classification":
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        elif self.config.problem_type == "multi_label_classification":
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return ((loss,) + output) if loss is not None else output

    return ImageClassifierOutput(
        loss=loss,
        logits=logits,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )

def MyEmbeddingsForward(
    self,
    pixel_values: torch.Tensor,
    bool_masked_pos: Optional[torch.BoolTensor] = None,
    interpolate_pos_encoding: bool = False,
) -> torch.Tensor:
    batch_size, num_channels, height, width = pixel_values.shape
    embeddings = self.patch_embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
    # add position embeddings w/o cls token
    embeddings = embeddings + self.position_embeddings[:, 1:, :]
    if bool_masked_pos is not None:
        # embeddings, position_embeddings = select_masking(embeddings, self.position_embeddings, bool_masked_pos)
        # embeddings, position_embeddings = select_masking_batch(embeddings, self.position_embeddings, bool_masked_pos)
        embeddings = select_batch_removing(embeddings, bool_masked_pos)
    # add positional embeddings to cls token
    cls_tokens = self.cls_token.expand(batch_size, -1, -1) + self.position_embeddings[:, :1, :]
    # concatenate cls token and embeddings
    embeddings = torch.cat((cls_tokens, embeddings), dim=1)

    # if interpolate_pos_encoding:
    #     embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)

    embeddings = self.dropout(embeddings)

    return embeddings

def MyViTSelfAttentionforward(
    self, hidden_states, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False, mask_list = None
) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
    mixed_query_layer = self.query(hidden_states)

    key_layer = self.transpose_for_scores(self.key(hidden_states))
    value_layer = self.transpose_for_scores(self.value(hidden_states))
    query_layer = self.transpose_for_scores(mixed_query_layer)
    # Take the dot product between "query" and "key" to get the raw attention scores.
    attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    if mask_list is not None:

        B, H, _, _ = query_layer.shape 
        new_mask = torch.zeros(B, 197)
        # attn = torch.randn(, 3, 197, 197)
        attn_permute = attention_scores.permute((1, 0, 3, 2))
        for index, mask in enumerate(mask_list):
            mask = np.array(mask) + 1 # class token shift
            mask = np.append(mask, 0) # add cls token
            new_mask[index, mask] = 1
        attn_permute[:, new_mask != 1] = -torch.tensor(float('inf')).type(attn_permute.dtype)
        attention_scores = attn_permute.permute((1, 0, 3, 2))

    attention_scores = attention_scores / math.sqrt(self.attention_head_size)

    # Normalize the attention scores to probabilities.
    attention_probs = nn.functional.softmax(attention_scores, dim=-1)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attention_probs = self.dropout(attention_probs)

    # Mask heads if we want to
    if head_mask is not None:
        attention_probs = attention_probs * head_mask

    context_layer = torch.matmul(attention_probs, value_layer)

    context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
    context_layer = context_layer.view(new_context_layer_shape)

    outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
    return outputs

def MyViTEncoderForward(
    self,
    hidden_states: torch.Tensor,
    head_mask: Optional[torch.Tensor] = None,
    output_attentions: bool = False,
    output_hidden_states: bool = False,
    return_dict: bool = True,
    mask_list = None,
    target_mask_layer = None,
) -> Union[tuple, BaseModelOutput]:
    all_hidden_states = () if output_hidden_states else None
    all_self_attentions = () if output_attentions else None
    flag = False

    for i, layer_module in enumerate(self.layer):

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        layer_head_mask = head_mask[i] if head_mask is not None else None
        
        if self.gradient_checkpointing and self.training:
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs, output_attentions)

                return custom_forward

            layer_outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(layer_module),
                hidden_states,
                layer_head_mask,
            )
        else:
            if i == target_mask_layer or flag == True:
                flag = True
                layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions, mask_list=mask_list)
            else: 
                layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions)

            hidden_states = layer_outputs[0]

        if output_attentions:
            all_self_attentions = all_self_attentions + (layer_outputs[1],)

    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

    if not return_dict:
        return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
    return BaseModelOutput(
        last_hidden_state=hidden_states,
        hidden_states=all_hidden_states,
        attentions=all_self_attentions,
    )

def MyViTAttentionForward(
    self,
    hidden_states: torch.Tensor,
    head_mask: Optional[torch.Tensor] = None,
    output_attentions: bool = False,
    mask_list = None,
) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
    self_outputs = self.attention(hidden_states, head_mask, output_attentions, mask_list=mask_list)

    attention_output = self.output(self_outputs[0], hidden_states)

    outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
    return outputs

def MyViTLayerForward(
    self,
    hidden_states: torch.Tensor,
    head_mask: Optional[torch.Tensor] = None,
    output_attentions: bool = False,
    mask_list = None,
) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
    self_attention_outputs = self.attention(
        self.layernorm_before(hidden_states),  # in ViT, layernorm is applied before self-attention
        head_mask,
        output_attentions=output_attentions,
        mask_list = mask_list,
    )
    attention_output = self_attention_outputs[0]
    outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

    # first residual connection
    hidden_states = attention_output + hidden_states

    # in ViT, layernorm is also applied after self-attention
    layer_output = self.layernorm_after(hidden_states)
    layer_output = self.intermediate(layer_output)

    # second residual connection is done here
    layer_output = self.output(layer_output, hidden_states)

    outputs = (layer_output,) + outputs

    return outputs

def select_masking(x, position_embeddings, mask):
    mask = torch.tensor(mask).cuda(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    x_masked = torch.index_select(x, 1, mask)

    # 位置エンコーディングをマスクする
    position_embeddings_masked = torch.index_select(position_embeddings[:, 1:, :], 1, mask)
    cls_token = position_embeddings[:, 0:1, :]
    new_position_embeddings = torch.cat((cls_token, position_embeddings_masked), dim=1)
    del mask, position_embeddings_masked, cls_token
    return x_masked, new_position_embeddings

def select_batch_removing(x, mask_list):

    x_masked_list = []
    for index, mask in enumerate(mask_list):
        mask = torch.tensor(mask).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        x_masked = torch.index_select(x[index:index+1, :, :], 1, mask)
        x_masked_list.append(x_masked)
    x_masked = torch.cat(x_masked_list, dim=0)
    del mask
    return x_masked

def select_masking_batch(x, position_embeddings, mask_list):
    x_masked_list = []
    new_position_embeddings_list = []
    if not mask_list:
        return x, position_embeddings
    else:
        cls_token = position_embeddings[:, 0:1, :]
        for mask in mask_list:
            mask = torch.tensor(mask).cuda(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
            x_masked = torch.index_select(x, 1, mask)

            # 位置エンコーディングをマスクする
            position_embeddings_masked = torch.index_select(position_embeddings[:, 1:, :], 1, mask)
            print("cls_token:", cls_token.shape)
            print("embeddings:", position_embeddings.shape)
            print(position_embeddings_masked.shape)
            new_position_embeddings = torch.cat((cls_token, position_embeddings_masked), dim=1)
            x_masked_list.append(x_masked)
            new_position_embeddings_list.append(new_position_embeddings)
        x_masked = torch.cat(x_masked_list, dim=0)
        new_position_embeddings = torch.cat(new_position_embeddings_list, dim=0)
        del mask, position_embeddings_masked, cls_token
        return x_masked, new_position_embeddings