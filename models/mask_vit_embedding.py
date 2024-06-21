import numpy as np
import copy
import torch
from typing import Union, Optional, Tuple
from transformers import ViTForImageClassification, ViTConfig, ViTModel
from transformers.models.vit.modeling_vit import ViTSelfAttention, ViTAttention, ViTEncoder, ViTLayer, ViTModel, ViTEmbeddings, BaseModelOutputWithPooling, ImageClassifierOutput
from transformers.modeling_outputs import BaseModelOutput
from torch import nn
import math

def MoXIViTForward(
    self,
    pixel_values: Optional[torch.Tensor] = None,
    embedding_mask: Optional[torch.BoolTensor] = None,
    head_mask: Optional[torch.Tensor] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    interpolate_pos_encoding: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    target_mask_layer = None,
    mask_list = None,
) -> Union[Tuple, BaseModelOutputWithPooling]:
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
        pixel_values, embedding_mask=embedding_mask, interpolate_pos_encoding=interpolate_pos_encoding
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

def MoXIViTForImageClassificationForward(
    self,
    pixel_values: Optional[torch.Tensor] = None,
    embedding_mask: Optional[torch.BoolTensor] = None,
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
        embedding_mask=embedding_mask,
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

def MoXIEmbeddingsForward(
    self,
    pixel_values: torch.Tensor,
    embedding_mask: Optional[torch.BoolTensor] = None,
    interpolate_pos_encoding: bool = False,
) -> torch.Tensor:
    batch_size, num_channels, height, width = pixel_values.shape
    embeddings = self.patch_embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
    # add position embeddings w/o cls token
    embeddings = embeddings + self.position_embeddings[:, 1:, :]
    if embedding_mask is not None:
        embeddings = select_batch_removing(embeddings, embedding_mask)
    # add positional embeddings to cls token
    cls_tokens = self.cls_token.expand(batch_size, -1, -1) + self.position_embeddings[:, :1, :]
    # concatenate cls token and embeddings
    embeddings = torch.cat((cls_tokens, embeddings), dim=1)
    embeddings = self.dropout(embeddings)

    return embeddings

def select_batch_removing(x, mask_list):

    x_masked_list = []
    for index, mask in enumerate(mask_list):
        mask = torch.tensor(mask, dtype=torch.int64).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        x_masked = torch.index_select(x[index:index+1, :, :], 1, mask)
        x_masked_list.append(x_masked)
    x_masked = torch.cat(x_masked_list, dim=0)
    del mask
    return x_masked