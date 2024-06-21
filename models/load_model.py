from transformers import ViTForImageClassification, ViTImageProcessor, ViTConfig, ViTModel, ResNetForImageClassification, AutoImageProcessor
from transformers.models.vit.modeling_vit import ViTEmbeddings,  ViTSelfAttention, ViTAttention, ViTEncoder, ViTLayer
from .mask_vit_embedding import MyViTForward, MyEmbeddingsForward, MyViTForImageClassificationForward, MyViTSelfAttentionforward, MyViTEncoderForward, MyViTAttentionForward, MyViTLayerForward
import os

# load_model
def load_model(args):
    if args.checkpoint_dir is not None:
        config_path = os.path.join(args.checkpoint_dir, 'config.json')
        processor_path = os.path.join(args.checkpoint_dir, 'preprocessor_config.json')
        model_path = os.path.join(args.checkpoint_dir, 'pytorch_model.bin')
        config = ViTConfig.from_pretrained(config_path)
        model = ViTForImageClassification.from_pretrained(model_path, config=config)
        image_processor = ViTImageProcessor.from_pretrained(processor_path)
    else:
        checkpoint = load_checkpoint(args)
        if args.model_name == 'vit-b' or args.model_name == 'vit-t' or args.model_name == 'vit-s' or args.model_name == 'deit-b' or args.model_name == 'deit-t' or args.model_name == 'deit-s':
            model = ViTForImageClassification.from_pretrained(checkpoint)
            image_processor = ViTImageProcessor.from_pretrained(checkpoint)
        elif args.model_name == 'resnet':
            model = ResNetForImageClassification.from_pretrained(checkpoint)
            image_processor = AutoImageProcessor.from_pretrained(checkpoint)

    assert 'classifier.weight' in model.state_dict(), 'Model is not fully trained.'
    if args.isTraining:
        model.train()
    else:
        model.eval()
    model.to(args.device)
    return model, image_processor

# when using the mask method with a removal patch, vit_embedding is replaced.
def replace_vit_embedding_mask(args, model):
    if args.interaction_method == 'vit_embedding' and not args.isUsedTargetLayer:
        print('vit embedding')
        print('Target Layer is not used')
        model.vit.embeddings.forward = MyEmbeddingsForward.__get__(model.vit.embeddings, ViTEmbeddings)
        model.vit.forward = MyViTForward.__get__(model.vit, ViTModel)
        model.forward = MyViTForImageClassificationForward.__get__(model, ViTForImageClassification)
    elif args.interaction_method == 'vit_embedding' and args.isUsedTargetLayer:
        print('vit embedding')
        print('Target Layer is used')
        model.vit.encoder.forward = MyViTEncoderForward.__get__(model.vit.encoder, ViTEncoder)
        model.vit.forward = MyViTForward.__get__(model.vit, ViTModel)
        model.forward = MyViTForImageClassificationForward.__get__(model, ViTForImageClassification)
        print(f'all layer numbers: {len(model.vit.encoder.layer)}')
        for index in range(len(model.vit.encoder.layer)):
            model.vit.encoder.layer[index].attention.attention.forward = MyViTSelfAttentionforward.__get__(model.vit.encoder.layer[index].attention.attention, ViTSelfAttention)
            model.vit.encoder.layer[index].attention.forward = MyViTAttentionForward.__get__(model.vit.encoder.layer[index].attention, ViTAttention)
            model.vit.encoder.layer[index].forward = MyViTLayerForward.__get__(model.vit.encoder.layer[index], ViTLayer)
    elif args.interaction_method == 'pixel_zero_input':
        print('pixel zero input')
        pass
    return model

# load model checkpoint
def load_checkpoint(args):
    if args.model_name == 'vit-b':
        checkpoint = 'google/vit-base-patch16-224'
    elif args.model_name == 'vit-t':
        checkpoint = 'WinKawaks/vit-tiny-patch16-224'
    elif args.model_name == 'vit-s':
        checkpoint = 'WinKawaks/vit-small-patch16-224'
    elif args.model_name == 'deit-b':
        checkpoint = "facebook/deit-base-patch16-224"
    elif args.model_name == 'deit-t':
        checkpoint = "facebook/deit-tiny-patch16-224"
    elif args.model_name == 'deit-s':
        checkpoint = "facebook/deit-small-patch16-224"
    elif args.model_name == 'resnet':
        checkpoint = "microsoft/resnet-18"

    return checkpoint