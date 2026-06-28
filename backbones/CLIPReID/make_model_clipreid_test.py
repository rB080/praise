import os
import torch
import torch.nn as nn
import numpy as np
from .clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import clip
_tokenizer = _Tokenizer()
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

class CLIPWrapper(nn.Module):
    def __init__(self, size=(256, 128), stride_size=(16, 16), device="cuda"):
        super().__init__()
        self.h_resolution = int((size[0]-16)//stride_size[0] + 1)
        self.w_resolution = int((size[1]-16)//stride_size[1] + 1)
        self.vision_stride_size = stride_size[0]
        clip_model = load_clip_to_cpu("ViT-B-16", self.h_resolution, self.w_resolution, self.vision_stride_size)
        clip_model.to(device)
        self.clip_model = clip_model
        self.device = device

    def forward(self, x=None, text=None, batch_size=64):
        #breakpoint()
        if x is not None and text is not None:
            #   breakpoint()
            text = [text for i in range(batch_size)]
            text = clip.tokenize(text).to(self.device)
            text_feats = self.clip_model.encode_text(text)
            x_feats = self.clip_model.encode_image(x)
            x_feats = x_feats[-1][:,0]
            # normalized features
            x_feats = x_feats / x_feats.norm(dim=-1, keepdim=True)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
            logits = (self.clip_model.logit_scale * x_feats) * text_feats
            return logits
        if x is not None:
            return self.clip_model.encode_image(x)[-1][:,0]
        if text is not None:
            text = [text for i in range(batch_size)]
            text = clip.tokenize(text).to(self.device)
            return self.clip_model.encode_text(text)
        
        if x and text is None: return None


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts): 
        x = prompts + self.positional_embedding.type(self.dtype) 
        x = x.permute(1, 0, 2)  # NLD -> LND 
        x = self.transformer(x) 
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype) 

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection 
        return x

class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg):
        super(build_transformer, self).__init__()
        self.model_name = cfg.MODEL.NAME
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        if self.model_name == 'ViT-B-16':
            self.in_planes = 768
            self.in_planes_proj = 512
        elif self.model_name == 'RN50':
            self.in_planes = 2048
            self.in_planes_proj = 1024
        self.num_classes = num_classes
        self.camera_num = camera_num
        self.view_num = view_num
        self.sie_coe = cfg.MODEL.SIE_COE   

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.classifier_proj = nn.Linear(self.in_planes_proj, self.num_classes, bias=False)
        self.classifier_proj.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_proj = nn.BatchNorm1d(self.in_planes_proj)
        self.bottleneck_proj.bias.requires_grad_(False)
        self.bottleneck_proj.apply(weights_init_kaiming)

        self.h_resolution = int((cfg.INPUT.SIZE_TRAIN[0]-16)//cfg.MODEL.STRIDE_SIZE[0] + 1)
        self.w_resolution = int((cfg.INPUT.SIZE_TRAIN[1]-16)//cfg.MODEL.STRIDE_SIZE[1] + 1)
        self.vision_stride_size = cfg.MODEL.STRIDE_SIZE[0]
        path = self.model_name
        # path = '/export/livia/home/vision/Rbhattacharya/work/clipseg/weights/clipseg_weights/rd64-uni.pth'
        clip_model = load_clip_to_cpu(path, self.h_resolution, self.w_resolution, self.vision_stride_size)
        clip_model.to("cuda")

        self.image_encoder = clip_model.visual

        if cfg.MODEL.SIE_CAMERA and cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num * view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(camera_num))
        elif cfg.MODEL.SIE_CAMERA:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(camera_num))
        elif cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(view_num))

        dataset_name = cfg.DATASETS.NAMES
        self.prompt_learner = PromptLearner2(num_classes, camera_num, dataset_name, clip_model.dtype, clip_model.token_embedding)
        self.prompt_learner_og = PromptLearner(num_classes, dataset_name, clip_model.dtype, clip_model.token_embedding)
        self.cam_prompt_learner = PromptLearner_background(camera_num, dataset_name, clip_model.dtype, clip_model.token_embedding)
        self.text_encoder = TextEncoder(clip_model)

    def forward(self, x = None, label=None, get_image = False, get_text = False, cam_label= None, view_label=None, tta_module=None, tta=False, stage2=False):
        #breakpoint()
        if get_text == True:
            #prompts = self.prompt_learner(label, stage2=stage2) 
            if not stage2:
                prompts = self.prompt_learner(label[0], label[1], stage2=stage2)
                text_features = self.text_encoder(prompts, self.prompt_learner.tokenized_prompts)
            else:
                self.prompt_learner_og.cls_ctx = self.prompt_learner.cls_ctx
                prompts = self.prompt_learner_og(label, stage2=stage2)
                text_features = self.text_encoder(prompts, self.prompt_learner_og.tokenized_prompts)
            return text_features

        if get_image == True:
            image_features_last, image_features, image_features_proj = self.image_encoder(x) 
            if self.model_name == 'RN50':
                return image_features_proj[0]
            elif self.model_name == 'ViT-B-16':
                return image_features_proj[:,0]
        
        


        if self.model_name == 'RN50':
            image_features_last, image_features, image_features_proj = self.image_encoder(x) 
            img_feature_last = nn.functional.avg_pool2d(image_features_last, image_features_last.shape[2:4]).view(x.shape[0], -1) 
            img_feature = nn.functional.avg_pool2d(image_features, image_features.shape[2:4]).view(x.shape[0], -1) 
            img_feature_proj = image_features_proj[0]

        elif self.model_name == 'ViT-B-16':
            if cam_label != None and view_label!=None:
                cv_embed = self.sie_coe * self.cv_embed[cam_label * self.view_num + view_label]
            elif cam_label != None:
                cv_embed = self.sie_coe * self.cv_embed[cam_label]
            elif view_label!=None:
                cv_embed = self.sie_coe * self.cv_embed[view_label]
            else:
                cv_embed = None
            image_features_last, image_features, image_features_proj = self.image_encoder(x, cv_embed) 
            img_feature_last = image_features_last[:,0]
            img_feature = image_features[:,0]
            img_feature_proj = image_features_proj[:,0]

        if tta_module is not None:
            img_feature_proj = tta_module.image_adapter(img_feature_proj)

        feat = self.bottleneck(img_feature) 
        feat_proj = self.bottleneck_proj(img_feature_proj) 
        # breakpoint()
        if self.training:
            cls_score = self.classifier(feat)
            cls_score_proj = self.classifier_proj(feat_proj)
            return [cls_score, cls_score_proj], [img_feature_last, img_feature, img_feature_proj], img_feature_proj

        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return torch.cat([feat, feat_proj], dim=1)
            else:
                if not tta:
                    return torch.cat([img_feature, img_feature_proj], dim=1)
                else:
                    return torch.cat([img_feature, img_feature_proj], dim=1), [self.classifier(feat), self.classifier_proj(feat_proj)]

    def TTA_forward(self, x):
        _, image_features, image_features_proj = self.image_encoder(x) 
        if self.model_name == 'RN50':
            img_feat = image_features_proj[0]
        elif self.model_name == 'ViT-B-16':
            img_feat = image_features_proj[:,0]

        labels = torch.tensor([i for i in range(self.num_classes)]).int().to(x.device)
        cam_labels = torch.tensor([i for i in range(self.camera_num)]).int().to(x.device)
        prompts = self.prompt_learner(labels)
        cam_prompts = self.cam_prompt_learner(cam_labels)
        text_features = self.text_encoder(prompts, self.prompt_learner.tokenized_prompts)
        cam_text_features = self.text_encoder(cam_prompts, self.cam_prompt_learner.tokenized_prompts)
        IF = img_feat # / img_feat.norm(dim=-1, keepdim=True)
        TF = text_features # / text_features.norm(dim=-1, keepdim=True)
        CTF = cam_text_features # / cam_text_features.norm(dim=-1, keepdim=True)
        similarity = ((IF @ TF.t())).softmax(dim=-1)
        cam_similarity = ((IF @ CTF.t())).softmax(dim=-1)
        #breakpoint()
        return torch.cat([image_features[:,0], img_feat], dim=1), similarity, cam_similarity

    def use_image_encoder(self, x):
        _, _, image_features_proj = self.image_encoder(x) 
        return image_features_proj[:,0]
    
    def use_text_encoder(self, label):
        prompts = self.prompt_learner(label) 
        text_features = self.text_encoder(prompts, self.prompt_learner.tokenized_prompts)
        return text_features


    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        #breakpoint()
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))

def make_model(cfg, num_class, camera_num, view_num):
    model = build_transformer(num_class, camera_num, view_num, cfg)
    return model


from .clip import clip
def load_clip_to_cpu(backbone_name, h_resolution, w_resolution, vision_stride_size):
    if os.path.exists(backbone_name):
        model_path = backbone_name
    else:
        url = clip._MODELS[backbone_name]
        model_path = clip._download(url)
    # breakpoint()
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict(), h_resolution, w_resolution, vision_stride_size)

    return model

class PromptLearner(nn.Module):
    def __init__(self, num_class, dataset_name, dtype, token_embedding):
        super().__init__()
        if dataset_name == "VehicleID" or dataset_name == "veri":
            ctx_init = "A photo of a X X X X vehicle."
            #ctx_init = "A surveillance image of a X X X X vehicle."
        else:
            ctx_init = "A photo of a X X X X person."
            #ctx_init = "A photo of a X X X X person wearing Y Y Y Y clothes in a Z Z Z Z view."

            #ctx_init = "A surveillance image of a X X X X vehicle."
            # ctx_init = "A photo of a X X X X."
        #breakpoint()
        ctx_dim = 512
        # use given words to initialize context vectors
        ctx_init = ctx_init.replace("_", " ")
        n_ctx = 4
        
        tokenized_prompts = clip.tokenize(ctx_init).cuda() 
        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts).type(dtype) 
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor

        n_cls_ctx = 4
        cls_vectors = torch.empty(num_class, n_cls_ctx, ctx_dim, dtype=dtype) 
        nn.init.normal_(cls_vectors, std=0.02)
        self.cls_ctx = nn.Parameter(cls_vectors) 

        
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :n_ctx + 1, :])  
        self.register_buffer("token_suffix", embedding[:, n_ctx + 1 + n_cls_ctx: , :])  
        self.num_class = num_class
        self.n_cls_ctx = n_cls_ctx

    def forward(self, label, stage2=False):
        #breakpoint()
        cls_ctx = self.cls_ctx[label] 
        b = label.shape[0]
        prefix = self.token_prefix.expand(b, -1, -1) 
        suffix = self.token_suffix.expand(b, -1, -1) 
            
        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                cls_ctx,     # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        ) 

        return prompts 
    
class PromptLearner_background(nn.Module):
    def __init__(self, num_cams, dataset_name, dtype, token_embedding):
        super().__init__()
        if dataset_name == "VehicleID" or dataset_name == "veri":
            ctx_init = "A photo of a X X X X vehicle."
            #ctx_init = "A surveillance image of a X X X X vehicle."
        else:
            ctx_init = "A photo in a X X X X background."
            #ctx_init = "A photo of a X X X X person wearing Y Y Y Y clothes in a Z Z Z Z view."

            #ctx_init = "A surveillance image of a X X X X vehicle."
            # ctx_init = "A photo of a X X X X."
        #breakpoint()
        ctx_dim = 512
        # use given words to initialize context vectors
        ctx_init = ctx_init.replace("_", " ")
        n_ctx = 4
        
        tokenized_prompts = clip.tokenize(ctx_init).cuda() 
        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts).type(dtype) 
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor

        n_cls_ctx = 4
        cls_vectors = torch.empty(num_cams, n_cls_ctx, ctx_dim, dtype=dtype) 
        nn.init.normal_(cls_vectors, std=0.02)
        self.cls_ctx = nn.Parameter(cls_vectors) 

        
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :n_ctx + 1, :])  
        self.register_buffer("token_suffix", embedding[:, n_ctx + 1 + n_cls_ctx: , :])  
        self.num_cams = num_cams
        self.n_cls_ctx = n_cls_ctx

    def forward(self, label, stage2=False):
        #breakpoint()
        cls_ctx = self.cls_ctx[label] 
        b = label.shape[0]
        prefix = self.token_prefix.expand(b, -1, -1) 
        suffix = self.token_suffix.expand(b, -1, -1) 
            
        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                cls_ctx,     # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        ) 

        return prompts 


class PromptLearner2(nn.Module):
    def __init__(self, num_class, num_cams, dataset_name, dtype, token_embedding):
        super().__init__()
        if dataset_name == "VehicleID" or dataset_name == "veri":
            ctx_init = "A photo of a X X X X vehicle."
            #ctx_init = "A surveillance image of a X X X X vehicle."
        else:
            # ctx_init = "A photo of a X X X X person."
            ctx_init = "A photo of a X X X X person in a Y Y Y Y background."

            #ctx_init = "A surveillance image of a X X X X vehicle."
            # ctx_init = "A photo of a X X X X."
        #breakpoint()
        ctx_dim = 512
        # use given words to initialize context vectors
        ctx_init = ctx_init.replace("_", " ")
        n_ctx = 4
        
        tokenized_prompts = clip.tokenize(ctx_init).cuda() 
        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts).type(dtype) 
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor

        n_cls_ctx = 4
        cls_vectors = torch.empty(num_class, n_cls_ctx, ctx_dim, dtype=dtype) 
        nn.init.normal_(cls_vectors, std=0.02)
        self.cls_ctx = nn.Parameter(cls_vectors) 
        cls_vectors = torch.empty(num_cams, n_cls_ctx, ctx_dim, dtype=dtype) 
        nn.init.normal_(cls_vectors, std=0.02)
        self.cls_ctx2 = nn.Parameter(cls_vectors) 
        # cls_vectors = torch.empty(num_class, n_cls_ctx, ctx_dim, dtype=dtype) 
        # nn.init.normal_(cls_vectors, std=0.02)
        # self.cls_ctx3 = nn.Parameter(cls_vectors) 

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :n_ctx + 1, :])  
        self.register_buffer("token_suffix_1", embedding[:, n_ctx + 1 + n_cls_ctx:n_ctx + 3 + n_cls_ctx , :])  
        #self.register_buffer("token_suffix_2", embedding[:, n_ctx + 3 + 2*n_cls_ctx:n_ctx + 6 + 2*n_cls_ctx , :])  
        self.register_buffer("token_suffix", embedding[:, n_ctx + 3 + 2*n_cls_ctx: , :])  
        # breakpoint()
        self.num_class = num_class
        self.n_cls_ctx = n_cls_ctx

    def forward(self, label, cam_label, stage2=False):
        #breakpoint()
        cls_ctx = self.cls_ctx[label] 
        cls_ctx2 = self.cls_ctx2[cam_label] 
        #cls_ctx3 = self.cls_ctx3[label] 
        b = label.shape[0]
        prefix = self.token_prefix.expand(b, -1, -1) 
        suffix = self.token_suffix.expand(b, -1, -1) 
        suffix_1 = self.token_suffix_1.expand(b, -1, -1) 
        #suffix_2 = self.token_suffix_2.expand(b, -1, -1) 

        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                cls_ctx, # (n_cls, n_ctx, dim)
                suffix_1,
                cls_ctx2,
                #suffix_2,
                #cls_ctx3,
                suffix  # (n_cls, *, dim)
            ],
            dim=1,
        )

        return prompts

class TestTimeAdapter(nn.Module):
    def __init__(self, dtype, token_embedding, num_classes=3060, dims=1280):
        super().__init__()
        self.prompt_learner = PromptLearner(num_class=num_classes, dataset_name="msmt17", dtype=dtype, token_embedding=token_embedding)
        self.text_adapter = nn.Sequential(
            nn.Linear(dims, dims), nn.LayerNorm(dims), nn.Linear(dims, dims)
        )
        self.image_adapter = nn.Sequential(
            nn.Linear(dims, dims), nn.LayerNorm(dims), nn.Linear(dims, dims)
        )