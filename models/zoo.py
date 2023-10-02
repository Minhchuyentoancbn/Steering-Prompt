import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import numpy as np
from .vit import VisionTransformer
from sklearn.cluster import SpectralClustering


class CodaPrompt(nn.Module):
    def __init__(self, emb_d, n_tasks, prompt_param, key_dim=768):
        super().__init__()
        self.task_count = 0
        self.emb_d = emb_d
        self.key_d = key_dim
        self.n_tasks = n_tasks
        self._init_smart(emb_d, prompt_param)

        # e prompt init
        for e in self.e_layers:
            # for model saving/loading simplicity, we init the full paramaters here
            # however, please note that we reinit the new components at each task
            # in the "spirit of continual learning", as we don't know how many tasks
            # we will encounter at the start of the task sequence
            #
            # in the original paper, we used ortho init at the start - this modification is more 
            # fair in the spirit of continual learning and has little affect on performance
            e_l = self.e_p_length
            p = tensor_prompt(self.e_pool_size, e_l, emb_d)
            k = tensor_prompt(self.e_pool_size, self.key_d)
            a = tensor_prompt(self.e_pool_size, self.key_d)
            p = self.gram_schmidt(p)
            k = self.gram_schmidt(k)
            a = self.gram_schmidt(a)
            setattr(self, f'e_p_{e}',p)
            setattr(self, f'e_k_{e}',k)
            setattr(self, f'e_a_{e}',a)

    def _init_smart(self, emb_d, prompt_param):

        # prompt basic param
        self.e_pool_size = int(prompt_param[0])
        self.e_p_length = int(prompt_param[1])
        self.e_layers = [0,1,2,3,4]

        # strenth of ortho penalty
        self.ortho_mu = prompt_param[2]
        
    def process_task_count(self):
        self.task_count += 1

        # in the spirit of continual learning, we will reinit the new components
        # for the new task with Gram Schmidt
        #
        # in the original paper, we used ortho init at the start - this modification is more 
        # fair in the spirit of continual learning and has little affect on performance
        # 
        # code for this function is modified from:
        # https://github.com/legendongary/pytorch-gram-schmidt/blob/master/gram_schmidt.py
        for e in self.e_layers:
            K = getattr(self,f'e_k_{e}')
            A = getattr(self,f'e_a_{e}')
            P = getattr(self,f'e_p_{e}')
            k = self.gram_schmidt(K)
            a = self.gram_schmidt(A)
            p = self.gram_schmidt(P)
            setattr(self, f'e_p_{e}',p)
            setattr(self, f'e_k_{e}',k)
            setattr(self, f'e_a_{e}',a)

    # code for this function is modified from:
    # https://github.com/legendongary/pytorch-gram-schmidt/blob/master/gram_schmidt.py
    def gram_schmidt(self, vv):

        def projection(u, v):
            denominator = (u * u).sum()

            if denominator < 1e-8:
                return None
            else:
                return (v * u).sum() / denominator * u

        # check if the tensor is 3D and flatten the last two dimensions if necessary
        is_3d = len(vv.shape) == 3
        if is_3d:
            shape_2d = copy.deepcopy(vv.shape)
            vv = vv.view(vv.shape[0],-1)

        # swap rows and columns
        vv = vv.T

        # process matrix size
        nk = vv.size(1)
        uu = torch.zeros_like(vv, device=vv.device)

        # get starting point
        pt = int(self.e_pool_size / (self.n_tasks))
        s = int(self.task_count * pt)
        f = int((self.task_count + 1) * pt)
        if s > 0:
            uu[:, 0:s] = vv[:, 0:s].clone()
        for k in range(s, f):
            redo = True
            while redo:
                redo = False
                vk = torch.randn_like(vv[:,k]).to(vv.device)
                uk = 0
                for j in range(0, k):
                    if not redo:
                        uj = uu[:, j].clone()
                        proj = projection(uj, vk)
                        if proj is None:
                            redo = True
                            print('restarting!!!')
                        else:
                            uk = uk + proj
                if not redo: uu[:, k] = vk - uk
        for k in range(s, f):
            uk = uu[:, k].clone()
            uu[:, k] = uk / (uk.norm())

        # undo swapping of rows and columns
        uu = uu.T 

        # return from 2D
        if is_3d:
            uu = uu.view(shape_2d)
        
        return torch.nn.Parameter(uu) 

    def forward(self, x_querry, l, x_block, train=False, task_id=None):

        # e prompts
        e_valid = False
        if l in self.e_layers:
            e_valid = True
            B, C = x_querry.shape

            K = getattr(self,f'e_k_{l}')
            A = getattr(self,f'e_a_{l}')
            p = getattr(self,f'e_p_{l}')
            pt = int(self.e_pool_size / (self.n_tasks))
            s = int(self.task_count * pt)
            f = int((self.task_count + 1) * pt)
            
            # freeze/control past tasks
            if train:
                if self.task_count > 0:
                    K = torch.cat((K[:s].detach().clone(),K[s:f]), dim=0)
                    A = torch.cat((A[:s].detach().clone(),A[s:f]), dim=0)
                    p = torch.cat((p[:s].detach().clone(),p[s:f]), dim=0)
                else:
                    K = K[s:f]
                    A = A[s:f]
                    p = p[s:f]
            else:
                K = K[0:f]
                A = A[0:f]
                p = p[0:f]

            # with attention and cosine sim
            # (b x 1 x d) * soft([1 x k x d]) = (b x k x d) -> attention = k x d
            a_querry = torch.einsum('bd,kd->bkd', x_querry, A)
            # # (b x k x d) - [1 x k x d] = (b x k) -> key = k x d
            n_K = nn.functional.normalize(K, dim=1)
            q = nn.functional.normalize(a_querry, dim=2)
            aq_k = torch.einsum('bkd,kd->bk', q, n_K)
            # (b x 1 x k x 1) * [1 x plen x k x d] = (b x plen x d) -> prompt = plen x k x d
            P_ = torch.einsum('bk,kld->bld', aq_k, p)

            # select prompts
            i = int(self.e_p_length/2)
            Ek = P_[:,:i,:]
            Ev = P_[:,i:,:]

            # ortho penalty
            if train and self.ortho_mu > 0:
                loss = ortho_penalty(K) * self.ortho_mu
                loss += ortho_penalty(A) * self.ortho_mu
                loss += ortho_penalty(p.view(p.shape[0], -1)) * self.ortho_mu
            else:
                loss = 0
        else:
            loss = 0

        # combine prompts for prefix tuning
        if e_valid:
            p_return = [Ek, Ev]
        else:
            p_return = None

        # return
        return p_return, loss, x_block

def ortho_penalty(t):
    return ((t @t.T - torch.eye(t.shape[0]).cuda())**2).mean()

# @article{wang2022dualprompt,
#   title={DualPrompt: Complementary Prompting for Rehearsal-free Continual Learning},
#   author={Wang, Zifeng and Zhang, Zizhao and Ebrahimi, Sayna and Sun, Ruoxi and Zhang, Han and Lee, Chen-Yu and Ren, Xiaoqi and Su, Guolong and Perot, Vincent and Dy, Jennifer and others},
#   journal={European Conference on Computer Vision},
#   year={2022}
# }
class DualPrompt(nn.Module):
    def __init__(self, emb_d, n_tasks, prompt_param, key_dim=768):
        super().__init__()
        self.task_count = 0
        self.emb_d = emb_d
        self.key_d = key_dim
        self.n_tasks = n_tasks
        self._init_smart(emb_d, prompt_param)

        # g prompt init
        for g in self.g_layers:
            p = tensor_prompt(self.g_p_length, emb_d)
            setattr(self, f'g_p_{g}',p)

        # e prompt init
        for e in self.e_layers:
            p = tensor_prompt(self.e_pool_size, self.e_p_length, emb_d)
            k = tensor_prompt(self.e_pool_size, self.key_d)
            setattr(self, f'e_p_{e}',p)
            setattr(self, f'e_k_{e}',k)

    def _init_smart(self, emb_d, prompt_param):
        
        self.top_k = 1
        self.task_id_bootstrap = True

        # prompt locations
        self.g_layers = [0,1]
        self.e_layers = [2,3,4]

        # prompt pool size
        self.g_p_length = int(prompt_param[2])
        self.e_p_length = int(prompt_param[1])
        self.e_pool_size = int(prompt_param[0])

    def process_task_count(self):
        self.task_count += 1

    def forward(self, x_querry, l, x_block, train=False, task_id=None):

        # e prompts
        e_valid = False
        if l in self.e_layers:
            e_valid = True
            B, C = x_querry.shape
            K = getattr(self,f'e_k_{l}') # 0 based indexing here
            p = getattr(self,f'e_p_{l}') # 0 based indexing here
            
            # cosine similarity to match keys/querries
            n_K = nn.functional.normalize(K, dim=1)
            q = nn.functional.normalize(x_querry, dim=1).detach()
            cos_sim = torch.einsum('bj,kj->bk', q, n_K)
            
            if train:
                # dual prompt during training uses task id
                if self.task_id_bootstrap:
                    loss = (1.0 - cos_sim[:,task_id]).sum()
                    P_ = p[task_id].expand(len(x_querry),-1,-1)
                else:
                    top_k = torch.topk(cos_sim, self.top_k, dim=1)
                    k_idx = top_k.indices
                    loss = (1.0 - cos_sim[:,k_idx]).sum()
                    P_ = p[k_idx]
            else:
                top_k = torch.topk(cos_sim, self.top_k, dim=1)
                k_idx = top_k.indices
                P_ = p[k_idx]
                
            # select prompts
            if train and self.task_id_bootstrap:
                i = int(self.e_p_length / 2)
                Ek = P_[:,:i,:].reshape((B,-1,self.emb_d))
                Ev = P_[:,i:,:].reshape((B,-1,self.emb_d))
            else:
                i = int(self.e_p_length/2)
                Ek = P_[:,:,:i,:].reshape((B,-1,self.emb_d))
                Ev = P_[:,:,i:,:].reshape((B,-1,self.emb_d))
        
        # g prompts
        g_valid = False
        if l in self.g_layers:
            g_valid = True
            j = int(self.g_p_length/2)
            p = getattr(self,f'g_p_{l}') # 0 based indexing here
            P_ = p.expand(len(x_querry),-1,-1)
            Gk = P_[:,:j,:]
            Gv = P_[:,j:,:]

        # combine prompts for prefix tuning
        if e_valid and g_valid:
            Pk = torch.cat((Ek, Gk), dim=1)
            Pv = torch.cat((Ev, Gv), dim=1)
            p_return = [Pk, Pv]
        elif e_valid:
            p_return = [Ek, Ev]
        elif g_valid:
            p_return = [Gk, Gv]
            loss = 0
        else:
            p_return = None
            loss = 0

        # return
        if train:
            return p_return, loss, x_block
        else:
            return p_return, 0, x_block

# @inproceedings{wang2022learning,
#   title={Learning to prompt for continual learning},
#   author={Wang, Zifeng and Zhang, Zizhao and Lee, Chen-Yu and Zhang, Han and Sun, Ruoxi and Ren, Xiaoqi and Su, Guolong and Perot, Vincent and Dy, Jennifer and Pfister, Tomas},
#   booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
#   pages={139--149},
#   year={2022}
# }
class L2P(DualPrompt):
    def __init__(self, emb_d, n_tasks, prompt_param, key_dim=768):
        super().__init__(emb_d, n_tasks, prompt_param, key_dim)

    def _init_smart(self, emb_d, prompt_param):
        self.top_k = 5
        self.task_id_bootstrap = False

        # prompt locations
        self.g_layers = []
        if prompt_param[2] > 0:
            self.e_layers = [0,1,2,3,4]
        else:
            self.e_layers = [0]

        # prompt pool size
        self.g_p_length = -1
        self.e_p_length = int(prompt_param[1])
        self.e_pool_size = int(prompt_param[0])


class CPP(nn.Module):
    """
    Contrastive Prototypical Prompt
    """
    def __init__(self, emb_d, n_tasks, prompt_param, key_dim=768):
        super().__init__()
        self.task_count = 0
        self.emb_d = emb_d
        self.key_d = key_dim
        self.n_tasks = n_tasks
        self._init_smart(emb_d, prompt_param)

        self.e_layers = np.arange(12)  # Default: all layers, Deep Prompt
        for e in self.e_layers:
            e_l = self.e_p_length
            p = tensor_prompt(self.e_pool_size, e_l, emb_d)
            setattr(self, f'e_p_{e}', p)

        self.key_prototypes = torch.zeros(self.num_centroids * self.num_classes, self.key_d)
        self.num_cls_per_task = self.num_classes // self.n_tasks
        # Task with respect to each key prototype
        self.key_task = torch.arange(self.n_tasks).repeat_interleave(self.num_centroids * self.num_cls_per_task)

        # A 3-layer MLP head
        self.mlp_head = nn.Sequential(
            nn.Linear(768, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 768)
        )


    def _init_smart(self, emb_d, prompt_param):
        # Prompt basic param
        self.e_p_length = 1
        self.e_pool_size = int(prompt_param[0])
        assert self.e_pool_size == self.n_tasks, "Number of prompts must be equal to number of tasks"
        self.num_centroids = int(prompt_param[1])
        self.num_neighbours = int(prompt_param[2])
        self.num_classes = int(prompt_param[3])

    def reset_head(self):
        for m in self.mlp_head:
            if isinstance(m, nn.Linear):
                m.reset_parameters()

    def process_task_count(self):
        self.task_count += 1
        # Reset the head
        self.reset_head()

    def forward(self, x_querry, l, x_block, train=False, task_id=None):
        if l in self.e_layers:
            B, C = x_querry.shape
            p = getattr(self, f'e_p_{l}') # 0 based indexing here
            P_ = p[task_id].expand(B, -1, -1)  # 0 based indexing here
            P_ = P_.reshape((B, -1, self.emb_d))
            p_return = [P_, ]  # Prompt tuning
            # return
            return p_return, 0, x_block
        else:
            raise ValueError("The layer index is not valid")

    def compute_centroids(self, X, y):
        """
        Compute the centroids of each class using spectral clustering

        Arguments:
        ----------
        X: torch.Tensor
            The input tensor, shape (N, d_k)

        y: int
            The label of the input
        """
        assert X.shape[1] == self.key_d, "The input dimension must be equal to the key dimension"

        X = F.normalize(X, dim=1)

        # Compute the cosine similarity matrix and set the diagonal to 0
        S = (torch.mm(X, X.t()) + 1) / 2 - torch.eye(X.shape[0], device=X.device)

        clustering = SpectralClustering(
            n_clusters=self.num_centroids, affinity='precomputed',
            random_state=42, n_jobs=-1
        ).fit(S.cpu().numpy())
        cluster = clustering.labels_

        for i in range(self.num_centroids):
            # Compute the centroid of each cluster
            idx = np.where(cluster == i)[0]
            self.key_prototypes[y * self.num_centroids + i] = torch.mean(X[idx], dim=0)


# note - ortho init has not been found to help l2p/dual prompt
def tensor_prompt(a, b, c=None, ortho=False):
    if c is None:
        p = torch.nn.Parameter(torch.FloatTensor(a,b), requires_grad=True)
    else:
        p = torch.nn.Parameter(torch.FloatTensor(a,b,c), requires_grad=True)
    if ortho:
        nn.init.orthogonal_(p)
    else:
        nn.init.uniform_(p)
    return p    


class ViTZoo(nn.Module):
    def __init__(self, num_classes=10, pt=False, prompt_flag=False, prompt_param=None):
        super(ViTZoo, self).__init__()

        # get last layer
        self.last = nn.Linear(512, num_classes)
        self.prompt_flag = prompt_flag
        self.task_id = None

        # get feature encoder
        if pt:
            zoo_model = VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12,
                                        num_heads=12, ckpt_layer=0,
                                        drop_path_rate=0
                                        )
            from timm.models import vit_base_patch16_224
            load_dict = vit_base_patch16_224(pretrained=True).state_dict()
            del load_dict['head.weight']; del load_dict['head.bias']
            zoo_model.load_state_dict(load_dict)

        # classifier
        self.last = nn.Linear(768, num_classes)

        # create prompting module
        if self.prompt_flag == 'l2p':
            self.prompt = L2P(768, prompt_param[0], prompt_param[1])
        elif self.prompt_flag == 'dual':
            self.prompt = DualPrompt(768, prompt_param[0], prompt_param[1])
        elif self.prompt_flag == 'coda':
            self.prompt = CodaPrompt(768, prompt_param[0], prompt_param[1])
        else:
            self.prompt = None
        
        # feature encoder changes if transformer vs resnet
        self.feat = zoo_model
        
    # pen: get penultimate features    
    def forward(self, x, pen=False, train=False):
        if self.prompt is not None:
            with torch.no_grad():
                q, _ = self.feat(x)
                q = q[:, 0, :]
            out, prompt_loss = self.feat(x, prompt=self.prompt, q=q, train=train, task_id=self.task_id)
            out = out[:, 0,:]
        else:
            out, _ = self.feat(x)
            out = out[:, 0, :]
        out = out.view(out.size(0), -1)
        if not pen:
            out = self.last(out)
        if self.prompt is not None and train:
            return out, prompt_loss
        else:
            return out


class ViTFree(nn.Module):
    """
    A baseline that use NCM for classification
    """
    def __init__(self, num_classes=10, pt=False, prompt_flag=False, prompt_param=None):

        super(ViTFree, self).__init__()
        self.num_classes = num_classes
        self.prompt_flag = prompt_flag
        self.task_id = None

        # get feature encoder
        if pt:
            zoo_model = VisionTransformer(
                img_size=224, patch_size=16, embed_dim=768, depth=12,
                num_heads=12, ckpt_layer=0,
                drop_path_rate=0
            )
            from timm.models import vit_base_patch16_224
            load_dict = vit_base_patch16_224(pretrained=True).state_dict()
            del load_dict['head.weight']; del load_dict['head.bias']
            zoo_model.load_state_dict(load_dict)

        # feature encoder changes if transformer vs resnet
        self.feat = zoo_model

        # create prompting module
        if self.prompt_flag == 'cpp':
            self.prompt = CPP(768, prompt_param[0], prompt_param[1])
        else:
            print("No prompt module is used")
            self.prompt = None

        # classifier
        self.num_centroids = int(prompt_param[1][1])
        self.num_tasks = int(prompt_param[0])
        self.value_prototypes = torch.zeros((num_classes * self.num_centroids, 768))
        self.prototype_std = torch.zeros((num_classes * self.num_centroids, 768))

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

 
    def forward(self, x, pen=False, train=False):
        if self.prompt is not None:
            q = self.get_query_features(x)
            if train:
                out, _ = self.feat(x, prompt=self.prompt, q=q, train=train, task_id=self.task_id)
            else:
                with torch.no_grad():
                    out, _ = self.feat(x, prompt=self.prompt, q=q, train=train, task_id=self.task_id)
            out = out[:, 0, :]
            if not pen:
                out = self.prompt.mlp_head(out)
        else:  # No prompt
            out, _ = self.feat(x)
            out = out[:, 0, :]
        out = out.view(out.size(0), -1)

        if train:
            return out, torch.zeros(1).to(self._device)
        else:
            return out
        

    def get_query_features(self, x):
        """
        Get the query features of the input
        """
        # Get query features
        with torch.no_grad():
            q, _ = self.feat(x)
            q = q[:, 0, :]
        return q
    

    def predict(self, x):
        """
        Predict the class of the input
        """
        B = x.shape[0]
        # Get query features
        q = self.get_query_features(x)
        n_q = F.normalize(q, dim=1).detach().cpu()

        max_idx = (self.task_id + 1) * self.prompt.num_centroids * self.prompt.num_cls_per_task
        n_K = F.normalize(self.prompt.key_prototypes[:max_idx], dim=1)
        tasks = self.prompt.key_task[:max_idx]
        num_neighbours = self.prompt.num_neighbours

        # Find potential task ids
        cos_sim = torch.mm(n_q, n_K.t())
        top_k = torch.topk(cos_sim, num_neighbours, dim=1)
        top_task = tasks[top_k.indices]

        # values = torch.zeros(B, num_neighbours, 768)
        task_value = torch.zeros(B, self.num_tasks, 768)
        mask = torch.zeros(B, self.num_tasks)
        for i in range(num_neighbours):
            # Get the value features
            task_id = top_task[:, i].to(self._device)
            with torch.no_grad():
                out, _ = self.feat(x, prompt=self.prompt, q=q, train=False, task_id=task_id)
            out = out[:, 0, :]
            out = out.view(out.size(0), -1).cpu()
            # out = F.normalize(out, dim=1)
            task_value[torch.arange(B), top_task[:, i]] = out
            mask[torch.arange(B), top_task[:, i]] = 1

        # Compute the distance between value features and value prototypes
        value_prototypes = self.value_prototypes[:max_idx]
        # value_prototypes = F.normalize(value_prototypes, dim=1)
        dist = torch.cdist(task_value, value_prototypes)  # (B, T, C)
        dist = dist * mask.unsqueeze(-1)

        # Get the sum distance for each value feature -> (B, C)
        dist = dist.sum(dim=1)
        # Get the minimum distance for each value feature -> (B)
        _, min_idx = torch.min(dist, dim=1)
        min_idx = min_idx // self.num_centroids

        return min_idx

    
    def compute_key_prototypes(self, X, y):
        """
        Compute the key prototypes of each class using spectral clustering

        Arguments:
        ----------
        X: torch.Tensor
            The input tensor, shape (N, d)

        y: int
            The label of the input
        """
        self.prompt.compute_centroids(X, y)
    

    def update_value_prototypes(self, X, cls):
        """
        Update the prototypes
        """
        n_X = F.normalize(X, dim=1)

        # Compute the cosine similarity matrix and set the diagonal to 0
        S = (torch.mm(n_X, n_X.t()) + 1) / 2 - torch.eye(X.shape[0])

        clustering = SpectralClustering(
            n_clusters=self.num_centroids, affinity='precomputed',
            random_state=42, n_jobs=-1
        ).fit(S.cpu().numpy())
        cluster = clustering.labels_

        for i in range(self.num_centroids):
            # Compute the centroid of each cluster
            idx = np.where(cluster == i)[0]
            self.value_prototypes[cls * self.num_centroids + i] = torch.mean(X[idx], dim=0)
            self.prototype_std[cls * self.num_centroids + i] = torch.std(X[idx], dim=0)


    def sample_prototypes(self):
        if self.task_id == 0:
            return None
        
        max_idx = self.task_id * self.prompt.num_cls_per_task * self.num_centroids
        # Sample the prototypes from the previous tasks
        sampled_prototypes = self.value_prototypes[:max_idx] + torch.randn_like(self.value_prototypes[:max_idx]) * \
                             self.prototype_std[:max_idx]
        sampled_prototypes = self.prompt.mlp_head(sampled_prototypes.to(self._device))
        return sampled_prototypes

        

def vit_pt_imnet(out_dim, prompt_flag = 'None', prompt_param=None):
    return ViTZoo(num_classes=out_dim, pt=True, prompt_flag=prompt_flag, prompt_param=prompt_param)

def vit_free_imnet(out_dim, prompt_flag = 'None', prompt_param=None):
    print("Using training free baseline")
    return ViTFree(num_classes=out_dim, pt=True, prompt_flag=prompt_flag, prompt_param=prompt_param)