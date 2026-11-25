"""Online TD-JEPA-inspired successor features, adapted to categorical traffic control.

Not full TD-JEPA: one traffic task, behavior next actions, fixed TD horizon,
EMA targets and variance/covariance anti-collapse instead of the paper's
task-conditioned bidirectional predictors and Gram orthogonality objective.
"""
import copy
import math
import torch
from torch import nn
import torch.nn.functional as F

VERSION = 1
DEFAULT_CONFIG = {
    'actor_kl_beta': 0.05337935120662157,
    'critic_kl_beta': 0.05419863990588785,
    'free_nats': 0.1,
    'latent_dim': 32, 'td_discount': 0.9, 'ema_decay': 0.99,
    'loss_coefficient': 0.001, 'variance_coefficient': 1.0,
    'covariance_coefficient': 0.01, 'initialization_seed': 9017,
    'log_interval': 50,
}


def validate_config(config):
    if not isinstance(config,dict) or set(config)!=set(DEFAULT_CONFIG):
        raise ValueError(f'advanced_vae_config must contain {sorted(DEFAULT_CONFIG)}')
    for key in ('latent_dim','initialization_seed','log_interval'):
        if type(config[key]) is not int:
            raise ValueError(f'advanced_vae_config.{key} must be an integer')
    if config['latent_dim']<2 or config['log_interval']<1 or not 0<=config['initialization_seed']<2**32:
        raise ValueError('Invalid advanced VAE dimension, log interval or initialization seed')
    for key in set(config)-{'latent_dim','initialization_seed','log_interval'}:
        v=config[key]
        if type(v) not in (int,float) or not math.isfinite(v) or v<0:
            raise ValueError(f'advanced_vae_config.{key} must be finite and nonnegative')
    if not 0<=config['td_discount']<1 or not 0<=config['ema_decay']<1:
        raise ValueError('Advanced VAE TD discount and EMA decay must be in [0,1)')
    if any(config[k]<=0 for k in ('actor_kl_beta','critic_kl_beta','loss_coefficient')):
        raise ValueError('Advanced VAE KL and auxiliary coefficients must be positive')
    return dict(config)


def deterministic_features(module, obs, phase, mask, neighbor=None):
    """Exactly the deterministic policy/value input prefix, excluding VAE z."""
    t, n = obs.shape[:2]
    s = module.normalization(obs.reshape(-1, module.flat_dim)).reshape(t,n,module.flat_dim)
    h = module.linear_s(s); memory, _ = module.gru(h, None); h = h+memory
    p = module.linear_a(phase.reshape(t*n,-1,module.flat_phase_vec_dim))
    h = h.reshape(t*n,-1,module.hidden_dim)
    if neighbor is None:
        h = module.decoder(p,h,mask.reshape(t*n,-1))
    else:
        h = module.decoder_s(p,h,mask.reshape(t*n,-1))
        a = module.linear_neighbor_a(neighbor).reshape(t*n,1,module.hidden_dim)
        h = module.decoder_a(h,a)
    return h


def pool(features, mask):
    h = features.reshape(*mask.shape,-1)
    legal = (~mask.bool()).unsqueeze(-1)
    if not legal.any(-2).all():
        raise ValueError('Advanced VAE needs a legal phase for every junction')
    return h.masked_fill(~legal,0).sum(-2)/legal.sum(-2)


class AdvancedVAE:
    """Training-only J3 module. Frozen policies need no auxiliary architecture."""
    def __init__(self,model,config):
        self.config=validate_config(config); self.model=model
        self.completed_updates=0; self.actor_steps=0; self.critic_steps=0
        self.sides={}; self.logs={'actor':[],'critic':[]}
        device=next(model.parameters()).device
        with torch.random.fork_rng(devices=[device.index or 0] if device.type=='cuda' else []):
            torch.manual_seed(self.config['initialization_seed'])
            for side in ('actor','critic'):
                self.sides[side]=Side(getattr(model,side+'_network'),self.config).to(device)
        self.attach_optimizers()

    def attach_optimizers(self):
        for side,obj in self.sides.items():
            optimizer=getattr(self.model,side+'_optimizer')
            existing={id(p) for group in optimizer.param_groups for p in group['params']}
            params=obj.trainable()
            if all(id(p) not in existing for p in params):optimizer.add_param_group({'params':params})
            elif not all(id(p) in existing for p in params):raise ValueError('Partial auxiliary optimizer group')

    def begin_batch(self):
        self.logs={'actor':[],'critic':[]}

    def vae_loss(self,prediction,mu,logvar,target,side):
        rec=(target.reshape_as(prediction)-prediction).square().mean()
        # Stable Gaussian KL to a fixed N(0,I): sum latent dims, mean valid samples.
        v=logvar.double()
        kl=(0.5*(mu.double().square()+torch.expm1(v)-v)).sum(-1).to(mu.dtype)
        applied=kl.clamp_min(self.config['free_nats']).mean()*self.config[side+'_kl_beta']
        return rec+applied,{'prediction_mse':rec.item(),'kl_per_vector':kl.detach().mean().item(),
            'kl_regularizer':applied.item(),'below_free_nats_fraction':(kl.detach()<self.config['free_nats']).float().mean().item()}

    def auxiliary_loss(self,side,features,obs,phase,mask,actions,next_obs,neighbor,vae_metrics):
        variables={'batch_obs':obs,'batch_phase_vec':phase,'batch_phase_mask':mask,
            'batch_actions':actions,'batch_target_prediction':next_obs,'batch_neighbor_actions_vec':neighbor}
        measure=(self.completed_updates==0 or (self.completed_updates+1)%self.config['log_interval']==0) and not self.logs[side]
        loss,metrics=self.sides[side].loss(features,getattr(self.model,side+'_network'),variables,side=='actor',measure)
        self.logs[side].append({**metrics,**vae_metrics})
        return loss

    def clip(self,side):
        norm=torch.nn.utils.clip_grad_norm_(self.sides[side].trainable(),10.)
        if not torch.isfinite(norm):raise ValueError('Non-finite advanced VAE auxiliary gradient')
        self.logs[side][-1]['auxiliary_gradient_norm']=norm.item()
        if side=='actor':self.actor_steps+=1
        else:self.critic_steps+=1

    def finish_batch(self,epochs):
        if any(len(self.logs[s])!=epochs for s in self.sides):raise ValueError('Incomplete advanced VAE update')
        for side,obj in self.sides.items():obj.ema(getattr(self.model,side+'_network'))
        self.completed_updates+=1

    def diagnostics(self):
        from statistics import fmean
        result={}
        for side,rows in self.logs.items():
            for key in sorted(set().union(*(r.keys() for r in rows))):
                result[side+'/'+key]=fmean(r[key] for r in rows if key in r)
        return result

    def state_dict(self):
        return {'version':VERSION,'config':self.config,'completed_updates':self.completed_updates,
            'actor_steps':self.actor_steps,'critic_steps':self.critic_steps,
            'sides':{s:m.state_dict() for s,m in self.sides.items()}}

    def load_state_dict(self,state,epochs):
        if not isinstance(state,dict) or state.get('version')!=VERSION or state.get('config')!=self.config:
            raise ValueError('Missing/incompatible advanced VAE training state')
        count=state['completed_updates']
        if state['actor_steps']!=count*epochs or state['critic_steps']!=count*epochs:
            raise ValueError('Advanced VAE optimizer/EMA clock mismatch')
        for s,obj in self.sides.items():obj.load_state_dict(state['sides'][s])
        self.completed_updates=count;self.actor_steps=state['actor_steps'];self.critic_steps=state['critic_steps']


class Side(nn.Module):
    def __init__(self, module, config):
        super().__init__(); self.config=dict(config)
        d=config['latent_dim']; hidden=module.hidden_dim; a=module.flat_phase_vec_dim
        self.projector=nn.Sequential(nn.Linear(hidden,d),nn.LayerNorm(d))
        self.predictor=nn.Sequential(nn.Linear(d+a,hidden//2),nn.ReLU(),nn.Linear(hidden//2,d))
        self.target_projector=copy.deepcopy(self.projector)
        self.target_predictor=copy.deepcopy(self.predictor)
        # Only clone deterministic encoder components, not unused VAE/policy layers.
        self.target_encoder=copy.deepcopy(module)
        for name in list(self.target_encoder._modules):
            if name not in ('normalization','linear_s','gru','linear_a','decoder','decoder_s','decoder_a','linear_neighbor_a'):
                del self.target_encoder._modules[name]
        for target in (self.target_encoder,self.target_projector,self.target_predictor):
            target.requires_grad_(False)

    def trainable(self):
        return [*self.projector.parameters(),*self.predictor.parameters()]

    def loss(self, online_features, module, variables, actor, measure=False):
        obs=variables['batch_obs']; phase=variables['batch_phase_vec']; mask=variables['batch_phase_mask']
        action=variables['batch_actions']; neighbor=None if actor else variables['batch_neighbor_actions_vec']
        if obs.shape[0]<2:
            raise ValueError('TD auxiliary requires at least two consecutive transitions')
        # Original rollout buffers are one contiguous fragment per junction.
        if not torch.equal(variables['batch_target_prediction'][:-1],obs[1:]):
            raise ValueError('TD auxiliary cannot bootstrap across shuffled/discontinuous time')
        current=self.projector(pool(online_features,mask))
        encoded_action=phase.gather(2,action[...,None,None].expand(*action.shape,1,phase.shape[-1])).squeeze(2)
        prediction=self.predictor(torch.cat((current[:-1],encoded_action[:-1]),-1))
        with torch.no_grad():
            h=deterministic_features(self.target_encoder,obs,phase,mask,neighbor)
            target=self.target_projector(pool(h,mask))
            next_prediction=self.target_predictor(torch.cat((target[1:],encoded_action[1:]),-1))
            desired=target[1:]+self.config['td_discount']*next_prediction
        td=(prediction-desired).square().mean()
        flat=current.reshape(-1,current.shape[-1]); centered=flat-flat.mean(0)
        std=(centered.square().mean(0)+1e-4).sqrt()
        variance=F.relu(1-std).square().mean()
        cov=centered.T@centered/max(1,flat.shape[0]-1)
        off=cov-torch.diag_embed(cov.diagonal())
        covariance=off.square().sum()/flat.shape[-1]
        total=td+self.config['variance_coefficient']*variance+self.config['covariance_coefficient']*covariance
        stats={'td_mse':td.item(),'variance_penalty':variance.item(),'covariance_penalty':covariance.item(),
               'representation_std_mean':std.mean().item(),'td_pairs':prediction.shape[0]*prediction.shape[1],
               'target_requires_grad':desired.requires_grad,'weighted_loss':self.config['loss_coefficient']*total.item()}
        if measure:
            grad=torch.autograd.grad(total,module.linear_s[0].weight,retain_graph=True)[0]
            stats['weighted_backbone_probe_gradient_norm']=self.config['loss_coefficient']*grad.detach().norm().item()
        return self.config['loss_coefficient']*total,stats

    @torch.no_grad()
    def ema(self,module):
        decay=self.config['ema_decay']
        for source,target in ((self.projector,self.target_projector),(self.predictor,self.target_predictor),(module,self.target_encoder)):
            src=dict(source.named_parameters())
            for name,p in target.named_parameters():
                p.mul_(decay).add_(src[name],alpha=1-decay)
            buffers=dict(source.named_buffers())
            for name,b in target.named_buffers():
                b.copy_(buffers[name])

