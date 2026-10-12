"""Small classroom PPO: warm-start owned Actor/Critic, freeze HIM, fresh Adam.

The original research pipeline also trained an HIM estimator and used RowAdam.
Here the sensor estimator is fixed and the PPO recipe is identical in B/C.
There is no vendor ONNX, expert target action, or privileged Actor input.
"""
from __future__ import annotations
import os
os.environ.setdefault('OMP_WAIT_POLICY', 'PASSIVE')
os.environ.setdefault('GOMP_SPINCOUNT', '0')
import argparse, copy, hashlib, json, math, random, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from scratch_policy.policy import mlp, state_digest
from scratch_policy.policy_yaw_him_v25 import load_actor as load_him_actor
from scratch_policy.policy import load_actor as load_base_actor
from scratch_policy.yaw_course_env_v7 import YawCourseCfgV7, YawCourseEnvV7
from kudada_motrix.randomization import DomainRandCfg

ROOT = Path(__file__).resolve().parents[1]

def write_json(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

class StandingCourseEnv(YawCourseEnvV7):
    """One visible experiment factor. Cost uses magnitude, never signed average."""
    def __init__(self, cfg, num_envs, standing_speed_cost):
        self.standing_speed_cost = float(standing_speed_cost)
        super().__init__(cfg, num_envs)

    def _reward_terms(self, phase):
        terms = super()._reward_terms(phase)
        resting = np.all(np.abs(self.filtered_commands) < .1, axis=1)
        speed_squared = np.sum(self.sim_data['world_lin_vel'][:, :2]**2, axis=1)
        terms['standing_speed_cost'] = np.asarray(
            -self.standing_speed_cost * speed_squared * resting * self.cfg.ctrl_dt,
            dtype=np.float32)
        return terms

def train(config_path: Path, run_dir: Path | None = None):
    recipe = json.loads(Path(config_path).read_text(encoding='utf-8-sig'))
    required = {'checkpoint','seed','iterations','num_envs','rollout_steps','epochs',
                'minibatches','lr','critic_lr','std','gamma','gae_lambda','clip_ratio',
                'target_kl','standing_speed_cost','environment','run_dir'}
    if set(recipe) != required:
        raise ValueError(f'Configuration keys mismatch: {set(recipe)^required}')
    for key in ('iterations','num_envs','rollout_steps','epochs','minibatches'):
        if type(recipe[key]) is not int or recipe[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if recipe['num_envs'] % 4 or recipe['num_envs'] < 4:
        raise ValueError('num_envs must be a multiple of four')
    if recipe['minibatches'] > recipe['num_envs'] * recipe['rollout_steps']:
        raise ValueError('Too many minibatches')
    for key in ('lr','critic_lr','std','gamma','gae_lambda','clip_ratio','target_kl'):
        if not math.isfinite(recipe[key]) or recipe[key] <= 0:
            raise ValueError(f'{key} must be finite and positive')
    if max(recipe['gamma'],recipe['gae_lambda']) > 1 or recipe['clip_ratio'] >= 1:
        raise ValueError('Invalid GAE/clip configuration')
    if not math.isfinite(recipe['standing_speed_cost']) or recipe['standing_speed_cost'] < 0:
        raise ValueError('standing_speed_cost must be finite and nonnegative')
    seed = recipe['seed']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(2)
    # Native Motrix batch physics and small CPU networks keep lab reproducible.
    device = torch.device('cpu')
    source = (ROOT / recipe['checkpoint']).resolve()
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    parent = torch.load(source, map_location='cpu', weights_only=False)
    has_him = parent['actor_config'].get('architecture') == 'scratch_him_yaw_command_phase_bypass_v25'
    actor = (load_him_actor(parent) if has_him else load_base_actor(parent)).to(device).train()
    if has_him:
        for p in actor.him_encoder.parameters(): p.requires_grad_(False)
    actor_parameters = [p for p in actor.parameters() if p.requires_grad]
    critic = mlp(233, parent['critic_hidden'], 1).to(device)
    critic.load_state_dict(parent['critic_state_dict'], strict=True)
    actor_opt = torch.optim.Adam(actor_parameters, lr=recipe['lr'])
    critic_opt = torch.optim.Adam(critic.parameters(), lr=recipe['critic_lr'])
    logstd = torch.full((10,), math.log(recipe['std']), device=device)
    kwargs = copy.deepcopy(recipe['environment'])
    kwargs['randomization'] = DomainRandCfg(**kwargs['randomization'])
    cfg = YawCourseCfgV7(seed=seed, **kwargs)
    env = StandingCourseEnv(cfg, recipe['num_envs'], recipe['standing_speed_cost'])
    env.set_course_stage_update(100); env.set_curriculum(1.); env.init_state()
    target = (ROOT / (run_dir or recipe['run_dir'])).resolve()
    target.mkdir(parents=True, exist_ok=True)
    if (target/'latest.pt').exists():
        raise FileExistsError(f'{target} already has a model; choose a fresh run directory')
    write_json(target/'config.json',recipe)
    metadata = dict(method='classroom_PPO_warm_start_fresh_Adam_v1',
        parent_sha256=source_sha, parent_actor_sha256=state_digest(parent['actor_state_dict']),
        optimizer_initialization='fresh Adam for Actor core/yaw branch and Critic; no inherited moments',
        estimator='sensor-only HIM weights retained and frozen' if has_him else 'none; ordinary five-frame actor', exploration='fixed latent Gaussian std',
        actor_input='sensor history924 only; privileged13 exclusively Critic/reward',
        training_seed=seed, hardware_tested=False,
        source_sha256={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                       for d in ('course','scratch_policy','tinker_policy','kudada_motrix')
                       for p in (ROOT/d).glob('*.py')})
    write_json(target/'initialization.json',metadata)
    initial_him = state_digest(actor.him_encoder.state_dict()) if has_him else None
    history=[]; observations=[]; started=time.perf_counter()

    def save(iteration):
        # Retain truthful lineage and exact architecture/migration declarations.
        cp = {key:copy.deepcopy(parent[key]) for key in (
            'format','actor_config','critic_hidden','critic_privileged_size',
            'hardware_contract','action_bounds','provenance','him_migration','yaw_phase_migration')
              if key in parent}
        cp.update(actor_state_dict={k:v.detach().cpu() for k,v in actor.state_dict().items()},
            critic_state_dict={k:v.detach().cpu() for k,v in critic.state_dict().items()},
            logstd=logstd.cpu(), training_method=metadata['method'],
            course_training=metadata, config=recipe, completed_updates=iteration,
            actor_optimizer=actor_opt.state_dict(), critic_optimizer=critic_opt.state_dict(),
            total_transitions=iteration*recipe['num_envs']*recipe['rollout_steps'])
        torch.save(cp,target/'latest.pt')
        torch.save(cp,target/f'model_{iteration:04d}.pt')
        np.savez_compressed(target/'observations.npz',observations=np.asarray(observations[:256]))
        write_json(target/'history.json',history)

    for iteration in range(1, recipe['iterations']+1):
        buffer={k:[] for k in ('obs','priv','latent','mean','logp','value','reward','done','terminated','next_value')}
        failures=0; timeouts=0; rewards=[]
        for _ in range(recipe['rollout_steps']):
            obs_np=env.policy_input()
            if len(observations)<256: observations.extend(obs_np[:4].copy())
            obs=torch.from_numpy(obs_np).to(device)
            priv=torch.from_numpy(env.privileged()).to(device)
            with torch.no_grad():
                mean=actor.latent(obs); dist=Normal(mean,logstd.exp())
                latent=dist.sample(); action=actor.actions_from_latent(latent)
                logp=dist.log_prob(latent).sum(-1)
                value=critic(torch.cat((actor.features(obs),priv),-1)).squeeze(-1)
            state=env.step(action.cpu().numpy())
            d=env.transition_data
            done=state.done.copy(); terminated=d['terminated'].copy()
            truncated=d['truncated'] & ~terminated
            next_obs=env.policy_input(); next_priv=env.privileged()
            # Bootstrap true timeout states, never post-autoreset observations.
            if truncated.any():
                ids=np.flatnonzero(truncated)
                terminal=obs_np[ids].reshape(-1,21,44).copy()
                terminal[:,:-1]=terminal[:,1:].copy(); terminal[:,-1]=d['next_frame'][ids]
                next_obs[ids]=terminal.reshape(-1,924)
                next_priv[ids]=env.privileged(terminal=True)[ids]
            with torch.no_grad():
                nt=torch.from_numpy(next_obs).to(device)
                nv=critic(torch.cat((actor.features(nt),torch.from_numpy(next_priv).to(device)),-1)).squeeze(-1)
            values=dict(obs=obs,priv=priv,latent=latent,mean=mean,logp=logp,value=value,
                reward=torch.from_numpy(state.reward.copy()).to(device),
                done=torch.from_numpy(done).to(device),terminated=torch.from_numpy(terminated).to(device),next_value=nv)
            for k,v in values.items(): buffer[k].append(v.detach())
            failures+=int(terminated.sum()); timeouts+=int(truncated.sum()); rewards.append(float(state.reward.mean()))
        b={k:torch.stack(v) for k,v in buffer.items()}
        advantages=torch.zeros_like(b['reward']); last=torch.zeros(recipe['num_envs'],device=device)
        for t in reversed(range(recipe['rollout_steps'])):
            delta=b['reward'][t]+recipe['gamma']*b['next_value'][t]*(~b['terminated'][t])-b['value'][t]
            last=delta+recipe['gamma']*recipe['gae_lambda']*(~b['done'][t])*last
            advantages[t]=last
        returns=(advantages+b['value']).flatten()
        adv=advantages.flatten(); adv=(adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
        flat={k:v.flatten(0,1) for k,v in b.items()}
        losses=[]; kls=[]; accepted=0; stop=False
        for epoch in range(recipe['epochs']):
            for ids in torch.randperm(len(adv),device=device).chunk(recipe['minibatches']):
                mean=actor.latent(flat['obs'][ids]); dist=Normal(mean,logstd.exp())
                # Analytic Gaussian KL avoids a noisy sample estimate.
                kl=((mean-flat['mean'][ids]).square()/(2*logstd.exp().square())).sum(-1).mean()
                kls.append(float(kl.detach()))
                if float(kl.detach())>recipe['target_kl']:
                    stop=True; break
                ratio=(dist.log_prob(flat['latent'][ids]).sum(-1)-flat['logp'][ids]).exp()
                loss=-torch.minimum(ratio*adv[ids],ratio.clamp(1-recipe['clip_ratio'],1+recipe['clip_ratio'])*adv[ids]).mean()
                actor_opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(actor_parameters,1.)
                actor_opt.step(); accepted+=1; losses.append(float(loss.detach()))
                pred=critic(torch.cat((actor.features(flat['obs'][ids]),flat['priv'][ids]),-1)).squeeze(-1)
                value_loss=(pred-returns[ids]).square().mean()
                critic_opt.zero_grad(); value_loss.backward(); nn.utils.clip_grad_norm_(critic.parameters(),1.)
                critic_opt.step()
            if stop: break
        row=dict(update=iteration,reward=float(np.mean(rewards)),failures=failures,timeouts=timeouts,
            kl_max=max(kls,default=0.),ppo_minibatches=accepted,
            policy_loss=float(np.mean(losses)) if losses else 0.,wall_seconds=time.perf_counter()-started)
        history.append(row); print(json.dumps(row),flush=True)
        if iteration%5==0 or iteration==recipe['iterations']: save(iteration)
    assert not has_him or state_digest(actor.him_encoder.state_dict())==initial_him
    result=dict(**metadata,wall_seconds=time.perf_counter()-started,
                total_transitions=recipe['iterations']*recipe['num_envs']*recipe['rollout_steps'],
                him_unchanged=True,actor_changed=state_digest(actor.state_dict())!=metadata['parent_actor_sha256'],
                final_actor_sha256=state_digest(actor.state_dict()),latest_checkpoint_sha256=hashlib.sha256((target/'latest.pt').read_bytes()).hexdigest())
    write_json(target/'verification.json',result)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True); p.add_argument('--run-dir',type=Path)
    a=p.parse_args(); print(json.dumps(train(a.config,a.run_dir),ensure_ascii=False))

if __name__=='__main__': main()
