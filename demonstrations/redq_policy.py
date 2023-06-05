from tianshou.policy import REDQPolicy
from tianshou.utils.net.common import EnsembleLinear, Net
from tianshou.utils.net.continuous import ActorProb, Critic
import torch
import numpy as np
from mujoco_redq import get_args

def get_redq_policy(env, **kwargs):
    args = get_args()
    args.task = kwargs["task"]
    args.hidden_sizes = kwargs["hidden_sizes"]

    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    args.max_action = env.action_space.high[0]
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    print("Action range:", np.min(env.action_space.low), np.max(env.action_space.high))
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # model
    net_a = Net(args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device)
    actor = ActorProb(
        net_a,
        args.action_shape,
        max_action=args.max_action,
        device=args.device,
        unbounded=True,
        conditioned_sigma=True,
    ).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)

    def linear(x, y):
        return EnsembleLinear(args.ensemble_size, x, y)

    net_c = Net(
        args.state_shape,
        args.action_shape,
        hidden_sizes=args.hidden_sizes,
        concat=True,
        device=args.device,
        linear_layer=linear,
    )
    critics = Critic(
        net_c,
        device=args.device,
        linear_layer=linear,
        flatten_input=False,
    ).to(args.device)
    critics_optim = torch.optim.Adam(critics.parameters(), lr=args.critic_lr)

    if args.auto_alpha:
        target_entropy = -np.prod(env.action_space.shape)
        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        args.alpha = (target_entropy, log_alpha, alpha_optim)

    policy = REDQPolicy(
        actor,
        actor_optim,
        critics,
        critics_optim,
        args.ensemble_size,
        args.subset_size,
        tau=args.tau,
        gamma=args.gamma,
        alpha=args.alpha,
        estimation_step=args.n_step,
        actor_delay=args.update_per_step,
        target_mode=args.target_mode,
        action_space=env.action_space,
    )

    return policy, args
