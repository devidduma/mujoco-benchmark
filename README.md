# Tianshou's Mujoco Benchmark

In this project, I follow the instructions and try to reproduce the results from [Tianshou MuJoCo benchmark](https://github.com/thu-ml/tianshou/tree/master/examples/mujoco).

I benchmarked 9 Tianshou algorithm implementations in 11 out of 11 environments from [Gymnasium MuJoCo task suite](https://gymnasium.farama.org/environments/mujoco/) provided by Gymnasium.<sup>[[1]](#footnote1)</sup>.

For each supported algorithm and supported MuJoCo environments, we provide:
- Default hyperparameters used for benchmark and scripts to reproduce the benchmark.
- Pretrained agents and log details can all be found in the folder [second_benchmark/log](https://github.com/devidduma/mujoco-benchmark/tree/master/second_benchmark/log).

Supported algorithms are listed below:
- [Deep Deterministic Policy Gradient (DDPG)](https://arxiv.org/pdf/1509.02971.pdf), [commit id](https://github.com/thu-ml/tianshou/tree/e605bdea942b408126ef4fbc740359773259c9ec)
- [Twin Delayed DDPG (TD3)](https://arxiv.org/pdf/1802.09477.pdf), [commit id](https://github.com/thu-ml/tianshou/tree/e605bdea942b408126ef4fbc740359773259c9ec)
- [Soft Actor-Critic (SAC)](https://arxiv.org/pdf/1812.05905.pdf), [commit id](https://github.com/thu-ml/tianshou/tree/e605bdea942b408126ef4fbc740359773259c9ec)
- [REINFORCE algorithm](https://papers.nips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf), [commit id](https://github.com/thu-ml/tianshou/tree/e27b5a26f330de446fe15388bf81c3777f024fb9)
- [Natural Policy Gradient](https://proceedings.neurips.cc/paper/2001/file/4b86abe48d358ecf194c56c69108433e-Paper.pdf), [commit id](https://github.com/thu-ml/tianshou/tree/844d7703c313009c4c364edb4018c91de93439ca)
- [Advantage Actor-Critic (A2C)](https://openai.com/blog/baselines-acktr-a2c/), [commit id](https://github.com/thu-ml/tianshou/tree/1730a9008ad6bb67cac3b21347bed33b532b17bc)
- [Proximal Policy Optimization (PPO)](https://arxiv.org/pdf/1707.06347.pdf), [commit id](https://github.com/thu-ml/tianshou/tree/6426a39796db052bafb7cabe85c764db20a722b0)
- [Trust Region Policy Optimization (TRPO)](https://arxiv.org/pdf/1502.05477.pdf), [commit id](https://github.com/thu-ml/tianshou/tree/5057b5c89e6168220272c9c28a15b758a72efc32)
- [Hindsight Experience Replay (HER)](https://arxiv.org/abs/1707.01495)

## Usage

For each environment, a Jupyter Notebook is available to train 9 Deep Reinforcement Learning algorithms.  Logs are then saved in `./log/` and can be monitored with tensorboard.

```bash
$ tensorboard --logdir log
```

Tensorboard is the preferred way to analyze the results. Additionally, bash scripts are also available that can convert all tfevent files into csv files, then try plotting the results.

```bash
# generate csv
$ ./tools.py --root-dir ./results/Ant-v4/sac
# generate figures
$ ./plotter.py --root-dir ./results/Ant-v4 --shaded-std --legend-pattern "\\w+"
# generate numerical result (support multiple groups: `--root-dir ./` instead of single dir)
$ ./analysis.py --root-dir ./results --norm
```

## Note

<a name="footnote1">[1]</a>  Supported environments include HalfCheetah-v4, Hopper-v4, Swimmer-v4, Walker2d-v4, Ant-v4, Humanoid-v4, Reacher-v4, InvertedPendulum-v4, InvertedDoublePendulum-v4, Pusher-v4 and HumanoidStandup-v4.