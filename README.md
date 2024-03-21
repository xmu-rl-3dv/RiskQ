### RiskQ: Risk-sensitive Multi-Agent Reinforcement Learning Value Factorization  

-----

RiskQ (https://proceedings.neurips.cc/paper_files/paper/2023/file/6d3040941a2d57ead4043556a70dd728-Paper-Conference.pdf) is a novel Multi-Agent Reinforcement Learning method that adheres to the Risk-sensitive Individual-Global-Max (RIGM) principle, enabling effective coordination of risk-sensitive policies among agents, with its proven performance and source code publicly accessible.



This code is build based on PyMARL2 and PyMARL. We assume that you have experience with PyMARL.
The requirements are the same as PyMARL2.

#### Run an experiment 

```
cd RiskQ
python3 src/main.py --config=RiskQ --env-config=sc2 with env_args.map_name=2s3z
```

RiskQ is merely one example; you can generate more variants based on different risk functions and other configurations.

The config files act as defaults for an algorithm or environment. 

They are all located in `src/config`.
`--config` refers to the config files in `src/config/algs`
`--env-config` refers to the config files in `src/config/envs`

All results will be stored in the `Results` folder.

#### Citing RiskQ

If you use RiskQ in your research, please cite our paper:

```
@misc{shen2023riskq,
      title={RiskQ: Risk-sensitive Multi-Agent Reinforcement Learning Value Factorization}, 
      author={Siqi Shen and Chennan Ma and Chao Li and Weiquan Liu and Yongquan Fu and Songzhu Mei and Xinwang Liu and Cheng Wang},
      year={2023},
      eprint={2311.01753},
      archivePrefix={arXiv},
      primaryClass={cs.MA}
}
```



## License

Code licensed under the Apache License v2.0
