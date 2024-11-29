# Don't Command, Cultivate: an Exploratory Study of System-2 Alignment

<div align="center">
Yuhang Wang, Jitao Sang* 
</div>
<div align="center">
Department of Computer Science
</div>
<div align="center">
Beijing Jiaotong University
</div>


<div align="center">
    <a href="https://arxiv.org/pdf/2411.17075"><img src="images/Paper-Arxiv-orange.svg" ></a>
</div>

The o1 system card identifies the o1 models as the most robust within OpenAI, with
their defining characteristic being the progression from rapid, intuitive thinking
(System-1) to more deliberate, reasoned thought (System-2). This observation
motivated us to investigate the influence of System-2 thinking patterns on model
safety.
In our preliminary research, we conducted safety evaluations of the o1 model,
including complex jailbreak attack scenarios using adversarial natural language
prompts and mathematical encoding prompts. Our findings indicate that the o1
model demonstrates relatively improved safety performance, though vulnerabilities
remain, especially against attacks leveraging mathematical encoding. Through
detailed analysis, we identified specific response patterns associated with these
vulnerabilities.
We further explored System-2 Alignment on open-source models using prompt
engineering and supervised fine-tuning techniques. Experimental results suggest
that methods encouraging models to carefully analyze user inputs improve safety.
Additionally, we proposed an implementation framework for reinforcement learning with process supervision to enhance safety alignment. The implementation
details and experimental results will be presented in future versions.


## Experimental dataset
The dataset utilized in our study was derived through sampling from the wildjailbreak dataset (see Raw Data [LINK](https://huggingface.co/datasets/allenai/wildjailbreak) for more details). 




## Acknowledgement
- Hugging Face for their open-source transformer models.

## Citation
```
@misc{Wang2024DontCC,
      title={Don't Command, Cultivate: An Exploratory Study of System-2 Alignment}, 
      author={Yuhang Wang and Jitao Sang},
      year={2024}
}
```

