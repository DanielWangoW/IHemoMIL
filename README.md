# Hemodynamic Fluctuations with MIL

This repo contains the implementation, which is a method that leverages weakly-supervised multiple instance learning and ranking-based aggregation to effectively recognize hemodynamic fluctuations within PPG waveforms.
<p align="center">
    <img src="https://image-oss-danielwangow.oss-cn-shanghai.aliyuncs.com/github-extern/IHemoMIL-Concept.png" alt="IHemoMIL Concept">
    <b>Fig. 1</b> Human cardiovascular system, PPG, motivation, and paradigm of MIL
</p>

## Rethinking PPG Pattern Recognition with MIL and Ranking-based Strategy
Following the problem formulation, as illustrated in Fig. 2, this section outlines our methodology for leveraging AI to make interpretable decisions in recognizing hemodynamic fluctuation patterns. This study addresses the challenges of current automatic PPG-based diagnostics, including ***interpretability for hemodynamic anomalies and collaboration between computer-assisted healthcare and human clinical workflows***.   


<p align="center">
    <img src="https://image-oss-danielwangow.oss-cn-shanghai.aliyuncs.com/github-extern/Workflow4IHemoMIL.png" alt="Workflow4IHemoMIL">
    <b>Fig. 2</b> Proposed end-to-end interpretable framework of IHemoMIL, encompassing the entire pipeline from data generation to decision making for ensuring transparency and explainability throughout the process
</p>

We utilize **causality-informed synthetic PPG datasets (Fig.2a)**, incorporating morphological perturbation-based anomaly injection and vasculature response-based waveform simulation, to thoroughly examine cause-and-effect relationships in peripheral pulse waves.  We propose IHemoMIL, **a general framework for widespread adaptability and preceptive explainability via pinpoint discriminatory motifs under a weakly supervised multiple instance learning paradigm**. IHemoMIL incorporates InceptionTime for multi-scale receptive fields (Fig.2b) and salience-induced MIL aggregation via ranking-based poolings (Fig.2c) to enhance both predictive and interpretable performance. 

## Causality-informed Synthetic PPG Dataset

Our causality-informed synthetic dataset generation pipeline, as illustrated in Fig. 2a, focuses on modeling physiological changes (**Causes**) and their effects on PPG waveforms (**Effects**) due to hemodynamic fluctuations.   We utilized nine datasets, including five synthetic PPG datasets, three realistic PPG datasets for hypertension/hypotension identification, AFib detection, and HAR, and the WebTraffic dataset for general time series benchmarking.  

**Data availability.** The datasets are released under different access terms:

* **PRRB, BIDMC, VitalDB, PulseDB-VitalDB, PERFormAF, and DaLia** — we provide the processing scripts under `ihemomil/data`; the raw data should be downloaded from their original sources.
* **SimHF-3K** — distributed under a limited authorization; data available on request.
* **WebTraffic** — we forked the generation and processing scripts, so the data can be synthesized locally and independently.

## Usage

To use the models and algorithms in this project, refer to the example notebooks in the notebooks directory. These notebooks provide step-by-step instructions on how to load data, train models, and evaluate results.

* **config** - Base configuration files for the models.
* **data** - Datasets used for training and evaluation.
* **ihemomil** - Data processing, model, and the necessary components.
* **notebook** - Notebooks providing visualization.
* **ranking_backend** - Pre-compiled classes that can be invoked directly.
* **train.py** - Training script.

## Dependencies

- Python 3.10
- matplotlib == 3.7.0
- numpy == 1.23.5
- pandas == 1.5.3
- scikit_learn == 1.2.2
- torch == 1.13.0

Dependencies can be installed using the following command:

```bash
pip install -r requirements.txt
```

## Reproducibility

To easily reproduce the results you can follow the next steps:

1. Initialize: Create a python virtual environment, follow the required dependencies.
2. Obtain the datasets and put them at `./data/`.
3. Run script autotask_trainer.sh.

```bash
python train.py \
  --is_train True \
  --dataset "web_traffic" \
  --data_path "data" \
  --checkpoint "checkpoint" \
  --channel 1 \
  --backbone "inceptiontime"\
  --pooling "conj" \
  --d_model 128 \
  --apply_positional_encoding True \
  --batch_size 512 \
  --epochs 1500 \
  --learning_rate 0.001 \
  --use_gpu True \
  --gpu_id 1
```

For convenience, we include a pre-generated WebTraffic dataset for quick testing. Note that while the generation process is fairly stochastic, the ranking-based methods remain consistently better than the baselines overall.

## Notice on public release

For intellectual property reasons, the source code of the ranking-based pooling is not yet released.
Instead, we ship a pre-compiled implementation with a callable interface, so the methods can still be invoked and evaluated as usual.

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

## Contact

If you have any questions, feel free to contact through Email (daomiao.wang@live.com). Pull requests are highly welcomed!

## Acknowledgments

This library follows the implementation of baseline models: 
- InceptionTime: https://github.com/hfawaz/InceptionTime
- MILLET: https://github.com/JAEarly/MILTimeSeriesClassification
- Time Series Library (TSLib): https://github.com/thuml/Time-Series-Library  

Thanks for the in-house cardiovascular simulation model Nektar1D for vasculature response-based waveform generation. At the same time, thank you all for your attention to this work! 



