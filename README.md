## Some code used in DeepMET trainings
(Code Taken from Jan Steggemann and Markus Seidel)

First install the necessary packages with [MiniConda](https://docs.conda.io/en/latest/miniconda.html)
```
conda create -n METTraining python=3.7
conda install -n METTraining numpy h5py
conda install -n METTraining progressbar2
conda install -n METTraining conda-forge::uproot
conda install -n METTraining matplotlib pandas scikit-learn
conda install -n METTraining tensorflow-gpu=1.13.1 keras=2.2.4
conda install pydot
```
and activate the environment
```
conda activate METTraining
```

Prepare the `h5` training files
```
python convertNanoToHDF5.py -i /eos/cms/store/user/yofeng/WRecoilNanoAOD_Skimmed_v10_tempMET_TTJets/myNanoProdMc2016_NANO_2_Skim.root -o /eos/cms/store/user/yofeng/DeepMETTrainingFile/myNanoProdMc2016_NANO_ttbar_2_Skim.h5
```

Run the training
```
python train_ptmiss_mine.py -i input.txt
```
