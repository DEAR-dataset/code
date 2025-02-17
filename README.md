# DEAR

This the code for  a [`torch.data.Dataset`](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset) class and meant to be used in
conjunction with the data for the DEAR paper which can  be found on Zenodo.

# Usage

Copy the dear directory to your source repository, you are then be able to spawn a
related  evaluation task object by using the desired class.

## Environment

```python
environment_eval_dataset = EnvironmentDEARDataset(
    base_path=Path("/data/evaluation/dear"),
    split=DatasetType.TRAIN,
    target_variable_type=TargetVariableType.DISCRETE,
)
```

## Indoor or Outdoor

```python
indoor_or_outdoor_eval_dataset = IndoorOutdoorDEARDataset(
    base_path=Path("/data/evaluation/dear"),
    split=DatasetType.TRAIN,
    target_variable_type=TargetVariableType.DISCRETE,
)
```
## Stationary or Transient Noise

```python
noise_eval_dataset = StationaryTransientNoiseDEARDataset(
    base_path=Path("/data/evaluation/dear"),
    split=DatasetType.TRAIN,
    target_variable_type=TargetVariableType.DISCRETE,
)
```

## Signal to Noise Ration (SNR)

```python
snr_eval_dataset = SNRDEARDataset(
    base_path=Path("/data/evaluation/dear"),
    split=DatasetType.TRAIN,
    target_variable_type=TargetVariableType.CONTINUOUS,
)
```

## Speech Present

```python
speech_present_eval_dataset = SpeechDEARDataset(
    base_path=Path("/data/evaluation/dear"),
    split=DatasetType.TRAIN,
    speech_present=True,
    target_variable_type=TargetVariableType.DISCRETE,
)
```

## Speakers Active

```python
speakers_active_eval_dataset = SpeechDEARDataset(
    base_path=Path("/data/evaluation/dear"),
    split=DatasetType.TRAIN,
    speech_present=False,
    target_variable_type=TargetVariableType.CONTINUOUS,
)
```

## Direct-to-Reverberant Ratio (DRR)

```python
drr_eval_dataset = DRRDEARDataset(
    base_path=Path("/data/evaluation/dear"),
    split=DatasetType.TRAIN,
    target_variable_type=TargetVariableType.CONTINUOUS,
)
```

## RT60

```python
rt60_eval_dataset = RT60DEARDataset(
    base_path=Path("/data/evaluation/dear"),
    split=DatasetType.TRAIN,
    target_variable_type=TargetVariableType.CONTINUOUS,
)
```
You can then use the normal PyTorch pattern to run your evaluation

```python
model = Wav2Vec2Model()
for segments, labels in rt60_eval_dataset:
    predicted_labels = model(segments)
    score = metric(labels, predicted_labels)
```
