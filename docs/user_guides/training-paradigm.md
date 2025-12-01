# PRISM Training Paradigm

**Important**: PRISM uses a fundamentally different training paradigm than typical deep learning.

## Key Differences from Standard Deep Learning

### 1. **Generative, Not Discriminative**

**Standard Deep Learning (Discriminative)**:
```python
# Model takes input and produces output
output = model(input)
loss = criterion(output, target)
```

**PRISM (Generative)**:
```python
# Model generates output from learned latent vector (no input!)
output = model()  # No input parameter!
loss = criterion(output, measurement)
```

### 2. **Single Sample Training, Not Batches**

**Standard Deep Learning**:
```python
# Train on batches of data
for batch in dataloader:  # batch_size = 32, 64, etc.
    output = model(batch_input)
    loss = criterion(output, batch_target)
    loss.backward()
```

**PRISM**:
```python
# Train on ONE sample at a time, sequentially
for sample_idx, sample_center in enumerate(sample_centers):
    measurement = telescope.measure(...)  # Single measurement
    output = model()  # Generate output
    loss = criterion(output, measurement)
    loss.backward()
```

### 3. **Progressive Reconstruction, Not Inference**

**Standard Deep Learning**:
- **Training Phase**: Learn from many examples
- **Inference Phase**: Apply learned model to new data

**PRISM**:
- **Only Training**: The entire algorithm IS the training process
- **No Inference**: Each reconstruction is done by training the model from scratch
- The trained latent vector IS the reconstruction

## How PRISM Works

1. **Initialize**: Train model to match an initial measurement
2. **Progressive Training**: For each new sample position:
   - Measure at that position
   - Update model to match BOTH old and new measurements
   - Continue until convergence
3. **Result**: The model's output `model()` is the final reconstruction

## Implications for Code

### ❌ Don't Do This (Batch Thinking)
```python
# Wrong: No batches in PRISM!
dataloader = DataLoader(dataset, batch_size=32)
for batch in dataloader:
    output = model(batch)

# Wrong: No inference phase!
model.eval()
predictions = model(test_data)

# Wrong: Model doesn't take inputs!
output = model(input_tensor)
```

### ✅ Do This (PRISM Paradigm)
```python
# Correct: Single sample, generative training
for sample_idx, center in enumerate(sample_centers):
    measurement = telescope.measure(image, centers=[center])
    output = model()  # No input!
    loss = criterion(output, measurement)
    loss.backward()
    optimizer.step()

# Correct: Training IS the algorithm
final_reconstruction = model()  # This is the result!
```

## Mixed Precision (AMP) with PRISM

Even with AMP, the paradigm is the same:

```python
# Correct AMP usage with PRISM
scaler = torch.cuda.amp.GradScaler()

for sample_idx, center in enumerate(sample_centers):
    measurement = telescope.measure(...)

    with torch.cuda.amp.autocast(enabled=True):
        output = model()  # No input!
        loss = criterion(output, measurement)

    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

## Testing Considerations

When writing tests for PRISM:

1. **No batch_size needed**: Tests should work with single samples
2. **No DataLoader needed**: PRISM doesn't use DataLoader
3. **Call model() without inputs**: `output = model()`, not `output = model(input)`
4. **Focus on reconstruction quality**: Not classification accuracy or prediction metrics

## Summary

| Aspect | Standard DL | PRISM |
|--------|-------------|-------|
| Model Type | Discriminative | Generative |
| Input | Takes input data | No input (learns latent) |
| Training | Batches of samples | One sample at a time |
| Phases | Training → Inference | Only training (training IS the algorithm) |
| Output | Predictions on new data | Reconstruction from measurements |
| Paradigm | Learn general patterns | Optimize for specific instance |

**Remember**: PRISM is an optimization-based reconstruction method, not a learned inference model!
