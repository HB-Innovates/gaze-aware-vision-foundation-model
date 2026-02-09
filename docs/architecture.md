# Architecture Documentation

## System Overview

The Gaze-Aware Vision Foundation Model is a multi-modal deep learning system that combines vision understanding with gaze tracking for AR/VR applications.

## Component Architecture

### 1. Multi-Modal Foundation Model

```
┌────────────────────┐
│  Input Layer         │
│  - RGB Image         │
│  - Gaze Vector       │
└────────┬───────────┘
         │
    ┌────┼────┐
    │         │
┌───┴────┐  ┌┴──────┐
│ Vision   │  │ Gaze    │
│ Encoder  │  │ Encoder │
│ (CLIP)   │  │ (MLP)   │
└───┬────┘  └┬──────┘
    │         │
    └───┬────┘
        │
┌───────┴────────┐
│ Cross-Modal      │
│ Projection       │
└───────┬────────┘
        │
┌───────┴────────┐
│ Gaze-Guided      │
│ Attention        │
└───────┬────────┘
        │
┌───────┴────────┐
│ Output           │
│ Embeddings       │
└──────────────────┘
```

#### Vision Encoder
- **Architecture**: Vision Transformer (ViT-B/16)
- **Pretrained**: CLIP weights
- **Output**: 512-dimensional embeddings
- **Features**:
  - Patch-based image processing
  - Self-attention mechanisms
  - Transfer learning from CLIP

#### Gaze Encoder
- **Architecture**: Multi-layer MLP
- **Input**: 3D gaze vector (yaw, pitch, roll)
- **Output**: 512-dimensional embeddings
- **Layers**: [3 → 64 → 128 → 256 → 512]

#### Cross-Modal Projection
- Fuses vision and gaze embeddings
- Multiple fusion strategies:
  - Concatenation + Linear projection
  - Element-wise addition
  - Element-wise multiplication
- Output: 1024-dimensional fused embeddings

#### Gaze-Guided Attention
- Multi-head attention mechanism
- Gaze context modulates attention weights
- Enables focus on relevant visual regions
- 8 attention heads

### 2. Gaze Tracking Pipeline

```
Eye Images → CNN → Features → Regressor → Gaze Vector
                                    (yaw, pitch, roll)
```

#### CNN Architecture
- 4 convolutional blocks
- Progressive channel increase: [64 → 128 → 256 → 512]
- Batch normalization and dropout
- Global average pooling
- Fully connected regression head

#### Temporal Prediction
```
Gaze History → LSTM Encoder → Prediction Heads → Future Gaze
(t-9 to t)                                      (t+1 to t+5)
```

- LSTM with 2 layers, 128 hidden units
- Predicts 1-5 frames ahead
- Separate prediction head for each horizon

### 3. Power-Efficient Inference

#### Spiking Neural Network Conversion
1. **Rate Coding**: Convert activations to spike rates
2. **Temporal Dynamics**: Integrate-and-fire neurons
3. **Event-Driven**: Only active neurons compute
4. **Energy Savings**: 38x reduction in power consumption

#### Optimization Pipeline
```
Standard DNN → Quantization → Pruning → SNN Conversion → Optimized Model
```

## Data Flow

### Training
```
1. Load batch of (image, gaze) pairs
2. Forward pass through encoders
3. Compute cross-modal embeddings
4. Apply gaze-guided attention
5. Compute loss (MSE for gaze, contrastive for vision)
6. Backpropagate and update weights
```

### Inference
```
1. Preprocess input image and gaze
2. Extract features (CNN/ViT)
3. Predict gaze direction
4. Fuse modalities
5. Apply attention
6. Output predictions
```

## Performance Characteristics

### Model Sizes
- Vision Encoder: 86M parameters
- Gaze Encoder: 0.5M parameters
- Gaze Predictor: 8.2M parameters
- Total Multi-Modal: 95M parameters
- SNN Optimized: 24M parameters

### Computational Requirements
- Training: NVIDIA A100 (40GB) recommended
- Inference: NVIDIA T4 or better
- Mobile: Optimized for Apple Neural Engine / Qualcomm Hexagon

### Memory Footprint
- Training: 8-16GB GPU memory
- Inference (standard): 512MB
- Inference (optimized): 128MB

## Key Design Decisions

### 1. CLIP-based Vision Encoder
**Rationale**: Pretrained on large-scale vision-language data, strong transfer learning capabilities

### 2. MLP Gaze Encoder
**Rationale**: Gaze vectors are low-dimensional, simple architecture sufficient

### 3. Multi-Head Attention
**Rationale**: Enables model to attend to multiple visual regions guided by gaze

### 4. Temporal LSTM
**Rationale**: Captures sequential dependencies for predictive gaze forecasting

### 5. SNN Conversion
**Rationale**: Critical for mobile/wearable deployment, significant energy savings

## Future Enhancements

1. **Larger Vision Models**: Scale to ViT-L or ViT-H
2. **Multimodal Pretraining**: Self-supervised learning on unlabeled data
3. **Real-time Optimization**: TensorRT / CoreML integration
4. **Federated Learning**: Privacy-preserving personalization
5. **Multi-user Calibration**: Shared calibration across similar users
