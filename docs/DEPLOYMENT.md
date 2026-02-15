# Deployment Guide

Production deployment instructions.

## Docker Deployment

### Build Image
```dockerfile
FROM tensorflow/tensorflow:2.15.0-gpu

COPY . /app
WORKDIR /app
RUN pip install -e .

CMD ["python", "serve.py"]
```

Build: `docker build -t coffee-classifier .`
Run: `docker run -p 8000:8000 coffee-classifier`

## API Server

### Flask Example
```python
from flask import Flask, request
import tensorflow as tf

app = Flask(__name__)
model = tf.keras.models.load_model('best_model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    image = request.files['image']
    prediction = model.predict(preprocess(image))
    return {'class': int(prediction.argmax())}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
```

## Cloud Deployment

### AWS SageMaker
1. Package model
2. Upload to S3
3. Create endpoint
4. Deploy

### Google Cloud AI Platform
1. Export SavedModel
2. Upload to GCS
3. Create model version
4. Deploy endpoint

## Monitoring

Use TensorBoard for monitoring:
```bash
tensorboard --logdir=outputs/tensorboard
```

## Security

- API authentication
- Rate limiting
- Input validation
- Model encryption
