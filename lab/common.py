"""Shared utilities for CSA4712 Deep Learning experiments."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED)

def finish(name):
    out = Path(__file__).resolve().parent / 'outputs'
    out.mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(out / (name + '.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved:', out / (name + '.png'))

def tf_setup():
    try:
        import tensorflow as tf
    except ImportError as e:
        raise SystemExit('TensorFlow is required for this experiment. Install with: pip install tensorflow') from e
    tf.keras.utils.set_random_seed(SEED)
    return tf

def regression_sequence(n=500, length=12):
    x = np.linspace(0, 22, n, dtype=np.float32)
    signal = np.sin(x) + .2 * np.sin(3*x)
    X = np.array([signal[i:i+length] for i in range(n-length)], dtype=np.float32)
    y = signal[length:].astype(np.float32)
    return X[..., None], y[..., None]

if __name__ == '__main__':
    print('Shared utility module, not an experiment.')
