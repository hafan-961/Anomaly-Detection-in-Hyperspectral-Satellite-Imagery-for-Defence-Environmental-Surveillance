import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import pickle
import os


# 1.CONFIGURATION

INPUT_DIM = 30
LATENT_DIM = 8
BATCH_SIZE = 256
EPOCHS = 50
LR = 1e-3



print(f"\nTensorFlow version: {tf.__version__}")
print(f"Available devices:  {[d.name for d in tf.config.list_physical_devices()]}")


# 2. LOAD PREPROCESSED DATA

print("\n[1/5] Loading preprocessed data...")
from sklearn.decomposition import PCA

X_train = np.load("data/processed/X_train.npy").astype(np.float32)
X_val = np.load("data/processed/X_val.npy").astype(np.float32)
X_test = np.load("data/processed/X_test.npy").astype(np.float32)

print("Fitting PCA on training data...")
pca = PCA(n_components=INPUT_DIM, random_state=42)
X_train_pca = pca.fit_transform(X_train).astype(np.float32)
X_val_pca = pca.transform(X_val).astype(np.float32)
X_test_pca = pca.transform(X_test).astype(np.float32)

print(f"Train shape: {X_train_pca.shape}")
print(f"Val shape:   {X_val_pca.shape}")
print(f"Test shape:  {X_test_pca.shape}")

os.makedirs("models", exist_ok=True)
with open("models/pca_model.pkl", "wb") as f:
    pickle.dump(pca, f)
print("PCA model saved → models/pca_model.pkl")


# 3. BUILD AUTOENCODER (definition kept, training skipped)
print("\n[2/5] Building Autoencoder...")


def build_autoencoder(input_dim, latent_dim):
    # ── Encoder ──
    encoder_input = keras.Input(shape=(input_dim,), name="encoder_input")
    x = layers.Dense(128, activation="relu")(encoder_input)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(32, activation="relu")(x)
    encoded = layers.Dense(latent_dim, activation="relu", name="latent")(x)
    # ── Decoder ──
    x = layers.Dense(32, activation="relu")(encoded)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation="relu")(x)
    decoded = layers.Dense(input_dim, activation="linear", name="reconstruction")(x)

    autoencoder = keras.Model(encoder_input, decoded, name="Autoencoder")
    encoder = keras.Model(encoder_input, encoded, name="Encoder")
    return autoencoder, encoder





# 4. BUILD VAE (custom model class)
print("\n[3/5] Building VAE...")


class VAE(keras.Model):
    def __init__(self, input_dim, latent_dim, **kwargs):
        super().__init__(**kwargs)
        # ── Encoder ──
        self.enc1 = layers.Dense(128, activation="relu")
        self.enc_bn1 = layers.BatchNormalization()
        self.enc2 = layers.Dense(64, activation="relu")
        self.enc_bn2 = layers.BatchNormalization()
        self.enc3 = layers.Dense(32, activation="relu")
        self.mu_layer = layers.Dense(latent_dim, name="mu")
        self.log_var_layer = layers.Dense(latent_dim, name="log_var")
        # ── Decoder ──
        self.dec1 = layers.Dense(32, activation="relu")
        self.dec_bn1 = layers.BatchNormalization()
        self.dec2 = layers.Dense(64, activation="relu")
        self.dec_bn2 = layers.BatchNormalization()
        self.dec3 = layers.Dense(128, activation="relu")
        self.output_layer = layers.Dense(input_dim, activation="linear")
        self.input_dim = input_dim
        self.latent_dim = latent_dim

    def build(self, input_shape):
        super().build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,})
        return config

    def encode(self, x, training=False):
        x = self.enc_bn1(self.enc1(x), training=training)
        x = self.enc_bn2(self.enc2(x), training=training)
        x = self.enc3(x)
        return self.mu_layer(x), self.log_var_layer(x)

    def decode(self, z, training=False):
        z = self.dec_bn1(self.dec1(z), training=training)
        z = self.dec_bn2(self.dec2(z), training=training)
        z = self.dec3(z)
        return self.output_layer(z)

    def reparameterize(self, mu, log_var):
        eps = tf.random.normal(shape=tf.shape(mu))
        return mu + tf.exp(0.5 * log_var) * eps

    def call(self, x, training=False):
        mu, log_var = self.encode(x, training=training)
        z = self.reparameterize(mu, log_var)
        return self.decode(z, training=training)

    def train_step(self, data):
        x, _ = data
        with tf.GradientTape() as tape:
            mu, log_var = self.encode(x, training=True)
            z = self.reparameterize(mu, log_var)
            recon = self.decode(z, training=True)
            recon_loss = tf.reduce_mean(tf.square(x - recon))
            kl_loss = -0.5 * tf.reduce_mean(
                1 + log_var - tf.square(mu) - tf.exp(log_var))
            total_loss = recon_loss + 0.0001 * kl_loss  # reduced KL weight
        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {"loss": total_loss, "recon_loss": recon_loss, "kl_loss": kl_loss}

    def test_step(self, data):
        x, _ = data
        mu, log_var = self.encode(x, training=False)
        z = self.reparameterize(mu, log_var)
        recon = self.decode(z, training=False)
        recon_loss = tf.reduce_mean(tf.square(x - recon))
        kl_loss = -0.5 * tf.reduce_mean(1 + log_var - tf.square(mu) - tf.exp(log_var))
        total_loss = recon_loss + 0.0001 * kl_loss  # reduced KL weight
        return {"loss": total_loss, "recon_loss": recon_loss, "kl_loss": kl_loss}

    def anomaly_score(self, x):
        mu, log_var = self.encode(x, training=False)
        z = self.reparameterize(mu, log_var)
        recon = self.decode(z, training=False)
        recon_err = tf.reduce_mean(tf.square(x - recon), axis=1)
        kl = -0.5 * tf.reduce_mean(
            1 + log_var - tf.square(mu) - tf.exp(log_var), axis=1
        )
        return (recon_err + 0.0001 * kl).numpy()


vae = VAE(input_dim=INPUT_DIM, latent_dim=LATENT_DIM, name="VAE")
vae.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4))  # reduced LR
vae.build((None, INPUT_DIM))
vae.summary()


# 5 CALLBACKS
def get_callbacks(model_name):
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=f"models/{model_name}_best.keras",
            monitor="val_loss",
            save_best_only=True,
            verbose=1,),]



# 6. LOAD SAVED AUTOENCODER (training skipped)

print("\n[4/5] Loading saved Autoencoder...")
autoencoder = keras.models.load_model("models/autoencoder_final.keras")



# 7. TRAIN VAE

print("\n[5/5] Training VAE...")
vae_history = vae.fit(
    X_train_pca,
    X_train_pca,
    validation_data=(X_val_pca, X_val_pca),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1),],
    verbose=1,)
vae.save_weights("models/vae_weights.weights.h5")
print("VAE saved → models/vae_weights.weights.h5")


# 8 COMPUTE ANOMALY SCORES

print("\nComputing anomaly scores on test set...")

ae_recon = autoencoder.predict(X_test_pca, verbose=0)
ae_scores = np.mean((X_test_pca - ae_recon) ** 2, axis=1)
vae_scores = vae.anomaly_score(X_test_pca)

# ensemble
ensemble_scores = 0.4 * ae_scores + 0.6 * vae_scores


def normalize_scores(scores):
    return (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)


ae_scores_norm = normalize_scores(ae_scores)
vae_scores_norm = normalize_scores(vae_scores)
ensemble_scores_norm = normalize_scores(ensemble_scores)

print(
    f"AE  — mean: {ae_scores_norm.mean():.4f}  "
    f"max: {ae_scores_norm.max():.4f}"
)
print(
    f"VAE — mean: {vae_scores_norm.mean():.4f}  "
    f"max: {vae_scores_norm.max():.4f}"
)
print(f"      Ens — mean: {ensemble_scores_norm.mean():.4f}  "f"max: {ensemble_scores_norm.max():.4f}")

os.makedirs("data/processed", exist_ok=True)
np.save("data/processed/ae_scores.npy", ae_scores_norm)
np.save("data/processed/vae_scores.npy", vae_scores_norm)
np.save("data/processed/ensemble_scores.npy", ensemble_scores_norm)


# 9 PLOT RESULTS

print("\nGenerating plots...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("AVIRIS-NG — Model Training Results", fontsize=16, fontweight="bold")


axes[0, 0].text(
    0.5, 0.5,
    "Autoencoder loaded from\nsaved model (not retrained)",
    ha="center", va="center",
    transform=axes[0, 0].transAxes,
    fontsize=12, color="gray"
)
axes[0, 0].set_title("Autoencoder — Training Loss")
axes[0, 0].grid(True, alpha=0.3)

# VAE Loss
axes[0, 1].plot(vae_history.history["loss"], label="Train", color="blue")
axes[0, 1].plot(vae_history.history["val_loss"], label="Val", color="red")
axes[0, 1].set_title("VAE — Training Loss")
axes[0, 1].set_xlabel("Epoch")
axes[0, 1].set_ylabel("Loss")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

#Score Distributions
axes[1, 0].hist(ae_scores_norm, bins=100, alpha=0.5, color="blue", label="Autoencoder")
axes[1, 0].hist(vae_scores_norm, bins=100, alpha=0.5, color="red", label="VAE")
axes[1, 0].hist(
    ensemble_scores_norm, bins=100, alpha=0.5, color="green", label="Ensemble"
)
axes[1, 0].set_title("Anomaly Score Distributions")
axes[1, 0].set_xlabel("Anomaly Score (0=normal, 1=anomaly)")
axes[1, 0].set_ylabel("Pixel Count")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

#Threshold
threshold = np.percentile(ensemble_scores_norm, 95)
axes[1, 1].hist(ensemble_scores_norm, bins=100, color="steelblue", alpha=0.7)
axes[1, 1].axvline(
    x=threshold,
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"95th percentile = {threshold:.3f}",)
axes[1, 1].set_title("Ensemble Score — Anomaly Threshold")
axes[1, 1].set_xlabel("Anomaly Score")
axes[1, 1].set_ylabel("Pixel Count")
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/03_training.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved → outputs/03_training.png")


print("\nModels saved:")
print("models/autoencoder_final.keras")
print("models/vae_weights.weights.h5")
print("models/pca_model.pkl")
print("\nScores saved:")
print("data/processed/ae_scores.npy")
print("data/processed/vae_scores.npy")
print("data/processed/ensemble_scores.npy")