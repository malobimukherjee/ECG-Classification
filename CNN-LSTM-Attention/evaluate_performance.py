import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from tensorflow.keras.losses import CategoricalCrossentropy
from model_train import Config, FullModel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(csv_file, config):
    combined_beats_array = np.loadtxt(csv_file, delimiter=',', skiprows=1)
    labels = combined_beats_array[:, 0]
    ecg_values = combined_beats_array[:, 1:]
    scaler = StandardScaler()
    ecg_values_array = scaler.fit_transform(ecg_values)

    X_train, X_test, y_train, y_test = train_test_split(
        ecg_values_array, labels, test_size=0.2, random_state=42, stratify=labels
    )
    y_train = to_categorical(y_train, num_classes=config.num_classes)
    y_test = to_categorical(y_test, num_classes=config.num_classes)

    return X_train, X_test, y_train, y_test

def main():
    config = Config()

    # Load and preprocess data
    csv_file = 'C:/Users/DELL/OneDrive/Documents/combined_beats.csv'
    X_train, X_test, y_train, y_test = load_and_preprocess_data(csv_file, config)

    X_test = np.expand_dims(X_test, axis=-1)

    # Load model with custom objects
    model = load_model('C:/Users/DELL/OneDrive/Documents/trained_model.keras', custom_objects={'FullModel': FullModel, 'Config': Config})

    # Evaluate model
    test_loss, test_accuracy = model.evaluate(X_test, y_test, batch_size=config.batch_size)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # Predict and evaluate metrics
    y_pred = model.predict(X_test)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)

    accuracy = accuracy_score(y_test_labels, y_pred_labels)
    precision = precision_score(y_test_labels, y_pred_labels)
    recall = recall_score(y_test_labels, y_pred_labels)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    cm = confusion_matrix(y_test_labels, y_pred_labels)
    print("Confusion Matrix:")
    print(cm)

    # Plot loss distribution
    cce = CategoricalCrossentropy(reduction='none')
    per_sample_losses = cce(y_test, y_pred).numpy()

    normal_losses = per_sample_losses[y_test_labels == 0]
    abnormal_losses = per_sample_losses[y_test_labels == 1]

    plt.figure(figsize=(12, 6))
    plt.hist(normal_losses, bins=20, alpha=0.7, label='Normal Beats', color='blue')
    plt.hist(abnormal_losses, bins=20, alpha=0.7, label='Abnormal Beats', color='red')

    plt.xlabel('Loss')
    plt.ylabel('Frequency')
    plt.title('Loss Distribution')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
