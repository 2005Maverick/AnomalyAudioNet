import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self, model_path: str = "models"):
        """
        Initialize the model evaluator
        
        Args:
            model_path (str): Path to the trained model
        """
        self.model_path = Path(model_path)
        self.model = None
        
    def load_model(self, model_name: str):
        """
        Load the trained model
        
        Args:
            model_name (str): Name of the model file
        """
        try:
            logger.info(f"Loading model from {self.model_path / model_name}")
            # self.model = tf.keras.models.load_model(self.model_path / model_name)
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
            
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Evaluate the model on test data
        
        Args:
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test labels
            
        Returns:
            dict: Evaluation metrics
        """
        try:
            logger.info("Evaluating model...")
            # Add your evaluation logic here
            # Example:
            # predictions = self.model.predict(X_test)
            # metrics = {
            #     'accuracy': accuracy_score(y_test, predictions),
            #     'classification_report': classification_report(y_test, predictions),
            #     'confusion_matrix': confusion_matrix(y_test, predictions)
            # }
            return {}
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            raise
            
    def plot_confusion_matrix(self, confusion_mat: np.ndarray, class_names: list,
                            output_path: str = "results/confusion_matrix.png"):
        """
        Plot and save confusion matrix
        
        Args:
            confusion_mat (np.ndarray): Confusion matrix
            class_names (list): List of class names
            output_path (str): Path to save the plot
        """
        try:
            plt.figure(figsize=(10, 8))
            sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues',
                       xticklabels=class_names, yticklabels=class_names)
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig(output_path)
            plt.close()
        except Exception as e:
            logger.error(f"Error plotting confusion matrix: {str(e)}")
            raise

if __name__ == "__main__":
    evaluator = ModelEvaluator()
    # Load your test data here
    # evaluator.load_model("model.h5")
    # metrics = evaluator.evaluate(X_test, y_test)
    # evaluator.plot_confusion_matrix(metrics['confusion_matrix'], class_names) 