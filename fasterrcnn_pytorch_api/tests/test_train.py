import os

def test_train_function(trained_model_path):
    assert 'best_model.pth' in os.listdir(trained_model_path), f"No checkpoint was saved"
