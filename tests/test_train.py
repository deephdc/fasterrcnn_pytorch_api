import os
import shutil

#one may skip whole test with this decorator:
#@pytest.mark.skip(reason="takes too much resources")
def test_train_function(trained_model_path):
    checkpoint_file = os.path.join(
        trained_model_path, "last_model.pth"
    )
    assert "last_model.pth" in os.listdir(
        trained_model_path
    ), "No checkpoint was saved"
    assert os.path.exists(
        checkpoint_file
    ), "last_model.pth does not exist"
    # Remove the saved model directory and its contents
    shutil.rmtree(trained_model_path)
