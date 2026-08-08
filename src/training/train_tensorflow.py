import mlflow
import mlflow.keras

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("TensorFlow Experiments")

mlflow.keras.autolog()

with mlflow.start_run(run_name="tensorflow-baseline"):

    model.fit(
        X_train,
        y_train,
        epochs=10,
        validation_data=(X_val, y_val)
    )
