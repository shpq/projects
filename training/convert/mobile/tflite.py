import tensorflow as tf
import logging


def convert2tflite(model, save_path, inference_data=None, num_images=5):
    save_path = save_path.replace(".h5", ".tflite")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    logging.info("converting to tflite")
    tflite_model = converter.convert()

    with open(save_path, "wb") as f:
        f.write(tflite_model)

    if inference_data is None:
        return []

    logging.info("creating interpreter")
    interpreter = tf.lite.Interpreter(model_path=save_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    logging.info("starting inference")
    outputs = []

    for input_data in inference_data:
        input_data = input_data[0][:1, ...].astype("float32")
        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]["index"])
        outputs.append(output_data)
        if len(outputs) >= num_images:
            logging.info("done inference")
            break

    return outputs
