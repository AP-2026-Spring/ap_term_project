import tensorflow as tf
interpreter = tf.lite.Interpreter(model_path='best_full_integer_quant_edgetpu.tflite')
interpreter.allocate_tensors()
out_details = interpreter.get_output_details()[0]
print("Output Shape:", out_details['shape'])
print("Scale:", out_details['quantization'][0])
print("Zero Point:", out_details['quantization'][1])
