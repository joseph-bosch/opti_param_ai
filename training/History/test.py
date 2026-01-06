import tensorflow as tf
print("TF:", tf.__version__)
print("Devices:", tf.config.list_physical_devices())
print("GPUs:", tf.config.list_physical_devices("GPU"))
print("Build:", getattr(tf.sysconfig, "get_build_info", lambda: {})())
