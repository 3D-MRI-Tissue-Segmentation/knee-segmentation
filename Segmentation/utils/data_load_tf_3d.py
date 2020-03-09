import tensorflow as tf

class VolumeGenerator(tf.data.Dataset):
    def _generator(num_samples):
        # Opening the file
        time.sleep(0.03)
        
        for sample_idx in range(num_samples):
            # Reading data (line, record) from the file
            time.sleep(0.015)
            
            yield (sample_idx,)
    
    def __new__(cls, num_samples=3):
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_types=tf.dtypes.int64,
            output_shapes=(1,),
            args=(num_samples,)
        )

if __name__ == "__main__":
    import sys, os, time
    sys.path.insert(0, os.getcwd())

    def benchmark(dataset, num_epochs=2):
        start_time = time.perf_counter()
        for epoch_num in range(num_epochs):
            for sample in dataset:
                # Performing a training step
                time.sleep(0.01)
        tf.print("Execution time:", time.perf_counter() - start_time)
    
    benchmark(VolumeGenerator())
    benchmark(
        VolumeGenerator()
        .prefetch(tf.data.experimental.AUTOTUNE)
    )

    benchmark(
        VolumeGenerator()
        .prefetch(tf.data.experimental.AUTOTUNE)
    )
