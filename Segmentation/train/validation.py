from Segmentation.utils.data_loader import read_tfrecord_3d

def validate_best_model(chkpt_file, model, val_batch_size, buffer_size, tfrec_dir, multi_class,
                        crop_size, depth_crop_size, predict_slice):
    valid_ds = read_tfrecord_3d(tfrecords_dir=os.path.join(tfrec_dir, 'valid_3d/'),
                                is_training=False, predict_slice=predict_slice, 
                                batch_size=val_batch_size, buffer_size=buffer_size,
                                multi_class=multi_class)
    
    for x,y in valid_ds:
        tf.print(tf.shape(x), tf.shape(y))
    