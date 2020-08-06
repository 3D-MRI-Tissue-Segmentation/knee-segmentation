


from Segmentation.utils.evaluation_utils import get_bucket_weights


# TODO: update to use fewer directory names
def get_all_weights(bucket_name, target_weights_dir):
    """ Load the checkpoints in the specified log directory. The
    target weights directory should contain the weights that 
    you want to load or testing / evaluation. If a google cloud 
    storage bucket is given as bucket_name, load weights from 
    GCS, otherwise just return the weights directory. """

    # session_name = weights_dir.split('/')[3]
    # session_name = os.path.join(session_name, tpu_name, visual_file)
    session_name = target_weights_dir

    if bucket_name:
        # Get names within folder in gcloud
        storage_client = storage.Client()
        blobs = storage_client.list_blobs(bucket_name)
        session_content = []
        print('session_name',session_name)
        for blob in blobs:
            if session_name in blob.name:
                session_content.append(blob.name)

        session_weights = []
        for item in session_content:
            if ('_weights' in item) and ('.ckpt.index' in item):
                session_weights.append(item)

        ######################
        print("+========================================================")
        print('bucket_name', bucket_name)
        print("\n\nThe directories are:")
        print('target_weights_dir', target_weights_dir)
        for s in session_weights:
            print(s) #print all the checkpoint directories
        print("--")
        ######################

        return session_weights
    else:
        return target_weights_dir


class Evaluator:

    def __init__(self,
                 validation_steps,
                 run_eager,
                 target_weights_dir,
                 vis_out_dir,
                 vis_args,
                 vis_weight_name=''):
        """ Args:
                vis_args: Settings like gif_volume, gif_epochs,
                    gif_slice, num_classes, model_architecture
        """


                #   logdir=FLAGS.logdir,
                #   tpu_name=tpu,
                #   visual_file=FLAGS.visual_file,
                #   weights_dir=FLAGS.weights_dir,

                #   multi_as_binary=False,
                #   trained_model=model,
                #   model_architecture=FLAGS.model_architecture,
                #   callbacks=[tb],
                #   num_classes=num_classes)

        self.run_eager = run_eager
        self.val_steps = validation_steps
        self.target_weights_dir = target_weights_dir

    def eval_loop(self,
                  valid_ds,
                  train_ds,
                  trained_model,
                  chosen_epoch: int =-1):

        """ Custom loop to evaluate model and visualize as needed
        Args:
            chosen_epoch: weights from epoch number you want to load, -1 for last epoch"""

        # In case epoch given as -1, make into a real int index value
        chosen_epoch = session_weights.index(session_weights[chosen_epoch])

        # set run conditions
        # TODO: Make strategy.run()
        run_eval_strategy = 0
        if self.run_eager:
            run_eval_strategy = tf.function(run_eval_strategy)

        multi_class = num_classes > 1
        gif_dir=''

        # load the checkpoints in the specified log directory
        session_weights = get_all_weights(bucket_name, logdir, tpu_name, visual_file, weights_dir)
        last_epoch = len(session_weights)

        # trained_model.load_weights(weights_dir).expect_partial()
        # trained_model.evaluate(dataset, steps=validation_steps, callbacks=callbacks)


        # TODO: METRICS, CHANGE LATER
        # Callbacks (as in og conf matrix function)
        f = weights_dir.split('/')[-1]
        # Excluding parenthese before f too
        if weights_dir.endswith(f):
            writer_dir = weights_dir[:-(len(f)+1)]
        writer_dir = os.path.join(writer_dir, 'eval')
        eval_metric_writer = tf.summary.create_file_writer(writer_dir)
        #######################

        # Init visuals
        cm, classes = initialize_cm(multi_class, num_classes)
        fig, axes, images_gif = initialize_gif()
        target = 160 # how many slices in 1 vol
        sample_pred = []  # prediction for current 160,288,288 vol
        sample_y = []    # y for current 160,288,288 vol

        for epoch, chkpt in enumerate(session_weights):
            ### Skip to last chkpt if you only want evaluation

            name = chkpt.split('/')[-1]
            name = name.split('.inde')[0]
            # epoch = name.split('.')[1]

            #########################
            print("\n\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print(f"\t\tLoading weights from {epoch} epoch")
            print(f"\t\t  {name}")
            print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
            #########################

            # TODO: test that the chkpt is actually the full pathname to the target weight, incl bucket / local
            # --> may need to check what each 'blob' returned by the storage_client.list_blobs() 
            if bucket_name:
                trained_model.load_weights('gs://' + chkpt).expect_partial()
            else:
                trained_model.load_weights(chkpt).expect_partial()
            
            if epoch==chosen_epoch:
                trained_model.evaluate(dataset, steps=validation_steps, callbacks=callbacks)    


            # Initializing volume saving
            sample_pred = []  # prediction for current 160,288,288 vol
            sample_y = []    # y for current 160,288,288 vol

        def eval_step():

        