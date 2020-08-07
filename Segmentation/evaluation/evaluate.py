from Segmentation.utils.training_utils import visualise_binary, visualise_multi_class
from Segmentation.utils.evaluation_utils import update_cm, save_cm
from Segmentation.utils.evaluation_utils import update_gif_slice, update_volume_comp_gif, update_epoch_gif, update_volume_npy 
from Segmentation.utils.evaluation_utils import get_bucket_weights

# TODO: update to use fewer directory names
def get_bucket_weights(bucket_name, target_weights_dir):
    """ Load the checkpoints in the specified log directory. The
    target weights directory should contain the weights that 
    you want to load or testing / evaluation. """

    ######################
    """ Add the visualisation code here """
    print("+========================================================")
    print('bucket_name', bucket_name)
    print("\n\nThe directories are:")
    print('target_weights_dir', target_weights_dir)
    ######################

    # session_name = weights_dir.split('/')[3]
    # session_name = os.path.join(session_name, tpu_name, visual_file)
    session_name = target_weights_dir

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
    for s in session_weights:
        print(s) #print all the checkpoint directories
    print("--")
    ######################

    return session_weights

class Evaluator:

    def __init__(self,
                 validation_steps,
                 run_eager,
                 target_weights_dir,
                 visualisation_out_dir,
                 visualisation_args,
                 vis_weight_name=''):
        """ Args:
                visualisation_args: Settings like gif_volume, gif_epochs,
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
                  trained_model):

        """ Custom loop to evaluate model and visualize as needed """

        # set run conditions
        # TODO: Make strategy.run()
        run_eval_strategy = 0
        if self.run_eager:
            run_eval_strategy = tf.function(run_eval_strategy)

        multi_class = num_classes > 1
        gif_dir=''

        # load the checkpoints in the specified log directory
        session_weights = get_bucket_weights(bucket_name, logdir, tpu_name, visual_file, weights_dir)
        last_epoch = len(session_weights)

        # trained_model.load_weights(weights_dir).expect_partial()
        # trained_model.evaluate(dataset, steps=validation_steps, callbacks=callbacks)

        # Callbacks (as in og conf matrix function)
        f = weights_dir.split('/')[-1]
        # Excluding parenthese before f too
        if weights_dir.endswith(f):
            writer_dir = weights_dir[:-(len(f)+1)]
        writer_dir = os.path.join(writer_dir, 'eval')
        eval_metric_writer = tf.summary.create_file_writer(writer_dir)

        # Init visuals
        cm, classes = initialize_cm(multi_class, num_classes)
        fig, axes, images_gif = initialize_gif()
        target = 160 # how many slices in 1 vol
        sample_pred = []  # prediction for current 160,288,288 vol
        sample_y = []    # y for current 160,288,288 vol

        for chkpt in session_weights:
            ### Skip to last chkpt if you only want evaluation

            name = chkpt.split('/')[-1]
            name = name.split('.inde')[0]
            epoch = name.split('.')[1]

            #########################
            print("\n\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print(f"\t\tLoading weights from {epoch} epoch")
            print(f"\t\t  {name}")
            print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
            #########################

            trained_model.load_weights('gs://' + os.path.join(bucket_name,
                                                                weights_dir,
                                                                tpu_name,
                                                                visual_file,
                                                                name)).expect_partial()
            if epoch==last_epoch:
                trained_model.evaluate(dataset, steps=validation_steps, callbacks=callbacks)    


            # Initializing volume saving
            sample_pred = []  # prediction for current 160,288,288 vol
            sample_y = []    # y for current 160,288,288 vol

    def eval_step(self,
                  num_classes,
                  cm,
                  trained_model,
                  aug_strategy,
                  multi_as_binary,
                  multi_class,
                  which_epoch,
                  which_slice,
                  images_gif,
                  gif_dir,
                  pred,
                  target,
                  sample_pred,
                  sample_y,
                  visual_file,
                  name,
                  which_volume,
                  dataset):

        for step, (x, label) in enumerate(dataset):
            print('step ', step)
            pred = trained_model.predict(x)

            # Update visuals
            cm = update_cm(cm, num_classes)
            visualise_multi_class(label, pred)

            if step+1 == which_volume:
                update_gif_slice(x, label, trained_model,
                                 aug_strategy,
                                 multi_as_binary, multi_class,
                                 which_epoch, which_slice)

                images_gif = update_volume_comp_gif(x,label, images_gif, trained_model,
                                                    multi_class,
                                                    which_epoch,
                                                    gif_dir=gif_dir)

                images_gif = update_epoch_gif(x, trained_model, aug_strategy,
                                              multi_class, which_slice, 
                                              gif_dir=gif_dir)
            
                sample_pred, sample_y = update_volume_npy(label, pred, target, 
                                                          sample_pred, sample_y, 
                                                          visual_file, name, 
                                                          which_volume, multi_class)
                    
            iou = iou_loss_eval(label, pred) if multi_class else iou_loss(label, pred)
            dice = dice_coef_eval(label, pred) if multi_class else dice_coef(label, pred)

            with eval_metric_writer.as_default():
                tf.summary.scalar('iou eval validation', iou, step=step)
                tf.summary.scalar('dice eval validation', dice, step=step)

        # Save visuals
    #     save_cm(cm, model_architecture, fig_dir, classes)
    # pred_evolution_gif(fig, images_gif, save_dir=gif_dir, save=True, no_margins=False)