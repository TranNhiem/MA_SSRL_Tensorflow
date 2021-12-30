## 3. perform the train/eval loop
    #   (1) training framework :
    if "train" in FLAGS.mode:



        
        with self.strategy.scope():
            # Configure the learning rate
            if FLAGS.lr_strategies == "warmup_cos_lr":
                base_lr = FLAGS.base_lr
                scale_lr = FLAGS.lr_rate_scaling
                warmup_epochs = FLAGS.warmup_epochs
                train_epochs = FLAGS.train_epochs

                lr_schedule = WarmUpAndCosineDecay(
                    base_lr, train_global_batch, n_tra_sample, scale_lr, warmup_epochs,
                    train_epochs=train_epochs, train_steps=train_steps)

            elif FLAGS.lr_strategies == "cos_annealing_restart":
                base_lr = FLAGS.base_lr
                scale_lr = FLAGS.lr_rate_scaling
                # Control cycle of next step base of Previous step (2 times more steps)
                t_mul = 2.0
                # Control ititial Learning Rate Values (Next step equal to previous steps)
                m_mul = 1.0
                alpha = 0.0  # Final values of learning rate
                first_decay_steps = train_steps / (FLAGS.number_cycles * t_mul)
                lr_schedule = CosineAnnealingDecayRestarts(
                    base_lr, first_decay_steps, train_global_batch, scale_lr, t_mul=t_mul, m_mul=m_mul, alpha=alpha)

            # Current Implement the Mixpercision optimizer
            optimizer = all_model.build_optimizer(lr_schedule)

            # Build tracking metrics
            all_metrics = []
            # Linear classfiy metric
            weight_decay_metric = tf.keras.metrics.Mean('train/weight_decay')
            total_loss_metric = tf.keras.metrics.Mean('train/total_loss')
            all_metrics.extend([weight_decay_metric, total_loss_metric])

            if FLAGS.train_mode == 'pretrain':
                # for contrastive metrics
                contrast_loss_metric = tf.keras.metrics.Mean(
                    'train/non_contrast_loss')
                contrast_acc_metric = tf.keras.metrics.Mean(
                    "train/non_contrast_acc")
                contrast_entropy_metric = tf.keras.metrics.Mean(
                    'train/non_contrast_entropy')
                all_metrics.extend(
                    [contrast_loss_metric, contrast_acc_metric, contrast_entropy_metric])

            if FLAGS.train_mode == 'finetune' or FLAGS.lineareval_while_pretraining:
                logging.info(
                    "Apllying pre-training and Linear evaluation at the same time")
                # Fine-tune architecture metrics
                supervised_loss_metric = tf.keras.metrics.Mean(
                    'train/supervised_loss')
                supervised_acc_metric = tf.keras.metrics.Mean(
                    'train/supervised_acc')
                all_metrics.extend(
                    [supervised_loss_metric, supervised_acc_metric])




            # Check and restore Ckpt if it available
            # Restore checkpoint if available.
            checkpoint_manager = try_restore_from_checkpoint(
                online_model, optimizer.iterations, optimizer)

            global_step = optimizer.iterations
            for epoch in trange(FLAGS.train_epochs):

                total_loss = 0.0
                num_batches = 0

                for _, (ds_one, ds_two) in enumerate(train_ds):

                    total_loss += distributed_train_step(ds_one, ds_two)
                    num_batches += 1

                    # Update weight of Target Encoder Every Step
                    beta = 0.99
                    target_encoder_weights = target_model.get_weights()
                    online_encoder_weights = online_model.get_weights()

                    for i in range(len(online_encoder_weights)):
                        target_encoder_weights[i] = beta * target_encoder_weights[i] + (
                            1-beta) * online_encoder_weights[i]
                    target_model.set_weights(target_encoder_weights)

                    # if (global_step.numpy()+ 1) % checkpoint_steps==0:

                    with summary_writer.as_default():
                        cur_step = global_step.numpy()
                        checkpoint_manager.save(cur_step)
                        logging.info('Completed: %d / %d steps',
                                     cur_step, train_steps)
                        metrics.log_and_write_metrics_to_summary(
                            all_metrics, cur_step)
                        tf.summary.scalar('learning_rate', lr_schedule(tf.cast(global_step, dtype=tf.float32)),
                                          global_step)
                        summary_writer.flush()

                epoch_loss = total_loss/num_batches
                # Wandb Configure for Visualize the Model Training
                wandb.log({
                    "epochs": epoch+1,
                    "train_contrast_loss": contrast_loss_metric.result(),
                    "train_contrast_acc": contrast_acc_metric.result(),
                    "train_contrast_acc_entropy": contrast_entropy_metric.result(),
                    "train/weight_decay": weight_decay_metric.result(),
                    "train/total_loss": epoch_loss,
                    "train/supervised_loss":    supervised_loss_metric.result(),
                    "train/supervised_acc": supervised_acc_metric.result()
                })
                for metric in all_metrics:
                    metric.reset_states()
                # Saving Entire Model
                if (epoch+1) % 20 == 0:
                    save_encoder = os.path.join(
                        FLAGS.model_dir, "encoder_model_" + str(epoch) + ".h5")
                    save_online_model = os.path.join(
                        FLAGS.model_dir, "online_model_" + str(epoch) + ".h5")
                    save_target_model = os.path.join(
                        FLAGS.model_dir, "target_model_" + str(epoch) + ".h5")
                    online_model.resnet_model.save_weights(save_encoder)
                    online_model.save_weights(save_online_model)
                    target_model.save_weights(save_target_model)
            logging.info('Training Complete ...')

    #   (2) evaluation framework :
    if "eval" in FLAGS.mode:
        ckpt_iter = [checkpoint_manager.latest_checkpoint] if "train" not in FLAGS.mode \
                        else tf.train.checkpoints_iterator(FLAGS.model_dir, min_interval_secs=15)
        for ckpt in ckpt_iter:
            perform_evaluation(online_model, val_ds, eval_steps, ckpt, strategy)
            
            # global_step from ckpt
            if result['global_step'] >= train_steps:
                logging.info('Evaluation complete. Existing-->')