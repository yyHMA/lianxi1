import tensorflow as tf
import matplotlib.pyplot as plt
import argparse, math, os, time, traceback, librosa.display
from hparams import hparams
from models import create_model
from models.datafeeder import DataFeeder
from util import infolog, plot, ValueWindow
from util.text import sequence_to_text


log = infolog.log


def add_stats(model):
  with tf.variable_scope('stats') as scope:
    tf.summary.histogram('mel_outputs', model.mel_outputs)
    tf.summary.histogram('mel_targets', model.mel_targets)
    tf.summary.scalar('learning_rate', model.learning_rate)
    tf.summary.scalar('loss', model.loss)
    gradient_norms = [tf.norm(grad) for grad in model.gradients]
    tf.summary.histogram('gradient_norm', gradient_norms)
    tf.summary.scalar('max_gradient_norm', tf.reduce_max(gradient_norms))
    return tf.summary.merge_all()


def train(log_dir, args):
  checkpoint_path = os.path.join(log_dir, 'model.ckpt')
  input_path = os.path.join(args.base_dir, args.input)

  coord = tf.train.Coordinator()
  with tf.variable_scope('datafeeder') as scope:
    feeder = DataFeeder(coord, input_path, hparams)

  global_step = tf.Variable(0, name='global_step', trainable=False)
  with tf.variable_scope('model') as scope:
    model = create_model(args.model, hparams)
    model.initialize(feeder.inputs, feeder.input_lengths, feeder.mel_targets)
    model.add_loss()
    model.add_optimizer(global_step)
    stats = add_stats(model)

  # Bookkeeping
  time_window = ValueWindow(100)
  loss_window = ValueWindow(100)
  saver = tf.train.Saver(max_to_keep=300, keep_checkpoint_every_n_hours=2)

  # Train
  with tf.Session() as sess:
    try:
      summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
      sess.run(tf.global_variables_initializer())

      # 加 -20210303
      checkpoint_state = tf.train.get_checkpoint_state(log_dir)
      if (checkpoint_state and checkpoint_state.model_checkpoint_path):
        log('Loading checkpoint {}'.format(checkpoint_state.model_checkpoint_path), slack=True)
        saver.restore(sess, checkpoint_state.model_checkpoint_path)
      else:
        log('No model to load at {}'.format(log_dir), slack=True)
        saver.save(sess, checkpoint_path, global_step=global_step)

      if args.restore_step:
        restore_path = '%s-%d' % (checkpoint_path, args.restore_step)
        saver.restore(sess, restore_path)
        log('Resuming from checkpoint: %s' % (restore_path), slack=True)

      feeder.start_in_session(sess)

      while not coord.should_stop():
        start_time = time.time()
        step, loss, opt = sess.run([global_step, model.loss, model.optimize])
        time_window.append(time.time() - start_time)
        loss_window.append(loss)
        message = 'Step %-7d [%.03f sec/step, loss=%.05f, avg_loss=%.05f]' % (
          step, time_window.average, loss, loss_window.average)
        log(message, slack=(step % args.checkpoint_interval == 0))

        if step % args.summary_interval == 0:
          log('Writing summary at step: %d' % step)
          summary_writer.add_summary(sess.run(stats), step)

        if step % args.checkpoint_interval == 0:
          log('Saving checkpoint to: %s-%d' % (checkpoint_path, step))
          saver.save(sess, checkpoint_path, global_step=step)
          log('Saving alignment...')
          input_seq, alignment = sess.run([model.inputs[0], model.alignments[0]])
          input_seq = sequence_to_text(input_seq)
          print('-------------------1',input_seq)
          plot.plot_alignment(alignment, os.path.join(log_dir, 'step-%d-align.png' % step), input_seq,
            info='%s, step=%d, loss=%.5f' % (args.model, step, loss), istrain=1)
          log('Input: %s' % input_seq)

    except Exception as e:
      log('Exiting due to exception: %s' % e, slack=True)
      traceback.print_exc()
      coord.request_stop(e)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--base_dir', default=os.path.expanduser('/ssd-data/data'))
  parser.add_argument('--input', default='tacotron-train/train.txt')
  parser.add_argument('--model', default='tacotron')
  parser.add_argument('--restore_step', type=int)
  parser.add_argument('--summary_interval', type=int, default=100)
  parser.add_argument('--checkpoint_interval', type=int, default=1000)
  args = parser.parse_args()
  run_name = args.model
  log_dir = os.path.join(args.base_dir, 'logs-%s' % run_name)
  os.makedirs(log_dir, exist_ok=True)
  infolog.init(os.path.join(log_dir, 'train.log'), run_name)
  train(log_dir, args)


if __name__ == '__main__':
  main()
