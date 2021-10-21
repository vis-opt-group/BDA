import argparse
import pickle
from threading import Thread
import os

import tensorflow as tf
import numpy as np
import time
import boml
import boml.load_data as dl
from boml.utils import feed_dicts
from sklearn.manifold import TSNE
DATASETS_FOLDER = '../omniglot_resized'
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser()

parser.add_argument('-m', '--mode', type=str, default="train", metavar='STRING',
                    help='mode, can be train or test')

# Dataset/method options
parser.add_argument('-d', '--dataset', type=str, default='omniglot', metavar='STRING',
                    help='omniglot or miniimagenet.')
parser.add_argument('-nc', '--classes', type=int, default=5, metavar='NUMBER',
                    help='number of classes used in classification (c for  c-way classification).')
parser.add_argument('-etr', '--examples_train', type=int, default=1, metavar='NUMBER',
                    help='number of examples used for inner gradient update (k for k-shot learning).')
parser.add_argument('-etes', '--examples_test', type=int, default=15, metavar='NUMBER',
                    help='number of examples used for test sets')

# Training options
parser.add_argument('-sd', '--seed', type=int, default=0, metavar='NUMBER',
                    help='seed for random number generators')
parser.add_argument('-mbs', '--meta_batch_size', type=int, default=16, metavar='NUMBER',
                    help='number of tasks sampled per meta-updat                                                                                                                                                                  e')
parser.add_argument('-nmi', '--meta_train_iterations', type=int, default=3000, metavar='NUMBER',
                    help='number of metatraining iterations.')
parser.add_argument('-T', '--T', type=int, default=5, metavar='NUMBER',
                    help='number of inner updates during training.')
parser.add_argument('-xi', '--xavier', type=bool, default=False, metavar='BOOLEAN',
                    help='FFNN weights initializer')
parser.add_argument('-bn', '--batch-norm', type=bool, default=False, metavar='BOOLEAN',
                    help='Use batch normalization before classifier')
parser.add_argument('-mlr', '--meta-lr', type=float, default=0.001, metavar='NUMBER',
                    help='starting meta learning rate')
parser.add_argument('-mlrdr', '--meta-lr-decay-rate', type=float, default=1.e-5, metavar='NUMBER',
                    help='meta lr  inverse time decay rate')

parser.add_argument('-cv', '--clip_value', type=float, default=0., metavar='NUMBER',
                    help='meta gradient clip value (0. for no clipping)')
parser.add_argument('-lr', '--lr', type=float, default=0.1, metavar='NUMBER',
                    help='starting learning rate')

# settings of parameters for specific algorithms
parser.add_argument('-alpha_decay', '--alpha_decay', type=float, default=1.e-5, metavar='NUMBER',
                    help='alpha decay rate')
parser.add_argument('-alpha', '--alpha', type=float, default=0.6, metavar='NUMBER',
                    help='factor for controlling the ratio of gradients')
parser.add_argument('-ds', '--darts', type=bool, default=False, metavar='BOOLEAN',
                    help='whether to implement Darts Method')
parser.add_argument('-fo', '--first_order', type=bool, default=False, metavar='BOOLEAN',
                    help='whether to implement FOMAML, short for First Order MAML')
parser.add_argument('-ga', '--gamma', type=float, default=1.0, metavar='NUMBER',
                    help='coefficient for BA to be used in the UL calculation process')
parser.add_argument('-lrl', '--learn_lr', type=bool, default=False, metavar='BOOLEAN',
                    help='True if learning rate is an hyperparameter')
parser.add_argument('-lrst', '--learn_st', type=bool, default=False, metavar='BOOLEAN',
                    help='True if s and t are outer parameters')
parser.add_argument('-lralpha', '--learn_alpha', type=bool, default=False, metavar='BOOLEAN',
                    help='True if alpha is an hyperparameter')
parser.add_argument('-learn_alpha_itr', '--learn_alpha_itr', type=bool, default=False, metavar='BOOLEAN',
                    help='learn alpha iteration wise')

parser.add_argument('-md', '--method', type=str, default='MetaRepr', metavar='STRING',
                    help='choose which method to use,[Trad, Aggr,Simple]')
parser.add_argument('-scalor', '--scalor', type=float, default=0.0, metavar='NUMBER',
                    help='scalor for controlling the regularization coefficient')
parser.add_argument('-tr_ir', '--truncate_iter', type=int, default=-1, metavar='NUMBER',
                    help='truncated iterations ')
parser.add_argument('-i_d', '--inner_method', type=str, default='Aggr', metavar='STRING',
                    help='choose which method to use,[Trad, Aggr,Simple]')
parser.add_argument('-o_d', '--outer_method', type=str, default='Reverse', metavar='STRING',
                    help='choose which method to use,[Reverse,Implicit,Forward,Simple]')
parser.add_argument('-io', '--inner_opt', type=str, default='SGD', metavar='STRING',
                    help='the typer of inner optimizer, which should be listed in [SGD,Adam,Momentum]')
parser.add_argument('-oo', '--outer_opt', type=str, default='Adam', metavar='STRING',
                    help='the typer of outer optimizer, which should be listed in [SGD,Adam,Momentum]')
parser.add_argument('-u_t', '--use_t', type=bool, default=False, metavar='BOOLEAN',
                    help='whether use T-Net')
parser.add_argument('-u_w', '--use_warp', type=bool, default=False, metavar='BOOLEAN',
                    help='whether use Warp layer to implement Warp-MAML')
parser.add_argument('-bs', '--ba_s', type=float, default=1.0, metavar='NUMBER',
                    help='coefficient for UL objective for BDA algorithm')
parser.add_argument('-bt', '--ba_t', type=float, default=1.0, metavar='NUMBER',
                    help='coefficient for LL objectiv for BDA algorithm')
parser.add_argument('-la', '--warp_lambda', type=float, default=1.0, metavar='NUMBER',
                    help='coefficient for WarpGrad to be used in the UL calculation process')



# Logging, saving, and testing options
parser.add_argument('-log', '--log', type=bool, default=False, metavar='BOOLEAN',
                    help='if false, do not log summaries, for debugging code.')
parser.add_argument('-ld', '--logdir', type=str, default='logs/', metavar='STRING',
                    help='directory for summaries and checkpoints.')
parser.add_argument('-res', '--resume', type=bool, default=True, metavar='BOOLEAN',
                    help='resume training if there is a model available')
parser.add_argument('-pi', '--print-interval', type=int, default=1, metavar='NUMBER',
                    help='number of meta-train iterations before print')
parser.add_argument('-si', '--save_interval', type=int, default=1, metavar='NUMBER',
                    help='number of meta-train iterations before save')
parser.add_argument('-te', '--test_episodes', type=int, default=600, metavar='NUMBER',
                    help='number of episodes for testing')


# Testing options (put parser.mode = 'test')
parser.add_argument('-exd', '--expdir', type=str, default=None, metavar='STRING',
                    help='directory of the experiment model files')
parser.add_argument('-itt', '--iterations_to_test', type=str, default=[50000], metavar='STRING',
                    help='meta_iteration to test (model file must be in "exp_dir")')
parser.add_argument('-Notes', '--Notes', type=str, default='BDA_Notes',
                    help='Something important')
args = parser.parse_args()

exp_string = str(args.classes) + 'way_' + str(args.examples_train) + 'shot_' + str(args.meta_batch_size) + \
             'mbs' + str(args.T) + 'T' + str(args.clip_value) + 'clip' + str(args.alpha) + 'alpha' + str(args.inner_method) +\
             'inner_method' + str(args.outer_method) + 'outer_method'+str(args.meta_lr) + 'meta_lr' + str(args.lr) + 'lr' + str(args.Notes) + 'Notes'

def meta_train(exp_dir, metasets, exs, pybml_ho, saver, sess, n_test_episodes, MBS, seed, resume, T,
               n_meta_iterations, print_interval, save_interval):
    result_path = os.path.join(exp_dir, 'results.pickle')
    tf.global_variables_initializer().run(session=sess)
    n_test_batches = n_test_episodes // MBS
    rand = dl.get_rand_state(seed)

    results = {'train_train': {'mean': [], 'std': []}, 'train_test': {'mean': [], 'std': []},
               'test_test': {'mean': [], 'std': []}, 'valid_test': {'mean': [], 'std': []},
               'outer_losses': {'mean': [], 'std': []}, 'learning_rate': [], 'iterations': [],
               'episodes': [], 'time': [], 'alpha': [],'svd_layer':[],'weights':[],'orthogonality':[],
               'clip_weights':[]}

    resume_itr = 0
    if resume:
        model_file = tf.train.latest_checkpoint(exp_dir)
        if model_file:
            print("Restoring results from " + result_path)
            results = load_obj(result_path)

            ind1 = model_file.index('model')
            resume_itr = int(model_file[ind1 + 5:]) + 1
            print("Restoring model weights from " + model_file)
            saver.restore(sess, model_file)

    ''' Meta-Train '''
    train_batches = BatchQueueMock(metasets.train, 1, MBS, rand)
    valid_batches = BatchQueueMock(metasets.validation, n_test_batches, MBS, rand)
    test_batches = BatchQueueMock(metasets.test, n_test_batches, MBS, rand)


    print('\nIteration quantities: train_train acc, train_test acc, valid_test, acc'
          ' test_test acc mean(std) over %d episodes' % n_test_episodes)
    with sess.as_default():
        start_time = time.time()
        inner_losses = []
        for meta_it in range(resume_itr, n_meta_iterations):
            start_time = time.time()
            tr_fd, v_fd = feed_dicts(train_batches.get_all_batches()[0], exs)
            pybml_ho.run(tr_fd, v_fd)

            duration = time.time() - start_time

            results['time'].append(duration)
            outer_losses = []
            for _, ex in enumerate(exs):
                outer_losses.append(sess.run(ex.errors['validation'], boml.utils.merge_dicts(tr_fd, v_fd)))
            outer_losses_moments = (np.mean(outer_losses), np.std(outer_losses))
            results['outer_losses']['mean'].append(outer_losses_moments[0])
            results['outer_losses']['std'].append(outer_losses_moments[1])

            if meta_it % print_interval == 0 or meta_it == n_meta_iterations - 1:
                results['iterations'].append(meta_it)
                results['episodes'].append(meta_it * MBS)
                if 'alpha' in pybml_ho.param_dict.keys():
                    alpha_moment = pybml_ho.param_dict['alpha'].eval()
                    print('alpha_itr' + str(meta_it) + ': ', alpha_moment)
                    results['alpha'].append(alpha_moment)
                if 's' in pybml_ho.param_dict.keys():
                    s = sess.run(["s:0"])[0]
                    print('s: {}'.format(s))
                if 't' in pybml_ho.param_dict.keys():
                    t = sess.run(["t:0"])[0]
                    print('t: {}'.format(t))
                if hasattr(pybml_ho.meta_model,"svd_layer"):
                    svd_layer = sess.run(pybml_ho.meta_model.svd_layer)
                    results['svd_layer'].append(svd_layer)
                if hasattr(pybml_ho.meta_model, "orthogonality"):
                    orthogonality = sess.run(pybml_ho.meta_model.orthogonality)
                    results['orthogonality'].append(orthogonality)
                    print(orthogonality)
                if hasattr(pybml_ho.meta_model,"clip_weights"):
                    clip_weights = sess.run(pybml_ho.meta_model.clip_weights)
                    results['clip_weights'].append(clip_weights)

                train_result = accuracy_on(train_batches, exs, pybml_ho, sess, T)
                test_result = accuracy_on(test_batches, exs, pybml_ho, sess, T)
                valid_result = accuracy_on(valid_batches, exs, pybml_ho, sess, T)
                '''
                data = test_result[2][0]
                label = test_result[2][1]
                train_label = test_result[2][2]
                t_sne = TSNE(n_components=2, init='pca', random_state=0)
                result_orig = t_sne.fit_transform(data)
                plot_embedding(result_orig, label, 't-sne enbedding of the digits orig')
                plot_embedding(result_orig, train_label, 't-sne enbedding of the digits train')
            '''
                train_train = (np.mean(train_result[0]), np.std(train_result[0]))
                train_test = (np.mean(train_result[1]), np.std(train_result[1]))
                valid_test = (np.mean(valid_result[1]), np.std(valid_result[1]))
                test_test = (np.mean(test_result[1]), np.std(test_result[1]))

                results['train_train']['mean'].append(train_train[0])
                results['train_test']['mean'].append(train_test[0])
                results['valid_test']['mean'].append(valid_test[0])
                results['test_test']['mean'].append(test_test[0])

                results['train_train']['std'].append(train_train[1])
                results['train_test']['std'].append(train_test[1])
                results['valid_test']['std'].append(valid_test[1])
                results['test_test']['std'].append(test_test[1])

                results['inner_losses'] = inner_losses

                print('mean outer losses: {}'.format(outer_losses_moments[0]))



                print(
                    'it %d, ep %d (%.5fs): %.5f, %.5f, %.5f, %.5f' % (meta_it, meta_it * MBS, duration, train_train[0],
                                                                      train_test[0], valid_test[0], test_test[0]))

                lr = sess.run(["lr:0"])[0]
                print('lr: {}'.format(lr))

                # do_plot(logdir, results)

            if meta_it % save_interval == 0 or meta_it == n_meta_iterations - 1:
                saver.save(sess, exp_dir + '/model' + str(meta_it))
                save_obj(result_path, results)



        return results


def meta_test(exp_dir, metasets, exs, pybml_ho, saver, sess, c_way, k_shot, lr, n_test_episodes, MBS, seed, T,
              iterations=list(range(10000))):
    meta_test_str = str(c_way) + 'way_' + str(k_shot) + 'shot_' \
                    + str(T) + 'T' + str(lr) + 'lr' + str(n_test_episodes) + 'ep'

    n_test_batches = n_test_episodes // MBS
    rand = dl.get_rand_state(seed)

    valid_batches = BatchQueueMock(metasets.validation, 1, MBS, rand)
    test_batches = BatchQueueMock(metasets.test, 1, MBS, rand)

    print('\nMeta-testing {} (over {} eps)...'.format(meta_test_str, n_test_episodes))

    test_results = {'test_test': {'mean': [], 'std': []}, 'valid_test': {'mean': [], 'std': []},
                    'cp_numbers': [], 'time': [],
                    'n_test_episodes': n_test_episodes, 'episodes': [], 'iterations': []}

    test_result_path = os.path.join(exp_dir, meta_test_str + '_results.pickle')

    start_time = time.time()
    for i in iterations:
        model_file = os.path.join(exp_dir, 'model' + str(i))
        if tf.train.checkpoint_exists(model_file):
            print("Restoring model weights from " + model_file)
            saver.restore(sess, model_file)

            test_results['iterations'].append(i)
            test_results['episodes'].append(i * MBS)

            valid_result = accuracy_on(valid_batches, exs, pybml_ho, sess, T)
            test_result = accuracy_on(test_batches, exs, pybml_ho, sess, T)
            data =test_result[2][0]
            label =test_result[2][1]
            train_label = test_result[2][2]
            t_sne = TSNE(n_components=2, init='pca', random_state=0)
            result_orig = t_sne.fit_transform(data)
            plot_embedding(result_orig, label, 't-sne enbedding of the digits orig')
            plot_embedding(result_orig, train_label, 't-sne enbedding of the digits train')
            duration = time.time() - start_time

            valid_test = (np.mean(valid_result[1]), np.std(valid_result[1]))
            test_test = (np.mean(test_result[1]), np.std(test_result[1]))

            test_results['time'].append(duration)

            test_results['valid_test']['mean'].append(valid_test[0])
            test_results['test_test']['mean'].append(test_test[0])

            test_results['valid_test']['std'].append(valid_test[1])
            test_results['test_test']['std'].append(test_test[1])

            print('valid-test_test acc (%d meta_it)(%.2fs): %.3f (%.3f),  %.3f (%.3f)' % (i, duration, valid_test[0],
                                                                                          valid_test[1], test_test[0],
                                                                                          test_test[1]))

            save_obj(test_result_path, test_results)

    return test_results


def meta_test_up_to_T(exp_dir, metasets, exs, pybml_ho, saver, sess, c_way, k_shot, lr, n_test_episodes, MBS, seed, T,
                      iterations=list(range(10000))):
    meta_test_str = str(c_way) + 'way_' + str(k_shot) + 'shot_' + str(lr) + 'lr' + str(n_test_episodes) + 'ep'

    n_test_batches = n_test_episodes // MBS
    rand = dl.get_rand_state(seed)

    valid_batches = BatchQueueMock(metasets.validation, n_test_batches, MBS, rand)
    test_batches = BatchQueueMock(metasets.test, n_test_batches, MBS, rand)
    train_batches = BatchQueueMock(metasets.train, n_test_batches, MBS, rand)

    print('\nMeta-testing {} (over {} eps)...'.format(meta_test_str, n_test_episodes))

    test_results = {'valid_test': [], 'test_test': [], 'train_test': [], 'time': [], 'n_test_episodes': n_test_episodes,
                    'episodes': [], 'iterations': []}

    test_result_path = os.path.join(exp_dir, meta_test_str + 'noTrain_results.pickle')

    start_time = time.time()
    for i in iterations:
        model_file = os.path.join(exp_dir, 'model' + str(i))
        if tf.train.checkpoint_exists(model_file):
            print("Restoring model weights from " + model_file)
            saver.restore(sess, model_file)

            test_results['iterations'].append(i)
            test_results['episodes'].append(i * MBS)

            valid_result = accuracy_on_up_to_T(valid_batches, exs, pybml_ho, sess, T)
            test_result = accuracy_on_up_to_T(test_batches, exs, pybml_ho, sess, T)
            train_result = accuracy_on_up_to_T(train_batches, exs, pybml_ho, sess, T)

            duration = time.time() - start_time

            test_results['time'].append(duration)

            for t in range(T):

                valid_test = (np.mean(valid_result[1][t]), np.std(valid_result[1][t]))
                test_test = (np.mean(test_result[1][t]), np.std(test_result[1][t]))
                train_test = (np.mean(train_result[1][t]), np.std(train_result[1][t]))

                if t >= len(test_results['valid_test']):
                    test_results['valid_test'].append({'mean': [], 'std': []})
                    test_results['test_test'].append({'mean': [], 'std': []})
                    test_results['train_test'].append({'mean': [], 'std': []})

                test_results['valid_test'][t]['mean'].append(valid_test[0])
                test_results['test_test'][t]['mean'].append(test_test[0])
                test_results['train_test'][t]['mean'].append(train_test[0])

                test_results['valid_test'][t]['std'].append(valid_test[1])
                test_results['test_test'][t]['std'].append(test_test[1])
                test_results['train_test'][t]['std'].append(train_test[1])

                print('valid-test_test acc T=%d (%d meta_it)(%.2fs): %.4f (%.4f), %.4f (%.4f),'
                      '  %.4f (%.4f)' % (t + 1, i, duration, train_test[0], train_test[1], valid_test[0], valid_test[1],
                                         test_test[0], test_test[1]))

                # print('valid-test_test acc T=%d (%d meta_it)(%.2fs): %.4f (%.4f),'
                #      '  %.4f (%.4f)' % (t+1, i, duration, valid_test[0], valid_test[1],
                #                         test_test[0], test_test[1]))

            save_obj(test_result_path, test_results)

    return test_results


def batch_producer(metadataset, batch_queue, n_batches, batch_size, rand=0):
    while True:
        batch_queue.put([d for d in metadataset.generate(n_batches, batch_size, rand)])


def start_batch_makers(number_of_workers, metadataset, batch_queue, n_batches, batch_size, rand=0):
    for w in range(number_of_workers):
        worker = Thread(target=batch_producer, args=(metadataset, batch_queue, n_batches, batch_size, rand))
        worker.setDaemon(True)
        worker.start()


# Class for debugging purposes for multi-thread issues (used now because it resolves rand issues)
class BatchQueueMock:
    def __init__(self, metadataset, n_batches, batch_size, rand):
        self.metadataset = metadataset
        self.n_batches = n_batches
        self.batch_size = batch_size
        self.rand = rand

    def get_all_batches(self):
        return [d for d in self.metadataset.generate(self.n_batches, self.batch_size, self.rand)]


def save_obj(file_path, obj):
    with open(file_path, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_obj(file_path):
    with open(file_path, 'rb') as handle:
        b = pickle.load(handle)
    return b


''' Useful Functions '''


def just_train_on_dataset(dat, exs, pybml_ho, sess, T):
    train_fd, valid_fd = feed_dicts(dat, exs)
    # print('train_feed:', train_fd)  # DEBUG
    sess.run(pybml_ho.outergradient.initialization)
    tr_acc, v_acc = [], []

    for ex in exs:
        [sess.run(ex.optimizers['apply_updates'], feed_dict={ex.x: train_fd[ex.x], ex.y: train_fd[ex.y]}) for _ in
         range(T)]
        tr_acc.append(sess.run(ex.scores['accuracy'], feed_dict={ex.x: train_fd[ex.x], ex.y: train_fd[ex.y]}))
        v_acc.append(sess.run(ex.scores['accuracy'], feed_dict={ex.x: valid_fd[ex.x_], ex.y: valid_fd[ex.y_]}))

    output_label = sess.run(exs[0].model.out, feed_dict={exs[0].x: valid_fd[exs[0].x_], exs[0].y: valid_fd[exs[0].y_]})
    return tr_acc, v_acc, [np.reshape(valid_fd[exs[0].x_],newshape=(75 ,-1)),np.argmax(valid_fd[exs[0].y_],axis=1), np.argmax(output_label,axis=1)]


def accuracy_on(batch_queue, exs, pybml_ho, sess, T):
    tr_acc, v_acc = [], []
    output = []
    for d in batch_queue.get_all_batches():
        result = just_train_on_dataset(d, exs, pybml_ho, sess, T)
        tr_acc.extend(result[0])
        v_acc.extend(result[1])
        output.extend(result[2])
    return tr_acc, v_acc, output


def just_train_on_dataset_up_to_T(dat, exs, pybml_ho, sess, T):
    train_fd, valid_fd = feed_dicts(dat, exs)
    # print('train_feed:', train_fd)  # DEBUG
    sess.run(pybml_ho.outergradient.initialization)
    tr_acc, v_acc = [[] for _ in range(T)], [[] for _ in range(T)]
    for ex in exs:
        # ex.model.initialize(session=sess)
        for t in range(T):
            sess.run(ex.optimizers['apply_updates'], feed_dict={ex.x: train_fd[ex.x], ex.y: train_fd[ex.y]})
            tr_acc[t].append(sess.run(ex.scores['accuracy'], feed_dict={ex.x: train_fd[ex.x], ex.y: train_fd[ex.y]}))
            v_acc[t].append(sess.run(ex.scores['accuracy'], feed_dict={ex.x: valid_fd[ex.x_], ex.y: valid_fd[ex.y_]}))
    return tr_acc, v_acc


def accuracy_on_up_to_T(batch_queue, exs, pybml_ho, sess, T):
    tr_acc, v_acc = [[] for _ in range(T)], [[] for _ in range(T)]
    for d in batch_queue.get_all_batches():
        result = just_train_on_dataset_up_to_T(d, exs, pybml_ho, sess, T)
        [tr_acc[T].extend(r) for T, r in enumerate(result[0])]
        [v_acc[T].extend(r) for T, r in enumerate(result[1])]

    return tr_acc, v_acc


def get_rand_state(rand):
    """
    Utility methods for getting a `RandomState` object.

    :param rand: rand can be None (new State will be generated),
                    np.random.RandomState (it will be returned) or an integer (will be treated as seed).

    :return: a `RandomState` object
    """
    if isinstance(rand, np.random.RandomState):
        return rand
    elif isinstance(rand, (int, np.ndarray, list)) or rand is None:
        return np.random.RandomState(rand)
    else:
        raise ValueError('parameter rand {} has wrong type'.format(rand))


