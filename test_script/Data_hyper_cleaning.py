import tensorflow.contrib.layers as layers
import sys
import copy,random
import inspect, os
sys.path.append('../')
os.environ['DATASETS_FOLDER'] = '../'
os.environ['EXPERIMENTS_FOLDER'] = '../'
from test_script.script_helper import *
import sklearn
from sklearn.metrics import f1_score
import boml as boml

import numpy as np
import tensorflow as tf

sess = tf.InteractiveSession()

parser = argparse.ArgumentParser()
# Dataset/method options
parser.add_argument('-d', '--dataset', type=str, default='miniimagenet', metavar='STRING',
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
parser.add_argument('-mbs', '--meta_batch_size', type=int, default=2, metavar='NUMBER',
                    help='number of tasks sampled per meta-update')
parser.add_argument('-nmi', '--meta_train_iterations', type=int, default=3000, metavar='NUMBER',
                    help='number of metatraining iterations.')
parser.add_argument('-T', '--T', type=int, default=5, metavar='NUMBER',
                    help='number of inner updates during training.')
parser.add_argument('-xi', '--xavier', type=bool, default=False, metavar='BOOLEAN',
                    help='FFNN weights initializer')
parser.add_argument('-bn', '--batch-norm', type=bool, default=False, metavar='BOOLEAN',
                    help='Use batch normalization before classifier')
parser.add_argument('-mlr', '--meta-lr', type=float, default=0.3, metavar='NUMBER',
                    help='starting meta learning rate')
parser.add_argument('-mlrdr', '--meta-lr-decay-rate', type=float, default=1.e-5, metavar='NUMBER',
                    help='meta lr  inverse time decay rate')

parser.add_argument('-cv', '--clip_value', type=float, default=0., metavar='NUMBER',
                    help='meta gradient clip value (0. for no clipping)')
parser.add_argument('-lr', '--lr', type=float, default=0.4, metavar='NUMBER',
                    help='starting learning rate')

# settings of parameters for specific algorithms
parser.add_argument('-alpha_decay', '--alpha_decay', type=float, default=1.e-5, metavar='NUMBER',
                    help='alpha decay rate')
parser.add_argument('-alpha', '--alpha', type=float, default=0.0, metavar='NUMBER',
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

parser.add_argument('-md', '--method', type=str, default='Simple', metavar='STRING',
                    help='choose which method to use,[Trad, Aggr,Simple]')
parser.add_argument('-scalor', '--scalor', type=float, default=0.0, metavar='NUMBER',
                    help='scalor for controlling the regularization coefficient')
parser.add_argument('-tr_ir', '--truncate_iter', type=int, default=-1, metavar='NUMBER',
                    help='truncated iterations ')
parser.add_argument('-i_d', '--inner_method', type=str, default='Trad', metavar='STRING',
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
                    help='coefficient for UL objective for BA algorithm')
parser.add_argument('-bt', '--ba_t', type=float, default=1.0, metavar='NUMBER',
                    help='coefficient for LL objectiv for BA algorithm')
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
parser.add_argument('-Notes', '--Notes', type=str, default='Notes',
                    help='Something important')
args = parser.parse_args()

def save_obj(file_path, obj):
    with open(file_path, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def cross_entropy_loss(pred,label,method='MetaRepr'):
    var=tf.get_collection(boml.extension.GraphKeys.METAPARAMETERS)
    return tf.reduce_mean(tf.sigmoid(tf.get_collection(boml.extension.GraphKeys.METAPARAMETERS)[0])
                          * tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=pred))

def outer_cross_entropy_loss(pred,label,method='MetaRepr'):
    return  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=pred))


def F1_score(y_pred, y_true, average):
    '''

    y_pred: encoded in one hot
    y_true：array, a vector
    '''

    y_pred = np.argmax(y_pred, 1)
    y_true =np.argmax(y_true, 1)
    return sklearn.metrics.f1_score(y_true, y_pred, average=average)


def pollute(datasets_, n):
    correct_label = copy.deepcopy(datasets_.target)
    datasets = copy.deepcopy(datasets_)
    incorrect_local = random.sample(range(correct_label.shape[0]), n)
    incorrect_local_hot = np.zeros(correct_label.shape[0])
    incorrect_local_hot[incorrect_local] = 1.0

    for i in range(n):
        while (all(correct_label[incorrect_local[i]] == datasets.target[incorrect_local[i]])):
            np.random.shuffle(datasets.target[incorrect_local[i]])
    return datasets, correct_label, incorrect_local, incorrect_local_hot


def get_data():
    # load a small portion of mnist data
    datasets = boml.load_data.mnist(folder=os.path.join(os.getcwd(), 'MNIST_DATA'), partitions=(1/14, 1/14))
    return datasets.train, datasets.validation, datasets.test

def get_fashion_mnist_data():
    # load a small portion of mnist data
    datasets = boml.load_data.fashion_mnist(partitions=(1/14, 1/14))
    return datasets.train, datasets.validation, datasets.test


def g_logits(x,y):
    with tf.variable_scope('model'):
        h1 = layers.fully_connected(x, 300)
        logits = layers.fully_connected(h1, int(y.shape[1]))
    return logits

# set up model
#datasets=boml.load_data.mnist(folder=os.path.join(os.getcwd(), 'MNIST_DATA'), partitions=(1/14, 1/14))
fashion_mnist_datasets=boml.load_data.mnist(partitions=(1/14, 1/14))
ex = boml.BOMLExperiment(datasets=fashion_mnist_datasets)

# hyper model
hyper_model = boml.BOMLNetMnistMetaReprV3(_input=ex.x)

# task model
ex.model = boml.BOMLNetFC(_input=ex.x)

# define LL and UL problems
boml_ho = boml.BOMLOptimizer(method='MetaRepr', inner_method=args.inner_method,outer_method=args.outer_method,truncate_iter=args.truncate_iter)
boml_ho.param_dict['loss_func'] = cross_entropy_loss #损失函数有问题
boml_ho.param_dict['outer_loss_func'] = outer_cross_entropy_loss #损失函数有问题
boml_ho.param_dict['meta_learner'] = hyper_model #损失函数有问题
repr_out= ex.model.re_forward(new_input=ex.x).out
ce = tf.nn.softmax_cross_entropy_with_logits(labels=ex.y, logits=repr_out)
result = tf.math.argmax(repr_out, 1)
correct = tf.math.argmax(ex.y, 1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(result, correct), tf.float32))
ex.errors['training'] = cross_entropy_loss(pred=repr_out,label=ex.y)


repr_out_val = ex.model.re_forward(new_input=ex.x_).out
ce_val = tf.nn.softmax_cross_entropy_with_logits(labels=ex.y_, logits=repr_out_val)

ex.errors['outer_training'] = outer_cross_entropy_loss(pred=repr_out_val, label=ex.y_)
ex.errors['validation'] = outer_cross_entropy_loss(pred=repr_out_val, label=ex.y_)

logits_val = tf.nn.softmax(repr_out_val)
#f1_score_micro = F1_score(y_pred=logits_val,y_true=ex.y_,average='micro')
#f1_score_macro = F1_score(y_pred=logits_val,y_true=ex.y_,average='macro')
#f1_score_weighted = F1_score(y_pred=logits_val,y_true=ex.y_,average='weighted')
#lr = boml.get_outerparameter('lr', initializer=0.01)

# ce = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
# L = tf.reduce_mean(tf.sigmoid(lambdas)*ce)
# E = tf.reduce_mean(ce)

optim_dict = boml_ho.ll_problem(inner_objective=ex.errors['training'], learning_rate=args.lr, s=args.ba_s, t=args.ba_t,
                                inner_objective_optimizer='SGD', outer_objective=ex.errors['validation'],
                                alpha_init=args.alpha, T=args.T, experiment=ex, gamma=1.0, learn_lr=args.learn_lr,
                                learn_alpha_itr=False, learn_alpha=False,loss_func=cross_entropy_loss,outer_loss_func=outer_cross_entropy_loss,
                                var_list=ex.model.var_list)


boml_ho.ul_problem(outer_objective=ex.errors['validation'], inner_grad=optim_dict,
                   outer_objective_optimizer='Adam', meta_learning_rate=0.001, mlr_decay=1.e-5,
                   meta_param=tf.get_collection(boml.extension.GraphKeys.METAPARAMETERS))
hyper_step = boml_ho.aggregate_all()

## Load data
#train_set, validation_set, test_set = get_data()
train_set, validation_set, test_set = fashion_mnist_datasets.train,fashion_mnist_datasets.validation,fashion_mnist_datasets.test
train_set_polluted, correct_label, incorrect_local, incorrect_local_hot = pollute(train_set,
                                                                                  int(0.5 * train_set.target.shape[0]))


# Number of inner iterations
train_inner_set_supplier = train_set_polluted.create_supplier(ex.x,ex.y)
train_outer_set_supplier = validation_set.create_supplier(ex.x_, ex.y_)
validation_set_supplier = validation_set.create_supplier(ex.x, ex.y)
correct_train_set_supplier = train_set.create_supplier(ex.x, ex.y)
test_set_supplier = test_set.create_supplier(ex.x, ex.y)

results = {'train_train': [], 'train_test': [],
           'test_test': [], 'valid_test': [],
           'inner_losses': [],'outer_losses': [], 'learning_rate': [], 'iterations': [],
           'episodes': [], 'time': [], 'alpha': [],
           'F1_score_micro':[], 'F1_score_macro':[], 'F1_score_average':[]}

tr_loss = []
val_loss = []
tr_accs = []
val_accs = []
test_accs = []
lambdas_h = []
correctly_identifying = []

tf.global_variables_initializer().run()
ex.model.initialize()
print('inner:', ex.errors['training'].eval(train_inner_set_supplier()))
print('outer:', ex.errors['validation'].eval(train_outer_set_supplier()))
# print('-'*50)
start_time=time.time()
n_hyper_iterations = args.meta_train_iterations
for _ in range(n_hyper_iterations):

    hyper_step(inner_objective_feed_dicts=train_inner_set_supplier(),
               outer_objective_feed_dicts=train_outer_set_supplier())
    results['time'].append(time.time() -start_time)
    print('iter:', _)
    results['iterations'].append(_)
    print('duration',results['time'][_])
    results['inner_losses'].append(ex.errors['training'].eval(correct_train_set_supplier()))
    print('inner:', results['inner_losses'][_])
    results['outer_losses'].append(ex.errors['validation'].eval(train_outer_set_supplier()))
    print('outer:', results['outer_losses'][_])
    # print('grad_lr_first_part:', sess.run(boml_ho.outergradient.grad_lr,train_outer_set_supplier()))
    results['train_train'].append(accuracy.eval(correct_train_set_supplier()))
    print('inner accuracy:', results['train_train'][_])
    results['valid_test'].append(accuracy.eval(validation_set_supplier()))
    print('outer accuracy:', results['valid_test'][_])
    results['test_test'].append(accuracy.eval(test_set_supplier()))
    print('test accuracy:', results['test_test'][_])
    results['F1_score_macro'].append(F1_score(y_pred=repr_out.eval(test_set_supplier()), y_true=test_set.target, average='micro'))
    print('F1_score_macro :', results['F1_score_macro'][_])
    results['F1_score_micro'].append(F1_score(y_pred=repr_out.eval(test_set_supplier()), y_true=test_set.target, average='macro'))
    print('test F1_score_micro:', results['F1_score_micro'][_])
    results['F1_score_average'].append(F1_score(y_pred=repr_out.eval(test_set_supplier()), y_true=test_set.target, average='weighted'))
    print('F1_score_average:', results['F1_score_average'][_])
    print('learning rate', boml_ho.param_dict['learning_rate'].eval())
    print('norm of examples weight', np.linalg.norm(hyper_model.outer_param_dict['lambda'].eval(),ord=np.inf))
    print('-'*50)
    save_obj(os.path.join(os.getcwd(), str(args.Notes)+'.pickle'), results)
    start_time = time.time()
# # plt.plot(tr_accs, label='training accuracy')
# # plt.plot(val_accs, label='validation accuracy')
# # plt.legend(loc=0, frameon=True)
# # # plt.xlim(0, 19)
