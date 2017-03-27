# -*- coding: utf-8 -*-

"""
    kitt_lib.kitt_monkey
    ~~~~~~~~~~~~~~~~~~~~
    Monkey work.
"""

from termcolor import colored
from numpy import count_nonzero

pruning_print_template = '{0:<8}{1:<18}{2:<20}{3:<23}{4:<15}{5:<15}{6:<15}'
learning_print_template = '{0:<8}{1:<22}{2:<25}{3:<20}'

def print_message(message):
    print colored('\n--------------------------------------------------------------------', 'blue')
    print colored('-- '+message, 'blue')

def print_param(description, param_str):
    print colored('\t% '+description+': ', 'blue')+param_str

def print_initialized(net):
    print_message(message='Network initialized.')
    print_param(description='problem dimension', param_str=str(net.structure[0]))
    print_param(description='number of classes', param_str=str(net.structure[-1]))
    print_param(description='class labels', param_str=str(net.labels))
    print_param(description='net structure', param_str=str(net.structure))
    print_param(description='net transfer function', param_str=str(net.tf.__class__.__name__))

def print_learning_started(kw):
    if kw['verbose']:
        print_message(message='Learning has started...')
        print_param(description='problem dimension', param_str=str(len(kw['self'].t_data[0][0])))
        print_param(description='number of training samples', param_str=str(len(kw['self'].t_data)))
        if kw['self'].v_data is not None:
            print_param(description='number of validation samples', param_str=str(len(kw['self'].v_data)))
        print_param(description='learning rate', param_str=str(kw['learning_rate']))
        print_param(description='mini-batch size', param_str=str(kw['batch_size']))
        print_param(description='maximum number of epochs (t.c.)', param_str=str(kw['n_epoch']))
        print_param(description='maximum number of stable epochs (t.c.)', param_str=str(kw['c_stable']))
        print_param(description='required accuracy (t.c.)', param_str=str(kw['req_acc']))
        print_param(description='required error (t.c.)', param_str=str(kw['req_err']))
        print '\n'
        print learning_print_template.format('epoch', 'on training data', 'on validation data', 'epoch time')
        print '-------------------------------------------------------------------'

def print_and_check_epoch(stats, kw):
    line = ' '+str(stats['i_epoch'])+'\t  '+colored(str(format(stats['t_acc'][-1], '.2f')), 'green')
    if stats['t_err'][-1] < stats['t_err_best']:
        stats['t_err_best'] = stats['t_err'][-1]
        col = 'red'
        stats['c_stable'] = 0
    else:
        col = 'magenta'
        stats['c_stable'] += 1
    line += '/'+colored(str(format(stats['t_err'][-1], '.4f')), col)
    try:
        line += colored('\t\t'+str(format(stats['v_acc'][-1], '.2f')), 'green')
        if stats['v_err'][-1] < stats['v_err_best']:
            stats['v_err_best'] = stats['v_err'][-1]
            col = 'red'
        else:
            col = 'magenta'
        line += '/'+colored(str(format(stats['v_err'][-1], '.4f')), col)
    except IndexError:
        line += colored('\t\t'+'no data', 'green')+'/'+colored('no data', 'red')
    line += '\t\t'+colored(str(format(stats['ep_time'][-1], '.4f'))+' s', 'cyan')
    if kw['verbose']:
        print line

    # Check termination conditions
    if kw['strict_termination']:
        if stats['t_acc'][-1] >= kw['req_acc'] and stats['t_err'][-1] <= kw['req_err']:
            return True
    else:
        if stats['c_stable'] == kw['c_stable']:
            print_learning_finished(why='Given number of stable epochs performed ('+str(kw['c_stable'])+').', t=sum(stats['ep_time']), verbose=kw['verbose'])
            return True
        if stats['t_acc'][-1] >= kw['req_acc']:
            print_learning_finished(why='Trained to required accuracy ('+str(kw['req_acc'])+').', t=sum(stats['ep_time']), verbose=kw['verbose'])
            return True
        if stats['t_err'][-1] <= kw['req_err']:
            print_learning_finished(why='Error reduced to required number ('+str(kw['req_err'])+').', t=sum(stats['ep_time']), verbose=kw['verbose'])
            return True

def print_learning_finished(why, t, verbose):
    if verbose:
        print_message(message='Learning finished in '+str(round(t, 4))+'s. '+why)

def print_pruning_started(net, kw, vars, stats):
    if kw['verbose']:
        print_message(message='Pruning has started...')
        print_param(description='net initial structure', param_str=str(net.structure))
        print_param(description='net initial number of synapses (w, b)', param_str=str(net.count_synapses()))
        print_param(description='min required accuracy (s.c.)', param_str=str(kw['req_acc']))
        print_param(description='max required error (s.c.)', param_str=str(kw['req_err']))
        print_param(description='maximum number of re-training epochs (gu.c.)', param_str=str(kw['n_epoch']))
        print_param(description='number of stable iterations (gu.c.)', param_str=str(kw['c_stable']))
        print '\n'
        print pruning_print_template.format('step', 'trying to cut', 'structure', 'left synapses (w/b)', 'retrained', 'next level', 'step time')
        print '--------------------------------------------------------------------------------------------------------------'
        stats['structure'].append(vars['net_tmp'].structure)
        stats['n_synapses'].append(vars['net_tmp'].count_synapses())
        print colored(pruning_print_template.format(vars['step'], '0', stats['structure'][-1], stats['n_synapses'][-1], 'None', vars['level'], 
                                                    colored('None', 'cyan')), 'yellow')

def print_pruning_step(stats, vars):
    stats['structure'].append(vars['net_tmp'].structure)
    stats['n_synapses'].append(vars['net_tmp'].count_synapses())
    print colored(pruning_print_template.format(vars['step'], str(stats['n_to_cut'][-1]), 
                                                stats['structure'][-1], stats['n_synapses'][-1], 'yes' if stats['retrained'][-1] else 'no', vars['level'], 
                                                colored(str(round(stats['step_time'][-1], 4))+' s', 'cyan')), 'green' if stats['retrained'][-1] else 'red')

def print_pruning_finished(net, kw, stats):
    print_message(message='Pruning finished in '+str(sum(stats['step_time']))+' s.')
    print_param(description='net final structure', param_str=str(net.structure))
    print_param(description='net final number of synapses (w/b)', param_str=str(net.count_synapses()))
    res_t = net.evaluate_(data=net.t_data)
    res_v = net.evaluate_(data=net.v_data) if net.v_data else None
    print_param(description='classification accuracy on training data', param_str=str(res_t[1]))
    print_param(description='classification error on training data', param_str=str(res_t[0]))
    print_param(description='classification accuracy on validation data', param_str=str(res_v[1]))
    print_param(description='classification error on validation data', param_str=str(res_v[0]))