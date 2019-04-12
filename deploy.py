from collections import defaultdict, deque
from copy import deepcopy
from datetime import datetime
import logging
import multiprocessing
from multiprocessing import Process, Manager
from multiprocessing import get_logger
from multiprocessing.managers import SyncManager
import os
from os import getpid
import signal
import sys
import time
import traceback

import numpy as np
from PIL import Image

from exceptions import *
import func_library
from utils.utils import BlankHolder, GraphPlot, fprint, get_tabs, list_avg, repeator


class Color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'


class Component:
    """Component class that wraps a python function and controls its I/O among other things.

    Methods intended for user access:
        [Constructor]: To initialize a component, and attach a function to it
    Attributes intended for user access:
        runtime_args: Set this to provide runtime arguments to the python function being wrapped
        static_args: Set this to provide static arguments to the python function being wrapped
        to_record: Set this to store the required set of output variables after the function completes execution
        to_print:  Set this to print the required set of output variables after the function completes execution
        to_save:  Set this to save the required set of output variables after the function completes execution

    Refer the README at <github_link> to understand the details.

    """
    def __init__(self, name, func, run_async=True, queue_size=1, debug=False):
        """It is important that none of these attributes are initialized to be shared objects by
        multiprocessing.Manager. That is because when a pipeline uses a map method, it calls the run
        on same class objects. However, under the hood, it issues a fork() on every new Process() call.
        As such, any updates made to these class members are not reflected in the main process,
        and there is no data collision. Rather, if a shared object was made, they would collide.
        That is also the reason why, during compilation of the pipeline, if it is set to parallelize
        (different from Map) then it asks each Component to instantiate the required set of variables as
        shared variables."""
        self._name = name
        self._func = getattr(func_library, func) if isinstance(func, str) else func
        self._run_async = run_async
        self._queue_size = queue_size
        self._debug = debug
        self._to_record = {}
        self._static_args = {}
        self._runtime_args = {}
        # runtime_args will overwrite any static_args at runtime
        self.to_save = {}
        self.to_print = []
        self.__mapper_name_variable = {}  # Variable name -> Actual variable in memory
        # ^ If running in a parallelized pipeline, this variable type will change
        # ^ This variable should be initialized before trying to get any attribute of this class
        self.__mapper_name_fork = defaultdict(dict)
        # ^ Variable name -> Components that are using this as part of their runtime_args
        self.__pipeline = None  # Pipeline is responsible for setting this value
        self._queuing = False  # Pipeline is responsible for setting this value
        self._compute_lock = None  # Pipeline will set it if required
        self.FutureVariable = FutureVariable(self, self.__mapper_name_variable)
        # TODO: check all variables in init or not

    # User methods
    @property
    def runtime_args(self):
        return self._runtime_args

    @property
    def static_args(self):
        return self._static_args

    @property
    def to_record(self):
        return self._to_record

    @runtime_args.setter
    def runtime_args(self, val_dict):
        self._runtime_args = val_dict
        # Create a map counter, to help handle automatic deletion
        for k, v in val_dict.items():
            if not isinstance(v, _FutureEvaluator):
                raise KeyInvalidError(var_name=v, comp_obj=self, case='runtime_args')
            cls_object = v._get_class()
            if type(cls_object) == type(self):
                cls_object.__mapper_name_fork[v._get_var_name()].update(
                    {self: dict(obj=self, used=False, same_pipeline=None, user_varname=k)})

    @static_args.setter
    def static_args(self, var_dict):
        self._static_args = var_dict

    @to_record.setter
    def to_record(self, val_dict):
        self._to_record = val_dict
        # To initialize placeholders only
        for v in val_dict.values():
            if not v in self.__mapper_name_variable:
                self.__mapper_name_variable[v] = BlankHolder()

    # Private methods
    def __hash__(self):
        return hash(self._name + '_' + str(id(self)))

    def __eq__(self, other):
        return (self._name, str(id(self))) == (other._name, id(other))

    def __ne__(self, other):
        # Not strictly necessary, but to avoid having both x==y and x!=y
        # True at the same time
        return not(self == other)

    def _compile(self, queue=False, q_manager=None, sync_manager=None):
        if queue:
            self._compute_lock = q_manager.Lock()
            self._queuing = True
            self.__mapper_runtimearg_itercount = q_manager.dict()
            for k in self._runtime_args.keys():
                self.__mapper_runtimearg_itercount[k] = -1
            self.__mapper_name_variable = {}
            for v in self._to_record.values():
                q_dict = q_manager.dict()
                mutex = q_manager.Lock()
                not_full = q_manager.Condition(mutex)
                not_empty = q_manager.Condition(mutex)
                this_val_dict = sync_manager.QueueDict(q_dict, not_full, not_empty, maxsize=self._queue_size)
                keys_list = q_manager.list()
                q_status_dict = q_manager.dict()
                status_dict = sync_manager.StatusDict(q_status_dict, keys_list)  # q_manager.dict()

                self.__mapper_name_variable[v] = dict(status_dict=status_dict, value=this_val_dict)

    def _compute(self, pipeline, indent_level, run_count, run_timer, callback=None, **kwargs):
        current_pid = getpid()
        try:
            self.__compute(pipeline, indent_level, run_count, run_timer, current_pid, **kwargs)
            if callback is not None:
                callback(self._name, run_count)
        except Exception:
            self.__pipeline._report_exception(comp=self, exc_info=traceback.format_exc(),
                                              pid=current_pid, run_count=run_count)

    def __compute(self, pipeline, indent_level, run_count, run_timer, current_pid, **kwargs):
        compute_start_time = datetime.now()
        if self._queuing and self._run_async:
            self._compute_lock.acquire()
        fprint(pid=current_pid, indent_level=indent_level, run_count=run_count, comp_name=self._name,
               msg='Compute Starting...')

        # ------------Run------------
        fprint(pid=current_pid, indent_level=indent_level, run_count=run_count, comp_name=self._name,
               msg='Running Function...')
        startTime = datetime.now()
        # TODO: check error logging at self._func
        var_dict = self._func(**{**self._get_args_to_run(pipeline, pid=current_pid, indent_level=indent_level,
                                run_count=run_count), **kwargs})
        if self._debug:
            fprint(pid=current_pid, indent_level=indent_level, run_count=run_count, comp_name=self._name,
                   msg='Got function result...')

        # ------------If Queing, update self.__mapper_runtimearg_itercount------------
        if self._queuing:
            for k in self._runtime_args.keys():
                self.__mapper_runtimearg_itercount[k] += 1

        # ------------Record------------
        if self._debug:
            fprint(pid=current_pid, indent_level=indent_level, run_count=run_count, comp_name=self._name,
                   msg='Started recording...')
        for k, v in self._to_record.items():
            if not k in var_dict:
                raise KeyNonExistentError(var_name=k, comp_obj=self, pipeline_name=pipeline._name,
                                          case='to_record')
            # No need to check if 'v' exists in pipeline.__mapper_name_variable, because it is put
            # the moment a pipeline is complied.
            # TODO: check the above comment

            # If the var is being used in a component from any other pipeline, do not queue it.
            # Otherwise Queues will not know when to pop that object
            if self._queuing:
                self.__mapper_name_variable[v]['value'].put(value=var_dict[k], var_name=k,
                                                            status_dict=self.__mapper_name_variable[v]['status_dict'],
                                                            comp_name=self._name, debug=self._debug, pid=current_pid,
                                                            indent_level=indent_level, run_count=run_count)
            else:
                self.__mapper_name_variable[v] = var_dict[k]
        if self._debug:
            fprint(pid=current_pid, indent_level=indent_level, run_count=run_count, comp_name=self._name,
                   msg='Done recording...')

        # ------------Print------------
        if len(self.to_print) > 0:
            fprint(pid=current_pid, indent_level=indent_level, run_count=run_count, comp_name=self._name,
                   msg='to_print...')
            for k in self.to_print:
                if not k in var_dict:
                    raise KeyNonExistentError(var_name=k, comp_obj=self, pipeline_name=pipeline._name,
                                              case='to_print')
                fprint(pid=current_pid, indent_level=indent_level, run_count=run_count, comp_name=self._name,
                       msg='[to_print] Value of {}'.format(k))
                fprint(pid=current_pid, indent_level=indent_level, run_count=run_count, comp_name=self._name,
                       msg='[to_print] ' + str(var_dict[k]).replace('\n','\n\t{}'.format(get_tabs(indent_level))))

        # ------------Save------------
        if bool(self.to_save):
            fprint(pid=current_pid, indent_level=indent_level, run_count=run_count, comp_name=self._name,
                   msg='Saving...')
            for k, v in self.to_save.items():
                if k not in var_dict:
                    raise KeyNonExistentError(var_name=k, comp_obj=self, pipeline_name=pipeline._name,
                                              case='to_save')
                self._save(var_dict[k], v)

        # ------------Cleanup components' variables used------------
        # NOTE: Refer deletion policy to understand what will be deleted
        # Each forked component variable updates its parents that it has done using it, and calls the parent
        # to check for deletion
        if self._debug:
            fprint(pid=current_pid, indent_level=indent_level, run_count=run_count,
                   comp_name=self._name, msg='Starting Cleanup...')
        for k, v in self._runtime_args.items():
            # If v is from a component, process it. We do not deal with cleanup of pipeline variables at this time
            cls_object = v._get_class()
            if type(cls_object) == type(self):
                if self._queuing:
                    cls_object.__mapper_name_fork[v._get_var_name()][self]['used_counter'] = self.__mapper_runtimearg_itercount[k]
                    cls_object.__check_and_delete(v._get_var_name(), self, self.__mapper_runtimearg_itercount[k], k,
                                                  caller_pid=current_pid, indent_level=indent_level,
                                                  run_count=run_count)
                else:
                    cls_object.__mapper_name_fork[v._get_var_name()][self]['used'] = True
                    cls_object.__check_and_delete(v._get_var_name(), self, 0, k, caller_pid=current_pid,
                                                  indent_level=indent_level, run_count=run_count)
        if self._debug:
            fprint(pid=current_pid, indent_level=indent_level, run_count=run_count,
                   comp_name=self._name, msg='Done with Cleanup...')
        compute_run_time = datetime.now() - compute_start_time
        existing_dict = run_timer[self._name]
        existing_dict[run_count] = compute_run_time
        run_timer[self._name] = existing_dict
        if self._queuing and self._run_async:
            self._compute_lock.release()
        fprint(pid=current_pid, indent_level=indent_level, run_count=run_count, comp_name=self._name,
               msg='Compute Ended. Took {}'.format(compute_run_time))

    def __check_and_delete(self, name, caller, caller_count, var_name, caller_pid, indent_level=None, run_count=None):
        # Check the self.__mapper_name_fork and delete variables as per the deletion policy
        if self._debug:
            fprint(pid=caller_pid, indent_level=indent_level, run_count=run_count, comp_name=self._name,
                   msg='Got check_and_delete call from {} of {} for: {} of comp {} at idx {}'\
                       .format(var_name, caller._name, name, self._name, caller_count))
        if self._queuing:
            delete, overwrite = True, True
            for v in self.__mapper_name_fork[name].values():
                if not v['same_pipeline']:
                    delete = False
                if v['used_counter'] < caller_count:
                    overwrite = False
                    delete = False
            if delete:
                self.__mapper_name_variable[name]['value'].delete(
                    idx=caller_count, var_name=name, comp_name=self._name,
                    status_dict=self.__mapper_name_variable[name]['status_dict'],
                    pid=caller_pid, indent_level=indent_level, run_count=run_count)
            else:
                if overwrite:
                    raise NotImplementedError('Overwrite not supported/required')
                    # self.__mapper_name_variable[name]['status_dict'].increment('overwrite', 1)
        else:
            for v in self.__mapper_name_fork[name].values():
                if not (v['same_pipeline'] and v['used']):
                    delete = False
                    if self._debug:
                        fprint(pid=caller_pid, indent_level=indent_level, run_count=run_count, comp_name=self._name,
                               msg='Not deleting {} because: {} {}'.format(name, v['same_pipeline'], v['used']))
                    break
            else:
                delete = True
            if delete:
                if self._debug:
                    fprint(pid=caller_pid, indent_level=indent_level, run_count=run_count, comp_name=self._name,
                           msg='Deleting {}'.format(name))
                self.__mapper_name_variable[name] = BlankHolder()
                # Reset the 'used' to False, in case of future calls
                for v in self.__mapper_name_fork[name].values():
                    v['used'] = False

    def _get_args_to_run(self, pipeline, pid=None, indent_level=None, run_count=None):
        if self._debug:
            fprint(pid=pid, indent_level=indent_level, run_count=run_count, comp_name=self._name,
                   msg='Called to get runtime args...')
        for k, v in self._runtime_args.items():
            # TODO: Check for BlankHolder -> either not assigned or got deleted
            # If pipeline arg, give it the usual way
            if type(v._get_class()) != type(self):
                self._static_args[k] = v()
            else:
                if v._get_class()._queuing:
                    if self._debug:
                        fprint(pid=pid, indent_level=indent_level, run_count=run_count, comp_name=self._name,
                               msg='Trying to get runtime arg for {} at {}'\
                                   .format(k, self.__mapper_runtimearg_itercount[k]+1))
                    self._static_args[k] = v(idx=self.__mapper_runtimearg_itercount[k]+1, debug=self._debug, pid=pid,
                                             indent_level=indent_level, run_count=run_count, comp_name=self._name)
                else:
                    self._static_args[k] = v(debug=self._debug, pid=pid, indent_level=indent_level,
                                             run_count=run_count, comp_name=self._name)
        if self._debug:
            fprint(pid=pid, indent_level=indent_level, run_count=run_count, comp_name=self._name,
                   msg='Got all runtime args...')
        return self._static_args

    def _get_pipeline(self):
        return self.__pipeline

    def _get_pipeline_name(self):
        return self.__pipeline._name

    def _preset(self):
        # Check to make sure all self._to_record variables are being used somewhere. Else, this could cause
        # any infinite pause on this component's execution (especially when multiprocessing), and might also result in
        # memory leaks
        for v in self._to_record.values():
            if not v in self.__mapper_name_fork:
                raise ValueError('Run of the pipeline \"{0}\" failed. Its component \"{1}\" records '\
                                 'the variable \"{2}\", but it does not seem like any other component is '\
                                 'using it. If this is not the case, make sure that assignment happens '\
                                 'before compiling this pipeline.'
                                 .format(self._get_pipeline_name(), self._name, v))
        # Update self.__mapper_name_fork to record if the forked component variables are from the same pipeline or not
        # This is helpful when handling variable deletion
        for name, fork_dict in self.__mapper_name_fork.items():
            for v in fork_dict.values():
                if v['obj']._get_pipeline() is None:
                    raise ValueError('Run of the pipeline \"{0}\" failed. Its component \"{1}\" gives '\
                                     'the variable \"{2}\" to another component \"{3}\" which has not been '\
                                     'assigned to any pipeline yet. Please assign it to a pipeline, compile '\
                                     'that pipeline, and then call this.'
                                     .format(self._get_pipeline_name(), self._name, name, v['obj']._name))
                if self._get_pipeline_name() == v['obj']._get_pipeline_name():
                    v['same_pipeline'] = True
                else:
                    v['same_pipeline'] = False
        # Check to make sure all component variables coming from components that are not part of this
        # component's pipeline have been compiled elsewhere.
        for k, v in self._runtime_args.items():
            v_class = v._get_class()
            if type(v_class) == type(self):
                if v_class._get_pipeline() is None:
                    raise ValueError('Run of the pipeline \"{0}\" failed. Its component \"{1}\" receives the '\
                                     'variable \"{2}\" from another component \"{3}\" which has not been assigned '\
                                     'to any pipeline yet. Please assign it to a pipeline, compile that pipeline, '\
                                     'run it, and then call this.'\
                                     .format(self._get_pipeline_name(), self._name, k, v._get_class()._name))
                else:
                    if not v_class._get_pipeline()._run_called:
                        raise ValueError('Run of the pipeline \"{0}\" failed. Its component \"{1}\" receives '\
                                         'the variable \"{2}\" from another component \"{3}\" which as been assigned '\
                                         'to the pipeline \"{4}\", but that pipeline has not started running yet. '\
                                         'Run it, and then call this.'\
                                         .format(self._get_pipeline_name(), self._name, k, v._get_class()._name,
                                                 v_class._get_pipeline_name()))

    def _set_pipeline(self, pipeline):
        if self.__pipeline is not None:
            raise ValueError('This component has already been used while compiling the pipeline \"{}\"'\
                              .format(self.__pipeline))
        self.__pipeline = pipeline

    def _save(self, var, form):
        if isinstance(form, str):
            if form.endswith('.npy'):
                np.save(form, var)
            elif form.endswith('.png') or form.endswith('.jpg'):
                # import datetime
                # form = form.replace('.png', '__' + str(datetime.datetime.now()) + '.png')
                Image.fromarray((var.astype('uint8')).squeeze()).save(form)
            else:
                raise ValueError('Saving not supported for this extension')
        elif callable(form):
            form(var)
        else:
            raise RuntimeError('Incorrect way to save the argument')

    def _get_mapper_name_variable(self):
        return self.__mapper_name_variable


class Pipeline:
    """Pipeline class that takes in a bunch of components to be executed and executes them accordingly.

    Methods intended for user access:
        [Constructor]: To initialize a pipeline
        add_pipeline_variable: To add static pipeline variables
        compile_pipeline: To compile a pipeline, and to make it ready for execution
        run: Run the pipeline

    Refer the README at <github_link> to understand the details.

    """
    __variable_counter = 0
    __indent_level = -1
    _mapper_pipeline_name_obj = {}
    __manager = Manager()
    _pipelines_run_order = __manager.list()
    __sync_manager = SyncManager()

    def __init__(self, name):
        if name in self.__class__._mapper_pipeline_name_obj:
            raise ValueError('A pipeline with that name already exists')
        self.__compiled = False
        self.__component_list = []
        self.__future_jobs = deque()
        self._default_component_status_dict = defaultdict(bool)
        self._map_count = 0
        self._map_parallelize = False
        self.__mapper_name_variable = {}
        self._name = name
        self._parallelize = False
        self._process_list = []
        self._run_count = 0
        self._run_called = False
        self._sync_manager = None
        self._component_run_timer = self.__class__.__manager.dict()
        self._self_run_timer = self.__class__.__manager.dict()
        self.FutureVariable = FutureVariable(self, self.__mapper_name_variable)

    # User methods
    def add_pipeline_variable(self, inp_dict):
        for k, v in inp_dict.items():
            if k in self.__mapper_name_variable:
                raise KeyExistingError(var_name=k, pipeline_name=self._name, case='add_variable')
            self.__mapper_name_variable[k] = v

    def compile_pipeline(self, components_iterable, parallelize=False, logging_level='INFO', logging_file=None):
        self.__class__._mapper_pipeline_name_obj[self._name] = self
        self.__main_pid = getpid()
        self._parallelize = parallelize
        self._new_process_lock = self.__class__.__manager.Lock()
        self._run_init_lock = self.__class__.__manager.Lock()
        if self._parallelize:
            self.__class__.__sync_manager.start()
            self._component_callback_lock = self.__class__.__manager.Lock()
            self._component_run_status = self.__class__.__manager.dict()

        # https://docs.python.org/2/library/logging.html#logging-levels
        if logging_file is not None:
            if logging_file == 'auto':
                logging_file = self._name + '.log'
            logger = get_logger()
            level = logging.getLevelName(logging_level)
            logger.setLevel(level)
            fh = logging.FileHandler(logging_file, mode='w')
            fh.setLevel(level)
            # ch = logging.StreamHandler()
            # ch.setLevel(level)
            formatter = logging.Formatter('[%(asctime)s][PID %(process)d] - %(levelname)s - %(message)s',
                                           datefmt='%Y-%m-%d %H:%M:%S')
            # ch.setFormatter(formatter)
            fh.setFormatter(formatter)
            # logger.addHandler(ch)
            logger.addHandler(fh)

        self.__component_list = components_iterable
        for comp in components_iterable:
            if comp._name in self._default_component_status_dict:
                raise ValueError('Component with this name already exists in this pipeline')
            self._component_run_timer[comp._name] = {}
            self._default_component_status_dict[comp._name] = True
            comp._set_pipeline(self)
            if self._parallelize:
                comp._compile(self._parallelize, self.__class__.__manager, self.__class__.__sync_manager)
            else:
                comp._compile(self._parallelize)
        self.__compiled = True

    def map(self, kwargs, num_cores=-1):
        if self._parallelize:
            raise RuntimeError('Pipeline received a call for map method, but it was set to parallelize at compile time.')

        self._map_parallelize = True
        num_available = multiprocessing.cpu_count()
        if num_cores == -1:
            num_cores = num_available
        else:
            num_cores = min(num_cores, num_available)
            # give USERWARNING of this reset

        total_len = len(next(iter(kwargs.values())))
        for iter_count in range(total_len):
            current_kwargs = {key: kwargs[key][iter_count] for key in kwargs}
            self._new_process_lock.acquire()
            p = Process(target=self.run, kwargs=dict(kwargs=current_kwargs))
            p.daemon = True
            p.start()
            self._process_list.append(p)
            self._new_process_lock.release()
            self._map_count += 1
            if (iter_count + 1) % num_cores == 0 or (iter_count + 1) == total_len:
                self.wait(future_jobs=False, async_jobs=True)

    def run(self, kwargs=None):
        if kwargs is None:
            kwargs = {}
        if not self.__compiled:
            raise ValueError('Compile the pipeline \"{}\" before running'.format(self._name))
        run_count = self._map_count if self._map_parallelize else self._run_count

        self._run_init_lock.acquire()
        self.__class__.__indent_level += 1
        current_run_order = self.__class__._pipelines_run_order
        current_run_order.append(dict(name=self._name, run_count=run_count,
                                      indent_level=self.__class__.__indent_level))
        self.__class__._pipelines_run_order = current_run_order

        current_run_time_dict = self._self_run_timer
        current_run_time_dict[run_count] = {'start_datetime': datetime.now()}
        self._self_run_timer = current_run_time_dict
        if self._parallelize:
            self._component_run_status[run_count] = deepcopy(self._default_component_status_dict)
        self._run_init_lock.release()

        fprint(pid=getpid(), indent_level=self.__class__.__indent_level, run_count=run_count, comp_name=None,
               msg='{}RUNNING PIPELINE \"{}\"...{}{}'.format(Color.BOLD, self._name, Color.END, Color.END))

        if run_count == 0:
            self._run_called = True
            self._preset()

        if self._parallelize:
            if self.__component_list[0]._run_async:
                self._run(comp=self.__component_list[0], kwargs=kwargs, parallelize=True,
                          indent_level=self.__class__.__indent_level, run_count=run_count)
            else:
                self.__future_jobs.append([0, kwargs, self.__class__.__indent_level, run_count])
        else:
            self._run(comp=self.__component_list[0], kwargs=kwargs, parallelize=False,
                      indent_level=self.__class__.__indent_level, run_count=run_count)

        for iter_comp, comp in enumerate(self.__component_list[1:]):
            if self._parallelize:
                if comp._run_async:
                    self._run(comp=comp, kwargs=None, parallelize=True,
                              indent_level=self.__class__.__indent_level, run_count=run_count)
                else:
                    self.__future_jobs.append([iter_comp+1, None, self.__class__.__indent_level, run_count])
            else:
                self._run(comp=comp, kwargs=None, parallelize=False,
                          indent_level=self.__class__.__indent_level, run_count=run_count)

        self.__class__.__indent_level -= 1
        self._run_count += 1
        # Get run time of this pipeline
        if not self._parallelize:
            current_run_time_dict = self._self_run_timer[run_count]
            current_run_time_dict['end_datetime'] = datetime.now()
            self._self_run_timer[run_count] = current_run_time_dict
            # Else, we will get this in wait()

    @classmethod
    def summarize(cls, visualize_path=None):
        if visualize_path is not None:
            plotter = GraphPlot(visualize_path)
        else:
            plotter = None
        print('\n\nSUMMARY:')
        num_pipelines = len(cls._pipelines_run_order)
        current_pipeline_idxs = [0]
        current_pipeline_name = cls._pipelines_run_order[0]['name']
        current_indent_level = cls._pipelines_run_order[0]['indent_level']
        iter_pipelines = 1
        while iter_pipelines < num_pipelines:
            while cls._pipelines_run_order[iter_pipelines]['name'] == current_pipeline_name:
                current_pipeline_idxs.append(iter_pipelines)
                iter_pipelines += 1
                if iter_pipelines == num_pipelines:
                    break
            cls._summarize(idxs=current_pipeline_idxs, name=current_pipeline_name,
                           indent_level=current_indent_level, plotter=plotter)
            if iter_pipelines == num_pipelines:
                break
            # Reset
            current_pipeline_idxs = [iter_pipelines]
            current_pipeline_name = cls._pipelines_run_order[iter_pipelines]['name']
            current_indent_level = cls._pipelines_run_order[iter_pipelines]['indent_level']
            iter_pipelines += 1
            if iter_pipelines == num_pipelines:
                cls._summarize(idxs=current_pipeline_idxs, name=current_pipeline_name,
                               indent_level=current_indent_level, plotter=plotter)
        if visualize_path is not None:
            plotter.plot()

    @classmethod
    def _summarize(cls, idxs, name, indent_level, plotter=None):
        pipeline_run_times = []
        component_run_times = defaultdict(list)
        current_pipeline = cls._mapper_pipeline_name_obj[name]
        for idx in idxs:
            current_run_count = cls._pipelines_run_order[idx]['run_count']
            current_start_datetime = current_pipeline._self_run_timer[current_run_count]['start_datetime']
            current_end_datetime = current_pipeline._self_run_timer[current_run_count]['end_datetime']
            pipeline_run_times.append(current_end_datetime - current_start_datetime)
            # Now get the component runtimes for this run
            for k, v in current_pipeline._component_run_timer.items():
                component_run_times[k].append(v[current_run_count])
        # Now print the statistics
        print('{}Pipeline: {}'.format(get_tabs(indent_level), name))
        print('{}Total number of runs: {}. Runtime Statistics: Max = {}'\
               .format(get_tabs(indent_level), len(pipeline_run_times), max(pipeline_run_times)))
        print('{}                                             Min = {}'\
               .format(get_tabs(indent_level), min(pipeline_run_times)))
        print('{}                                             Avg = {}'\
               .format(get_tabs(indent_level), list_avg(pipeline_run_times)))

        print('{}Components:'.format(get_tabs(indent_level)))
        comp_names = [comp._name for comp in current_pipeline._Pipeline__component_list]
        for comp_name in comp_names:
            print('{}Component: {}'.format(get_tabs(indent_level), comp_name))
            current_com_runtimes = component_run_times[comp_name]
            print('{}Total number of runs: {}. Runtime Statistics: Max = {}'\
                   .format(get_tabs(indent_level), len(current_com_runtimes), max(current_com_runtimes)))
            print('{}                                             Min = {}'\
                   .format(get_tabs(indent_level), min(current_com_runtimes)))
            print('{}                                             Avg = {}'\
                   .format(get_tabs(indent_level), list_avg(current_com_runtimes)))
        print('\n')

        if plotter is not None:
            plotter.add_pipeline(dict(name=name, indent_level=indent_level,
                                      pipeline_run_times=pipeline_run_times,
                                      component_run_times=component_run_times,
                                      component_names=comp_names))

    def wait(self, future_jobs=True, async_jobs=True):
        # Complete any future jobs, and then wait for any pending parallel processes
        if future_jobs:
            num_pending = len(self.__future_jobs)
            for _ in range(num_pending):
                current_job = self.__future_jobs.popleft()
                self._run(comp=self.__component_list[current_job[0]], kwargs=current_job[1],
                          parallelize=False, indent_level=current_job[2], run_count=current_job[3])
                self._component_callback(comp_name=self.__component_list[current_job[0]]._name,
                                         run_count=current_job[3])

        if async_jobs:
            _ = [p.join() for p in self._process_list]

        # Reset
        self._process_list = []

    # Private methods
    def _component_callback(self, comp_name, run_count):
        # Helpful when running in parallelize=True mode. Tells when a component has finished processing
        # without having the parent regularly checking on it
        self._component_callback_lock.acquire()
        current_status_dict = self._component_run_status[run_count]
        del current_status_dict[comp_name]
        if len(current_status_dict) == 0:
            # All components for this run have finished executing
            current_run_time_dict = self._self_run_timer[run_count]
            current_run_time_dict['end_datetime'] = datetime.now()
            self._self_run_timer[run_count] = current_run_time_dict
            del self._component_run_status[run_count]
        else:
            self._component_run_status[run_count] = current_status_dict
        self._component_callback_lock.release()

    def _report_exception(self, comp, exc_info, pid, run_count):
        self._new_process_lock.acquire()  # This is not explicitly released. The script is set to terminate.
        print('\n')
        fprint(pid=pid, indent_level=0, run_count=run_count, comp_name=comp._name,
               msg='Exception reported')
        print(exc_info)
        print('Terminating the parent process and all children (if any)...')
        os.kill(self.__main_pid, signal.SIGTERM)

    def _run(self, comp, indent_level, run_count, kwargs=None, parallelize=False):
        if kwargs is None:
            kwargs = {}

        if parallelize:
            self._new_process_lock.acquire()
            comp_process = Process(target=comp._compute,
                                   kwargs=dict(pipeline=self, indent_level=indent_level,
                                               run_count=run_count, run_timer=self._component_run_timer,
                                               callback=self._component_callback, **kwargs))
            comp_process.daemon = True
            comp_process.start()
            self._process_list.append(comp_process)
            self._new_process_lock.release()
        else:
            comp._compute(pipeline=self, indent_level=indent_level, run_count=run_count,
                          run_timer=self._component_run_timer, **kwargs)

    def _preset(self):
        for comp in self.__component_list:
            comp._preset()

    def _reset(self):
        for comp in self.__component_list:
            comp._reset()


class FutureVariable:
    """To be used by the Component/Pipeline as a placeholder for a future variable."""
    def __init__(self, cls_object, mapper_name_variable):
        self.__cls_object = cls_object
        self.__mapper_name_variable = mapper_name_variable

    def __getattribute__(self, name):
        if name.startswith('__'):
            return super().__getattribute__(name)
        else:
            return _FutureEvaluator(super().__getattribute__('_FutureVariable__cls_object'),
                                    super().__getattribute__('_FutureVariable__mapper_name_variable'),
                                    name)


class _FutureEvaluator:
    """To be used by the Component/Pipeline as a placeholder for a future evaluator."""

    def __init__(self, cls_object, mapper_name_variable, name):
        self._cls_object = cls_object
        self.__mapper_name_variable = mapper_name_variable
        self.__name = name

    def __call__(self, idx=None, comp_name=None, debug=False, pid=None, indent_level=None, run_count=None):
        # idx is not None => Queuing
        if idx is None:
            return self.__mapper_name_variable[self.__name]
        else:
            out = self._cls_object._get_mapper_name_variable()[self.__name]['value'].get(idx, comp_name=comp_name,
                                                                                         var_name=self.__name, debug=debug,
                                                                                         pid=pid, indent_level=indent_level,
                                                                                         run_count=run_count)
            return out

    def _get_class(self):
        return self._cls_object

    def _get_var_name(self):
        return self.__name


class QueueDict:
    """A Queue like shared dictionary with a preset max-size, used by components running in parallel fashion.

    The dictionary is managed by a multiprocessing.Manager object, and as such, anything going in
    needs to be serializable. It mimicks multiprocessing.Queue, but adds features like `peek ith element`,
    `delete ith element` in O(1). Also implements `blocking` on insertion when the dict is full.

    """
    def __init__(self, q_dict, not_full, not_empty, maxsize=1):
        assert maxsize >= 1
        self.maxsize = maxsize
        self.dict = q_dict
        self.not_full = not_full
        self.not_empty = not_empty
        self.q_size = 0

    def delete(self, idx, var_name, status_dict, comp_name, debug=False, pid=None, indent_level=None, run_count=None):
        if debug:
            fprint(pid=pid, indent_level=indent_level, run_count=run_count, comp_name=comp_name,
                   msg='Requesting to delete {}...'.format(var_name))
        self.not_empty.acquire()
        while self.q_size == 0:
            self.not_empty.wait()
        while not idx in self.dict:
            # This should ideally not happen
            if debug:
                fprint(pid=pid, indent_level=indent_level, run_count=run_count, comp_name=comp_name,
                       msg='**Delete is BLOCKED for {}...**'.format(var_name))
                time.sleep(2)
            continue
        self.__delete(idx, status_dict)
        self.not_full.notify()
        self.not_empty.release()
        if debug:
            fprint(pid=pid, indent_level=indent_level, run_count=run_count, comp_name=comp_name,
                   msg='Successfully deleted {}. q_size now is: '.format(var_name, self.q_size))

    def get(self, idx, var_name, comp_name=None, debug=False, pid=None, indent_level=None, run_count=None):
        if debug:
            fprint(pid=pid, indent_level=indent_level, run_count=run_count, comp_name=comp_name,
                   msg='Requesting to get {}...'.format(var_name))
        """Returns the value at idx, but does not delete it."""
        while not idx in self.dict:
            if debug:
                fprint(pid=pid, indent_level=indent_level, run_count=run_count, comp_name=comp_name,
                       msg='**Get is BLOCKED for {}...**'.format(var_name))
                time.sleep(2)
            continue
        if debug:
            fprint(pid=pid, indent_level=indent_level, run_count=run_count, comp_name=comp_name,
                   msg='Successfully got {}. q_size now is: {}'.format(var_name, self.q_size))
        return self.dict[idx]

    def put(self, value, var_name, status_dict, comp_name, debug=False, pid=None, indent_level=None, run_count=None):
        if debug:
            fprint(pid=pid, indent_level=indent_level, run_count=run_count, comp_name=comp_name,
                   msg='Requesting to put {} to record...'.format(var_name))
        self.not_full.acquire()
        while self.q_size == self.maxsize:
            if debug:
                fprint(pid=pid, indent_level=indent_level, run_count=run_count, comp_name=comp_name,
                       msg='**Queue is BLOCKED to put {} to record...**'.format(var_name))
            self.not_full.wait()
        self._put(value, status_dict, debug)
        self.not_empty.notify()
        self.not_full.release()
        if debug:
            fprint(pid=pid, indent_level=indent_level, run_count=run_count, comp_name=comp_name,
                   msg='Successfully put {} to record. q_size now is: {}'.format(var_name, self.q_size))

    def __delete(self, key, status_dict):
        del self.dict[key]
        self.q_size -= 1
        # Also delete from `keys_list`
        list_index = status_dict.dict['keys_list'].index(key)
        temp_list = status_dict.dict['keys_list']
        del temp_list[list_index]
        status_dict.dict['keys_list'] = temp_list

    def _put(self, value, status_dict, debug=False):
        self.dict[status_dict.dict['next_counter']] = value

        temp_list = status_dict.dict['keys_list']
        temp_list.append(status_dict.dict['next_counter'])
        status_dict.dict['keys_list'] = temp_list

        status_dict.dict['next_counter'] += 1
        self.q_size += 1


class StatusDict:
    """A shared dictionary used by components running in parallel fashion.

    A component stores and shares some crucial status info about each of its to_record and runtime_args variables.
    The dictionary is managed by a multiprocessing.Manager object.

    """
    def __init__(self, q_dict, keys_list):
        self.dict = q_dict
        self.dict['next_counter'] = 0
        self.dict['overwrite'] = -1
        self.dict['keys_list'] = keys_list

    def increment(self, name, val):
        self.dict[name] += val


def sigterm_handler(signal, frame):
    sys.exit()


SyncManager.register('QueueDict', QueueDict)
SyncManager.register('StatusDict', StatusDict)
signal.signal(signal.SIGTERM, sigterm_handler)
