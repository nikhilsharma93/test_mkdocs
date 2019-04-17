class PipelineErros(Exception):
    supported_cases = ['add_variable']
    def __init__(self, var_name, pipeline_name, case, child_msg):
        if case == 'add_variable':
            msg = 'Error trying to add \"{}\" as a pipeline variable to the pipeline \"{}\"'\
                   .format(var_name, pipeline_name)
        super().__init__(msg + '\n' + child_msg)


class ComponentErrors(Exception):
    supported_cases = ['to_record', 'to_save', 'to_print', 'runtime_args']
    def __init__(self, var_name, case, child_msg, pipeline_name=None, comp_obj=None):
        if pipeline_name is None:
            msg = 'Error while running component \"{}\"'.format(comp_obj._name)
        else:
            msg = 'Error while running component \"{}\" of pipeline \"{}\"'.format(comp_obj._name, pipeline_name)
        super().__init__(msg + '\n' + child_msg)



class KeyExistingError(PipelineErros, ComponentErrors):
    def __init__(self, var_name, case, pipeline_name=None, comp_obj=None):
        if case == 'add_variable':
            msg = 'A variable by that name already exists in the pipeline and cannot be overwritten. ' + \
                  'Did you already set it somewhere else?'
        elif case == 'runtime_args':
            msg = 'Variable with value \"{}\" was provided as part of \"runtime_args\", '.format(var_name) + \
                  'but it cannot be accepted. You should only provide variables that have been recorded by some ' + \
                  'component, or added to some pipeline. In any case, access them via ' + \
                  '<component_name>.FutureVariable.<var_name> or <pipeline_name>.FutureVariable.<var_name>\n' + \
                  'Also make sure the variable name does not start with single or double underscores.'
        else:
            msg = ''
        if case in PipelineErros.supported_cases:
            PipelineErros.__init__(self, var_name=var_name, pipeline_name=pipeline_name, case=case, child_msg=msg)
        elif case in ComponentErrors.supported_cases:
            ComponentErrors.__init__(self, var_name=var_name, pipeline_name=pipeline_name, case=case,
                                     child_msg=msg, comp_obj=comp_obj)
        else:
            raise NotImplementedError("Case \"{}\" not supported for KeyNonExistentError".format(case))


class KeyNonExistentError(PipelineErros, ComponentErrors):
    def __init__(self, var_name, case, pipeline_name=None, comp_obj=None):
        if case == 'to_record':
            msg = 'Variable with name \"{}\" was requested as part of \"to_record\", '.format(var_name) + \
                  'but it does not exist in the component function.'
        elif case == 'to_save':
            msg = 'Variable with name \"{}\" was requested as part of \"to_save\", '.format(var_name) + \
                   'but it does not exist in the component function.'
        elif case == 'to_print':
            msg = 'Variable with name \"{}\" was requested as part of \"to_print\", '.format(var_name) + \
                  'but it does not exist in the component function.'
        elif case == 'runtime_args':
            msg = 'Variable with name \"{}\" was provided as part of \"runtime_args\", '.format(var_name) + \
                  'but it does not exist in this pipeline, and hence cannot be fed to the function.'
            msg += '\nDid you forget to add this variable as a pipeline variable?'
        elif case in ['share_from', 'share_with']:
            msg = 'A variable by that name does not exist in the former.'
        else:
            msg = ''
        if case in PipelineErros.supported_cases:
            PipelineErros.__init__(self, var_name=var_name, pipeline_name=pipeline_name, case=case, child_msg=msg)
        elif case in ComponentErrors.supported_cases:
            ComponentErrors.__init__(self, var_name=var_name, pipeline_name=pipeline_name, case=case,
                                     child_msg=msg, comp_obj=comp_obj)
        else:
            raise NotImplementedError("Case \"{}\" not supported for KeyNonExistentError".format(case))
