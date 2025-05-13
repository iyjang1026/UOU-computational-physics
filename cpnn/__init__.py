is_simple_core = False

if is_simple_core:
    from cpnn.core_simple import Variable
    from cpnn.core_simple import Function
    from cpnn.core_simple import using_config
    from cpnn.core_simple import no_grad
    from cpnn.core_simple import as_array
    from cpnn.core_simple import as_variable
    from cpnn.core_simple import setup_variable
else:
    from cpnn.core import Variable
    from cpnn.core import Function
    from cpnn.core import using_config
    from cpnn.core import no_grad
    from cpnn.core import as_array
    from cpnn.core import as_variable
    from cpnn.core import setup_variable

    import cpnn.functions
    import cpnn.cuda

setup_variable()
