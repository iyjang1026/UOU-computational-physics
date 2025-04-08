is_simple_core = True
if is_simple_core:
	from cpnn.core_simple import Variable
	from cpnn.core_simple import Function
	from cpnn.core_simple import using_config
	from cpnn.core_simple import no_grad
	from cpnn.core_simple import as_array

else:
	from cpnn.core import Variable
	from cpnn.core import Function
	from cpnn.core import using_config
	from cpnn.core import no_grad
	from cpnn.core import as_array



