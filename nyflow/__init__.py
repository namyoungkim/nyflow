# =============================================================================
is_simple_core = True
# is_simple_core = False
# =============================================================================
if is_simple_core:
    from nyflow.core_simple import Variable
    from nyflow.core_simple import Function
    from nyflow.core_simple import using_config
    from nyflow.core_simple import no_grad
    from nyflow.core_simple import as_array
    from nyflow.core_simple import as_variable
    from nyflow.core_simple import setup_variable

else:
    pass

setup_variable()	