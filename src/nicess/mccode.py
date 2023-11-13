from mccode_antlr.assembler import Assembler


def ensure_user_var(a: Assembler, dtype: str, name: str, description: str):
    a.ensure_user_var(f'{dtype} {name}; // {description}')


def declare_array(instrument: Assembler, element_type: str, name: str, description: str, values):
    instrument.declare_array(element_type, name, values)


def ensure_parameter(instrument: Assembler, data_type: str, name: str, description: str):
    instrument.parameter(f'{data_type} {name}; // {description}', ignore_repeated=True)
