# from mcstasscript.interface.instr import McStas_instr as ScriptInstrument
# from mcstasscript.helper.mcstas_objects import Component as ScriptComponent


def ensure_user_var(instrument, dtype: str, name: str, description: str):
    types_names = [(uv.type, uv.name) for uv in instrument.user_var_list]
    if (dtype, name) in types_names:
        return
    try:
        instrument.add_user_var(dtype, name=name, comment=description)
    except NameError:
        if name in [n for _, n in types_names]:
            print(f"A USERVARS variable with name {name} but type different than {dtype} has already been define.")
        print(f"A variable named {name} exists but is not a USERVARS")


def declare_array(instrument, element_type: str, name: str, description: str, values):
    try:
        return instrument.add_declare_var(element_type, name=name, comment=description, array=len(values), value=values)
    except NameError:
        print(f"Failed to create a declare variable named {name} -- does it already exist?")


def ensure_parameter(instrument, data_type: str, name: str, description: str):
    types_names = [(p.type, p.name) for p in instrument.parameters]
    if (data_type, name) in types_names:
        pass
    try:
        instrument.add_parameter(data_type, name=name, comment=description)
    except NameError:
        print(f'Failed to create parameter named {name} -- does it exist already?')
